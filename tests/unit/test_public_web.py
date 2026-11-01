"""Public retrieval is explicit, bounded, and testable without a network."""

import asyncio
import json
import re
from contextlib import AsyncExitStack
from pathlib import Path

import httpx
import pytest
from pydantic_ai.messages import ModelResponse, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from petri.agents.config import AgentConfig, load_agent_config
from petri.agents.contract import AgentName
from petri.agents.factory import build_agents, run_agent
from petri.research.tools import ResearchContext
from petri.research.web import MAX_BYTES, MAX_TEXT, PublicWebClient, PublicWebError
from petri.storage import database
from petri.harness.pi.model import PiModel
from petri.harness.pi.transport import PiRpcTransport

DEFAULTS = Path(__file__).parents[2] / "petri/defaults/petri.yaml"


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["provider", "pi"])
@pytest.mark.parametrize("enabled", [False, True])
async def test_investigator_attempts_retrieval_before_honest_inconclusive_output(fake_pi, backend, enabled, monkeypatch):
    raw = load_agent_config(DEFAULTS).model_dump(mode="json")
    raw["research"] = {"public_web": enabled}
    raw["harness"] = {"backend": backend, "model": "local-test", "output_mode": "prompted"}
    config = AgentConfig.model_validate(raw)
    final = {"verdict": "NEEDS_MORE_EVIDENCE", "sources_cited": []}
    model_requests = []
    attempts = []

    def respond(request):
        payload = json.loads(request.content)
        model_requests.append(payload)
        message = {"role": "assistant", "content": json.dumps(final)}
        if len(model_requests) == 2:
            message = {"role": "assistant", "tool_calls": [{"id": "search", "type": "function", "function": {
                "name": "web_search", "arguments": json.dumps({"query": "fixture evidence"})}}]}
        return httpx.Response(200, json={"id": "local-response", "object": "chat.completion", "created": 0,
            "model": "local-test", "choices": [{"index": 0, "message": message,
            "finish_reason": "tool_calls" if "tool_calls" in message else "stop"}]})

    def unavailable(request):
        attempts.append(request)
        return httpx.Response(503)

    async with AsyncExitStack() as stack:
        if backend == "pi":
            fake_pi.scenario({"text": json.dumps(final)},
                             {"text": json.dumps({"web_search": [{"query": "fixture evidence"}]})},
                             {"text": json.dumps(final)})
            transport = PiRpcTransport(binary=str(fake_pi.path))
            original = transport.request

            async def record(request):
                if request["type"] == "prompt":
                    model_requests.append(request)
                return await original(request)

            monkeypatch.setattr(transport, "request", record)
            model = await stack.enter_async_context(PiModel("local-test", transport=transport))
        else:
            http = await stack.enter_async_context(httpx.AsyncClient(transport=httpx.MockTransport(respond)))
            model = OpenAIChatModel("local-test", provider=OpenAIProvider(
                base_url="http://local.invalid/v1", api_key="local-test", http_client=http))
        agent = build_agents(config, constitution="Preserve evidence.", model=model,
                             research_client=client(unavailable))[AgentName.INVESTIGATOR]
        result = await agent.run("Investigate the fixture claim.")
        assert (model.round_trips if backend == "pi" else len(model_requests)) == (3 if enabled else 1)
    assert len(attempts) == (1 if enabled else 0)
    assert result.output.verdict == "NEEDS_MORE_EVIDENCE" and not result.output.sources_cited
    if enabled and backend == "provider":
        assert all(request.get("tool_choice") == "required" and "response_format" not in request
                   for request in model_requests[:2])
        assert model_requests[2].get("tool_choice") != "required"
        assert "response_format" not in model_requests[2]
        instructions = "\n".join(message["content"] for message in model_requests[2]["messages"]
                                 if isinstance(message.get("content"), str))
        assert re.search(r'"required"\s*:\s*\[\s*"verdict"\s*\]', instructions)
    elif enabled:
        for request in model_requests[:2]:
            prompt = request["message"]
            assert "The next response must be a host function-call envelope" in prompt
            assert '"name": "web_search"' in prompt and '"name": "web_fetch"' in prompt
            assert '"required": ["verdict"]' not in prompt
        assert "The next response must be a host function-call envelope" not in model_requests[2]["message"]
        assert '"required": ["verdict"]' in model_requests[2]["message"]


async def public(host):
    return ["93.184.216.34"]


def client(handler):
    return PublicWebClient(transport=httpx.MockTransport(handler), resolve_host=public)


@pytest.mark.asyncio
async def test_search_and_fetch_parse_real_transport_responses_without_scripts():
    requests = []

    def handle(request):
        requests.append(str(request.url))
        if request.url.host == "html.duckduckgo.com":
            return httpx.Response(
                200,
                headers={"content-type": "text/html"},
                text="""
            <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Fpaper">Research <b>paper</b></a>
            <a class="result__a" href="https://example.org/paper">Duplicate</a>""",
            )
        return httpx.Response(
            200,
            headers={"content-type": "text/html"},
            text="""<html><title>Paper title</title>
        <script>secret script</script><style>hidden CSS</style><p>Measured evidence &amp; results.</p></html>""",
        )

    web = client(handle)
    hits = await web.search("measured evidence", limit=1)
    assert len(hits) == 1 and hits[0].url == "https://example.org/paper"
    page = await web.fetch(hits[0].url)
    assert page.title == "Paper title" and page.text == "Measured evidence & results."
    assert page.untrusted_evidence and len(page.content_sha256) == 64
    assert len(requests) == 2


@pytest.mark.asyncio
async def test_fetch_limits_bytes_and_text_and_rejects_nontext():
    page = await client(
        lambda request: httpx.Response(
            200, headers={"content-type": "text/plain"}, content=b"x" * (MAX_BYTES + 1)
        )
    ).fetch("https://example.org/large")
    assert page.truncated and len(page.text) == MAX_TEXT
    with pytest.raises(PublicWebError, match="HTML, plain text, or JSON"):
        await client(
            lambda request: httpx.Response(
                200, headers={"content-type": "application/pdf"}, content=b"%PDF"
            )
        ).fetch("https://example.org/file")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "https://user:password@example.org",
        "http://example.org:8080",
        "http://example.org:bad",
    ],
)
async def test_nonpublic_url_forms_are_rejected_before_http(url):
    requests = []
    web = client(lambda request: requests.append(request))
    with pytest.raises(PublicWebError):
        await web.fetch(url)
    assert requests == []


@pytest.mark.asyncio
async def test_private_addresses_and_redirects_are_rejected():
    requests = []

    async def resolve(host):
        return ["127.0.0.1"] if host == "localhost" else ["93.184.216.34"]

    def handle(request):
        requests.append(str(request.url))
        return httpx.Response(302, headers={"location": "http://localhost/internal"})

    web = PublicWebClient(transport=httpx.MockTransport(handle), resolve_host=resolve)
    with pytest.raises(PublicWebError, match="public IP"):
        await web.fetch("http://localhost/internal")
    assert requests == []
    with pytest.raises(PublicWebError, match="public IP"):
        await web.fetch("https://example.org/redirect")
    assert requests == ["https://example.org/redirect"]


@pytest.mark.asyncio
async def test_total_retrieval_deadline_is_bounded(monkeypatch):
    monkeypatch.setattr("petri.research.web.MAX_SECONDS", 0.01)

    async def slow(request):
        await asyncio.sleep(0.1)
        return httpx.Response(200, headers={"content-type": "text/plain"}, text="Late")

    with pytest.raises(PublicWebError, match="time limit"):
        await client(slow).fetch("https://example.org/slow")


def test_opt_in_registration_covers_evidence_roles_and_makes_no_requests():
    calls = []
    web = client(lambda request: calls.append(request))
    config = load_agent_config(DEFAULTS)
    assert not config.research.public_web
    agents = build_agents(
        config, constitution="Evidence first.", model=TestModel(), research_client=web
    )
    assert all(not agent._function_toolset.tools for agent in agents.values())
    raw = config.model_dump()
    raw["research"] = {"public_web": True}
    config = AgentConfig.model_validate(raw)
    agents = build_agents(
        config, constitution="Evidence first.", model=TestModel(), research_client=web
    )
    for name, agent in agents.items():
        assert set(agent._function_toolset.tools) == (
            {"web_search", "web_fetch"} if name not in {AgentName.DECOMPOSITION_LEAD, AgentName.CELL_LEAD} else set()
        )
    assert calls == []


@pytest.mark.asyncio
async def test_real_agent_tool_loop_records_retrieved_source_provenance_once(tmp_path):
    config = load_agent_config(DEFAULTS)
    raw = config.model_dump()
    raw["research"] = {"public_web": True}
    config = AgentConfig.model_validate(raw)
    requests = []

    def handle(request):
        requests.append(request)
        if request.url.host == "html.duckduckgo.com":
            return httpx.Response(
                200,
                headers={"content-type": "text/html"},
                text='<a class="result__a" href="https://example.org/paper">Actual fixture source</a>',
            )
        return httpx.Response(
            200,
            headers={"content-type": "text/html"},
            text="<title>Fixture source</title><p>Observed fixture evidence.</p>",
        )

    def respond(messages, info):
        returns = [
            part
            for message in messages
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        if not returns:
            return ModelResponse(parts=[ToolCallPart("web_search", {"query": "fixture evidence"})])
        if len(returns) == 1:
            assert returns[-1].content["results"][0]["url"] == "https://example.org/paper"
            return ModelResponse(
                parts=[ToolCallPart("web_fetch", {"url": "https://example.org/paper"})]
            )
        assert returns[-1].content["text"] == "Observed fixture evidence."
        assert returns[-1].content["untrusted_evidence"] is True
        return ModelResponse(
            parts=[
                ToolCallPart(
                    info.output_tools[0].name,
                    {
                        "verdict": "EVIDENCE_SUFFICIENT",
                        "sources_cited": [
                            {
                                "url": "https://example.org/paper",
                                "title": "Fixture source",
                                "hierarchy_level": 2,
                            }
                        ],
                    },
                )
            ]
        )

    agents = build_agents(
        config,
        constitution="Evidence first.",
        model=FunctionModel(respond),
        output_mode="tool",
        research_client=client(handle),
    )
    context = ResearchContext(
        tmp_path / "petri.sqlite", "dish-alpha-001-001", "investigator", 1, 1, "research"
    )
    for _ in range(2):
        result = await run_agent(
            agents, AgentName.INVESTIGATOR, "Assess the fixture claim.", config, deps=context
        )
        assert result.output.sources_cited[0].url == "https://example.org/paper"
    events = database.read_events(context.db_path)
    assert len(events) == 2 and len(requests) == 4
    search, source = events
    assert search["type"] == "search_executed" and search["data"]["source_urls"] == [
        "https://example.org/paper"
    ]
    assert source["type"] == "source_reviewed"
    assert source["data"]["finding"] == "Observed fixture evidence."
    assert (
        source["data"]["retrieval_method"] == "public-web"
        and len(source["data"]["content_sha256"]) == 64
    )
    assert source["data"]["retrieved_at"]


@pytest.mark.asyncio
async def test_retrieval_failure_returns_an_error_without_source_events(tmp_path):
    config = load_agent_config(DEFAULTS)
    raw = config.model_dump()
    raw["research"] = {"public_web": True}
    config = AgentConfig.model_validate(raw)

    def respond(messages, info):
        returns = [
            part
            for message in messages
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        if not returns:
            return ModelResponse(
                parts=[ToolCallPart("web_fetch", {"url": "https://example.org/unavailable"})]
            )
        assert "error" in returns[-1].content and returns[-1].content["untrusted_evidence"]
        return ModelResponse(
            parts=[
                ToolCallPart(
                    info.output_tools[0].name, {"verdict": "CANNOT_DETERMINE", "sources_cited": []}
                )
            ]
        )

    agents = build_agents(
        config,
        constitution="Evidence first.",
        model=FunctionModel(respond),
        output_mode="tool",
        research_client=client(lambda request: httpx.Response(503)),
    )
    context = ResearchContext(
        tmp_path / "petri.sqlite", "dish-alpha-001-001", "investigator", 1, 1, "research"
    )
    result = await run_agent(
        agents, AgentName.INVESTIGATOR, "Assess the fixture.", config, deps=context
    )
    assert result.output.sources_cited == [] and not context.db_path.exists()


@pytest.mark.asyncio
async def test_opt_in_output_rejects_a_citation_that_was_never_retrieved():
    from petri.agents.factory import AgentRunError

    config = load_agent_config(DEFAULTS)
    raw = config.model_dump()
    raw["research"] = {"public_web": True}
    config = AgentConfig.model_validate(raw)
    requests = []

    def respond(messages, info):
        return ModelResponse(
            parts=[
                ToolCallPart(
                    info.output_tools[0].name,
                    {
                        "verdict": "EVIDENCE_SUFFICIENT",
                        "sources_cited": [
                            {"url": "https://example.org/invented", "hierarchy_level": 2}
                        ],
                    },
                )
            ]
        )

    agents = build_agents(
        config,
        constitution="Evidence first.",
        model=FunctionModel(respond),
        output_mode="tool",
        retries=1,
        research_client=client(lambda request: requests.append(request)),
    )
    with pytest.raises(AgentRunError, match="retries"):
        await run_agent(agents, AgentName.INVESTIGATOR, "Assess the fixture.", config)
    assert requests == []
