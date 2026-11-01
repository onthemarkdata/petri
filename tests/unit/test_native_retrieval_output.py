"""Native prompted roles can search, fetch, and validate a strict final result."""

import json
import re
from pathlib import Path

import httpx
import pytest
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from petri.agents.config import AgentConfig, load_agent_config
from petri.agents.contract import AgentName
from petri.agents.factory import build_agents
from petri.research.web import RetrievedPage, SearchHit


@pytest.mark.asyncio
@pytest.mark.parametrize("name", [name for name in AgentName
                                 if name not in {AgentName.DECOMPOSITION_LEAD, AgentName.CELL_LEAD}])
async def test_native_role_searches_then_fetches_before_strict_final_output(name):
    raw = load_agent_config(Path(__file__).parents[2] / "petri/defaults/petri.yaml").model_dump(mode="json")
    raw["harness"] = {"backend": "provider", "model": "local-test", "output_mode": "prompted"}
    raw["research"] = {"public_web": True}
    raw["model_settings"] = {"max_tokens": 321}
    config = AgentConfig.model_validate(raw)
    requests, calls = [], []
    url = "https://example.org/primary"

    class Sources:
        async def search(self, query, *, limit=5):
            calls.append(("search", query))
            return [SearchHit(url=url, title="Primary source")]

        async def fetch(self, source_url):
            calls.append(("fetch", source_url))
            return RetrievedPage(url=url, title="Primary source", text="Measured finding.",
                retrieved_at="2026-11-01T00:00:00Z", content_sha256="a" * 64, truncated=False)

    def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        if len(requests) <= 2:
            tool = "web_search" if len(requests) == 1 else "web_fetch"
            arguments = {"query": "primary evidence"} if len(requests) == 1 else {"url": url}
            message = {"role": "assistant", "tool_calls": [{"id": tool, "type": "function",
                "function": {"name": tool, "arguments": json.dumps(arguments)}}]}
        else:
            message = {"role": "assistant", "content": json.dumps({
                "verdict": "NOT_ALLOWED" if len(requests) == 3 else config.agents[name].verdicts_pass[0],
                "sources_cited": [{"url": url, "hierarchy_level": 2, "finding": "Measured finding."}],
            })}
        return httpx.Response(200, json={"id": "fixture", "object": "chat.completion", "created": 0,
            "model": "local-test", "choices": [{"index": 0, "message": message,
            "finish_reason": "tool_calls" if "tool_calls" in message else "stop"}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        model = OpenAIChatModel("local-test", provider=OpenAIProvider(
            base_url="http://local.invalid/v1", api_key="local-test", http_client=http))
        agent = build_agents(config, constitution="Preserve attribution.", model=model,
                             research_client=Sources())[name]
        result = await agent.run("Assess the public fixture claim.")
    assert calls == [("search", "primary evidence"), ("fetch", url)]
    assert len(requests) == 4
    assert all("response_format" not in request and request["max_completion_tokens"] == 321 for request in requests)
    assert all({tool["function"]["name"] for tool in request["tools"]} == {"web_search", "web_fetch"}
               for request in requests)
    instructions = "\n".join(message["content"] for message in requests[2]["messages"]
                             if isinstance(message.get("content"), str))
    assert re.search(r'"required"\s*:\s*\[\s*"verdict"\s*\]', instructions)
    assert '"title": "SourceCitation"' not in instructions
    assert result.output.verdict == config.agents[name].verdicts_pass[0]
    assert result.output.sources_cited[0].url == url


def _syntax_retry_fixture():
    raw = load_agent_config(Path(__file__).parents[2] / "petri/defaults/petri.yaml").model_dump(mode="json")
    raw["harness"] = {"backend": "provider", "model": "local-test", "output_mode": "prompted"}
    raw["research"] = {"public_web": True}
    calls = []
    url = "https://example.org/primary"

    class Sources:
        async def search(self, query, *, limit=5):
            calls.append(("search", query))
            return [SearchHit(url=url, title="Primary source")]

        async def fetch(self, source_url):
            calls.append(("fetch", source_url))
            return RetrievedPage(url=url, title="Primary source", text="Measured finding.",
                retrieved_at="2026-11-01T00:00:00Z", content_sha256="a" * 64, truncated=False)

    return raw, Sources(), calls, url


@pytest.mark.asyncio
@pytest.mark.parametrize("supports_json", [False, True])
@pytest.mark.parametrize("name", [name for name in AgentName
                                 if name not in {AgentName.DECOMPOSITION_LEAD, AgentName.CELL_LEAD}])
async def test_syntax_retry_is_constrained_then_citation_feedback_restores_tools(name, supports_json):
    raw, sources, calls, url = _syntax_retry_fixture()
    config = AgentConfig.model_validate(raw)
    requests = []
    malformed = '{"verdict":'
    final = json.dumps({"verdict": config.agents[name].verdicts_pass[0],
        "sources_cited": [{"url": url, "hierarchy_level": 2, "finding": "Measured finding."}]})

    def respond(request):
        requests.append(json.loads(request.content))
        step = len(requests)
        if step in {1, 4}:
            tool = "web_search" if step == 1 else "web_fetch"
            arguments = {"query": "primary evidence"} if step == 1 else {"url": url}
            message = {"role": "assistant", "tool_calls": [{"id": tool, "type": "function",
                "function": {"name": tool, "arguments": json.dumps(arguments)}}]}
        else:
            message = {"role": "assistant", "content": malformed if step == 2 else final}
        return httpx.Response(200, json={"id": "fixture", "object": "chat.completion", "created": 0,
            "model": "local-test", "choices": [{"index": 0, "message": message,
            "finish_reason": "tool_calls" if "tool_calls" in message else "stop"}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        model = OpenAIChatModel("local-test", profile={"supports_json_object_output": supports_json},
            provider=OpenAIProvider(base_url="http://local.invalid/v1", api_key="local-test", http_client=http))
        agent = build_agents(config, constitution="Preserve attribution.", model=model, research_client=sources)[name]
        result = await agent.run("Assess the public fixture claim.")
    assert len(requests) == 5
    assert calls == [("search", "primary evidence"), ("fetch", url)]
    if supports_json:
        assert requests[2]["response_format"] == {"type": "json_object"}
        assert not requests[2].get("tools") and "tool_choice" not in requests[2]
    else:
        assert "response_format" not in requests[2] and len(requests[2]["tools"]) == 2
    assert all("response_format" not in request and len(request["tools"]) == 2
               for request in [*requests[:2], *requests[3:]])
    assert any(message.get("content") == malformed for message in requests[2]["messages"])
    assert "json_invalid" in requests[2]["messages"][-1]["content"]
    assert "web_fetch" in requests[3]["messages"][-1]["content"]
    assert result.output.sources_cited[0].url == url


@pytest.mark.asyncio
async def test_constrained_syntax_retries_keep_the_output_retry_budget():
    from pydantic_ai import UnexpectedModelBehavior, capture_run_messages

    raw, sources, _, _ = _syntax_retry_fixture()
    requests = []
    malformed = '{"verdict":'

    def respond(request):
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            message = {"role": "assistant", "tool_calls": [{"id": "search", "type": "function",
                "function": {"name": "web_search", "arguments": '{"query":"evidence"}'}}]}
        else:
            message = {"role": "assistant", "content": malformed}
        return httpx.Response(200, json={"id": "fixture", "object": "chat.completion", "created": 0,
            "model": "local-test", "choices": [{"index": 0, "message": message,
            "finish_reason": "tool_calls" if "tool_calls" in message else "stop"}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        model = OpenAIChatModel("local-test", provider=OpenAIProvider(
            base_url="http://local.invalid/v1", api_key="local-test", http_client=http))
        agent = build_agents(raw, constitution="Preserve attribution.", model=model,
                             research_client=sources)[AgentName.INVESTIGATOR]
        with capture_run_messages() as messages, pytest.raises(UnexpectedModelBehavior, match="retries \\(2\\)"):
            await agent.run("Investigate the fixture claim.")
    assert len(requests) == 4
    assert all(request["response_format"] == {"type": "json_object"} and not request.get("tools")
               for request in requests[2:])
    assert sum(getattr(part, "content", None) == malformed
               for message in messages for part in message.parts) == 3


@pytest.mark.asyncio
async def test_pi_keeps_host_tools_available_after_syntax_and_citation_feedback(fake_pi):
    from petri.harness.pi.model import PiModel
    from petri.harness.pi.transport import PiRpcTransport

    raw, sources, calls, url = _syntax_retry_fixture()
    raw["harness"]["backend"] = "pi"
    final = json.dumps({"verdict": "EVIDENCE_SUFFICIENT",
        "sources_cited": [{"url": url, "hierarchy_level": 2, "finding": "Measured finding."}]})
    fake_pi.scenario({"text": json.dumps({"web_search": [{"query": "primary evidence"}]})},
        {"text": '{"verdict":'}, {"text": final},
        {"text": json.dumps({"web_fetch": [{"url": url}]})}, {"text": final})
    async with PiModel("local-test", transport=PiRpcTransport(binary=str(fake_pi.path))) as model:
        agent = build_agents(raw, constitution="Preserve attribution.", model=model,
                             research_client=sources)[AgentName.INVESTIGATOR]
        result = await agent.run("Investigate the fixture claim.")
        assert model.round_trips == 5
    assert calls == [("search", "primary evidence"), ("fetch", url)]
    assert result.output.sources_cited[0].url == url


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["provider", "pi"])
@pytest.mark.parametrize("tool", ["web_search", "web_fetch"])
async def test_invalid_tool_arguments_do_not_replace_an_executed_retrieval(fake_pi, monkeypatch, backend, tool):
    from contextlib import AsyncExitStack

    from pydantic_ai.messages import ToolReturnPart

    from petri.harness.pi.model import PiModel
    from petri.harness.pi.transport import PiRpcTransport
    from petri.research.web import PublicWebClient

    raw, _, _, url = _syntax_retry_fixture()
    raw["harness"]["backend"] = backend
    final = json.dumps({"verdict": "NEEDS_MORE_EVIDENCE", "sources_cited": [],
                        "summary": "Retrieval failed; no source was verified."})
    arguments = {"query": "primary evidence"} if tool == "web_search" else {"url": url}
    requests, retrievals = [], []

    def unavailable(request):
        retrievals.append(request)
        return httpx.Response(503)

    async def public(host):
        return ["93.184.216.34"]

    def respond(request):
        requests.append(json.loads(request.content))
        step = len(requests)
        if step in {1, 3}:
            message = {"role": "assistant", "tool_calls": [{"id": f"call-{step}", "type": "function",
                "function": {"name": tool, "arguments": json.dumps({} if step == 1 else arguments)}}]}
        else:
            message = {"role": "assistant", "content": final}
        return httpx.Response(200, json={"id": "fixture", "object": "chat.completion", "created": 0,
            "model": "local-test", "choices": [{"index": 0, "message": message,
            "finish_reason": "tool_calls" if "tool_calls" in message else "stop"}]})

    async with AsyncExitStack() as stack:
        if backend == "pi":
            fake_pi.scenario({"text": json.dumps({tool: [{}]})}, {"text": final},
                             {"text": json.dumps({tool: [arguments]})}, {"text": final})
            transport = PiRpcTransport(binary=str(fake_pi.path))
            original = transport.request

            async def record(request):
                if request["type"] == "prompt":
                    requests.append(request)
                return await original(request)

            monkeypatch.setattr(transport, "request", record)
            model = await stack.enter_async_context(PiModel("local-test", transport=transport))
        else:
            http = await stack.enter_async_context(httpx.AsyncClient(transport=httpx.MockTransport(respond)))
            model = OpenAIChatModel("local-test", provider=OpenAIProvider(
                base_url="http://local.invalid/v1", api_key="local-test", http_client=http))
        agent = build_agents(raw, constitution="Preserve attribution.", model=model,
            research_client=PublicWebClient(transport=httpx.MockTransport(unavailable), resolve_host=public))[
                AgentName.INVESTIGATOR]
        result = await agent.run("Investigate the fixture claim.")

    assert len(requests) == 4
    assert len(retrievals) == 1
    returns = [part for message in result.all_messages() for part in message.parts
               if isinstance(part, ToolReturnPart) and part.tool_name == tool]
    assert [part.outcome for part in returns] == ["retried", "success"]
    assert "HTTPStatusError" in returns[-1].content["error"]
    assert result.output.verdict == "NEEDS_MORE_EVIDENCE"
    assert result.output.summary == "Retrieval failed; no source was verified."
    assert not result.output.sources_cited
    feedback = "Call web_search or web_fetch before completing the investigation"
    assert feedback in str(requests[2])
    if backend == "provider":
        assert all(request.get("tool_choice") == "required" for request in requests[:3])
        assert requests[-1].get("tool_choice") != "required"
        assert all("response_format" not in request for request in requests)
    else:
        required = "The next response must be a host function-call envelope"
        assert all(required in request["message"] for request in requests[:3])
        assert required not in requests[-1]["message"]
