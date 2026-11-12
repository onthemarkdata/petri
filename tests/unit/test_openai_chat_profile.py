"""Dish profile selection reaches the native wire without changing research gates."""

import json
import socket

import httpx
import pytest
import pytest_asyncio
import yaml
from pydantic_ai import capture_run_messages
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel

from petri.agents.config import AgentConfig
from petri.agents.contract import AgentName
from petri.agents.factory import AgentRunError, build_agents, run_agent, typed_role_instruction
from petri.agents.outputs import SourceCitation
from petri.agents.source_policy import source_hierarchy_instructions
from petri.config import load_config, load_dish_config
from petri.harness.resolve import resolve_model
from petri.research.web import RetrievedPage, SearchHit

PROFILE_KEY = "openai_chat_supports_multiple_system_messages"
URL = "https://example.org/primary"
CONSTITUTION = "Preserve attribution."
REQUIRED_INSTRUCTIONS = (
    "This request is the required investigation stage. Call web_search or web_fetch. "
    "Wait for the tool result before returning the final assessment. A failed retrieval "
    "attempt is not evidence and may justify an inconclusive result."
)
SYNTAX_INSTRUCTIONS = (
    "The previous final assessment was invalid JSON. Return one complete JSON "
    "object matching the schema. This is a final-output syntax retry; tools are "
    "unavailable for this request. Report missing source verification honestly. "
    "Evidence and verdict validation still apply."
)


class Sources:
    def __init__(self):
        self.calls = []

    async def search(self, query, *, limit=5):
        self.calls.append(("search", query))
        return [SearchHit(url=URL, title="Primary source")]

    async def fetch(self, url):
        self.calls.append(("fetch", url))
        return RetrievedPage(url=URL, title="Primary source", text="Measured finding.",
            retrieved_at="2026-11-01T00:00:00Z", content_sha256="a" * 64, truncated=False)


def final_message(verdict="EVIDENCE_SUFFICIENT"):
    return {"role": "assistant", "content": json.dumps({
        "verdict": verdict,
        "sources_cited": [{"url": URL, "hierarchy_level": 2, "finding": "Measured finding."}],
    })}


def tool_message(name):
    arguments = {"query": "primary evidence"} if name == "web_search" else {"url": URL}
    return {"role": "assistant", "tool_calls": [{"id": name, "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)}}]}


@pytest_asyncio.fixture
async def wire(monkeypatch):
    requests, script = [], []

    def no_network(*args, **kwargs):
        raise AssertionError("Only mocked HTTP is allowed in this test.")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket.socket, "connect_ex", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.invalid/prefix/engines/fixture/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-profile-key")

    def respond(request):
        requests.append(request)
        assert script, "Unexpected model request"
        message = script.pop(0)
        return httpx.Response(200, json={"id": "fixture", "object": "chat.completion", "created": 0,
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "model": "fixture:variant", "choices": [{"index": 0, "message": message,
            "finish_reason": "tool_calls" if "tool_calls" in message else "stop"}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        monkeypatch.setattr("pydantic_ai.providers._openai_compatible.create_async_httpx2_client", lambda: http)
        yield requests, script


def dish(tmp_path, *, value=False, inherited=False, output_mode="prompted"):
    harness = {"backend": "provider", "output_mode": output_mode}
    if value != "omitted":
        harness[PROFILE_KEY] = value
    values = {"harness": harness, "research": {"public_web": True}}
    if inherited:
        values["model"] = "openai-chat:fixture:variant"
    else:
        values["model"] = "openai:unused-responses"
        harness["model"] = "openai-chat:fixture:variant"
    path = tmp_path / "defaults" / "petri.yaml"
    path.parent.mkdir(exist_ok=True)
    path.write_text(yaml.safe_dump(values))
    return load_dish_config(tmp_path)


def payloads(requests):
    return [json.loads(request.content) for request in requests]


def leading_system(request):
    messages = request["messages"]
    result = []
    for message in messages:
        if message["role"] != "system":
            break
        result.append(message["content"])
    assert sum(message["role"] == "system" for message in messages) == len(result)
    return result


@pytest.mark.asyncio
@pytest.mark.parametrize("inherited", [False, True])
@pytest.mark.parametrize("value", [False, True, None, "omitted"])
async def test_dish_profile_preserves_environment_and_provider_capabilities(tmp_path, wire, value, inherited):
    requests, script = wire
    config = dish(tmp_path, value=value, inherited=inherited)
    assert config.harness.openai_chat_supports_multiple_system_messages is (None if value == "omitted" else value)
    resolved = resolve_model(config.model_dump(mode="json"))
    if value is None or value == "omitted":
        assert resolved == "openai-chat:fixture:variant"
        return
    assert isinstance(resolved, OpenAIChatModel)
    assert resolved.model_name == "fixture:variant"
    assert resolved._profile == {PROFILE_KEY: value}
    ordinary = OpenAIChatModel("fixture:variant", provider="openai")
    assert {k: v for k, v in resolved.profile.items() if k != PROFILE_KEY} == {
        k: v for k, v in ordinary.profile.items() if k != PROFILE_KEY}
    assert resolved.profile[PROFILE_KEY] is value
    assert resolved.system == ordinary.system == "openai"
    assert resolved.base_url == "https://gateway.invalid/prefix/engines/fixture/v1/"
    script.append(final_message("PIPELINE_COMPLETE"))
    agents = build_agents(config, constitution=CONSTITUTION, model=resolved)
    await run_agent(agents, AgentName.CELL_LEAD, "Complete the fixture.", config)
    assert str(requests[0].url) == "https://gateway.invalid/prefix/engines/fixture/v1/chat/completions"
    assert requests[0].headers["authorization"] == "Bearer synthetic-profile-key"
    assert payloads(requests)[0]["model"] == "fixture:variant"


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [False, True])
async def test_prompted_investigator_wire_and_syntax_retry_preserve_research_gates(tmp_path, wire, value):
    requests, script = wire
    config = dish(tmp_path, value=value)
    sources = Sources()
    malformed = '{"verdict":'
    script.extend([tool_message("web_search"), {"role": "assistant", "content": malformed},
                   final_message(), tool_message("web_fetch"), final_message()])
    agents = build_agents(config, constitution=CONSTITUTION, research_client=sources)
    agent = agents[AgentName.INVESTIGATOR]
    original = agent._instructions[0].instruction
    assert original.startswith(CONSTITUTION + "\n\n" + typed_role_instruction(config.agents[AgentName.INVESTIGATOR].instruction))
    assert source_hierarchy_instructions(config.source_hierarchy) in original
    result = await run_agent(agents, AgentName.INVESTIGATOR, "Assess the public fixture claim.", config)
    sent = payloads(requests)
    assert len(sent) == 5 and not script
    assert sources.calls == [("search", "primary evidence"), ("fetch", URL)]
    assert sent[0]["tool_choice"] == "required" and "response_format" not in sent[0]
    assert [message["role"] for message in sent[0]["messages"]] == (["system", "user"] if not value else ["system", "system", "user"])
    assert leading_system(sent[0]) == ([original + "\n\n" + REQUIRED_INSTRUCTIONS]
                                      if not value else [original, REQUIRED_INSTRUCTIONS])
    assert sent[1].get("tool_choice") != "required"
    for request in sent[1:]:
        assert '"verdict"' in "\n".join(leading_system(request))
        assert "sources_cited" in "\n".join(leading_system(request))
        if not value:
            assert len(leading_system(request)) == 1
            assert leading_system(request)[0].startswith(original + "\n\n")
    assert SYNTAX_INSTRUCTIONS in "\n".join(leading_system(sent[2]))
    assert sent[2]["response_format"] == {"type": "json_object"}
    assert not sent[2].get("tools") and "tool_choice" not in sent[2]
    assert any(message.get("content") == malformed for message in sent[2]["messages"])
    assert "json_invalid" in sent[2]["messages"][-1]["content"]
    assert "Cite only final URLs returned by a successful web_fetch in this run." in sent[3]["messages"][-1]["content"]
    assert {tool["function"]["name"] for tool in sent[3]["tools"]} == {"web_search", "web_fetch"}
    assert all("response_format" not in sent[i] for i in [0, 1, 3, 4])
    assert result.output.agent == AgentName.INVESTIGATOR
    assert result.output.verdict == "EVIDENCE_SUFFICIENT"
    assert isinstance(result.output.sources_cited[0], SourceCitation)
    assert result.output.sources_cited[0].url == URL


@pytest.mark.asyncio
async def test_profile_keeps_retrieval_required_before_parsing(tmp_path, wire):
    requests, script = wire
    config = dish(tmp_path)
    sources = Sources()
    malformed = '{"verdict":'
    script.extend([{"role": "assistant", "content": malformed}, tool_message("web_fetch"), final_message()])
    agents = build_agents(config, constitution=CONSTITUTION, research_client=sources)
    result = await run_agent(agents, AgentName.INVESTIGATOR, "Investigate the fixture.", config)
    sent = payloads(requests)
    assert len(sent) == 3
    assert sources.calls == [("fetch", URL)]
    for request in sent[:2]:
        assert request["tool_choice"] == "required" and "response_format" not in request
        assert leading_system(request)[0].endswith("\n\n" + REQUIRED_INSTRUCTIONS)
    assert "Call web_search or web_fetch before completing the investigation." in sent[1]["messages"][-1]["content"]
    assert "json_invalid" not in sent[1]["messages"][-1]["content"]
    assert any(message.get("content") == malformed for message in sent[1]["messages"])
    assert result.output.sources_cited[0].url == URL


@pytest.mark.asyncio
async def test_profile_keeps_syntax_retry_limit(tmp_path, wire):
    requests, script = wire
    config = dish(tmp_path)
    malformed = '{"verdict":'
    script.extend([tool_message("web_search"), *[{"role": "assistant", "content": malformed}] * 3])
    agents = build_agents(config, constitution=CONSTITUTION, research_client=Sources())
    with capture_run_messages() as messages, pytest.raises(AgentRunError, match=r"retries \(2\)"):
        await run_agent(agents, AgentName.INVESTIGATOR, "Investigate the fixture.", config)
    sent = payloads(requests)
    assert len(sent) == 4 and not script
    assert all(len(leading_system(request)) == 1 for request in sent)
    assert all(request["response_format"] == {"type": "json_object"} and not request.get("tools")
               for request in sent[2:])
    assert sum(getattr(part, "content", None) == malformed for message in messages for part in message.parts) == 3


@pytest.mark.asyncio
async def test_profile_keeps_tool_output_mode(tmp_path, wire):
    requests, script = wire
    config = dish(tmp_path, output_mode="tool")
    script.extend([tool_message("web_fetch"), {"role": "assistant", "tool_calls": [{
        "id": "final", "type": "function", "function": {
            "name": "final_result", "arguments": final_message()["content"],
        },
    }]}])
    agents = build_agents(config, constitution=CONSTITUTION, research_client=Sources())
    result = await run_agent(agents, AgentName.INVESTIGATOR, "Investigate the fixture.", config)
    sent = payloads(requests)
    assert len(sent) == 2 and not script
    assert sent[0]["tool_choice"] == "required"
    assert all("final_result" in {tool["function"]["name"] for tool in request["tools"]} for request in sent)
    assert all("response_format" not in request for request in sent)
    assert result.output.sources_cited[0].url == URL


@pytest.mark.asyncio
async def test_profile_keeps_terminal_source_policy(tmp_path, wire):
    requests, script = wire
    config = dish(tmp_path)
    weak = json.loads(final_message("EVIDENCE_CONFIRMS")["content"])
    weak["sources_cited"][0]["hierarchy_level"] = 6
    script.extend([tool_message("web_fetch"), {"role": "assistant", "content": json.dumps(weak)},
                   final_message("EVIDENCE_CONFIRMS")])
    agents = build_agents(config, constitution=CONSTITUTION, research_client=Sources())
    result = await run_agent(agents, AgentName.EVIDENCE_EVALUATOR, "Evaluate the fixture.", config)
    sent = payloads(requests)
    assert len(sent) == 3 and not script
    assert all(len(leading_system(request)) == 1 for request in sent)
    assert "source" in sent[2]["messages"][-1]["content"].lower()
    assert result.output.verdict == "EVIDENCE_CONFIRMS"
    assert result.output.sources_cited[0].hierarchy_level == 2


@pytest.mark.asyncio
async def test_per_agent_selection_stays_independent(tmp_path, wire):
    config = dish(tmp_path)
    raw = config.model_dump(mode="json")
    raw["agents"]["triage"]["model"] = "anthropic:independent:variant"
    config = AgentConfig.model_validate(raw)
    seen = []

    def record(values):
        selected = resolve_model(values)
        seen.append((values["harness"], selected))
        return selected

    agents = build_agents(config, constitution=CONSTITUTION, model_resolver=record, research_client=Sources())
    assert agents[AgentName.TRIAGE].model == "anthropic:independent:variant"
    assert isinstance(agents[AgentName.INVESTIGATOR].model, OpenAIChatModel)
    independent = [(harness, model) for harness, model in seen if isinstance(model, str)]
    assert independent == [({"backend": "provider", "model": "anthropic:independent:variant"},
                            "anthropic:independent:variant")]
    assert all(model.profile[PROFILE_KEY] is False for _, model in seen if isinstance(model, OpenAIChatModel))


@pytest.mark.asyncio
async def test_absent_option_preserves_responses_selection(wire):
    from pydantic_ai.models import infer_model

    raw = load_config().model_dump(mode="json")
    for prefix in ("openai", "openai-responses"):
        raw["harness"] = {"backend": "provider", "model": f"{prefix}:fixture:variant"}
        selected = resolve_model(raw)
        assert selected == f"{prefix}:fixture:variant"
        assert isinstance(infer_model(selected), OpenAIResponsesModel)
