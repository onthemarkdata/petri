"""Typed references reuse earlier premises instead of creating duplicate cells."""

import json

import pytest
from pydantic_ai import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from petri.harness.legacy_bridge import HarnessInferenceProvider
from petri.reasoning.decomposer import decompose_claim
from petri.reasoning.decomposition_agents import DecompositionError, DecompositionInput, initial_decomposer, run_decomposition_agent
from petri.reasoning.decomposition_settings import DecompositionSettings
from petri.storage import database, edge_registry


@pytest.fixture
def foundation(tmp_path):
    path = tmp_path / "dish.sqlite"
    database.put_cell(path, {"id": "dish-foundation-001-001", "colony": "dish-foundation", "level": 1,
                      "seq": 1, "claim_text": "Solar panels generate electricity", "created_at": "2026-11-01T00:00:00Z"})
    return path


def test_search_tool_then_reference_does_not_create_or_expand_a_duplicate(foundation):
    calls = []

    async def response(messages, info):
        calls.append(messages)
        assert any(tool.name == "search_cells" for tool in info.function_tools)
        if len(calls) == 1:
            return ModelResponse(parts=[ToolCallPart("search_cells", {"query": "Generation reduces grid imports"})])
        if len(calls) == 2:
            return ModelResponse(parts=[TextPart(json.dumps({"nodes": [{"claim_text": "Generation reduces grid imports"}]}))])
        if len(calls) == 3:
            return ModelResponse(parts=[ToolCallPart("search_cells", {"query": "Solar panels generate electricity"})])
        assert "dish-foundation-001-001" in str(messages)
        return ModelResponse(parts=[TextPart(json.dumps({"is_atomic": False, "sub_premises": [
            {"existing_cell_id": "dish-foundation-001-001", "reason": "The foundation already represents generation"},
        ]}))])

    with HarnessInferenceProvider(FunctionModel(response)) as runner:
        result = decompose_claim("Grid imports fall", [], "dish", "dependent", None, agent_runner=runner,
                                 db_path=foundation, settings=DecompositionSettings(depth=4, per_layer=10, model="test"))
    assert len(calls) == 4
    assert len(result.cells) == 2 and len(database.list_cells(foundation)) == 3
    references = [edge for edge in edge_registry.read(foundation) if edge.edge_type == "cross_colony"]
    assert len(references) == 1
    assert references[0].from_cell == "dish-dependent-001-001"
    assert references[0].to_cell == "dish-foundation-001-001"
    assert references[0].reason == "The foundation already represents generation"
    assert result.cells[1].dependencies == ["dish-foundation-001-001"]


@pytest.mark.asyncio
async def test_unknown_reference_retries_then_accepts_an_existing_id(foundation):
    calls = []

    async def response(messages, info):
        calls.append(messages)
        if len(calls) == 2:
            return ModelResponse(parts=[ToolCallPart("search_cells", {"query": "Solar panels generate electricity"})])
        identifier = "missing" if len(calls) == 1 else "dish-foundation-001-001"
        return ModelResponse(parts=[TextPart(json.dumps({"nodes": [
            {"existing_cell_id": identifier, "reason": "An existing premise"},
        ]}))])

    result = await run_decomposition_agent(initial_decomposer, DecompositionInput("Claim", db_path=foundation,
                                             colony_id="dish-dependent"), model=FunctionModel(response))
    assert len(calls) == 3
    assert result.nodes[0].existing_cell_id == "dish-foundation-001-001"


@pytest.mark.asyncio
async def test_unknown_reference_exhaustion_fails_loudly(foundation):
    calls = []

    async def response(messages, info):
        calls.append(messages)
        return ModelResponse(parts=[TextPart(json.dumps({"nodes": [
            {"existing_cell_id": "unknown", "reason": "A fabricated identifier"},
        ]}))])

    with pytest.raises(DecompositionError, match="initial_decomposer failed"):
        await run_decomposition_agent(initial_decomposer, DecompositionInput("Claim", db_path=foundation),
                                      model=FunctionModel(response))
    assert len(calls) == 3
    assert len(database.list_cells(foundation)) == 1


def test_exact_duplicate_requests_search_then_reference(foundation):
    calls = []

    async def response(messages, info):
        calls.append(messages)
        if len(calls) == 1:
            return ModelResponse(parts=[TextPart(json.dumps({"nodes": [{
                "claim_text": "SOLAR panels generate electricity!", "relation": "supports",
            }]}))])
        if len(calls) == 2:
            return ModelResponse(parts=[ToolCallPart("search_cells", {"query": "Solar panels generate electricity"})])
        return ModelResponse(parts=[TextPart(json.dumps({"nodes": [{
            "existing_cell_id": "dish-foundation-001-001", "reason": "An existing generation premise",
        }]}))])

    with HarnessInferenceProvider(FunctionModel(response)) as runner:
        result = decompose_claim("Electricity supply", [], "dish", "dependent", None, agent_runner=runner,
                                 db_path=foundation, settings=DecompositionSettings(depth=1, per_layer=10, model="test"))
    assert len(calls) == 3
    assert len(result.cells) == 1
    assert len(database.list_cells(foundation)) == 2
    assert result.edges[0].reason == "An existing generation premise"


def test_search_only_handle_does_not_materialize_pending_api_cells(foundation):
    calls = []

    async def response(messages, info):
        calls.append(messages)
        if len(calls) == 1:
            return ModelResponse(parts=[ToolCallPart("search_cells", {"query": "Solar panels generate electricity"})])
        return ModelResponse(parts=[TextPart(json.dumps({"nodes": [{
            "existing_cell_id": "dish-foundation-001-001", "reason": "Existing solar generation",
        }]}))])

    with HarnessInferenceProvider(FunctionModel(response)) as runner:
        result = decompose_claim("Electricity supply", [], "dish", "dependent", None, agent_runner=runner,
                                 search_db_path=foundation,
                                 settings=DecompositionSettings(depth=1, per_layer=10, model="test"))
    assert len(result.cells) == 1 and result.edges[0].reason == "Existing solar generation"
    assert len(database.list_cells(foundation)) == 1
    assert edge_registry.read(foundation) == []


@pytest.mark.asyncio
async def test_paraphrases_cannot_skip_search_when_earlier_colonies_exist(foundation):
    calls = []

    async def response(messages, info):
        calls.append(messages)
        if len(calls) == 2:
            return ModelResponse(parts=[ToolCallPart("search_cells", {"query": "Sunlight supports daytime power"})])
        return ModelResponse(parts=[TextPart(json.dumps({"nodes": [{
            "claim_text": "Sunlight supports daytime power", "relation": "supports",
        }]}))])

    result = await run_decomposition_agent(initial_decomposer,
                    DecompositionInput("Daytime imports decline", db_path=foundation, colony_id="dish-dependent"),
                    model=FunctionModel(response))
    assert len(calls) == 3
    assert result.nodes[0].claim_text == "Sunlight supports daytime power"


@pytest.mark.asyncio
async def test_search_is_required_only_until_the_first_tool_result(foundation):
    settings = []
    modes = []

    class RecordingModel(FunctionModel):
        async def request(self, messages, model_settings, parameters):
            modes.append(parameters.output_mode)
            return await super().request(messages, model_settings, parameters)

    async def response(messages, info):
        settings.append(info.model_settings)
        if len(settings) == 1:
            return ModelResponse(parts=[ToolCallPart("search_cells", {"query": "Solar panels generate electricity"})])
        return ModelResponse(parts=[TextPart(json.dumps({"nodes": [{
            "existing_cell_id": "dish-foundation-001-001", "reason": "The generation premise already exists",
        }]}))])

    await run_decomposition_agent(initial_decomposer,
          DecompositionInput("A dependent claim", db_path=foundation, colony_id="dish-dependent"),
          model=RecordingModel(response))
    assert modes == ["text", "prompted"]
    assert settings[0]["tool_choice"] == "required"
    assert not settings[1] or settings[1].get("tool_choice") != "required"


def test_named_initial_agent_executes_pi_search_envelope(foundation, fake_pi):
    from petri.harness.pi.model import PiModel
    from petri.harness.pi.transport import PiRpcTransport

    fake_pi.scenario(
        {"text": '{"tool_calls": [{"name": "search_cells", "arguments": {"query": "Solar panels generate electricity"}}]}'},
        {"text": '{"nodes": [{"existing_cell_id": "dish-foundation-001-001", "reason": "Existing generation"}]}'},
    )
    model = PiModel("fake", transport=PiRpcTransport(binary=str(fake_pi.path)))
    with HarnessInferenceProvider(model) as runner:
        result = runner.initial_decomposition(DecompositionInput("A dependent claim", db_path=foundation,
                                                                colony_id="dish-dependent"))
    assert result.nodes[0].existing_cell_id == "dish-foundation-001-001"
    assert model.round_trips == 2


@pytest.mark.asyncio
async def test_openai_wire_omits_json_mode_only_during_required_search(foundation):
    import httpx
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    requests = []

    def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        if len(requests) == 1:
            message = {"role": "assistant", "content": None, "tool_calls": [{
                "id": "search-call", "type": "function", "function": {
                    "name": "search_cells", "arguments": json.dumps({"query": "Solar panels generate electricity"}),
                },
            }]}
        else:
            message = {"role": "assistant", "content": json.dumps({"nodes": [{
                "existing_cell_id": "dish-foundation-001-001", "reason": "Existing solar generation",
            }]})}
        return httpx.Response(200, json={"id": "response", "object": "chat.completion", "created": 0,
                              "model": "local-test", "choices": [{"index": 0, "message": message,
                              "finish_reason": "tool_calls" if len(requests) == 1 else "stop"}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        model = OpenAIChatModel("local-test", provider=OpenAIProvider(base_url="http://local.invalid/v1",
                                api_key="test-only", http_client=client))
        result = await run_decomposition_agent(initial_decomposer,
                 DecompositionInput("A dependent claim", db_path=foundation, colony_id="dish-dependent"), model=model)
    assert result.nodes[0].existing_cell_id == "dish-foundation-001-001"
    assert requests[0]["tool_choice"] == "required"
    assert not requests[0].get("response_format")
    assert requests[1]["response_format"] == {"type": "json_object"}
