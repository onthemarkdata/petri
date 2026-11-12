"""Typed references reuse earlier premises instead of creating duplicate cells."""

from copy import deepcopy
import json

import pytest
from pydantic_ai import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.messages import ToolReturnPart, UserPromptPart
from pydantic_ai.models.function import FunctionModel

from petri.harness.legacy_bridge import HarnessInferenceProvider
from petri.reasoning.decomposer import decompose_claim
from petri.reasoning.decomposition_agents import DecompositionError, DecompositionInput, five_whys, initial_decomposer, run_decomposition_agent
from petri.reasoning.decomposition_schemas import InitialDecomposition, PremiseCandidate, ReferencePremise
from petri.reasoning.decomposition_settings import DecompositionSettings
from petri.storage import database, edge_registry


@pytest.fixture
def foundation(tmp_path):
    path = tmp_path / "dish.sqlite"
    database.put_cell(path, {"id": "dish-foundation-001-001", "colony": "dish-foundation", "level": 1,
                      "seq": 1, "claim_text": "Solar panels generate electricity", "created_at": "2026-11-01T00:00:00Z"})
    return path


@pytest.fixture
def reference_pair(tmp_path, monkeypatch):
    monkeypatch.setattr("pydantic_ai.models.ALLOW_MODEL_REQUESTS", False)
    path = tmp_path / "dish.sqlite"
    cells = [
        {"id": "dish-foundation-000-000", "level": 0, "seq": 0,
         "claim_text": "Solar generation reduces daytime grid demand"},
        {"id": "dish-foundation-001-001", "level": 1, "seq": 1,
         "claim_text": "Rooftop panels convert sunlight into electricity"},
    ]
    for cell in cells:
        database.put_cell(path, {**cell, "colony": "dish-foundation", "created_at": "2026-11-01T00:00:00Z"})
    return path, cells


def search_calls(claims):
    return [ToolCallPart("search_cells", {"query": claim}, tool_call_id=f"lookup-{index}")
            for index, claim in enumerate(claims, 1)]


def reference_script(actions):
    exchanges = {"search": [], "output": []}
    count = 0

    async def response(messages, info):
        nonlocal count
        assert any(tool.name == "search_cells" for tool in info.function_tools)
        action = actions[min(count, len(actions) - 1)]
        count += 1
        if isinstance(action, dict):
            exchanges["output"].append(deepcopy(messages))
            return ModelResponse(parts=[TextPart(json.dumps(action))])
        exchanges["search"].append(deepcopy(messages))
        return ModelResponse(parts=action)

    return FunctionModel(response), exchanges


def retry_feedback(messages):
    return [part.content for message in messages for part in message.parts
            if isinstance(part, UserPromptPart) and isinstance(part.content, str)
            and "<retry_feedback>" in part.content]


def assert_duplicate_matches(feedback, cells, mechanism_ids=None):
    expected = [
        f"Candidate 1: claim_text={cells[0]['claim_text']!r}; matching existing_cell_ids={[cells[0]['id']]!r}.",
        f"Candidate 2: claim_text={cells[1]['claim_text']!r}; matching existing_cell_ids={mechanism_ids or [cells[1]['id']]!r}.",
    ]
    assert [line for line in feedback.splitlines() if line.startswith("Candidate ")] == expected
    assert "Retain every relevant selected premise by converting each matching premise to a ReferencePremise" in feedback
    assert "correct the whole list" in feedback
    assert "Do not retain irrelevant or redundant premises merely to meet a count" in feedback
    assert "Each ReferencePremise has only existing_cell_id and reason" in feedback
    assert "do not add claim_text or relation" in feedback
    assert "Call search_cells for each listed claim" in feedback
    assert "IDs in this feedback do not replace search results" in feedback


def test_all_duplicate_feedback_retains_root_and_mechanism_references_in_materialization(reference_pair):
    path, cells = reference_pair
    earlier_rows = database.list_cells(path)
    references = [
        {"existing_cell_id": cells[0]["id"], "reason": "The foundation covers the demand outcome"},
        {"existing_cell_id": cells[1]["id"], "reason": "The foundation covers the conversion mechanism"},
    ]
    model, exchanges = reference_script([
        search_calls([cell["claim_text"] for cell in cells]),
        {"nodes": [{"claim_text": cell["claim_text"], "relation": "supports"} for cell in cells]},
        {"nodes": references},
    ])
    with HarnessInferenceProvider(model) as runner:
        result = decompose_claim("Daytime imports decline", [], "dish", "dependent", None, agent_runner=runner,
                                 db_path=path, settings=DecompositionSettings(depth=4, per_layer=10, model="test"))

    assert len(exchanges["search"]) == 1
    assert len(exchanges["output"]) == 2
    messages = exchanges["output"][1]
    feedback = retry_feedback(messages)
    assert len(feedback) == 1
    assert_duplicate_matches(feedback[0], cells)
    tool_returns = [part for message in exchanges["output"][0] for part in message.parts
                    if isinstance(part, ToolReturnPart)]
    assert [part.tool_call_id for part in tool_returns] == ["lookup-1", "lookup-2"]
    assert all(cell["id"] in str(tool_returns) for cell in cells)
    journal = database.read_events(path, event_type="cell_expansion_completed")
    assert len(journal) == 1
    typed = InitialDecomposition.model_validate(journal[0]["data"]["output"])
    assert all(isinstance(node, ReferencePremise) for node in typed.nodes)
    assert [node.model_dump() for node in typed.nodes] == references
    assert all(set(node.model_dump()) == {"existing_cell_id", "reason"} for node in typed.nodes)
    assert [cell.id for cell in result.cells] == ["dish-dependent-000-000"]
    assert result.cells[0].dependencies == [cell["id"] for cell in cells]
    assert len(database.list_cells(path)) == len(earlier_rows) + 1
    assert database.list_cells(path, colony="dish-foundation") == earlier_rows
    edges = edge_registry.read(path)
    assert len(edges) == len(result.edges) == 2
    assert [(edge.from_cell, edge.to_cell, edge.edge_type, edge.reason) for edge in edges] == [
        (result.cells[0].id, reference["existing_cell_id"], "cross_colony", reference["reason"])
        for reference in references
    ]


@pytest.mark.asyncio
async def test_duplicate_feedback_lists_all_matching_ids_deterministically_until_initial_retries_exhaust(reference_pair):
    path, cells = reference_pair
    for seq in (3, 2):
        database.put_cell(path, {
            "id": f"dish-foundation-001-{seq:03d}", "colony": "dish-foundation", "level": 1, "seq": seq,
            "claim_text": cells[1]["claim_text"].upper() + "!", "created_at": "2026-11-01T00:00:00Z",
        })
    payload = {"nodes": [{"claim_text": cell["claim_text"]} for cell in cells]}
    model, exchanges = reference_script([search_calls([cell["claim_text"] for cell in cells]), payload])
    request = DecompositionInput("Daytime imports decline", db_path=path, colony_id="dish-dependent")
    with pytest.raises(DecompositionError, match=r"retries \(2\)"):
        await run_decomposition_agent(initial_decomposer, request, model=model)
    assert len(exchanges["search"]) == 1
    assert len(exchanges["output"]) == 3
    first_feedback = retry_feedback(exchanges["output"][1])
    final_feedback = retry_feedback(exchanges["output"][2])
    assert len(first_feedback) == 1 and len(final_feedback) == 2
    assert final_feedback == first_feedback * 2
    assert_duplicate_matches(first_feedback[0], cells, [
        "dish-foundation-001-001", "dish-foundation-001-002", "dish-foundation-001-003",
    ])
    assert len(database.list_cells(path)) == 4


@pytest.mark.asyncio
@pytest.mark.parametrize("reference_kind", ["unknown", "unsearched", "current-colony"])
async def test_duplicate_feedback_does_not_grant_reference_provenance(reference_pair, reference_kind):
    path, cells = reference_pair
    bad_id = {
        "unknown": "dish-missing-001-001",
        "unsearched": cells[1]["id"],
        "current-colony": "dish-dependent-001-001",
    }[reference_kind]
    if reference_kind == "current-colony":
        database.put_cell(path, {
            "id": bad_id, "colony": "dish-dependent", "level": 1, "seq": 1,
            "claim_text": "A local draft premise", "created_at": "2026-11-01T00:00:00Z",
        })
    model, exchanges = reference_script([
        search_calls([cells[0]["claim_text"]]),
        {"nodes": [{"claim_text": cell["claim_text"]} for cell in cells]},
        {"nodes": [{"existing_cell_id": bad_id, "reason": "A proposed reference"}]},
    ])
    request = DecompositionInput("Daytime imports decline", db_path=path, colony_id="dish-dependent")
    with pytest.raises(DecompositionError, match=r"retries \(2\)"):
        await run_decomposition_agent(initial_decomposer, request, model=model)
    assert len(exchanges["search"]) == 1
    assert len(exchanges["output"]) == 3
    assert request.search_results == {cells[0]["id"]}
    feedback = retry_feedback(exchanges["output"][1])
    assert len(feedback) == 1
    assert_duplicate_matches(feedback[0], cells)
    final_feedback = retry_feedback(exchanges["output"][2])
    assert len(final_feedback) == 2
    assert "Reference an existing cell returned by search_cells from an earlier colony" in final_feedback[-1]


@pytest.mark.parametrize("include_new", [False, True], ids=["root-only-singleton", "mixed-new-reference"])
def test_generic_root_reference_and_mixed_output_keep_materializing(reference_pair, include_new):
    path, cells = reference_pair
    reference = {"existing_cell_id": cells[0]["id"], "reason": "The foundation already covers the demand outcome"}
    new_claim = "Battery tariffs determine overnight operating costs"
    nodes = [reference] + ([{"claim_text": new_claim, "relation": "supports"}] if include_new else [])
    model, exchanges = reference_script([
        search_calls([cells[0]["claim_text"]] + ([new_claim] if include_new else [])),
        {"nodes": nodes},
        {"is_atomic": True, "sub_premises": [], "bedrock_reason": "A directly verifiable tariff measurement"},
    ])
    with HarnessInferenceProvider(model) as runner:
        result = decompose_claim("Daytime imports decline", [], "dish", "dependent", None, agent_runner=runner,
                                 db_path=path, settings=DecompositionSettings(depth=4, per_layer=10, model="test"))
    assert len(exchanges["search"]) == 1
    assert len(exchanges["output"]) == 1 + include_new
    assert all(retry_feedback(messages) == [] for messages in exchanges["output"])
    assert len(result.cells) == 1 + include_new
    assert len(database.list_cells(path)) == 3 + include_new
    references = [edge for edge in edge_registry.read(path) if edge.edge_type == "cross_colony"]
    assert len(references) == 1
    assert references[0].from_cell == "dish-dependent-000-000"
    assert references[0].to_cell == cells[0]["id"]
    assert references[0].reason == reference["reason"]
    if include_new:
        assert result.cells[1].claim_text == new_claim
        assert result.cells[1].is_atomic


@pytest.mark.asyncio
async def test_five_whys_duplicate_feedback_repairs_the_whole_list_in_one_retry(reference_pair):
    path, cells = reference_pair
    counterargument = {"claim_text": "Winter heating demand can exceed stored solar energy", "relation": "limits"}
    references = [{"existing_cell_id": cell["id"], "reason": f"Existing selected premise {index}"}
                  for index, cell in enumerate(cells, 1)]
    model, exchanges = reference_script([
        search_calls([cell["claim_text"] for cell in cells] + [counterargument["claim_text"]]),
        {"is_atomic": False, "sub_premises": [{"claim_text": cell["claim_text"]} for cell in cells] + [counterargument]},
        {"is_atomic": False, "sub_premises": references + [counterargument]},
    ])
    request = DecompositionInput("Storage reduces annual grid imports", db_path=path, colony_id="dish-dependent")
    result = await run_decomposition_agent(five_whys, request, model=model)
    assert len(exchanges["search"]) == 1
    assert len(exchanges["output"]) == 2
    feedback = retry_feedback(exchanges["output"][1])
    assert len(feedback) == 1
    assert_duplicate_matches(feedback[0], cells)
    assert [reference.model_dump() for reference in result.sub_premises[:2]] == references
    assert isinstance(result.sub_premises[2], PremiseCandidate)
    assert result.sub_premises[2].model_dump() == counterargument


@pytest.mark.asyncio
async def test_same_tokens_in_a_different_order_are_not_exact_duplicates(reference_pair):
    path, cells = reference_pair
    reordered = "Daytime grid demand reduces solar generation"
    model, exchanges = reference_script([
        search_calls([reordered]), {"nodes": [{"claim_text": reordered}]},
    ])
    result = await run_decomposition_agent(initial_decomposer,
             DecompositionInput("Daytime imports decline", db_path=path, colony_id="dish-dependent"), model=model)
    assert len(exchanges["output"]) == len(exchanges["search"]) == 1
    assert result.nodes[0].claim_text == reordered
    assert retry_feedback(exchanges["output"][0]) == []


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
