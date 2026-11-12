"""Bedrock stops and decomposition quality are enforced before cells are stored."""

from copy import deepcopy
import json

import pytest
from pydantic import ValidationError
from pydantic_ai import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.messages import ToolReturnPart, UserPromptPart
from pydantic_ai.models.function import FunctionModel

from petri.harness.legacy_bridge import HarnessInferenceProvider
from petri.models import Cell
from petri.reasoning.decomposer import decompose_claim
from petri.reasoning.decomposition_agents import (
    DecompositionError, DecompositionInput, five_whys, initial_instructions,
    parent_overlap, run_decomposition_agent, why_instructions,
)
from petri.reasoning.decomposition_schemas import PremiseCandidate, ReferencePremise, WhyExpansion
from petri.reasoning.decomposition_settings import DecompositionSettings
from petri.storage import database


def scripted_model(payloads):
    calls = []

    async def response(messages, info):
        calls.append(messages)
        return ModelResponse(parts=[TextPart(json.dumps(payloads[min(len(calls) - 1, len(payloads) - 1)]))])

    return FunctionModel(response), calls


def expansion(*children):
    return {"is_atomic": False, "sub_premises": list(children)}


def child(text, relation="supports"):
    return {"claim_text": text, "relation": relation}


@pytest.fixture
def reference_foundation(tmp_path, monkeypatch):
    monkeypatch.setattr("pydantic_ai.models.ALLOW_MODEL_REQUESTS", False)
    path = tmp_path / "dish.sqlite"
    claims = [
        "Solar panels generate electricity during daylight",
        "Battery storage loses energy during charging",
    ]
    references = []
    for seq, claim in enumerate(claims, 1):
        cell_id = f"dish-foundation-001-{seq:03d}"
        database.put_cell(path, {
            "id": cell_id, "colony": "dish-foundation", "level": 1, "seq": seq,
            "claim_text": claim, "created_at": "2026-11-01T00:00:00Z",
        })
        references.append({
            "existing_cell_id": cell_id,
            "reason": "The foundation records generation" if seq == 1 else "This is a counterargument about storage losses",
        })
    return path, claims, references


def searched_expansion_model(claims, first, corrected, *, new_claim=None):
    exchanges = {"search": [], "output": []}

    async def response(messages, info):
        assert any(tool.name == "search_cells" for tool in info.function_tools)
        snapshot = (deepcopy(messages), info.instructions)
        if not exchanges["search"]:
            exchanges["search"].append(snapshot)
            return ModelResponse(parts=[
                ToolCallPart("search_cells", {"query": claim}, tool_call_id=f"foundation-{seq}")
                for seq, claim in enumerate(claims, 1)
            ])
        if len(exchanges["output"]) == 1 and new_claim and len(exchanges["search"]) == 1:
            exchanges["search"].append(snapshot)
            return ModelResponse(parts=[ToolCallPart("search_cells", {"query": new_claim}, tool_call_id="counterargument")])
        exchanges["output"].append(snapshot)
        payload = first if len(exchanges["output"]) == 1 else corrected
        return ModelResponse(parts=[TextPart(json.dumps(payload))])

    return FunctionModel(response), exchanges


def assert_counterargument_shapes(text):
    assert "relation limits or rebuts only on a new PremiseCandidate" in text
    assert '{"claim_text": "<distinct testable claim>", "relation": "limits"}' in text
    assert '"rebuts"' in text
    assert "A ReferencePremise has only existing_cell_id and reason; do not add relation to it" in text
    assert "A reason describing a reference as a counterargument does not satisfy" in text
    assert "Retain valid references" in text
    assert "genuinely distinct, testable counterargument candidate" in text
    assert "Call search_cells for each proposed claim first" in text
    assert "reference existing matches instead of recreating them" in text
    assert "Do not invent a counterargument" in text
    assert "remove valid children to claim the singleton exemption" in text
    assert "mark a non-atomic parent atomic to evade this requirement" in text


def delivered_feedback(messages):
    return [
        part.content for message in messages for part in message.parts
        if isinstance(part, UserPromptPart) and isinstance(part.content, str)
        and "<retry_feedback>" in part.content
    ]


def delivered_counterargument_feedback(exchanges):
    messages, _ = exchanges["output"][1]
    feedback = delivered_feedback(messages)
    assert len(feedback) == 1
    assert isinstance(feedback[0], str)
    assert_counterargument_shapes(feedback[0])
    return feedback[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("relation", ["limits", "rebuts"])
async def test_reference_only_retry_accepts_distinct_counterargument_and_retains_references(reference_foundation, relation):
    path, claims, references = reference_foundation
    counterargument = child("Winter heating demand can exceed stored solar energy", relation)
    first = expansion(*references)
    WhyExpansion.model_validate(first)
    model, exchanges = searched_expansion_model(
        claims, first, expansion(*references, counterargument), new_claim=counterargument["claim_text"],
    )
    request = DecompositionInput("Solar storage reduces annual grid imports", db_path=path, colony_id="dish-dependent")
    result = await run_decomposition_agent(five_whys, request, model=model)

    assert len(exchanges["output"]) == 2
    assert len(exchanges["search"]) == 2
    assert request.search_performed
    assert {ref["existing_cell_id"] for ref in references} <= request.search_results
    messages, instructions = exchanges["output"][0]
    assert_counterargument_shapes(instructions)
    tool_returns = [part for message in messages for part in message.parts if isinstance(part, ToolReturnPart)]
    assert [part.tool_call_id for part in tool_returns] == ["foundation-1", "foundation-2"]
    for ref in references:
        assert ref["existing_cell_id"] in str(tool_returns)
    assert "Include the strongest counterargument" in delivered_counterargument_feedback(exchanges)
    assert not result.is_atomic
    assert all(isinstance(ref, ReferencePremise) for ref in result.sub_premises[:2])
    assert [ref.model_dump() for ref in result.sub_premises[:2]] == references
    assert all(set(ref.model_dump()) == {"existing_cell_id", "reason"} for ref in result.sub_premises[:2])
    assert isinstance(result.sub_premises[2], PremiseCandidate)
    assert result.sub_premises[2].model_dump() == counterargument
    assert len(result.sub_premises) == 3


@pytest.mark.asyncio
async def test_relation_on_reference_exhausts_the_single_output_retry(reference_foundation):
    path, claims, references = reference_foundation
    bad_correction = expansion(*[{**ref, "relation": "limits"} for ref in references])
    with pytest.raises(ValidationError) as schema_error:
        WhyExpansion.model_validate(bad_correction)
    assert [error["type"] for error in schema_error.value.errors()] == ["extra_forbidden", "extra_forbidden"]
    assert all(error["loc"][-1] == "relation" for error in schema_error.value.errors())
    model, exchanges = searched_expansion_model(claims, expansion(*references), bad_correction)
    request = DecompositionInput("Solar storage reduces annual grid imports", db_path=path, colony_id="dish-dependent")
    with pytest.raises(DecompositionError, match=r"retries \(1\)") as caught:
        await run_decomposition_agent(five_whys, request, model=model)
    assert "extra_forbidden" in str(caught.value.__cause__.__cause__)
    assert len(exchanges["output"]) == 2
    assert len(exchanges["search"]) == 1
    delivered_counterargument_feedback(exchanges)


@pytest.mark.asyncio
@pytest.mark.parametrize("required", [False, True])
async def test_reference_counterargument_reason_cannot_pass_typed_gate(reference_foundation, required):
    path, claims, references = reference_foundation
    assert "counterargument" in references[1]["reason"]
    first = expansion(*references)
    model, exchanges = searched_expansion_model(claims, first, first)
    request = DecompositionInput(
        "Solar storage reduces annual grid imports", db_path=path, colony_id="dish-dependent",
        require_counterargument=required,
    )
    expected = r"retries \(1\)" if required else "still lacks a counterargument"
    with pytest.raises(DecompositionError, match=expected):
        await run_decomposition_agent(five_whys, request, model=model)
    assert len(exchanges["output"]) == 2
    assert len(exchanges["search"]) == 1
    feedback = delivered_counterargument_feedback(exchanges)
    if required:
        assert "even for a singleton or atomic response" in feedback


@pytest.mark.asyncio
async def test_support_only_expansion_gets_one_retry_and_accepts_counterargument():
    supporting = expansion(child("Demand increases"), child("Costs fall"), child("Supply expands"))
    corrected = expansion(child("Demand increases"), child("Costs may erase the benefit", "limits"))
    model, calls = scripted_model([supporting, corrected])
    result = await run_decomposition_agent(five_whys, DecompositionInput("The market grows"), model=model)
    assert len(calls) == 2
    assert result.sub_premises[-1].relation == "limits"


@pytest.mark.asyncio
async def test_single_premise_does_not_require_counterargument():
    model, calls = scripted_model([expansion(child("Demand increases"))])
    await run_decomposition_agent(five_whys, DecompositionInput("The market grows"), model=model)
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_counterargument_still_missing_after_retry_is_an_error():
    model, calls = scripted_model([expansion(child("Demand increases"), child("Supply expands"))])
    with pytest.raises(DecompositionError, match="still lacks a counterargument"):
        await run_decomposition_agent(five_whys, DecompositionInput("The market grows"), model=model)
    assert len(calls) == 2


def test_relations_and_bedrock_reason_round_trip_through_typed_columns(tmp_path):
    model, calls = scripted_model([
        {"nodes": [child("A meter is a defined unit", "limits"), child("A second is a defined unit", "rebuts")]},
        {"is_atomic": True, "sub_premises": [], "bedrock_reason": "An SI unit definition"},
    ])
    path = tmp_path / "petri.sqlite"
    with HarnessInferenceProvider(model) as runner:
        result = decompose_claim("A physical claim", [], "dish", "colony", None, agent_runner=runner,
                                 db_path=path, settings=DecompositionSettings(depth=5, per_layer=10, model="test"))
    assert len(calls) == 3
    assert len(result.cells) == 3
    for cell in result.cells[1:]:
        row = database.get_cell(path, cell.id)
        restored = Cell(id=row["id"], colony_id=row["colony"], claim_text=row["claim_text"],
                        level=row["level"], relation=row["relation"], is_atomic=row["is_atomic"],
                        bedrock_reason=row["bedrock_reason"])
        assert restored.relation == cell.relation
        assert restored.is_atomic
        assert restored.bedrock_reason == "An SI unit definition"


def test_still_duplicate_is_dropped_and_warning_is_persisted(tmp_path):
    parent = "Solar generation reduces annual electricity costs"
    duplicate = child("SOLAR generation reduces annual electricity costs!", "limits")
    corrected = child("Nighttime imports can erase the savings", "rebuts")
    model, calls = scripted_model([
        {"nodes": [child(parent)]},
        expansion(duplicate, corrected),
        expansion(duplicate, corrected),
    ])
    path = tmp_path / "petri.sqlite"
    with HarnessInferenceProvider(model) as runner:
        result = decompose_claim("An energy claim", [], "dish", "colony", None, agent_runner=runner,
                                 db_path=path, settings=DecompositionSettings(depth=2, per_layer=10, model="test"))
    assert len(calls) == 3
    assert [cell.claim_text for cell in result.cells] == ["An energy claim", parent, corrected["claim_text"]]
    assert len(database.list_cells(path)) == 3
    events = database.read_events(path, event_type="decomposition_warning")
    assert len(events) == 1
    assert events[0]["cell_id"] == result.cells[1].id
    assert events[0]["data"]["claim_text"] == duplicate["claim_text"]
    assert events[0]["data"]["threshold"] == 0.8


@pytest.mark.asyncio
async def test_dropping_only_counterargument_cannot_leave_support_only_expansion():
    payload = expansion(child("parent premise", "rebuts"), child("Demand rises"), child("Supply falls"))
    model, calls = scripted_model([payload])
    with pytest.raises(DecompositionError, match="still lacks a counterargument"):
        await run_decomposition_agent(five_whys, DecompositionInput("parent premise"), model=model)
    assert len(calls) == 2


@pytest.mark.parametrize("relation", ["supports", "limits", "rebuts"])
def test_relation_is_a_closed_set(relation):
    assert PremiseCandidate(claim_text="Premise", relation=relation).relation == relation
    with pytest.raises(ValidationError):
        PremiseCandidate(claim_text="Premise", relation="agrees")
    with pytest.raises(ValidationError):
        Cell(id="dish-colony-001-001", colony_id="dish-colony", claim_text="Premise", level=1, relation="agrees")


def test_prompt_quality_bar_and_bedrock_first():
    initial = initial_instructions(DecompositionInput("claim"))
    for example in ("BAD CLAIM (obvious fact)", "BAD CLAIM (logical prerequisite)", "BAD CLAIM (too vague)"):
        assert example in initial
    assert "quantified thresholds" in initial and "exclusions" in initial
    why = why_instructions(DecompositionInput("claim"))
    assert why.startswith("FIRST — BEDROCK TEST.")
    assert "STOP: skip all brainstorming" in why
    assert '"is_atomic": true, "sub_premises": []' in why
    assert "strongest counterargument" in why and "relation limits or rebuts" in why
    assert_counterargument_shapes(why)
    assert "A single indispensable sub-premise is exempt" in why
    assert "Each new premise has only claim_text and relation" in initial
    assert "Each reference has only existing_cell_id and reason" in initial
    assert "Before proposing each premise, call search_cells" in initial


def test_parent_overlap_normalizes_case_and_punctuation():
    assert parent_overlap("A solar-panel output", "a SOLAR panel output!") == 1.0
    assert parent_overlap("six word parent claim text here", "six word parent claim text") >= 0.8
    assert parent_overlap("Different premise", "Other evidence") == 0


def test_bedrock_decision_precedes_children_in_output_schema():
    from petri.reasoning.decomposition_schemas import WhyExpansion

    assert list(WhyExpansion.model_json_schema()["properties"]) == [
        "is_atomic", "bedrock_reason", "sub_premises",
    ]


def test_flat_premise_errors_do_not_suggest_an_unrelated_reference_shape():
    from petri.reasoning.decomposition_schemas import WhyExpansion

    with pytest.raises(ValidationError) as caught:
        WhyExpansion.model_validate({"is_atomic": False, "sub_premises": [
            {"claim_text": "A testable premise", "relation": "supports", "sub_premises": []},
        ]})
    errors = caught.value.errors()
    assert len(errors) == 1 and errors[0]["type"] == "extra_forbidden"
    assert "existing_cell_id" not in str(caught.value)
    assert "Never put sub_premises" in why_instructions(DecompositionInput("claim"))


def test_why_guidance_is_explicit_and_empty_guidance_adds_no_section():
    assert "Guidance and trigger context:" not in why_instructions(DecompositionInput("claim"))
    assert "Guidance and trigger context:\nCRITICAL_FLAW_FOUND: revisit the causal assumption" in why_instructions(
        DecompositionInput("claim", guidance="CRITICAL_FLAW_FOUND: revisit the causal assumption"))


@pytest.mark.asyncio
@pytest.mark.parametrize("first", [
    {"is_atomic": True, "bedrock_reason": "An asserted definition", "sub_premises": []},
    expansion(child("A supporting mechanism")),
])
async def test_critical_flaw_requires_counterargument_even_for_singleton_or_atomic(first):
    model, calls = scripted_model([first, expansion(child("The mechanism can fail under overload", "limits"))])
    result = await run_decomposition_agent(five_whys, DecompositionInput("A causal claim", require_counterargument=True), model=model)
    assert len(calls) == 2 and result.sub_premises[0].relation == "limits"
    feedback = delivered_feedback(calls[1])
    assert len(feedback) == 1
    assert "even for a singleton or atomic response" in feedback[0]
    assert_counterargument_shapes(feedback[0])


@pytest.mark.asyncio
async def test_critical_flaw_counterargument_requirement_exhausts_loudly():
    model, calls = scripted_model([{"is_atomic": True, "bedrock_reason": "An asserted definition", "sub_premises": []}])
    with pytest.raises(DecompositionError, match="five_whys failed"):
        await run_decomposition_agent(five_whys, DecompositionInput("A causal claim", require_counterargument=True), model=model)
    assert len(calls) == 2
