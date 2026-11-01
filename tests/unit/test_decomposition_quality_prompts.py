"""Bedrock stops and decomposition quality are enforced before cells are stored."""

import json

import pytest
from pydantic import ValidationError
from pydantic_ai import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel

from petri.harness.legacy_bridge import HarnessInferenceProvider
from petri.models import Cell
from petri.reasoning.decomposer import decompose_claim
from petri.reasoning.decomposition_agents import (
    DecompositionError, DecompositionInput, five_whys, initial_instructions,
    parent_overlap, run_decomposition_agent, why_instructions,
)
from petri.reasoning.decomposition_schemas import PremiseCandidate
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


@pytest.mark.asyncio
async def test_critical_flaw_counterargument_requirement_exhausts_loudly():
    model, calls = scripted_model([{"is_atomic": True, "bedrock_reason": "An asserted definition", "sub_premises": []}])
    with pytest.raises(DecompositionError, match="five_whys failed"):
        await run_decomposition_agent(five_whys, DecompositionInput("A causal claim", require_counterargument=True), model=model)
    assert len(calls) == 2
