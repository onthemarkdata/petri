"""Proposals reserve a bounded budget and publish deferred cells atomically."""

from contextlib import closing

import pytest

from petri.lifecycle.config import LifecycleConfig
from petri.lifecycle.proposals import get_proposal, journal_output, materialize_proposal, reserve_proposal
from petri.lifecycle.triggers import RedecompositionTrigger
from petri.models import CellStatus, validate_event_data
from petri.reasoning.decomposition_schemas import WhyExpansion
from petri.storage import database
from petri.storage.colonies import load_colony, save_colony


def trigger(cell_id, generation=0):
    return RedecompositionTrigger(cell_id=cell_id, trigger_verdict="EVIDENCE_INCONCLUSIVE",
        agent="evidence_evaluator", fire_point="settled_round", generation=generation)


def setup(tmp_path, canonical_colony):
    path = tmp_path / "petri.sqlite"
    save_colony(path, canonical_colony["graph"], canonical_colony["colony_model"])
    return path, canonical_colony["cell1"].id


def output():
    return WhyExpansion.model_validate({"is_atomic": False, "sub_premises": [
        {"claim_text": "The protocol gives the expected measurement", "relation": "supports"},
        {"claim_text": "Instrument drift can reverse the result", "relation": "limits"},
        {"claim_text": "The sample excludes a known confounder", "relation": "supports"},
    ]})


def test_proposal_is_deferred_truncated_and_replay_safe(tmp_path, canonical_colony):
    path, parent = setup(tmp_path, canonical_colony)
    proposal = reserve_proposal(path, trigger(parent), LifecycleConfig(per_parent_cap=2))
    journal_output(path, proposal.id, output())
    materialized = materialize_proposal(path, proposal.id)
    assert materialized.state == "proposed" and materialized.capacity == 2
    assert materialized.reason == "truncated"
    for cell_id in materialized.child_ids:
        cell = database.get_cell(path, cell_id)
        assert cell["status"] == CellStatus.DEFER_CLOSED.value
        assert cell["level"] == 3
    assert database.get_cell(path, parent)["status"] == "NEW"
    graph, _ = load_colony(path, canonical_colony["colony_id"])
    assert set(materialized.child_ids) <= set(graph.get_dependencies(parent))
    events = database.read_events(path, event_type="redecomposition_proposed")
    assert len(events) == 1
    assert validate_event_data(events[0]["type"], events[0]["data"])
    assert reserve_proposal(path, trigger(parent), LifecycleConfig()) == materialized
    assert materialize_proposal(path, proposal.id) == materialized
    assert database.read_events(path, event_type="redecomposition_proposed") == events


@pytest.mark.parametrize("guard", ["rounds", "bedrock", "budget"])
def test_guards_halt_before_inference(tmp_path, canonical_colony, guard):
    path, parent = setup(tmp_path, canonical_colony)
    if guard == "bedrock":
        with closing(database.connect(path)) as conn:
            conn.execute("UPDATE cells SET is_atomic=1 WHERE id=?", (parent,))
            conn.commit()
    config = LifecycleConfig(max_rounds_per_cell=0 if guard == "rounds" else 2,
        dish_proposal_budget=0 if guard == "budget" else 100)
    proposal = reserve_proposal(path, trigger(parent), config)
    assert proposal.state == "halted" and proposal.reason
    assert proposal.capacity == 0
    assert len(database.read_events(path, event_type="redecomposition_halted")) == 1


def test_reservations_share_budget_and_atomic_commit_rolls_back(tmp_path, canonical_colony, monkeypatch):
    path, parent = setup(tmp_path, canonical_colony)
    config = LifecycleConfig(per_parent_cap=4, dish_proposal_budget=5)
    first = reserve_proposal(path, trigger(parent), config)
    second = reserve_proposal(path, trigger(canonical_colony["cell2"].id), config)
    assert first.capacity == 4 and second.capacity == 1
    journal_output(path, first.id, output())
    before = database.list_cells(path), database.read_edges(path)
    from petri.lifecycle import proposals
    original = proposals._event
    monkeypatch.setattr(proposals, "_event", lambda *args: (_ for _ in ()).throw(RuntimeError("injected failure")))
    with pytest.raises(RuntimeError, match="injected"):
        materialize_proposal(path, first.id)
    assert (database.list_cells(path), database.read_edges(path)) == before
    assert get_proposal(path, first.id).state == "reserved"
    monkeypatch.setattr(proposals, "_event", original)
    assert materialize_proposal(path, first.id).state == "proposed"


def test_flaw_truncation_keeps_counterargument(tmp_path, canonical_colony):
    path, parent = setup(tmp_path, canonical_colony)
    flaw = trigger(parent).model_copy(update={"trigger_verdict": "CRITICAL_FLAW_FOUND", "agent": "skeptic"})
    proposal = reserve_proposal(path, flaw, LifecycleConfig(per_parent_cap=1))
    journal_output(path, proposal.id, output())
    result = materialize_proposal(path, proposal.id)
    assert database.get_cell(path, result.child_ids[0])["relation"] == "limits"


@pytest.mark.asyncio
async def test_function_model_proposal_retries_flaw_and_checkpoints_only_paths(tmp_path, canonical_colony):
    import json
    import pickle
    from dbos import DBOS
    from pydantic_ai.messages import ModelResponse, TextPart
    from pydantic_ai.models.function import FunctionModel
    from petri.execution.dbos_app import CHECKPOINT_LIMIT
    from petri.execution.dbos_backend import DBOSExecutionBackend
    from petri.harness.legacy_bridge import HarnessInferenceProvider
    from petri.lifecycle.redecompose import RedecomposeSpec, propose
    from tests.pipeline_stubs import StubPhases

    path = tmp_path / "petri-dishes/test-dish/petri.sqlite"
    save_colony(path, canonical_colony["graph"], canonical_colony["colony_model"])
    parent = canonical_colony["cell1"].id
    calls = []
    evidence = "Supplied measurement notes: " + "x" * 10000
    (tmp_path / "evidence.txt").write_text(evidence)

    def respond(messages, info):
        calls.append(messages)
        relation = "supports" if len(calls) == 1 else "rebuts"
        return ModelResponse(parts=[TextPart(json.dumps({"is_atomic": False, "sub_premises": [
            {"claim_text": "A calibration error could explain the discrepancy", "relation": relation},
        ]}))])

    runner = HarnessInferenceProvider(FunctionModel(respond))
    backend = DBOSExecutionBackend(config={"execution": {"backend": "dbos"}})
    await backend.start_async(tmp_path, db_path=path, dependencies=lambda _: StubPhases().dependencies(),
        seed_runner_factory=lambda _: runner)
    spec = RedecomposeSpec(trigger=trigger(parent).model_copy(update={
        "trigger_verdict": "CRITICAL_FLAW_FOUND", "agent": "skeptic", "flaw_text": "Calibration is uncertain",
    }), evidence_paths=("evidence.txt",))
    try:
        result = await propose(spec)
        assert result.state == "proposed"
        assert await propose(spec) == result
        assert len(calls) == 2
        assert database.get_cell(path, result.child_ids[0])["relation"] == "rebuts"
        steps = await DBOS.list_workflow_steps_async(spec.workflow_id)
        assert {step["function_name"] for step in steps} == {
            "petri.reserve_redecomposition", "petri.redecompose_why", "petri.materialize_redecomposition",
        }
        assert all(len(pickle.dumps(step["output"])) < CHECKPOINT_LIMIT for step in steps)
        assert len(pickle.dumps(spec)) < CHECKPOINT_LIMIT
        assert evidence.encode() not in pickle.dumps(spec)
        assert len(await DBOS.list_workflows_async()) == 1
        assert len(database.read_events(path, event_type="redecomposition_proposed")) == 1
    finally:
        await backend.close_async()
        runner.close()
