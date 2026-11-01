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
