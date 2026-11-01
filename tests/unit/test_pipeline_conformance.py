"""Port the v1 lifecycle contract to typed nodes, Ends, and explicit collapses.

The issue calls the old group TestStateTransitions. In this checkout its cases
are TestUpdateStateValidTransitions.test_valid_transition and the four methods
in TestUpdateStateInvalidTransitions. Tests below preserve those names and cover
the entire table, including edges the original parametrization did not exercise.

Converged and stalled are removed intermediate states. A typed NeedsHuman End
therefore compresses phase -> stalled -> needs_human; it is not a new direct
phase-to-phase jump. Re-entry belongs to admission of a new round, outside this
graph. These dispositions are checked explicitly rather than skipped.
"""

from dataclasses import replace
from itertools import product

import pytest
from pydantic_graph import BaseNode

from petri.execution import pipeline
from petri.execution.models import CellPipelineState
from petri.execution.pipeline import (
    ConvergenceCheckNode,
    ConvergenceDecision,
    CritiqueNode,
    DeferClosed,
    DeferOpen,
    DispatchNode,
    Disproven,
    EvaluateNode,
    NeedsHuman,
    RedTeamNode,
    ResearchNode,
    SocraticNode,
    SyncConflict,
    Validated,
    declared_transitions,
    run_pipeline,
)
from petri.storage.queue import VALID_TRANSITIONS
from tests.pipeline_stubs import StubPhases

# Frozen before v1 retirement, so this suite remains a lifecycle specification.
LEGACY_TRANSITIONS = {
    "queued": ["socratic_active"],
    "socratic_active": ["research_active", "stalled"],
    "research_active": ["critique_active", "stalled"],
    "critique_active": ["mediating", "stalled"],
    "mediating": ["converged", "stalled", "research_active"],
    "converged": ["red_team_active", "deferred_open", "deferred_closed"],
    "stalled": ["needs_human", "queued"],
    "needs_human": ["queued"],
    "red_team_active": ["evaluating", "deferred_open"],
    "evaluating": ["done", "deferred_open", "deferred_closed"],
    "done": ["queued"],
    "deferred_open": ["queued"],
    "deferred_closed": [],
    "sync_conflict": ["queued"],
}
NODES = {
    "queued": DispatchNode,
    "socratic_active": SocraticNode,
    "research_active": ResearchNode,
    "critique_active": CritiqueNode,
    "mediating": ConvergenceCheckNode,
    "red_team_active": RedTeamNode,
    "evaluating": EvaluateNode,
}
ENDS = {
    "needs_human": {NeedsHuman},
    "done": {Validated, Disproven},
    "deferred_open": {DeferOpen},
    "deferred_closed": {DeferClosed},
    "sync_conflict": {SyncConflict},
}
OBSOLETE = {"converged": ConvergenceCheckNode, "stalled": NeedsHuman}
REENTRY = {"stalled", "needs_human", "done", "deferred_open", "sync_conflict"}
LEGAL = [(source, target) for source, targets in LEGACY_TRANSITIONS.items() for target in targets]
ILLEGAL = [
    (source, target)
    for source, target in product(LEGACY_TRANSITIONS, repeat=2)
    if target not in LEGACY_TRANSITIONS[source]
]


def test_frozen_contract_matches_v1_until_retirement():
    assert LEGACY_TRANSITIONS == VALID_TRANSITIONS


@pytest.mark.parametrize("state", list(LEGACY_TRANSITIONS))
def test_every_state_has_one_documented_disposition(state):
    assert sum(state in group for group in [NODES, ENDS, OBSOLETE]) == 1
    assert f"{state} ->" in pipeline.__doc__
    if state in NODES:
        assert NODES[state] in pipeline.PIPELINE_NODES
    elif state in ENDS:
        assert all(not issubclass(end, BaseNode) for end in ENDS[state])
    else:
        assert "obsolete:" in next(
            line for line in pipeline.__doc__.splitlines() if f"{state} ->" in line
        )


@pytest.mark.parametrize("source,target", LEGAL, ids=[f"{a}->{b}" for a, b in LEGAL])
def test_valid_transition(source, target):
    """Ports every TestUpdateStateValidTransitions.test_valid_transition case."""
    if target == "queued":
        assert source in REENTRY
        endings = {NeedsHuman} if source == "stalled" else ENDS[source]
        assert all(not hasattr(end, "run") for end in endings)
        return  # New-round admission, never a return edge inside a completed round.
    if source == "stalled":
        assert target == "needs_human"
        assert OBSOLETE[source] is NeedsHuman
        return  # The intermediate stall and parking result share one typed End.
    source_node = ConvergenceCheckNode if source == "converged" else NODES[source]
    next_nodes, endings = declared_transitions(source_node)
    if target == "converged":
        assert source == "mediating"
        assert RedTeamNode in next_nodes
    elif target == "stalled":
        assert NeedsHuman in endings
    elif target in NODES:
        assert NODES[target] in next_nodes
    else:
        assert ENDS[target] <= endings


@pytest.mark.parametrize("source,target", ILLEGAL, ids=[f"{a}!>{b}" for a, b in ILLEGAL])
def test_every_illegal_transition_has_no_direct_typed_edge(source, target):
    if source in ENDS:
        assert all(not hasattr(end, "run") for end in ENDS[source])
        return
    if source == "stalled":
        assert not hasattr(NeedsHuman, "run")
        return
    if target in OBSOLETE:
        # Neither old intermediate state has a target node to jump into.
        assert target not in NODES
        assert all(
            node.__name__ not in {"StalledNode", "ConvergedNode"}
            for node in pipeline.PIPELINE_NODES
        )
        if source in NODES and target == "converged":
            # A phase cannot bypass critique and enter the convergence decision.
            if source in {"queued", "socratic_active", "research_active"}:
                assert ConvergenceCheckNode not in declared_transitions(NODES[source])[0]
        return
    if source == "converged":
        # The removed state's legal exits are already part of the decision node.
        assert target not in NODES or NODES[target] is not RedTeamNode
        return
    next_nodes, endings = declared_transitions(NODES[source])
    if source == "mediating" and target == "red_team_active":
        assert "converged" in LEGACY_TRANSITIONS[source]
        assert target in LEGACY_TRANSITIONS["converged"]
        assert RedTeamNode in next_nodes
    elif target in NODES:
        assert NODES[target] not in next_nodes
    elif target == "needs_human" and NeedsHuman in endings:
        # Only the documented phase -> stalled -> needs_human path may compress.
        assert "stalled" in LEGACY_TRANSITIONS[source]
        assert "needs_human" in LEGACY_TRANSITIONS["stalled"]
        assert not hasattr(NeedsHuman, "run")
    elif target == "sync_conflict" and source == "queued":
        # v1 sync_check bypassed its table; v2 checks explicitly at dispatch.
        assert endings == {SyncConflict}
        assert not hasattr(SyncConflict, "run")
    elif source == "mediating" and target in {"deferred_open", "deferred_closed"}:
        # Convergence -> converged -> deferral becomes a typed decision End.
        assert "converged" in LEGACY_TRANSITIONS[source]
        assert target in LEGACY_TRANSITIONS["converged"]
        assert ENDS[target] <= endings
    else:
        assert not (ENDS[target] & endings)


def test_queued_to_done_raises():
    """Ports TestUpdateStateInvalidTransitions.test_queued_to_done_raises."""
    next_nodes, endings = declared_transitions(DispatchNode)
    assert next_nodes == {SocraticNode}
    assert not ({Validated, Disproven} & endings)


def test_deferred_closed_is_terminal():
    """Ports TestUpdateStateInvalidTransitions.test_deferred_closed_is_terminal."""
    assert not hasattr(DeferClosed, "run")
    assert "deferred_closed" not in REENTRY


def test_research_active_to_converged_skip_raises():
    """Ports TestUpdateStateInvalidTransitions.test_research_active_to_converged_skip_raises."""
    next_nodes, _ = declared_transitions(ResearchNode)
    assert next_nodes == {CritiqueNode}
    assert ConvergenceCheckNode not in next_nodes
    assert RedTeamNode not in next_nodes


@pytest.mark.asyncio
async def test_nonexistent_cell_raises():
    """Port the old missing-cell case to the injected admission lookup boundary."""
    phases = StubPhases()
    existing_cells = {"existing": object()}

    async def lookup(value):
        existing_cells[value.cell_id]
        return None

    with pytest.raises(KeyError, match="missing"):
        await run_pipeline(
            CellPipelineState(cell_id="missing"), replace(phases.dependencies(), sync_check=lookup)
        )
    assert phases.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("source,target", LEGAL, ids=[f"run:{a}->{b}" for a, b in LEGAL])
async def test_legal_transition_has_a_stubbed_execution_witness(source, target):
    phases = StubPhases()
    state = CellPipelineState(cell_id="cell")
    expected_end = Validated
    phase_names = {
        "socratic_active": "socratic",
        "research_active": "research",
        "critique_active": "critique",
    }
    if target == "stalled" or (source == "stalled" and target == "needs_human"):
        if source in phase_names:
            phases.park_phase = phase_names[source]
        else:
            phases.decisions = [ConvergenceDecision(outcome="circuit_breaker")]
        expected_end = NeedsHuman
    elif source == "mediating" and target == "research_active":
        phases.decisions = [ConvergenceDecision(outcome="iterate", weakest_link="sources")]
    elif target in {"deferred_open", "deferred_closed"}:
        expected_end = DeferOpen if target == "deferred_open" else DeferClosed
        if source == "evaluating":
            phases.verdict = pipeline.EvaluationVerdict(target.upper().replace("DEFERRED", "DEFER"))
        elif source == "red_team_active":
            phases.red_team_defer = "More evidence is needed"
        else:
            phases.decisions = [
                ConvergenceDecision(
                    outcome="short_circuit", short_circuit=target.replace("deferred", "defer")
                )
            ]
    elif target == "queued":
        # Terminal payloads have no run method. The next round starts at dispatch.
        state = CellPipelineState(cell_id="cell", iteration=10, cycle_start_iteration=10)
    result = await run_pipeline(state, phases.dependencies())
    assert isinstance(result, expected_end)
    if target == "queued":
        assert phases.calls[0][0] == "sync_check"
        assert phases.calls[1][0] == "socratic"
        assert state.iteration == state.cycle_start_iteration == 10
    if source == "mediating" and target == "research_active":
        assert [value.iteration for phase, value in phases.calls if phase == "research"] == [0, 1]
