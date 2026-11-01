"""Run typed node paths with deterministic phase results."""

from dataclasses import replace
from typing import get_args, get_origin, get_type_hints

import pytest
from pydantic import ValidationError
from pydantic_graph import End, GraphRunContext

from petri.execution.models import CellPipelineState
from petri.execution.pipeline import (
    ConvergenceCheckNode,
    ConvergenceDecision,
    CritiqueNode,
    DeferClosed,
    DeferOpen,
    Disproven,
    EvaluateNode,
    EvaluationVerdict,
    NeedsHuman,
    PipelineInput,
    RedTeamNode,
    ResearchNode,
    SyncConflict,
    Validated,
    run_pipeline,
)
from tests.pipeline_stubs import StubPhases


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "verdict,end_type",
    [
        (EvaluationVerdict.VALIDATED, Validated),
        (EvaluationVerdict.DISPROVEN, Disproven),
        (EvaluationVerdict.DEFER_OPEN, DeferOpen),
        (EvaluationVerdict.DEFER_CLOSED, DeferClosed),
    ],
)
async def test_full_graph_evaluation_ends(verdict, end_type):
    phases = StubPhases(verdict=verdict)
    result = await run_pipeline(CellPipelineState(cell_id="cell"), phases.dependencies())
    assert isinstance(result, end_type)
    assert [name for name, _ in phases.calls] == [
        "sync_check",
        "socratic",
        "research",
        "critique",
        "convergence",
        "red_team",
        "evaluate",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["socratic", "research", "critique"])
async def test_phase_failure_parks_for_human(phase):
    phases = StubPhases(park_phase=phase)
    result = await run_pipeline(CellPipelineState(cell_id="cell"), phases.dependencies())
    assert isinstance(result, NeedsHuman)
    assert result.reason == "Human review requested"
    assert phases.calls[-1][0] == phase


@pytest.mark.asyncio
async def test_sync_conflict_parks_before_inference():
    phases = StubPhases(conflict="Two edits changed the same claim")
    result = await run_pipeline(CellPipelineState(cell_id="cell"), phases.dependencies())
    assert isinstance(result, SyncConflict)
    assert result.reason == phases.conflict
    assert [name for name, _ in phases.calls] == ["sync_check"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "decision,end_type",
    [
        (ConvergenceDecision(outcome="circuit_breaker"), NeedsHuman),
        (
            ConvergenceDecision(outcome="short_circuit", short_circuit="needs_experiment"),
            NeedsHuman,
        ),
        (ConvergenceDecision(outcome="short_circuit", short_circuit="defer_open"), DeferOpen),
        (ConvergenceDecision(outcome="short_circuit", short_circuit="defer_closed"), DeferClosed),
    ],
)
async def test_convergence_parking_and_short_circuits(decision, end_type):
    phases = StubPhases(decisions=[decision])
    result = await run_pipeline(CellPipelineState(cell_id="cell"), phases.dependencies())
    assert isinstance(result, end_type)
    if isinstance(result, NeedsHuman):
        assert result.reason
    assert phases.calls[-1][0] == "convergence"


@pytest.mark.asyncio
async def test_red_team_can_defer_open():
    phases = StubPhases(red_team_defer="Evidence remains incomplete")
    result = await run_pipeline(CellPipelineState(cell_id="cell"), phases.dependencies())
    assert isinstance(result, DeferOpen)
    assert result.reason == phases.red_team_defer
    assert phases.calls[-1][0] == "red_team"


@pytest.mark.asyncio
async def test_iteration_is_owned_by_runner_and_increases_once_per_retry():
    state = CellPipelineState(cell_id="cell", iteration=7, cycle_start_iteration=7)
    phases = StubPhases(
        decisions=[
            ConvergenceDecision(
                outcome="iterate", weakest_link="sources", focused_directive="Verify citation"
            ),
            ConvergenceDecision(outcome="iterate", weakest_link="logic"),
        ]
    )
    await run_pipeline(state, phases.dependencies())
    research_inputs = [value for name, value in phases.calls if name == "research"]
    assert [value.iteration for value in research_inputs] == [7, 8, 9]
    assert research_inputs[1].weakest_link == "sources"
    assert research_inputs[1].focused_directive == "Verify citation"
    assert research_inputs[2].focused_directive == "Focus on addressing logic concerns"
    assert state.iteration == 9
    assert state.cycle_start_iteration == 7
    for name in ["research", "critique", "convergence"]:
        assert [value.iteration for phase, value in phases.calls if phase == name] == [7, 8, 9]
    assert CellPipelineState.model_validate(state.model_dump()).iteration == 9


@pytest.mark.asyncio
async def test_callable_cannot_set_iteration_in_snapshot_or_live_state():
    state = CellPipelineState(cell_id="cell", iteration=2)
    phases = StubPhases()

    async def research(value: PipelineInput):
        with pytest.raises(ValidationError, match="frozen"):
            value.iteration = 100
        with pytest.raises(ValidationError, match="frozen"):
            state.iteration = 100
        return await phases.research(value)

    await run_pipeline(state, replace(phases.dependencies(), research=research))
    assert state.iteration == 2


@pytest.mark.asyncio
async def test_node_context_cannot_set_iteration():
    state = CellPipelineState(cell_id="cell", iteration=2)
    ctx = GraphRunContext(state=state, deps=StubPhases().dependencies())
    with pytest.raises(ValidationError, match="frozen"):
        ctx.state.iteration = 1
    assert state.iteration == 2


def test_return_annotations_exclude_phase_skips_and_expose_exact_ends():
    research = get_args(get_type_hints(ResearchNode.run)["return"])
    assert set(get_origin(item) or item for item in research) == {CritiqueNode, End}
    convergence = get_args(get_type_hints(ConvergenceCheckNode.run)["return"])
    assert set(get_origin(item) or item for item in convergence) == {RedTeamNode, ResearchNode, End}
    evaluation = get_type_hints(EvaluateNode.run)["return"]
    assert get_origin(evaluation) is End
    assert set(get_args(get_args(evaluation)[0])) == {Validated, Disproven, DeferOpen, DeferClosed}


def test_invalid_decisions_and_empty_parking_reasons_rejected():
    with pytest.raises(ValidationError):
        ConvergenceDecision(outcome="unknown")
    with pytest.raises(ValidationError, match="requires its destination"):
        ConvergenceDecision(outcome="short_circuit")
    with pytest.raises(ValidationError):
        ConvergenceDecision(outcome="iterate", iteration=100)
    for end_type in [NeedsHuman, SyncConflict]:
        with pytest.raises(ValidationError):
            end_type(reason="")


@pytest.mark.asyncio
async def test_graph_rejects_runtime_transition_outside_return_annotation(monkeypatch):
    from pydantic_graph.exceptions import GraphRuntimeError

    async def skip_critique(self, ctx):
        return EvaluateNode()

    monkeypatch.setattr(ResearchNode, "run", skip_critique)
    with pytest.raises(GraphRuntimeError):
        await run_pipeline(CellPipelineState(cell_id="cell"), StubPhases().dependencies())


@pytest.mark.asyncio
async def test_graph_rejects_undeclared_end_payload(monkeypatch):
    from pydantic_graph.exceptions import GraphRuntimeError

    async def skip_to_validated(self, ctx):
        return End(Validated())

    monkeypatch.setattr(ResearchNode, "run", skip_to_validated)
    with pytest.raises(GraphRuntimeError, match="Undeclared terminal result"):
        await run_pipeline(CellPipelineState(cell_id="cell"), StubPhases().dependencies())
