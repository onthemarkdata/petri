"""Decision routing, source policy, and terminal domain status parity."""

import random
from types import SimpleNamespace

import pytest
from pydantic_ai.models.test import TestModel

from petri.agents.contract import AGENT_SPECS, AgentName
from petri.analysis import convergence as legacy
from petri.execution.decisions import (
    build_pipeline_dependencies,
    decide_convergence,
    evaluation_status,
    run_convergence,
    run_evaluation,
    write_terminal_status,
)
from petri.execution.evidence import RecordedAssessment, render_evidence
from petri.execution.models import CellPipelineState, PipelineOutcome
from petri.execution.phases import _record_assessment
from petri.execution.pipeline import (
    DeferOpen,
    Disproven,
    EvaluationVerdict,
    NeedsHuman,
    Validated,
    run_pipeline,
)
from petri.models import CellStatus, validate_event_data
from petri.storage import database
from tests.unit import test_research_phases
from tests.unit.test_research_phases import SOURCE, inputs

phase_services = test_research_phases.phase_services


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "verdict,status,end",
    [
        ("EVIDENCE_CONFIRMS", CellStatus.VALIDATED, Validated),
        ("EVIDENCE_REFUTES", CellStatus.DISPROVEN, Disproven),
        ("EVIDENCE_INCONCLUSIVE", CellStatus.DEFER_OPEN, DeferOpen),
    ],
)
async def test_full_pipeline_writes_terminal_status(
    phase_services, canonical_colony, verdict, status, end
):
    services, _ = phase_services
    cell_id = canonical_colony["cell1"].id
    evaluator = services.agents[AgentName.EVIDENCE_EVALUATOR]
    with evaluator.override(
        model=TestModel(
            custom_output_args={
                "agent": "evidence_evaluator",
                "verdict": verdict,
                "sources_cited": [SOURCE],
            }
        )
    ):
        result = await run_pipeline(
            CellPipelineState(cell_id=cell_id), build_pipeline_dependencies(services)
        )
    assert isinstance(result, end)
    assert database.get_cell(services.db_path, cell_id)["status"] == status.value
    events = database.read_events(services.db_path, cell_id=cell_id)
    assert {event["agent"] for event in events} >= {"red_team_lead", "evidence_evaluator"}
    assert "### Red Team Review" in render_evidence(events)
    assert "### Evidence Evaluation" in render_evidence(events)
    for event in events:
        validate_event_data(event["type"], event["data"])
    before = database.get_cell(services.db_path, cell_id)
    assert write_terminal_status(inputs(cell_id), services, status).status == status
    assert database.get_cell(services.db_path, cell_id) == before
    assert not list(services.db_path.parent.rglob("metadata.json"))


@pytest.mark.asyncio
async def test_iterate_then_circuit_breaker_audits_and_parks(phase_services, canonical_colony):
    services, _ = phase_services
    cell_id = canonical_colony["cell1"].id
    services.config.limits.run.request_limit = 60
    services.config.limits.run.total_tokens_limit = 1_000_000
    with services.agents[AgentName.INVESTIGATOR].override(
        model=TestModel(
            custom_output_args={
                "agent": "investigator",
                "verdict": "NEEDS_MORE_EVIDENCE",
                "sources_cited": [SOURCE],
            }
        )
    ):
        state = CellPipelineState(cell_id=cell_id, max_iterations=1)
        result = await run_pipeline(state, build_pipeline_dependencies(services))
    assert isinstance(result, NeedsHuman)
    assert state.iteration == 1
    assert state.weakest_link == "investigator"
    assert state.focused_directive == "Focus on addressing investigator concerns"
    assert database.get_cell(services.db_path, cell_id)["status"] == "STALLED"
    events = database.read_events(services.db_path, cell_id=cell_id)
    assert len([event for event in events if event["type"] == "decomposition_audit"]) == 1
    assert "### Decomposition Audit" in render_evidence(events)
    assert not any(event["agent"] == "red_team_lead" for event in events)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "agent,verdict,end,status",
    [
        (AgentName.TRIAGE, "LOW_VALUE_DEFER", DeferOpen, "DEFER_OPEN"),
        (AgentName.INVESTIGATOR, "CANNOT_DETERMINE", NeedsHuman, "NEEDS_EXPERIMENT"),
    ],
)
async def test_short_circuit_skips_red_team(
    phase_services, canonical_colony, agent, verdict, end, status
):
    services, _ = phase_services
    cell_id = canonical_colony["cell1"].id
    with services.agents[agent].override(
        model=TestModel(
            custom_output_args={
                "agent": agent.value,
                "verdict": verdict,
                "sources_cited": [SOURCE],
            }
        )
    ):
        result = await run_pipeline(
            CellPipelineState(cell_id=cell_id), build_pipeline_dependencies(services)
        )
    assert isinstance(result, end)
    assert database.get_cell(services.db_path, cell_id)["status"] == status
    assert not database.read_events(services.db_path, cell_id=cell_id, agent="red_team_lead")


@pytest.mark.asyncio
async def test_mechanical_check_never_calls_an_agent(phase_services, canonical_colony, monkeypatch):
    services, _ = phase_services
    value = inputs(canonical_colony["cell1"].id)

    async def forbidden(*args, **kwargs):
        raise AssertionError("Convergence must not perform inference")

    for agent in [*services.agents.values(), services.socratic_agent]:
        monkeypatch.setattr(agent, "run", forbidden)
    decision = await run_convergence(value, services)
    assert decision.outcome == PipelineOutcome.iterate
    assert decision.weakest_link is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("level", [None, 5, 6])
async def test_evaluation_refuses_terminal_result_without_eligible_source(
    phase_services, canonical_colony, level, monkeypatch
):
    services, _ = phase_services
    value = inputs(canonical_colony["cell1"].id)
    if level is not None:
        _record_assessment(
            services,
            value,
            "research",
            RecordedAssessment(
                agent="investigator",
                verdict="EVIDENCE_SUFFICIENT",
                sources_cited=[{**SOURCE, "hierarchy_level": level}],
            ),
        )

    async def forbidden(*args, **kwargs):
        raise AssertionError("A missing-source gate must not request a terminal verdict")

    monkeypatch.setattr(services.agents[AgentName.EVIDENCE_EVALUATOR], "run", forbidden)
    result = await run_evaluation(value, services)
    assert result is EvaluationVerdict.DEFER_OPEN
    events = database.read_events(
        services.db_path, cell_id=value.cell_id, agent="evidence_evaluator"
    )
    assert events[0]["data"]["verdict"] == "EVIDENCE_INCONCLUSIVE"


@pytest.mark.asyncio
async def test_decision_replay_adds_no_events(phase_services, canonical_colony, monkeypatch):
    services, _ = phase_services
    cell_id = canonical_colony["cell1"].id
    await run_pipeline(CellPipelineState(cell_id=cell_id), build_pipeline_dependencies(services))
    before = database.read_events(services.db_path)

    async def forbidden(*args, **kwargs):
        raise AssertionError("Replay must reuse committed agent output")

    for agent in [*services.agents.values(), services.socratic_agent]:
        monkeypatch.setattr(agent, "run", forbidden)
    await run_pipeline(CellPipelineState(cell_id=cell_id), build_pipeline_dependencies(services))
    assert database.read_events(services.db_path) == before


def test_decision_matches_legacy_convergence_matrix():
    rng = random.Random(56)
    roles = legacy.load_agent_roles()
    for iteration in [0, 1, 3, 9]:
        for _ in range(75):
            rows = [
                SimpleNamespace(
                    agent=name.value,
                    verdict=rng.choice(
                        [*spec.verdicts_pass, *spec.verdicts_block, "EXECUTION_ERROR", ""]
                    ),
                )
                for name, spec in AGENT_SPECS.items()
                if rng.random() > 0.1
            ]
            short = legacy.evaluate_short_circuits(rows, roles)
            result = legacy.check_convergence(rows, roles)
            expected = (
                "short_circuit"
                if short
                else "converged"
                if result.converged
                else "circuit_breaker"
                if legacy.compute_circuit_breaker(iteration, 0, 3)
                else "iterate"
            )
            decision = decide_convergence(
                rows, iteration=iteration, cycle_start_iteration=0, max_iterations=3
            )
            assert decision.outcome.value == expected
            if short:
                assert decision.short_circuit == short.type
            if expected == "iterate":
                assert decision.weakest_link == result.weakest_link


@pytest.mark.parametrize(
    "verdict,expected",
    [
        ("EVIDENCE_CONFIRMS", "VALIDATED"),
        ("EVIDENCE_REFUTES", "DISPROVEN"),
        ("EVIDENCE_INCONCLUSIVE", "DEFER_OPEN"),
        ("unknown", "DEFER_OPEN"),
    ],
)
def test_evaluator_mapping_matches_v1(verdict, expected):
    assert evaluation_status(verdict).value == expected
