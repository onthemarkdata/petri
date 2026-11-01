"""Mechanical convergence, decision phases, and checkpointable terminal writes."""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import closing
from functools import partial

from pydantic import BaseModel, ConfigDict

from petri.agents import convergence
from petri.agents.contract import AgentName
from petri.agents.source_policy import check_terminal_sources
from petri.execution.evidence import RecordedAssessment
from petri.execution.models import PipelineOutcome
from petri.execution.phases import (
    PhaseServices,
    _assessment,
    _claim,
    _emit,
    _finish,
    _prior_evidence,
    _record_assessment,
    build_research_dependencies,
    span_context,
)
from petri.execution.pipeline import (
    ConvergenceDecision,
    EvaluationVerdict,
    NeedsHuman,
    PipelineDeps,
    PipelineEnd,
    PipelineInput,
    RedTeamResult,
    SyncConflict,
)
from petri.models import CellStatus
from petri.storage import database
from petri.storage.event_log import append_event
from petri.telemetry.attributes import phase_span


class TerminalStatusResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    cell_id: str
    status: CellStatus


def decide_convergence(
    verdicts: Iterable[convergence.VerdictRecord],
    *,
    iteration: int,
    cycle_start_iteration: int,
    max_iterations: int,
) -> ConvergenceDecision:
    """Apply the v1 short-circuit, convergence, and circuit-breaker order without inference."""
    result = convergence.check_convergence(verdicts)
    short = convergence.evaluate_short_circuits(result)
    if short is not None:
        return ConvergenceDecision(outcome=PipelineOutcome.short_circuit, short_circuit=short.type)
    if result.converged:
        return ConvergenceDecision(outcome=PipelineOutcome.converged)
    if convergence.compute_circuit_breaker(iteration, cycle_start_iteration, max_iterations):
        return ConvergenceDecision(outcome=PipelineOutcome.circuit_breaker)
    return ConvergenceDecision(
        outcome=PipelineOutcome.iterate,
        weakest_link=result.weakest_link.value if result.weakest_link else None,
    )


async def run_convergence(value: PipelineInput, services: PhaseServices) -> ConvergenceDecision:
    with phase_span("convergence", span_context(services, value), iteration=value.iteration,
                    agent=AgentName.CELL_LEAD, weakest_link=value.weakest_link) as span:
        result = await _run_convergence(value, services)
        span.set_attribute("petri.outcome", result.outcome.value)
        if result.weakest_link:
            span.set_attribute("petri.weakest_link", result.weakest_link)
        return result


async def _run_convergence(value: PipelineInput, services: PhaseServices) -> ConvergenceDecision:
    """Read the current iteration's verdicts and record a mechanical decision."""
    _emit(services, value, "convergence", "phase_start")
    records = [
        RecordedAssessment.model_validate({"agent": row["agent"], **row["data"]})
        for row in database.read_events(
            services.db_path,
            cell_id=value.cell_id,
            iteration=value.iteration,
            event_type="verdict_issued",
        )
    ]
    decision = decide_convergence(
        records,
        iteration=value.iteration,
        cycle_start_iteration=value.cycle_start_iteration,
        max_iterations=value.max_iterations,
    )
    if decision.outcome != PipelineOutcome.short_circuit:
        result = convergence.check_convergence(records)
        append_event(
            database.SQLiteEventLog(services.db_path, value.cell_id),
            value.cell_id,
            "convergence_checked",
            "cell_lead",
            value.iteration,
            {
                "converged": result.converged,
                "stall_reason": "circuit_breaker" if decision.outcome == PipelineOutcome.circuit_breaker else None,
                "blocking_verdicts": {
                    name.value: vote.verdict for name, vote in result.blocking_results.items()
                }
                or None,
                "weakest_link": result.weakest_link.value if result.weakest_link else None,
            },
            round_number=services.round_number,
            node="convergence",
        )
    _emit(services, value, "convergence", "phase_end")
    return decision


async def run_decomposition_audit(value: PipelineInput, services: PhaseServices) -> None:
    """Audit framing after a circuit break; this is separate from the mechanical check."""
    _emit(services, value, "decomposition_audit", "phase_start")
    assessment = await _assessment(
        services,
        value,
        node="decomposition_audit",
        phase="decomposition_audit",
        agent_name="socratic_decomposition_auditor",
        agent=services.socratic_agent,
        claim=_claim(services, value),
        prior_evidence=_prior_evidence(services, value),
        directive=AUDIT_DIRECTIVE,
    )
    suggestion = assessment.arguments
    append_event(
        database.SQLiteEventLog(services.db_path, value.cell_id),
        value.cell_id,
        "decomposition_audit",
        "decomposition_auditor",
        value.iteration,
        {
            "iteration": value.iteration,
            "suggestion": suggestion,
            "should_restructure": "restructur" in suggestion.lower()
            or "redefin" in suggestion.lower(),
        },
        round_number=services.round_number,
        node="decomposition_audit",
    )
    _finish(services, value, "decomposition_audit")


async def convergence_phase(value: PipelineInput, services: PhaseServices) -> ConvergenceDecision:
    decision = await run_convergence(value, services)
    if decision.outcome == PipelineOutcome.circuit_breaker:
        await run_decomposition_audit(value, services)
    return decision


async def run_red_team(value: PipelineInput, services: PhaseServices) -> RedTeamResult:
    _emit(services, value, "red_team", "phase_start")
    await _assessment(
        services,
        value,
        node="red_team",
        phase="red_team",
        agent_name=AgentName.RED_TEAM_LEAD.value,
        agent=services.agents[AgentName.RED_TEAM_LEAD],
        claim=_claim(services, value),
        prior_evidence=_prior_evidence(services, value),
    )
    _finish(services, value, "red_team")
    return RedTeamResult()


def evaluation_status(verdict: str) -> EvaluationVerdict:
    """Preserve v1's inconclusive fallback for every nonterminal evaluator verdict."""
    if verdict == "EVIDENCE_CONFIRMS":
        return EvaluationVerdict.VALIDATED
    if verdict == "EVIDENCE_REFUTES":
        return EvaluationVerdict.DISPROVEN
    return EvaluationVerdict.DEFER_OPEN


async def run_evaluation(value: PipelineInput, services: PhaseServices) -> EvaluationVerdict:
    _emit(services, value, "evaluation", "phase_start")
    sources = [
        row["data"]
        for row in database.read_events(
            services.db_path, cell_id=value.cell_id, event_type="source_reviewed"
        )
    ]
    policy = check_terminal_sources(
        sources, services.config.source_hierarchy.minimum_terminal_level
    )
    if not policy.passes:
        assessment = _record_assessment(
            services,
            value,
            "evaluation",
            RecordedAssessment(
                agent=AgentName.EVIDENCE_EVALUATOR.value,
                verdict="EVIDENCE_INCONCLUSIVE",
                summary=policy.detail,
            ),
        )
    else:
        assessment = await _assessment(
            services,
            value,
            node="evaluation",
            phase="evaluation",
            agent_name=AgentName.EVIDENCE_EVALUATOR.value,
            agent=services.agents[AgentName.EVIDENCE_EVALUATOR],
            claim=_claim(services, value),
            prior_evidence=_prior_evidence(services, value),
            directive=policy.detail,
        )
    status = evaluation_status(assessment.verdict)
    # Recheck at the phase boundary even when a caller supplies an agent without
    # the factory's source validator. Existing reviewed sources count toward policy.
    if status in {EvaluationVerdict.VALIDATED, EvaluationVerdict.DISPROVEN} and not policy.passes:
        status = EvaluationVerdict.DEFER_OPEN
    _finish(services, value, "evaluation")
    return status


def write_terminal_status(
    value: PipelineInput, services: PhaseServices, status: CellStatus
) -> TerminalStatusResult:
    """Write only the status column in one transaction; safe to replay as a step."""
    with closing(database.connect(services.db_path)) as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            result = conn.execute(
                "UPDATE cells SET status = ? WHERE id = ?", (status.value, value.cell_id)
            )
            if result.rowcount != 1:
                raise KeyError(f"Cell {value.cell_id!r} is not in the domain database")
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
    return TerminalStatusResult(cell_id=value.cell_id, status=status)


async def finish_pipeline(
    value: PipelineInput, result: PipelineEnd, services: PhaseServices
) -> None:
    if isinstance(result, NeedsHuman):
        status = CellStatus(result.cell_status)
    elif isinstance(result, SyncConflict):
        status = CellStatus.STALLED
    else:
        status = CellStatus(result.status.upper())
    write_terminal_status(value, services, status)


async def check_cell(value: PipelineInput, services: PhaseServices) -> SyncConflict | None:
    _claim(services, value)
    return None


def build_pipeline_dependencies(services: PhaseServices) -> PipelineDeps:
    """Wire research and decision adapters with one terminal persistence callback."""
    deps = build_research_dependencies(
        services,
        convergence=partial(convergence_phase, services=services),
        red_team=partial(run_red_team, services=services),
        evaluate=partial(run_evaluation, services=services),
        sync_check=partial(check_cell, services=services),
    )
    return PipelineDeps(
        deps.socratic,
        deps.research,
        deps.critique,
        deps.convergence,
        deps.red_team,
        deps.evaluate,
        deps.sync_check,
        partial(finish_pipeline, services=services),
    )


AUDIT_DIRECTIVE = "This cell has FAILED TO CONVERGE after multiple iterations. The agents could not agree on a verdict. This means the problem may be in the DECOMPOSITION, not the evidence. Ask: Are the assumptions for this cell actually the RIGHT assumptions? Should this claim be broken down DIFFERENTLY? Is the claim itself poorly formed or ambiguous? In 'arguments', explain what's structurally wrong with how this cell is framed. In 'evidence', suggest how it should be restructured."
