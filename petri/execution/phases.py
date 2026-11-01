"""Checkpoint-shaped research phases with explicit inputs and idempotent events.

Agent responses live in domain events. Graph state contains control fields and
identifiers, never model transcripts or evidence documents. A retried phase
reuses committed verdicts and debate results; a crash before the first verdict
commit can still repeat an external model request.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Mapping
from contextlib import closing
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict
from pydantic_ai import Agent, PromptedOutput, ToolOutput, capture_run_messages
from pydantic_ai.exceptions import UnexpectedModelBehavior, UsageLimitExceeded
from pydantic_ai.models import Model
from pydantic_ai.usage import RunUsage, UsageLimits

from petri.agents.config import AgentConfig
from petri.agents.contract import AGENT_SPECS, AgentName, Phase
from petri.agents.debates import DebateStatus, log_debate, run_debate
from petri.agents.outputs import AgentAssessment
from petri.execution.evidence import RecordedAssessment, format_socratic, render_evidence
from petri.execution.pipeline import (
    ConvergenceDecision,
    EvaluationVerdict,
    PhaseResult,
    PipelineDeps,
    PipelineInput,
    RedTeamResult,
    SyncConflict,
)
from petri.harness.errors import PermanentHarnessError
from petri.harness.diagnostics import model_failure_detail
from petri.reasoning.prompts import build_assess_cell_prompt
from petri.research.tools import ResearchContext
from petri.storage import database
from petri.storage.event_log import append_event, derive_event_id
from petri.models import parse_key
from petri.telemetry.attributes import CellSpanContext, phase_span, set_verdict_attributes


class SocraticAssessment(RecordedAssessment):
    agent: Literal["socratic_questioner"] = "socratic_questioner"
    verdict: Literal[
        "CLARIFIED", "ASSUMPTIONS_CHALLENGED", "EVIDENCE_MAPPED", "VERIFIED", "SOCRATIC_INCOMPLETE"
    ]


class PhaseProgress(BaseModel):
    model_config = ConfigDict(frozen=True)
    cell_id: str
    phase: str
    iteration: int
    kind: Literal["phase_start", "phase_end", "agent_start", "agent_end"]
    agent: str | None = None
    verdict: str | None = None


class PhaseRunResult(PhaseResult):
    cell_id: str
    phase: str
    iteration: int


@dataclass
class PhaseServices:
    db_path: Path
    round_number: int
    today_iso: str
    config: AgentConfig
    agents: Mapping[AgentName, Agent[Any, AgentAssessment]]
    socratic_agent: Agent[Any, SocraticAssessment]
    dish_id: str = ""
    on_progress: Callable[[PhaseProgress], None] | None = None
    usage: RunUsage = field(default_factory=RunUsage)

    def __post_init__(self):
        self.config = AgentConfig.model_validate(self.config.model_dump(mode="json"))
        if self.round_number < 0 or not self.today_iso:
            raise ValueError("Provide a nonnegative round number and an explicit date")
        missing = set(AgentName) - set(self.agents)
        if missing:
            raise ValueError(f"The phase roster is missing agents: {sorted(missing)}")
        pairs = [debate.pair for debate in self.config.debates]
        if len(pairs) != len(set(pairs)):
            raise ValueError("Each debate pairing must occur only once per research round")
        for debate in self.config.debates:
            if any(name not in self.agents for name in debate.pair):
                raise ValueError("Both debate participants must exist before graph construction")


def build_socratic_agent(
    config: AgentConfig,
    *,
    constitution: str,
    model: Model | None = None,
) -> Agent[Any, SocraticAssessment]:
    """Build the pre-pipeline questioner without adding it to the voting roster."""
    if not constitution.strip():
        raise ValueError("Provide the constitution text before building the Socratic agent")
    selected: Model | str
    if model is None:
        from petri.harness.resolve import resolve_model

        selected = resolve_model(config.model_dump(mode="json"))
    else:
        selected = model
    output = (
        ToolOutput(SocraticAssessment)
        if model is not None or config.harness.backend == "provider"
        else PromptedOutput(SocraticAssessment)
    )
    return Agent(
        selected,
        name="socratic_questioner",
        output_type=output,
        instructions=constitution + "\n\n" + config.pre_pipeline.socratic_questioner.instruction,
        retries=2,
    )


def _emit(
    services: PhaseServices, value: PipelineInput, phase: str, kind, *, agent=None, verdict=None
):
    if services.on_progress:
        services.on_progress(
            PhaseProgress(
                cell_id=value.cell_id,
                phase=phase,
                iteration=value.iteration,
                kind=kind,
                agent=agent,
                verdict=verdict,
            )
        )


def _claim(services: PhaseServices, value: PipelineInput) -> str:
    cell = database.get_cell(services.db_path, value.cell_id)
    if cell is None:
        raise KeyError(f"Cell {value.cell_id!r} is not in the domain database")
    return cell["claim_text"]


def _stored_payload(
    services: PhaseServices, value: PipelineInput, node: str, agent: str, event_type: str
) -> dict | None:
    identity = derive_event_id(
        value.cell_id,
        round_number=services.round_number,
        node=node,
        agent=agent,
        iteration=value.iteration,
        event_type=event_type,
    )
    with closing(database.connect(services.db_path)) as conn:
        row = conn.execute("SELECT payload FROM events WHERE id = ?", (identity,)).fetchone()
    return json.loads(row[0]) if row else None


def _record_assessment(
    services: PhaseServices, value: PipelineInput, node: str, assessment: RecordedAssessment
) -> RecordedAssessment:
    target = database.SQLiteEventLog(services.db_path, value.cell_id)
    event = append_event(
        target,
        value.cell_id,
        "verdict_issued",
        assessment.agent,
        value.iteration,
        assessment.model_dump(mode="json", exclude={"agent"}),
        round_number=services.round_number,
        node=node,
    )
    # Concurrent retries must derive source events from the winning verdict.
    recorded = RecordedAssessment.model_validate({"agent": assessment.agent, **event.data})
    for sequence, source in enumerate(recorded.sources_cited):
        append_event(
            target,
            value.cell_id,
            "source_reviewed",
            assessment.agent,
            value.iteration,
            source.model_dump(mode="json", exclude={"pub_year"}),
            round_number=services.round_number,
            node=node,
            sequence=sequence,
        )
    return recorded


def span_context(services: PhaseServices, value: PipelineInput) -> CellSpanContext:
    identity = parse_key(value.cell_id, services.dish_id, strict=True)
    return CellSpanContext(services.dish_id, str(identity["colony_prefix"]),
        int(identity["level"]), int(identity["seq"]))


def _measured_delta(before: RunUsage, after: RunUsage) -> RunUsage | None:
    if after.requests == before.requests:
        return None
    return RunUsage(requests=after.requests - before.requests,
        input_tokens=after.input_tokens - before.input_tokens,
        output_tokens=after.output_tokens - before.output_tokens,
        unmeasured_requests=after.unmeasured_requests - before.unmeasured_requests)


async def _assessment(
    services: PhaseServices, value: PipelineInput, *, node: str, phase: str,
    agent_name: str, agent: Agent[Any, Any], claim: str,
    prior_evidence: str = "", directive: str | None = None,
) -> RecordedAssessment:
    before = replace(services.usage)
    role = AgentName(agent_name) if agent_name in AgentName._value2member_map_ else None
    with phase_span(phase, span_context(services, value), iteration=value.iteration,
                    agent=role, weakest_link=value.weakest_link) as span:
        if role is None:
            span.set_attribute("petri.agent_role", agent_name)
            span.set_attribute("petri.agent_blocking", False)
        try:
            result = await _assessment_impl(services, value, node=node, phase=phase,
                agent_name=agent_name, agent=agent, claim=claim,
                prior_evidence=prior_evidence, directive=directive)
        except Exception:
            set_verdict_attributes(span, "EXECUTION_ERROR",
                usage=_measured_delta(before, services.usage), model=services.config.model,
                prices=services.config.prices, outcome="error")
            raise
        set_verdict_attributes(span, result.verdict,
            usage=_measured_delta(before, services.usage), model=services.config.model,
            prices=services.config.prices, outcome="completed")
        return result


async def _assessment_impl(
    services: PhaseServices,
    value: PipelineInput,
    *,
    node: str,
    phase: str,
    agent_name: str,
    agent: Agent[Any, Any],
    claim: str,
    prior_evidence: str = "",
    directive: str | None = None,
) -> RecordedAssessment:
    _emit(services, value, phase, "agent_start", agent=agent_name)
    cached = _stored_payload(services, value, node, agent_name, "verdict_issued")
    if cached is not None:
        output = RecordedAssessment.model_validate({"agent": agent_name, **cached})
    else:
        content = (
            services.config.pre_pipeline.socratic_questioner
            if agent_name.startswith("socratic_")
            else services.config.agents[AgentName(agent_name)]
        )
        prompt = build_assess_cell_prompt(
            value.cell_id,
            claim,
            {
                "iteration": value.iteration,
                "weakest_link": value.weakest_link,
                "focused_directive": directive or value.focused_directive,
                "phase": phase,
                "prior_evidence": prior_evidence,
            },
            "socratic_questioner" if agent_name.startswith("socratic_") else agent_name,
            valid_verdicts=content.verdicts_pass + content.verdicts_block,
            role_instruction=content.instruction,
            today_iso=services.today_iso,
        )
        with capture_run_messages() as messages:
            try:
                response = await agent.run(
                    prompt,
                    deps=ResearchContext(services.db_path,value.cell_id,agent_name,value.iteration,services.round_number,node),
                    usage=services.usage,
                    usage_limits=UsageLimits(
                        request_limit=services.config.limits.run.request_limit,
                        total_tokens_limit=services.config.limits.run.total_tokens_limit,
                        tool_calls_limit=services.config.limits.run.tool_calls_limit,
                    ),
                )
            except (UnexpectedModelBehavior, UsageLimitExceeded, PermanentHarnessError) as error:
                detail = model_failure_detail(error, messages)
                append_event(database.SQLiteEventLog(services.db_path, value.cell_id),
                    value.cell_id, "agent_execution_failed", agent_name, value.iteration,
                    {"phase": phase, "error_type": type(error).__name__,
                     "message": str(error)[-4096:], "failure_detail": detail.model_dump(mode="json")},
                    round_number=services.round_number, node=node)
                raise
        output = RecordedAssessment.model_validate(
            {**response.output.model_dump(mode="json"), "agent": agent_name}
        )
    recorded = _record_assessment(services, value, node, output)
    _emit(services, value, phase, "agent_end", agent=agent_name, verdict=recorded.verdict)
    return recorded


def _finish(services: PhaseServices, value: PipelineInput, phase: str) -> PhaseRunResult:
    append_event(
        database.SQLiteEventLog(services.db_path, value.cell_id),
        value.cell_id,
        "evidence_appended",
        "cell_lead",
        value.iteration,
        {"summary": f"Appended {phase} findings (iteration {value.iteration})"},
        round_number=services.round_number,
        node=phase,
    )
    _emit(services, value, phase, "phase_end")
    return PhaseRunResult(cell_id=value.cell_id, phase=phase, iteration=value.iteration)


def _prior_evidence(services: PhaseServices, value: PipelineInput) -> str:
    return render_evidence(database.read_events(services.db_path, cell_id=value.cell_id))


async def run_socratic(value: PipelineInput, services: PhaseServices) -> PhaseRunResult:
    """Clarify the claim, challenge assumptions, map evidence, then verify the analysis."""
    _emit(services, value, "socratic", "phase_start")
    claim = _claim(services, value)
    assessments = []
    for step, directive in SOCRATIC_STEPS:
        assessments.append(
            await _assessment(
                services,
                value,
                node=f"socratic:{step}",
                phase="socratic",
                agent_name=f"socratic_{step}",
                agent=services.socratic_agent,
                claim=claim,
                directive=directive,
            )
        )
    await _assessment(
        services,
        value,
        node="socratic:verification",
        phase="socratic",
        agent_name="socratic_verifier",
        agent=services.socratic_agent,
        claim=claim,
        prior_evidence=format_socratic(assessments),
        directive=SOCRATIC_VERIFY,
    )
    return _finish(services, value, "socratic")


async def run_research(value: PipelineInput, services: PhaseServices) -> PhaseRunResult:
    _emit(services, value, "research", "phase_start")
    claim = _claim(services, value)
    prior = _prior_evidence(services, value)
    for name, spec in AGENT_SPECS.items():
        if spec.phase == Phase.RESEARCH and not spec.is_lead:
            await _assessment(
                services,
                value,
                node="research",
                phase="research",
                agent_name=name.value,
                agent=services.agents[name],
                claim=claim,
                prior_evidence=prior,
            )
    return _finish(services, value, "research")


async def run_critique(value: PipelineInput, services: PhaseServices) -> PhaseRunResult:
    _emit(services, value, "critique", "phase_start")
    claim = _claim(services, value)
    prior = _prior_evidence(services, value)
    for name, spec in AGENT_SPECS.items():
        if spec.phase == Phase.CRITIQUE and not spec.is_lead:
            await _assessment(
                services,
                value,
                node="critique",
                phase="critique",
                agent_name=name.value,
                agent=services.agents[name],
                claim=claim,
                prior_evidence=prior,
            )
    for debate in services.config.debates:
        node = "debate:" + ":".join(name.value for name in debate.pair)
        cached = _stored_payload(services, value, node, "cell_lead", "debate_mediated")
        if cached is None:
            for name in debate.pair:
                _emit(services, value, "critique", "agent_start", agent=name.value)
            with phase_span("debate", span_context(services, value), iteration=value.iteration,
                            agent=AgentName.CELL_LEAD, weakest_link=value.weakest_link) as span:
                result = await run_debate(
                    debate.protocol, services.agents, claim, services.config.limits.debate
                )
                set_verdict_attributes(span, result.status.value, usage=result.usage,
                    model=services.config.model, prices=services.config.prices, outcome=result.status.value)
            log_debate(
                database.SQLiteEventLog(services.db_path, value.cell_id),
                value.cell_id,
                value.iteration,
                result,
                round_number=services.round_number,
            )
            for exchange in result.exchanges:
                _emit(
                    services,
                    value,
                    "critique",
                    "agent_end",
                    agent=exchange.speaker.value,
                    verdict=exchange.assessment.verdict,
                )
            status, detail = result.status.value, result.detail
        else:
            detail_payload = json.loads(cached["exchange_summary"])
            status, detail = detail_payload["status"], detail_payload["detail"]
        if status != DebateStatus.COMPLETE:
            _emit(services, value, "critique", "phase_end")
            return PhaseRunResult(
                cell_id=value.cell_id,
                phase="critique",
                iteration=value.iteration,
                needs_human_reason=detail or f"Debate ended with {status}",
            )
    return _finish(services, value, "critique")


def build_research_dependencies(
    services: PhaseServices,
    *,
    convergence: Callable[[PipelineInput], Awaitable[ConvergenceDecision]],
    red_team: Callable[[PipelineInput], Awaitable[RedTeamResult]],
    evaluate: Callable[[PipelineInput], Awaitable[EvaluationVerdict]],
    sync_check: Callable[[PipelineInput], Awaitable[SyncConflict | None]],
) -> PipelineDeps:
    """Wire research nodes only after the roster and pairings have validated."""
    services.__post_init__()
    return PipelineDeps(
        partial(run_socratic, services=services),
        partial(run_research, services=services),
        partial(run_critique, services=services),
        convergence,
        red_team,
        evaluate,
        sync_check,
    )


# Preserve the v1 Socratic prompts as fixed phase instructions.
SOCRATIC_STEPS = (
    (
        "clarify",
        'SOCRATIC STEP 1 — CLARIFY: What exactly is being claimed? Identify every key term that needs a precise definition. What is ambiguous? What could be interpreted multiple ways? Return JSON with: "definitions" (key terms and their meanings), "ambiguities" (what\'s unclear), "refined_claim" (the claim restated precisely). Set "verdict" to exactly "CLARIFIED" once the claim has been restated precisely and every key term has been defined.',
    ),
    (
        "challenge_assumptions",
        'SOCRATIC STEP 2 — CHALLENGE ASSUMPTIONS: What are we assuming is true without evidence? List every hidden premise. For each assumption, ask: Is this actually true, or do we just believe it? What would change if this assumption were false? Return JSON with: "assumptions" (list of hidden premises), "if_false" (what changes if each assumption fails), "strongest_assumption" (the most load-bearing one). Set "verdict" to exactly "ASSUMPTIONS_CHALLENGED" once every hidden assumption has been surfaced and pressure-tested.',
    ),
    (
        "identify_evidence_needed",
        'SOCRATIC STEP 3 — MAP EVIDENCE: Based on the definitions and assumptions identified, what specific evidence would confirm or deny this claim? What would prove it false (falsification conditions)? What\'s the minimum evidence needed for a judgment? Return JSON with: "evidence_needed" (specific evidence to seek), "falsification_conditions" (what would disprove the claim), "minimum_threshold" (least evidence needed for a judgment). Set "verdict" to exactly "EVIDENCE_MAPPED" once the evidence plan and falsification conditions are on the table.',
    ),
)
SOCRATIC_VERIFY = 'SOCRATIC STEP 4 — VERIFY: Verify that the Socratic analysis above is complete. Check: 1) Were key terms actually DEFINED (not just mentioned)? 2) Were ASSUMPTIONS explicitly identified and challenged? 3) Were specific FALSIFICATION CONDITIONS stated? If any step was superficial or missing, set "verdict" to "SOCRATIC_INCOMPLETE". If all three prior steps were genuinely performed, set "verdict" to exactly "VERIFIED".'
