"""Classify settled verdicts without inference, storage, or execution calls."""

from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from petri.lifecycle.config import LifecycleConfig
from petri.models import Verdict

TriggerVerdict = Literal[
    "EVIDENCE_INCONCLUSIVE", "CRITICAL_FLAW_FOUND",
    "DEFENSIBLE_WITH_CAVEATS", "PIPELINE_STALLED",
]


class RedecompositionTrigger(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    cell_id: str = Field(min_length=1)
    trigger_verdict: TriggerVerdict
    agent: str = Field(min_length=1)
    fire_point: Literal["settled_round", "circuit_breaker"]
    weakest_link: str | None = None
    focused_directive: str = ""
    caveat_text: str = ""
    flaw_text: str = ""
    generation: int = Field(default=0, ge=0)


class TriggerContext(BaseModel):
    """Provenance supplied by the caller from the settled round's recorded outcome."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    settled: bool = True
    generation: int = Field(default=0, ge=0)
    stall_reason: str | None = None
    weakest_link: str | None = None
    focused_directive: str = ""


_EXPECTED_AGENTS = {
    "EVIDENCE_INCONCLUSIVE": "evidence_evaluator",
    "CRITICAL_FLAW_FOUND": "skeptic",
    "DEFENSIBLE_WITH_CAVEATS": "champion",
    "PIPELINE_STALLED": "cell_lead",
}


def detect_triggers(
    verdicts: Sequence[Verdict],
    convergence_outcome: TriggerContext,
    agent_roles: Mapping[str, object],
    config: LifecycleConfig,
) -> list[RedecompositionTrigger]:
    """Use the latest verdict per cell and agent; exclude user and unknown stalls."""
    context = convergence_outcome
    if not context.settled or context.generation >= config.max_rounds_per_cell:
        return []
    latest = {(verdict.cell_id, verdict.agent): verdict for verdict in verdicts}
    result = []
    for key in sorted(latest):
        verdict = latest[key]
        if verdict.agent not in agent_roles or _EXPECTED_AGENTS.get(verdict.verdict) != verdict.agent:
            continue
        stalled = verdict.verdict == "PIPELINE_STALLED"
        if stalled and context.stall_reason != "circuit_breaker":
            continue
        result.append(RedecompositionTrigger.model_validate({
            "cell_id": verdict.cell_id, "trigger_verdict": verdict.verdict, "agent": verdict.agent,
            "fire_point": "circuit_breaker" if stalled else "settled_round",
            "weakest_link": context.weakest_link,
            "focused_directive": context.focused_directive or verdict.summary,
            "caveat_text": verdict.summary if verdict.verdict == "DEFENSIBLE_WITH_CAVEATS" else "",
            "flaw_text": verdict.summary if verdict.verdict == "CRITICAL_FLAW_FOUND" else "",
            "generation": context.generation,
        }))
    return result
