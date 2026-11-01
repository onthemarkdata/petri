"""Mechanical convergence on the typed roster; no configuration or inference I/O."""

from collections.abc import Iterable, Mapping
from enum import StrEnum
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict

from petri.agents.contract import AGENT_SPECS, AgentName, AgentSpec, BlockingMode


class VerdictRecord(Protocol):
    @property
    def agent(self) -> str: ...

    @property
    def verdict(self) -> str: ...


class AgentVote(BaseModel):
    model_config = ConfigDict(frozen=True)
    verdict: str | None
    passes: bool
    redirect: str | None = None


class ConvergenceResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    converged: bool
    blocking_results: dict[AgentName, AgentVote]
    non_blocking_results: dict[AgentName, AgentVote]
    missing_blocking: list[AgentName]
    weakest_link: AgentName | None


class ShortCircuit(BaseModel):
    model_config = ConfigDict(frozen=True)
    type: Literal["needs_experiment", "defer_open"]
    agent: AgentName
    verdict: str


class Outcome(StrEnum):
    CONVERGED = "converged"
    ITERATE = "iterate"
    CIRCUIT_BREAKER = "circuit_breaker"
    SHORT_CIRCUIT = "short_circuit"


class ConvergenceOutcome(BaseModel):
    outcome: Outcome
    type: Literal["needs_experiment", "defer_open"] | None = None
    weakest_link: AgentName | None = None


def identify_weakest_link(
    blocking_results: Mapping[AgentName, AgentVote],
    missing_blocking: Iterable[AgentName],
) -> AgentName | None:
    """Prefer missing verdicts, then failures, in canonical roster order."""
    missing = set(missing_blocking)
    for name in AgentName:
        if name in missing:
            return name
    for name in AgentName:
        if name in blocking_results and not blocking_results[name].passes:
            return name
    return None


def check_convergence(
    verdicts: Iterable[VerdictRecord],
    specs: Mapping[AgentName, AgentSpec] = AGENT_SPECS,
) -> ConvergenceResult:
    """Apply the last nonempty verdict per agent, excluding leads from voting."""
    latest = {row.agent: row.verdict for row in verdicts if row.agent and row.verdict}
    blocking: dict[AgentName, AgentVote] = {}
    advisory: dict[AgentName, AgentVote] = {}
    missing: list[AgentName] = []
    for name in AgentName:
        spec = specs.get(name)
        if spec is None or spec.is_lead:
            continue
        blocks = spec.blocking_mode != BlockingMode.NON_BLOCKING
        verdict = latest.get(name)
        if verdict is None:
            if blocks:
                missing.append(name)
            else:
                advisory[name] = AgentVote(verdict=None, passes=False)
            continue
        passes = verdict != "EXECUTION_ERROR" and verdict in spec.verdicts_pass
        redirect = None
        if spec.blocking_mode == BlockingMode.CONDITIONAL and verdict in spec.verdicts_block:
            redirect = spec.redirect_on_block
        vote = AgentVote(verdict=verdict, passes=passes, redirect=redirect)
        (blocking if blocks else advisory)[name] = vote
    return ConvergenceResult(
        converged=not missing and all(vote.passes for vote in blocking.values()),
        blocking_results=blocking,
        non_blocking_results=advisory,
        missing_blocking=missing,
        weakest_link=identify_weakest_link(blocking, missing),
    )


def evaluate_short_circuits(
    convergence: ConvergenceResult,
    specs: Mapping[AgentName, AgentSpec] = AGENT_SPECS,
) -> ShortCircuit | None:
    """Evaluate typed rules against an already-computed convergence result."""
    if convergence.missing_blocking:
        return None
    for name in AgentName:
        spec = specs.get(name)
        vote = convergence.blocking_results.get(name)
        if spec is None or vote is None or vote.verdict not in spec.verdicts_block:
            continue
        if spec.redirect_on_block:
            circuit_type = spec.redirect_on_block.lower()
        elif vote.verdict == "CANNOT_DETERMINE":
            circuit_type = "needs_experiment"
        else:
            continue
        if all(
            other.passes for agent, other in convergence.blocking_results.items() if agent != name
        ):
            return ShortCircuit.model_validate(
                {"type": circuit_type, "agent": name, "verdict": vote.verdict}
            )
    return None


def compute_circuit_breaker(
    iteration: int, cycle_start_iteration: int, max_iterations: int = 3
) -> bool:
    """Count iterations relative to the current cycle's starting point."""
    return iteration - cycle_start_iteration >= max_iterations
