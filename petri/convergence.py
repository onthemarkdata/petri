"""Convergence engine for Petri validation pipeline.

Checks whether all blocking agents have issued passing verdicts, identifies
the weakest link when convergence fails, evaluates short-circuit conditions,
and enforces the iteration circuit breaker.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from petri.models import AgentRole, ConvergenceCheckResult, Debate, ShortCircuitCondition, Verdict

# ── Defaults ──────────────────────────────────────────────────────────────

_DEFAULTS_DIR = Path(__file__).parent / "defaults"


# ── Config Loaders ────────────────────────────────────────────────────────


def load_agent_roles(config_path: Path | None = None) -> dict[str, AgentRole]:
    """Load agent roles from petri.yaml.

    Falls back to ``petri/defaults/petri.yaml`` when *config_path* is None
    or points to a file that does not exist.

    *config_path* may point to either ``petri.yaml`` (consolidated) or
    a legacy ``agents.yaml`` — both are supported.

    Returns a dict of agent name -> AgentRole.
    """
    path = config_path if config_path and config_path.exists() else _DEFAULTS_DIR / "petri.yaml"
    raw = yaml.safe_load(path.read_text())
    agents: dict[str, AgentRole] = {}
    for name, cfg in raw.get("agents", {}).items():
        # YAML parses bare ``true``/``false`` as booleans — coerce to the
        # string representation that AgentRole expects.
        if "blocking" in cfg:
            cfg["blocking"] = str(cfg["blocking"]).lower()
        agents[name] = AgentRole(name=name, **cfg)
    return agents


# ── Core Convergence ──────────────────────────────────────────────────────


def check_convergence(
    verdicts: list[Verdict],
    agent_roles: dict[str, AgentRole],
) -> ConvergenceCheckResult:
    """Check if all blocking verdicts are in their pass set.

    Parameters
    ----------
    verdicts:
        List of ``Verdict`` models (as returned by ``get_verdicts``).
    agent_roles:
        Dict of agent name -> AgentRole (from ``load_agent_roles``).

    Returns
    -------
    ConvergenceCheckResult with converged, blocking_results,
    non_blocking_results, missing_blocking, weakest_link.
    """
    # Index verdicts by agent — use latest verdict per agent.
    verdict_by_agent: dict[str, str] = {}
    for verdict_entry in verdicts:
        agent = verdict_entry.agent
        verdict_str = verdict_entry.verdict
        if agent and verdict_str:
            verdict_by_agent[agent] = verdict_str

    blocking_results: dict[str, dict] = {}
    non_blocking_results: dict[str, dict] = {}
    missing_blocking: list[str] = []
    weakest_link: str | None = None

    for name, role in agent_roles.items():
        # Skip lead agents entirely — they are orchestrators, not voters.
        if role.is_lead:
            continue

        is_blocking = role.blocking in ("true", True)
        is_conditional = role.blocking == "conditional"

        if name in verdict_by_agent:
            agent_verdict = verdict_by_agent[name]
            passes = agent_verdict in role.verdicts_pass

            entry: dict = {"verdict": agent_verdict, "passes": passes}

            # Conditional blocking with redirect.
            if is_conditional and agent_verdict in role.verdicts_block and role.redirect_on_block:
                entry["redirect"] = role.redirect_on_block

            if is_blocking or is_conditional:
                blocking_results[name] = entry
            else:
                non_blocking_results[name] = entry
        else:
            # Agent has not yet issued a verdict.
            if is_blocking or is_conditional:
                missing_blocking.append(name)
            else:
                non_blocking_results[name] = {"verdict": None, "passes": False}

    # Determine overall convergence.
    all_blocking_pass = all(result["passes"] for result in blocking_results.values())
    converged = all_blocking_pass and len(missing_blocking) == 0

    # Identify the weakest link.
    weakest_link = identify_weakest_link(blocking_results, missing_blocking)

    return ConvergenceCheckResult(
        converged=converged,
        blocking_results=blocking_results,
        non_blocking_results=non_blocking_results,
        missing_blocking=missing_blocking,
        weakest_link=weakest_link,
    )


# ── Weakest Link ──────────────────────────────────────────────────────────


def identify_weakest_link(
    blocking_results: dict[str, dict],
    missing_blocking: list[str],
) -> str | None:
    """Find the first blocking agent that failed.

    Returns the agent name or ``None`` if all blocking agents pass.
    """
    # Check missing blocking agents first — they are the most critical gap.
    if missing_blocking:
        return missing_blocking[0]

    # Then check for explicit failures among blocking results.
    for name, entry in blocking_results.items():
        if not entry.get("passes", False):
            return name

    return None


# ── Short-Circuit Evaluation ──────────────────────────────────────────────


def evaluate_short_circuits(
    verdicts: list[Verdict],
    agent_roles: dict[str, AgentRole],
) -> ShortCircuitCondition | None:
    """Check for short-circuit conditions.

    Derives short-circuit rules from agent config (redirect_on_block and
    CANNOT_DETERMINE verdicts) rather than hardcoding agent names.

    A short-circuit fires when one agent issues a triggering verdict AND
    all other blocking agents pass.

    Returns ``None`` if no short-circuit applies, otherwise a
    ``ShortCircuitCondition``.
    """
    from petri.config import get_short_circuit_rules

    convergence = check_convergence(verdicts, agent_roles)
    blocking = convergence.blocking_results

    if convergence.missing_blocking:
        return None

    for rule in get_short_circuit_rules():
        triggering_agent = rule["agent"]
        triggering_verdict = rule["verdict"]
        circuit_type = rule["type"]

        agent_result = blocking.get(triggering_agent)
        if agent_result and agent_result["verdict"] == triggering_verdict:
            others_pass = all(
                entry["passes"]
                for agent_name, entry in blocking.items()
                if agent_name != triggering_agent
            )
            if others_pass:
                return ShortCircuitCondition(
                    type=circuit_type,
                    agent=triggering_agent,
                    verdict=triggering_verdict,
                )

    return None


# ── Circuit Breaker ───────────────────────────────────────────────────────


def compute_circuit_breaker(
    iteration: int,
    cycle_start_iteration: int,
    max_iterations: int = 3,
) -> bool:
    """Return ``True`` if the circuit breaker should fire.

    The breaker fires when the number of iterations since the cycle started
    reaches or exceeds *max_iterations*.
    """
    return (iteration - cycle_start_iteration) >= max_iterations
