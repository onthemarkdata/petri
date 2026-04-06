"""Unit tests for petri/convergence.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from petri.convergence import (
    check_convergence,
    compute_circuit_breaker,
    evaluate_short_circuits,
    identify_weakest_link,
    load_agent_roles,
)
from petri.models import AgentRole


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def agent_roles():
    """Load the default agent roles from petri/defaults/agents.yaml."""
    return load_agent_roles()


def _make_verdict(agent: str, verdict: str) -> dict:
    """Helper to build a verdict dict."""
    return {"agent": agent, "verdict": verdict}


# ── load_agent_roles ─────────────────────────────────────────────────────


class TestLoadAgentRoles:
    def test_loads_default_agents(self, agent_roles):
        assert len(agent_roles) == 13
        assert "investigator" in agent_roles
        assert "node_lead" in agent_roles

    def test_leads_are_non_blocking(self, agent_roles):
        for name in ("decomposition_lead", "node_lead", "red_team_lead"):
            role = agent_roles[name]
            assert role.is_lead is True
            assert role.blocking == "false"

    def test_blocking_agents_count(self, agent_roles):
        blocking = [
            r for r in agent_roles.values()
            if r.blocking in ("true", "conditional") and not r.is_lead
        ]
        # 6 blocking: investigator, freshness_checker, dependency_auditor,
        # skeptic, champion, pragmatist + 1 conditional: triage = 7
        assert len(blocking) == 7

    def test_fallback_to_defaults_on_missing_path(self):
        roles = load_agent_roles(Path("/nonexistent/agents.yaml"))
        assert len(roles) == 13


# ── check_convergence ────────────────────────────────────────────────────


class TestCheckConvergence:
    def test_all_blocking_pass_converges(self, agent_roles):
        """All blocking verdicts in their pass set -> converged."""
        verdicts = [
            _make_verdict("investigator", "EVIDENCE_SUFFICIENT"),
            _make_verdict("freshness_checker", "EVIDENCE_CURRENT"),
            _make_verdict("dependency_auditor", "DEPENDENCIES_CLEAN"),
            _make_verdict("skeptic", "ARGUMENT_WITHSTANDS_CRITIQUE"),
            _make_verdict("champion", "STRONG_CASE"),
            _make_verdict("pragmatist", "PRODUCTION_READY"),
            _make_verdict("triage", "HIGH_VALUE"),
        ]
        result = check_convergence(verdicts, agent_roles)
        assert result["converged"] is True
        assert result["weakest_link"] is None
        assert len(result["missing_blocking"]) == 0

    def test_one_blocking_fails_not_converged(self, agent_roles):
        """One blocking verdict fails -> not converged."""
        verdicts = [
            _make_verdict("investigator", "NEEDS_MORE_EVIDENCE"),  # FAILS
            _make_verdict("freshness_checker", "EVIDENCE_CURRENT"),
            _make_verdict("dependency_auditor", "DEPENDENCIES_CLEAN"),
            _make_verdict("skeptic", "ARGUMENT_WITHSTANDS_CRITIQUE"),
            _make_verdict("champion", "STRONG_CASE"),
            _make_verdict("pragmatist", "PRODUCTION_READY"),
            _make_verdict("triage", "HIGH_VALUE"),
        ]
        result = check_convergence(verdicts, agent_roles)
        assert result["converged"] is False
        assert result["weakest_link"] == "investigator"

    def test_missing_blocking_agent_not_converged(self, agent_roles):
        """Missing a blocking agent -> not converged."""
        verdicts = [
            # investigator missing
            _make_verdict("freshness_checker", "EVIDENCE_CURRENT"),
            _make_verdict("dependency_auditor", "DEPENDENCIES_CLEAN"),
            _make_verdict("skeptic", "ARGUMENT_WITHSTANDS_CRITIQUE"),
            _make_verdict("champion", "STRONG_CASE"),
            _make_verdict("pragmatist", "PRODUCTION_READY"),
            _make_verdict("triage", "HIGH_VALUE"),
        ]
        result = check_convergence(verdicts, agent_roles)
        assert result["converged"] is False
        assert "investigator" in result["missing_blocking"]

    def test_non_blocking_verdicts_dont_affect_convergence(self, agent_roles):
        """Non-blocking agents (simplifier, impact_assessor) don't block convergence."""
        verdicts = [
            _make_verdict("investigator", "EVIDENCE_SUFFICIENT"),
            _make_verdict("freshness_checker", "EVIDENCE_CURRENT"),
            _make_verdict("dependency_auditor", "DEPENDENCIES_CLEAN"),
            _make_verdict("skeptic", "ARGUMENT_WITHSTANDS_CRITIQUE"),
            _make_verdict("champion", "STRONG_CASE"),
            _make_verdict("pragmatist", "PRODUCTION_READY"),
            _make_verdict("triage", "HIGH_VALUE"),
            # simplifier and impact_assessor have blocking verdicts but are non-blocking
            _make_verdict("simplifier", "OVERCOMPLICATED"),
            _make_verdict("impact_assessor", "ISOLATED_LOW_IMPACT"),
        ]
        result = check_convergence(verdicts, agent_roles)
        assert result["converged"] is True

    def test_lead_agents_excluded_from_convergence(self, agent_roles):
        """Lead agents are excluded from convergence checks entirely."""
        verdicts = [
            _make_verdict("investigator", "EVIDENCE_SUFFICIENT"),
            _make_verdict("freshness_checker", "EVIDENCE_CURRENT"),
            _make_verdict("dependency_auditor", "DEPENDENCIES_CLEAN"),
            _make_verdict("skeptic", "ARGUMENT_WITHSTANDS_CRITIQUE"),
            _make_verdict("champion", "STRONG_CASE"),
            _make_verdict("pragmatist", "PRODUCTION_READY"),
            _make_verdict("triage", "HIGH_VALUE"),
            # Lead agents' verdicts should not matter
            _make_verdict("node_lead", "PIPELINE_STALLED"),
        ]
        result = check_convergence(verdicts, agent_roles)
        assert result["converged"] is True
        # node_lead should not appear in blocking or non-blocking results
        assert "node_lead" not in result["blocking_results"]

    def test_latest_verdict_per_agent_used(self, agent_roles):
        """When an agent has multiple verdicts, the latest one wins."""
        verdicts = [
            _make_verdict("investigator", "NEEDS_MORE_EVIDENCE"),  # old
            _make_verdict("investigator", "EVIDENCE_SUFFICIENT"),  # latest
            _make_verdict("freshness_checker", "EVIDENCE_CURRENT"),
            _make_verdict("dependency_auditor", "DEPENDENCIES_CLEAN"),
            _make_verdict("skeptic", "ARGUMENT_WITHSTANDS_CRITIQUE"),
            _make_verdict("champion", "STRONG_CASE"),
            _make_verdict("pragmatist", "PRODUCTION_READY"),
            _make_verdict("triage", "HIGH_VALUE"),
        ]
        result = check_convergence(verdicts, agent_roles)
        assert result["converged"] is True

    def test_conditional_blocking_with_redirect(self, agent_roles):
        """Triage with LOW_VALUE_DEFER triggers redirect."""
        verdicts = [
            _make_verdict("investigator", "EVIDENCE_SUFFICIENT"),
            _make_verdict("freshness_checker", "EVIDENCE_CURRENT"),
            _make_verdict("dependency_auditor", "DEPENDENCIES_CLEAN"),
            _make_verdict("skeptic", "ARGUMENT_WITHSTANDS_CRITIQUE"),
            _make_verdict("champion", "STRONG_CASE"),
            _make_verdict("pragmatist", "PRODUCTION_READY"),
            _make_verdict("triage", "LOW_VALUE_DEFER"),
        ]
        result = check_convergence(verdicts, agent_roles)
        assert result["converged"] is False
        triage_result = result["blocking_results"]["triage"]
        assert triage_result["redirect"] == "DEFER_OPEN"

    def test_empty_verdicts_not_converged(self, agent_roles):
        """No verdicts at all -> not converged, all blocking agents missing."""
        result = check_convergence([], agent_roles)
        assert result["converged"] is False
        assert len(result["missing_blocking"]) > 0

    def test_evidence_contradicts_passes(self, agent_roles):
        """EVIDENCE_CONTRADICTS is a pass verdict for investigator."""
        verdicts = [
            _make_verdict("investigator", "EVIDENCE_CONTRADICTS"),
            _make_verdict("freshness_checker", "STALE_BUT_HOLDS"),
            _make_verdict("dependency_auditor", "DEPENDENCIES_CLEAN"),
            _make_verdict("skeptic", "ARGUMENT_WITHSTANDS_CRITIQUE"),
            _make_verdict("champion", "DEFENSIBLE_WITH_CAVEATS"),
            _make_verdict("pragmatist", "DIRECTIONALLY_CORRECT"),
            _make_verdict("triage", "MODERATE_VALUE"),
        ]
        result = check_convergence(verdicts, agent_roles)
        assert result["converged"] is True


# ── identify_weakest_link ────────────────────────────────────────────────


class TestIdentifyWeakestLink:
    def test_missing_agent_is_weakest(self):
        result = {"missing_blocking": ["investigator"], "blocking_results": {}}
        assert identify_weakest_link(result) == "investigator"

    def test_failing_agent_is_weakest(self):
        result = {
            "missing_blocking": [],
            "blocking_results": {
                "investigator": {"verdict": "EVIDENCE_SUFFICIENT", "passes": True},
                "skeptic": {"verdict": "CRITICAL_FLAW_FOUND", "passes": False},
            },
        }
        assert identify_weakest_link(result) == "skeptic"

    def test_all_pass_returns_none(self):
        result = {
            "missing_blocking": [],
            "blocking_results": {
                "investigator": {"verdict": "EVIDENCE_SUFFICIENT", "passes": True},
            },
        }
        assert identify_weakest_link(result) is None

    def test_missing_takes_priority_over_failure(self):
        result = {
            "missing_blocking": ["freshness_checker"],
            "blocking_results": {
                "investigator": {"verdict": "NEEDS_MORE_EVIDENCE", "passes": False},
            },
        }
        assert identify_weakest_link(result) == "freshness_checker"


# ── evaluate_short_circuits ──────────────────────────────────────────────


class TestEvaluateShortCircuits:
    def test_cannot_determine_triggers_needs_experiment(self, agent_roles):
        """CANNOT_DETERMINE from investigator + all others pass -> needs_experiment."""
        verdicts = [
            _make_verdict("investigator", "CANNOT_DETERMINE"),
            _make_verdict("freshness_checker", "EVIDENCE_CURRENT"),
            _make_verdict("dependency_auditor", "DEPENDENCIES_CLEAN"),
            _make_verdict("skeptic", "ARGUMENT_WITHSTANDS_CRITIQUE"),
            _make_verdict("champion", "STRONG_CASE"),
            _make_verdict("pragmatist", "PRODUCTION_READY"),
            _make_verdict("triage", "HIGH_VALUE"),
        ]
        sc = evaluate_short_circuits(verdicts, agent_roles)
        assert sc is not None
        assert sc["type"] == "needs_experiment"
        assert sc["agent"] == "investigator"

    def test_low_value_defer_triggers_defer_open(self, agent_roles):
        """LOW_VALUE_DEFER from triage + all others pass -> defer_open."""
        verdicts = [
            _make_verdict("investigator", "EVIDENCE_SUFFICIENT"),
            _make_verdict("freshness_checker", "EVIDENCE_CURRENT"),
            _make_verdict("dependency_auditor", "DEPENDENCIES_CLEAN"),
            _make_verdict("skeptic", "ARGUMENT_WITHSTANDS_CRITIQUE"),
            _make_verdict("champion", "STRONG_CASE"),
            _make_verdict("pragmatist", "PRODUCTION_READY"),
            _make_verdict("triage", "LOW_VALUE_DEFER"),
        ]
        sc = evaluate_short_circuits(verdicts, agent_roles)
        assert sc is not None
        assert sc["type"] == "defer_open"
        assert sc["agent"] == "triage"

    def test_no_short_circuit_when_all_pass(self, agent_roles):
        """All pass -> no short circuit."""
        verdicts = [
            _make_verdict("investigator", "EVIDENCE_SUFFICIENT"),
            _make_verdict("freshness_checker", "EVIDENCE_CURRENT"),
            _make_verdict("dependency_auditor", "DEPENDENCIES_CLEAN"),
            _make_verdict("skeptic", "ARGUMENT_WITHSTANDS_CRITIQUE"),
            _make_verdict("champion", "STRONG_CASE"),
            _make_verdict("pragmatist", "PRODUCTION_READY"),
            _make_verdict("triage", "HIGH_VALUE"),
        ]
        sc = evaluate_short_circuits(verdicts, agent_roles)
        assert sc is None

    def test_no_short_circuit_when_multiple_fail(self, agent_roles):
        """CANNOT_DETERMINE + another failure -> no short circuit."""
        verdicts = [
            _make_verdict("investigator", "CANNOT_DETERMINE"),
            _make_verdict("freshness_checker", "MATERIALLY_STALE"),  # also fails
            _make_verdict("dependency_auditor", "DEPENDENCIES_CLEAN"),
            _make_verdict("skeptic", "ARGUMENT_WITHSTANDS_CRITIQUE"),
            _make_verdict("champion", "STRONG_CASE"),
            _make_verdict("pragmatist", "PRODUCTION_READY"),
            _make_verdict("triage", "HIGH_VALUE"),
        ]
        sc = evaluate_short_circuits(verdicts, agent_roles)
        assert sc is None

    def test_no_short_circuit_with_missing_agents(self, agent_roles):
        """CANNOT_DETERMINE but missing agents -> no short circuit."""
        verdicts = [
            _make_verdict("investigator", "CANNOT_DETERMINE"),
            # freshness_checker missing
            _make_verdict("dependency_auditor", "DEPENDENCIES_CLEAN"),
            _make_verdict("skeptic", "ARGUMENT_WITHSTANDS_CRITIQUE"),
            _make_verdict("champion", "STRONG_CASE"),
            _make_verdict("pragmatist", "PRODUCTION_READY"),
            _make_verdict("triage", "HIGH_VALUE"),
        ]
        sc = evaluate_short_circuits(verdicts, agent_roles)
        assert sc is None


# ── compute_circuit_breaker ──────────────────────────────────────────────


class TestComputeCircuitBreaker:
    def test_fires_at_max(self):
        assert compute_circuit_breaker(iteration=3, cycle_start_iteration=0, max_iterations=3) is True

    def test_does_not_fire_below_max(self):
        assert compute_circuit_breaker(iteration=2, cycle_start_iteration=0, max_iterations=3) is False

    def test_relative_counting(self):
        """Circuit breaker uses relative counting: iteration - cycle_start."""
        assert compute_circuit_breaker(iteration=7, cycle_start_iteration=5, max_iterations=3) is False
        assert compute_circuit_breaker(iteration=8, cycle_start_iteration=5, max_iterations=3) is True

    def test_fires_past_max(self):
        assert compute_circuit_breaker(iteration=10, cycle_start_iteration=0, max_iterations=3) is True

    def test_zero_iterations_does_not_fire(self):
        assert compute_circuit_breaker(iteration=0, cycle_start_iteration=0, max_iterations=3) is False

    def test_custom_max_iterations(self):
        assert compute_circuit_breaker(iteration=4, cycle_start_iteration=0, max_iterations=5) is False
        assert compute_circuit_breaker(iteration=5, cycle_start_iteration=0, max_iterations=5) is True

    def test_max_iterations_of_one(self):
        assert compute_circuit_breaker(iteration=1, cycle_start_iteration=0, max_iterations=1) is True
        assert compute_circuit_breaker(iteration=0, cycle_start_iteration=0, max_iterations=1) is False
