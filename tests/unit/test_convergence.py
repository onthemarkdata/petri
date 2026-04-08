"""Unit tests for petri/convergence.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from petri.analysis.convergence import (
    check_convergence,
    compute_circuit_breaker,
    evaluate_short_circuits,
    identify_weakest_link,
    load_agent_roles,
)
from petri.models import AgentRole, Verdict

from tests.conftest import make_verdict


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def agent_roles():
    """Load the default agent roles from petri/defaults/agents.yaml."""
    return load_agent_roles()


# ── load_agent_roles ─────────────────────────────────────────────────────


class TestLoadAgentRoles:
    def test_loads_default_agents(self, agent_roles):
        # 13 original agents + socratic_questioner (exploratory phase 0)
        assert len(agent_roles) == 14
        assert "investigator" in agent_roles
        assert "node_lead" in agent_roles
        assert "socratic_questioner" in agent_roles

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
        # 13 original agents + socratic_questioner (exploratory phase 0)
        assert len(roles) == 14


# ── check_convergence ────────────────────────────────────────────────────


class TestCheckConvergence:
    def test_all_blocking_pass_converges(self, agent_roles):
        """All blocking verdicts in their pass set -> converged."""
        verdicts = [
            make_verdict("investigator", "EVIDENCE_SUFFICIENT"),
            make_verdict("freshness_checker", "EVIDENCE_CURRENT"),
            make_verdict("dependency_auditor", "DEPENDENCIES_CLEAN"),
            make_verdict("skeptic", "ARGUMENT_WITHSTANDS_CRITIQUE"),
            make_verdict("champion", "STRONG_CASE"),
            make_verdict("pragmatist", "PRODUCTION_READY"),
            make_verdict("triage", "HIGH_VALUE"),
        ]
        result = check_convergence(verdicts, agent_roles)
        assert result.converged is True
        assert result.weakest_link is None
        assert len(result.missing_blocking) == 0

    def test_one_blocking_fails_not_converged(self, agent_roles):
        """One blocking verdict fails -> not converged."""
        verdicts = [
            make_verdict("investigator", "NEEDS_MORE_EVIDENCE"),  # FAILS
            make_verdict("freshness_checker", "EVIDENCE_CURRENT"),
            make_verdict("dependency_auditor", "DEPENDENCIES_CLEAN"),
            make_verdict("skeptic", "ARGUMENT_WITHSTANDS_CRITIQUE"),
            make_verdict("champion", "STRONG_CASE"),
            make_verdict("pragmatist", "PRODUCTION_READY"),
            make_verdict("triage", "HIGH_VALUE"),
        ]
        result = check_convergence(verdicts, agent_roles)
        assert result.converged is False
        assert result.weakest_link == "investigator"

    def test_missing_blocking_agent_not_converged(self, agent_roles):
        """Missing a blocking agent -> not converged."""
        verdicts = [
            # investigator missing
            make_verdict("freshness_checker", "EVIDENCE_CURRENT"),
            make_verdict("dependency_auditor", "DEPENDENCIES_CLEAN"),
            make_verdict("skeptic", "ARGUMENT_WITHSTANDS_CRITIQUE"),
            make_verdict("champion", "STRONG_CASE"),
            make_verdict("pragmatist", "PRODUCTION_READY"),
            make_verdict("triage", "HIGH_VALUE"),
        ]
        result = check_convergence(verdicts, agent_roles)
        assert result.converged is False
        assert "investigator" in result.missing_blocking

    def test_non_blocking_verdicts_dont_affect_convergence(self, agent_roles):
        """Non-blocking agents (simplifier, impact_assessor) don't block convergence."""
        verdicts = [
            make_verdict("investigator", "EVIDENCE_SUFFICIENT"),
            make_verdict("freshness_checker", "EVIDENCE_CURRENT"),
            make_verdict("dependency_auditor", "DEPENDENCIES_CLEAN"),
            make_verdict("skeptic", "ARGUMENT_WITHSTANDS_CRITIQUE"),
            make_verdict("champion", "STRONG_CASE"),
            make_verdict("pragmatist", "PRODUCTION_READY"),
            make_verdict("triage", "HIGH_VALUE"),
            # simplifier and impact_assessor have blocking verdicts but are non-blocking
            make_verdict("simplifier", "OVERCOMPLICATED"),
            make_verdict("impact_assessor", "ISOLATED_LOW_IMPACT"),
        ]
        result = check_convergence(verdicts, agent_roles)
        assert result.converged is True

    def test_lead_agents_excluded_from_convergence(self, agent_roles):
        """Lead agents are excluded from convergence checks entirely."""
        verdicts = [
            make_verdict("investigator", "EVIDENCE_SUFFICIENT"),
            make_verdict("freshness_checker", "EVIDENCE_CURRENT"),
            make_verdict("dependency_auditor", "DEPENDENCIES_CLEAN"),
            make_verdict("skeptic", "ARGUMENT_WITHSTANDS_CRITIQUE"),
            make_verdict("champion", "STRONG_CASE"),
            make_verdict("pragmatist", "PRODUCTION_READY"),
            make_verdict("triage", "HIGH_VALUE"),
            # Lead agents' verdicts should not matter
            make_verdict("node_lead", "PIPELINE_STALLED"),
        ]
        result = check_convergence(verdicts, agent_roles)
        assert result.converged is True
        # node_lead should not appear in blocking or non-blocking results
        assert "node_lead" not in result.blocking_results

    def test_latest_verdict_per_agent_used(self, agent_roles):
        """When an agent has multiple verdicts, the latest one wins."""
        verdicts = [
            make_verdict("investigator", "NEEDS_MORE_EVIDENCE"),  # old
            make_verdict("investigator", "EVIDENCE_SUFFICIENT"),  # latest
            make_verdict("freshness_checker", "EVIDENCE_CURRENT"),
            make_verdict("dependency_auditor", "DEPENDENCIES_CLEAN"),
            make_verdict("skeptic", "ARGUMENT_WITHSTANDS_CRITIQUE"),
            make_verdict("champion", "STRONG_CASE"),
            make_verdict("pragmatist", "PRODUCTION_READY"),
            make_verdict("triage", "HIGH_VALUE"),
        ]
        result = check_convergence(verdicts, agent_roles)
        assert result.converged is True

    def test_conditional_blocking_with_redirect(self, agent_roles):
        """Triage with LOW_VALUE_DEFER triggers redirect."""
        verdicts = [
            make_verdict("investigator", "EVIDENCE_SUFFICIENT"),
            make_verdict("freshness_checker", "EVIDENCE_CURRENT"),
            make_verdict("dependency_auditor", "DEPENDENCIES_CLEAN"),
            make_verdict("skeptic", "ARGUMENT_WITHSTANDS_CRITIQUE"),
            make_verdict("champion", "STRONG_CASE"),
            make_verdict("pragmatist", "PRODUCTION_READY"),
            make_verdict("triage", "LOW_VALUE_DEFER"),
        ]
        result = check_convergence(verdicts, agent_roles)
        assert result.converged is False
        triage_result = result.blocking_results["triage"]
        assert triage_result["redirect"] == "DEFER_OPEN"

    def test_empty_verdicts_not_converged(self, agent_roles):
        """No verdicts at all -> not converged, all blocking agents missing."""
        result = check_convergence([], agent_roles)
        assert result.converged is False
        assert len(result.missing_blocking) > 0

    def test_evidence_contradicts_passes(self, agent_roles):
        """EVIDENCE_CONTRADICTS is a pass verdict for investigator."""
        verdicts = [
            make_verdict("investigator", "EVIDENCE_CONTRADICTS"),
            make_verdict("freshness_checker", "STALE_BUT_HOLDS"),
            make_verdict("dependency_auditor", "DEPENDENCIES_CLEAN"),
            make_verdict("skeptic", "ARGUMENT_WITHSTANDS_CRITIQUE"),
            make_verdict("champion", "DEFENSIBLE_WITH_CAVEATS"),
            make_verdict("pragmatist", "DIRECTIONALLY_CORRECT"),
            make_verdict("triage", "MODERATE_VALUE"),
        ]
        result = check_convergence(verdicts, agent_roles)
        assert result.converged is True


# ── identify_weakest_link ────────────────────────────────────────────────


class TestIdentifyWeakestLink:
    def test_missing_agent_is_weakest(self):
        blocking_results = {}
        missing_blocking = ["investigator"]
        assert identify_weakest_link(blocking_results, missing_blocking) == "investigator"

    def test_failing_agent_is_weakest(self):
        blocking_results = {
            "investigator": {"verdict": "EVIDENCE_SUFFICIENT", "passes": True},
            "skeptic": {"verdict": "CRITICAL_FLAW_FOUND", "passes": False},
        }
        missing_blocking = []
        assert identify_weakest_link(blocking_results, missing_blocking) == "skeptic"

    def test_all_pass_returns_none(self):
        blocking_results = {
            "investigator": {"verdict": "EVIDENCE_SUFFICIENT", "passes": True},
        }
        missing_blocking = []
        assert identify_weakest_link(blocking_results, missing_blocking) is None

    def test_missing_takes_priority_over_failure(self):
        blocking_results = {
            "investigator": {"verdict": "NEEDS_MORE_EVIDENCE", "passes": False},
        }
        missing_blocking = ["freshness_checker"]
        assert identify_weakest_link(blocking_results, missing_blocking) == "freshness_checker"


# ── evaluate_short_circuits ──────────────────────────────────────────────


class TestEvaluateShortCircuits:
    def test_cannot_determine_triggers_needs_experiment(self, agent_roles):
        """CANNOT_DETERMINE from investigator + all others pass -> needs_experiment."""
        verdicts = [
            make_verdict("investigator", "CANNOT_DETERMINE"),
            make_verdict("freshness_checker", "EVIDENCE_CURRENT"),
            make_verdict("dependency_auditor", "DEPENDENCIES_CLEAN"),
            make_verdict("skeptic", "ARGUMENT_WITHSTANDS_CRITIQUE"),
            make_verdict("champion", "STRONG_CASE"),
            make_verdict("pragmatist", "PRODUCTION_READY"),
            make_verdict("triage", "HIGH_VALUE"),
        ]
        sc = evaluate_short_circuits(verdicts, agent_roles)
        assert sc is not None
        assert sc.type == "needs_experiment"
        assert sc.agent == "investigator"

    def test_low_value_defer_triggers_defer_open(self, agent_roles):
        """LOW_VALUE_DEFER from triage + all others pass -> defer_open."""
        verdicts = [
            make_verdict("investigator", "EVIDENCE_SUFFICIENT"),
            make_verdict("freshness_checker", "EVIDENCE_CURRENT"),
            make_verdict("dependency_auditor", "DEPENDENCIES_CLEAN"),
            make_verdict("skeptic", "ARGUMENT_WITHSTANDS_CRITIQUE"),
            make_verdict("champion", "STRONG_CASE"),
            make_verdict("pragmatist", "PRODUCTION_READY"),
            make_verdict("triage", "LOW_VALUE_DEFER"),
        ]
        sc = evaluate_short_circuits(verdicts, agent_roles)
        assert sc is not None
        assert sc.type == "defer_open"
        assert sc.agent == "triage"

    def test_no_short_circuit_when_all_pass(self, agent_roles):
        """All pass -> no short circuit."""
        verdicts = [
            make_verdict("investigator", "EVIDENCE_SUFFICIENT"),
            make_verdict("freshness_checker", "EVIDENCE_CURRENT"),
            make_verdict("dependency_auditor", "DEPENDENCIES_CLEAN"),
            make_verdict("skeptic", "ARGUMENT_WITHSTANDS_CRITIQUE"),
            make_verdict("champion", "STRONG_CASE"),
            make_verdict("pragmatist", "PRODUCTION_READY"),
            make_verdict("triage", "HIGH_VALUE"),
        ]
        sc = evaluate_short_circuits(verdicts, agent_roles)
        assert sc is None

    def test_no_short_circuit_when_multiple_fail(self, agent_roles):
        """CANNOT_DETERMINE + another failure -> no short circuit."""
        verdicts = [
            make_verdict("investigator", "CANNOT_DETERMINE"),
            make_verdict("freshness_checker", "MATERIALLY_STALE"),  # also fails
            make_verdict("dependency_auditor", "DEPENDENCIES_CLEAN"),
            make_verdict("skeptic", "ARGUMENT_WITHSTANDS_CRITIQUE"),
            make_verdict("champion", "STRONG_CASE"),
            make_verdict("pragmatist", "PRODUCTION_READY"),
            make_verdict("triage", "HIGH_VALUE"),
        ]
        sc = evaluate_short_circuits(verdicts, agent_roles)
        assert sc is None

    def test_no_short_circuit_with_missing_agents(self, agent_roles):
        """CANNOT_DETERMINE but missing agents -> no short circuit."""
        verdicts = [
            make_verdict("investigator", "CANNOT_DETERMINE"),
            # freshness_checker missing
            make_verdict("dependency_auditor", "DEPENDENCIES_CLEAN"),
            make_verdict("skeptic", "ARGUMENT_WITHSTANDS_CRITIQUE"),
            make_verdict("champion", "STRONG_CASE"),
            make_verdict("pragmatist", "PRODUCTION_READY"),
            make_verdict("triage", "HIGH_VALUE"),
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
