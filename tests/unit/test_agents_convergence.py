"""Reuse legacy scenarios and compare randomized verdict histories against v2."""

import random
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from petri.agents import convergence as v2
from petri.agents.contract import AGENT_SPECS, AgentName, BlockingMode
from petri.analysis import convergence as legacy
from tests.conftest import make_verdict
from tests.unit import test_convergence as scenarios

GROUPS = [
    scenarios.TestCheckConvergence,
    scenarios.TestIdentifyWeakestLink,
    scenarios.TestEvaluateShortCircuits,
    scenarios.TestComputeCircuitBreaker,
]
TWINS = [
    (group, method) for group in GROUPS for method in vars(group) if method.startswith("test_")
]


def legacy_view(verdicts, _roles):
    return SimpleNamespace(**v2.check_convergence(verdicts).model_dump(mode="json"))


@pytest.mark.parametrize("group,method", TWINS, ids=[f"{g.__name__}.{m}" for g, m in TWINS])
def test_legacy_scenario_has_v2_twin(group, method, monkeypatch):
    monkeypatch.setattr(scenarios, "check_convergence", legacy_view)
    monkeypatch.setattr(scenarios, "compute_circuit_breaker", v2.compute_circuit_breaker)
    monkeypatch.setattr(
        scenarios,
        "evaluate_short_circuits",
        lambda verdicts, roles: v2.evaluate_short_circuits(v2.check_convergence(verdicts)),
    )
    monkeypatch.setattr(
        scenarios,
        "identify_weakest_link",
        lambda votes, missing: v2.identify_weakest_link(
            {AgentName(name): v2.AgentVote.model_validate(vote) for name, vote in votes.items()},
            [AgentName(name) for name in missing],
        ),
    )
    bound = getattr(group(), method)
    if group in {scenarios.TestCheckConvergence, scenarios.TestEvaluateShortCircuits}:
        bound(legacy.load_agent_roles())
    else:
        bound()


def passing():
    return [
        make_verdict(name.value, spec.verdicts_pass[0])
        for name, spec in AGENT_SPECS.items()
        if spec.blocking_mode != BlockingMode.NON_BLOCKING
    ]


def test_randomized_differential_parity():
    rng = random.Random(417)
    roles = legacy.load_agent_roles()
    roles = {name.value: roles[name.value] for name in AgentName}
    for _ in range(300):
        rows = []
        for name in AgentName:
            spec = AGENT_SPECS[name]
            for _ in range(rng.randrange(3)):
                rows.append(
                    make_verdict(
                        name.value,
                        rng.choice(
                            [*spec.verdicts_pass, *spec.verdicts_block, "EXECUTION_ERROR", ""]
                        ),
                    )
                )
        rng.shuffle(rows)
        old, new = legacy.check_convergence(rows, roles), v2.check_convergence(rows)
        assert new.converged == old.converged
        assert new.weakest_link == old.weakest_link
        assert new.missing_blocking == old.missing_blocking
        old_short = legacy.evaluate_short_circuits(rows, roles)
        new_short = v2.evaluate_short_circuits(new)
        assert (new_short.model_dump() if new_short else None) == (
            old_short.model_dump() if old_short else None
        )


def test_execution_failure_is_failing_not_missing():
    rows = passing() + [make_verdict("investigator", "EXECUTION_ERROR")]
    result = v2.check_convergence(rows)
    assert not result.converged
    assert result.missing_blocking == []
    assert result.weakest_link == AgentName.INVESTIGATOR
    assert result.blocking_results[AgentName.INVESTIGATOR].passes is False
    assert all("EXECUTION_ERROR" not in spec.verdicts_pass for spec in AGENT_SPECS.values())


def test_order_independent_of_spec_and_verdict_insertion():
    rows = passing() + [
        make_verdict("investigator", "NEEDS_MORE_EVIDENCE"),
        make_verdict("skeptic", "CRITICAL_FLAW_FOUND"),
    ]
    # Keep last-write ordering meaningful while permuting independent latest verdicts.
    latest = {row.agent: row for row in rows}
    items = list(AGENT_SPECS.items())
    rng = random.Random(3)
    for _ in range(20):
        rng.shuffle(items)
        distinct = list(latest.values())
        rng.shuffle(distinct)
        assert v2.check_convergence(distinct, dict(items)).weakest_link == AgentName.INVESTIGATOR


def test_no_io_or_recomputation_in_short_circuit():
    result = v2.check_convergence(passing() + [make_verdict("triage", "LOW_VALUE_DEFER")])
    with (
        patch("builtins.open", side_effect=AssertionError("I/O")),
        patch.object(v2, "check_convergence", side_effect=AssertionError("recompute")),
    ):
        assert v2.evaluate_short_circuits(result).type == "defer_open"


def test_outcome_rejects_unknown_state():
    with pytest.raises(ValidationError):
        v2.ConvergenceOutcome(outcome="looks_good")
    assert v2.ConvergenceOutcome(outcome="converged").outcome is v2.Outcome.CONVERGED
