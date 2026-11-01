"""Signal verdicts produce proposals only with the required provenance."""

import ast
from pathlib import Path

import pytest

from petri.lifecycle.config import LifecycleConfig
from petri.lifecycle.triggers import TriggerContext, detect_triggers
from petri.models import Verdict


@pytest.mark.parametrize("agent,verdict,reason", [
    ("evidence_evaluator", "EVIDENCE_INCONCLUSIVE", None),
    ("skeptic", "CRITICAL_FLAW_FOUND", None),
    ("champion", "DEFENSIBLE_WITH_CAVEATS", None),
    ("cell_lead", "PIPELINE_STALLED", "circuit_breaker"),
])
def test_each_signal_captures_context(agent, verdict, reason):
    item = Verdict(cell_id="dish-colony-000-000", agent=agent, iteration=2,
                   verdict=verdict, summary="The claim depends on an unmeasured assumption.")
    context = TriggerContext(stall_reason=reason, weakest_link="skeptic", focused_directive="Measure the assumption.")
    triggers = detect_triggers([item], context, {agent: object()}, LifecycleConfig())
    assert len(triggers) == 1
    result = triggers[0]
    assert result.agent == agent and result.trigger_verdict == verdict
    assert result.weakest_link == "skeptic" and result.focused_directive == "Measure the assumption."
    assert result.fire_point == ("circuit_breaker" if reason else "settled_round")
    if agent == "champion":
        assert result.caveat_text == item.summary
    if agent == "skeptic":
        assert result.flaw_text == item.summary


@pytest.mark.parametrize("reason", ["user_stop", "stop", "unknown", None])
def test_user_stop_or_missing_provenance_never_triggers(reason):
    verdict = Verdict(cell_id="cell", agent="cell_lead", iteration=1, verdict="PIPELINE_STALLED")
    assert not detect_triggers([verdict], TriggerContext(stall_reason=reason), {"cell_lead": object()}, LifecycleConfig())


def test_max_rounds_and_latest_verdict_guard():
    verdict = Verdict(cell_id="cell", agent="champion", iteration=1, verdict="DEFENSIBLE_WITH_CAVEATS")
    assert not detect_triggers([verdict], TriggerContext(generation=2), {"champion": object()}, LifecycleConfig())
    resolved = verdict.model_copy(update={"verdict": "DEFENSIBLE", "iteration": 2})
    assert not detect_triggers([verdict, resolved], TriggerContext(), {"champion": object()}, LifecycleConfig())
    assert not detect_triggers([verdict], TriggerContext(settled=False), {"champion": object()}, LifecycleConfig())
    assert not detect_triggers([verdict], TriggerContext(), {}, LifecycleConfig())


def test_classifier_has_no_inference_storage_or_execution_imports():
    path = Path(__file__).parents[2] / "petri/lifecycle/triggers.py"
    for node in ast.walk(ast.parse(path.read_text())):
        names = [node.module or ""] if isinstance(node, ast.ImportFrom) else (
            [alias.name for alias in node.names] if isinstance(node, ast.Import) else [])
        assert not any(name.startswith(("dbos", "pydantic_ai", "petri.storage", "petri.harness", "petri.execution"))
                       for name in names)
