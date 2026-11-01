"""Canonical v2 roster invariants and the temporary legacy boundary."""

from enum import StrEnum
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from petri.agents.contract import AGENT_SPECS, VERDICT_ENUMS, AgentName, AgentSpec, BlockingMode


def test_roster_shape():
    assert len(AGENT_SPECS) == len(AgentName) == 13
    assert list(AGENT_SPECS) == list(AgentName)
    assert "socratic_questioner" not in AgentName._value2member_map_
    assert sum(s.is_lead for s in AGENT_SPECS.values()) == 3
    assert sum(s.blocking_mode == BlockingMode.BLOCKING for s in AGENT_SPECS.values()) == 6
    conditional = [s for s in AGENT_SPECS.values() if s.blocking_mode == BlockingMode.CONDITIONAL]
    assert [s.name for s in conditional] == [AgentName.TRIAGE]
    assert conditional[0].redirect_on_block == "DEFER_OPEN"


def test_specs_are_frozen():
    with pytest.raises(ValidationError, match="frozen"):
        AGENT_SPECS[AgentName.TRIAGE].scope = "changed"


@pytest.mark.parametrize(
    "name,changes,message",
    [
        (AgentName.INVESTIGATOR, {"verdicts_block": ()}, "require verdicts_block"),
        (AgentName.TRIAGE, {"redirect_on_block": None}, "require redirect_on_block"),
        (AgentName.CELL_LEAD, {"blocking_mode": BlockingMode.BLOCKING}, "Lead agents"),
    ],
)
def test_invalid_structure(name, changes, message):
    data = AGENT_SPECS[name].model_dump()
    with pytest.raises(ValidationError, match=message):
        AgentSpec.model_validate({**data, **changes})


def test_advisory_block_vocabulary_is_representable():
    # Factory issue #31 decides how advisory vocabulary affects eager validation.
    spec = AGENT_SPECS[AgentName.SIMPLIFIER]
    assert spec.blocking_mode == BlockingMode.NON_BLOCKING
    assert spec.verdicts_block == ("OVERCOMPLICATED",)


def test_verdict_types_and_partitions():
    assert set(VERDICT_ENUMS) == set(AgentName)
    assert all(issubclass(v, StrEnum) for v in VERDICT_ENUMS.values())
    triage = AGENT_SPECS[AgentName.TRIAGE]
    assert triage.verdicts_pass == ("HIGH_VALUE", "MODERATE_VALUE")
    assert triage.verdicts_block == ("LOW_VALUE_DEFER",)
    assert {v.value for v in VERDICT_ENUMS[AgentName.TRIAGE]} == {
        *triage.verdicts_pass,
        *triage.verdicts_block,
    }
    evaluator = AGENT_SPECS[AgentName.EVIDENCE_EVALUATOR]
    assert evaluator.verdicts_block == ()
    assert set(evaluator.verdicts_pass) == {
        "EVIDENCE_CONFIRMS",
        "EVIDENCE_REFUTES",
        "EVIDENCE_INCONCLUSIVE",
    }


def test_utility_content_is_preserved_outside_roster():
    defaults = Path(__file__).parents[2] / "petri/defaults"
    current = yaml.safe_load((defaults / "petri.yaml").read_text())
    legacy = yaml.safe_load((defaults / "legacy-petri.yaml").read_text())
    assert set(current["agents"]) == set(AgentName)
    utility = current["pre_pipeline"]["socratic_questioner"]
    for key in ("verdicts_pass", "verdicts_block"):
        assert utility[key] == legacy["agents"]["socratic_questioner"][key]
    assert "identify the current sources that research must retrieve" in utility["instruction"]
    assert "WebSearch" not in utility["instruction"]
