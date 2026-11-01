"""Check shipped vocabularies directly, independent of the v2 configuration loader."""

import ast
from pathlib import Path

import pytest
import yaml

from petri.agents.contract import AGENT_SPECS, VERDICT_ENUMS, AgentName

DEFAULTS = Path(__file__).parents[2] / "petri/defaults/petri.yaml"


def assert_conforms(name, path):
    content = yaml.safe_load(path.read_text())["agents"][name]
    enum_values = {member.value for member in VERDICT_ENUMS[name]}
    yaml_values = set(content["verdicts_pass"]) | set(content["verdicts_block"])
    assert enum_values == yaml_values, (
        f"{name}: missing-from-YAML={sorted(enum_values - yaml_values)}; "
        f"missing-from-enum={sorted(yaml_values - enum_values)}"
    )
    for partition in ("verdicts_pass", "verdicts_block"):
        expected = set(getattr(AGENT_SPECS[name], partition))
        actual = set(content[partition])
        assert expected == actual, (
            f"{name}.{partition}: missing-from-YAML={sorted(expected - actual)}; "
            f"missing-from-enum={sorted(actual - expected)}"
        )


@pytest.mark.parametrize("name", list(AgentName))
def test_default_verdict_partition(name):
    assert_conforms(name, DEFAULTS)


def test_roster_membership():
    roster = yaml.safe_load(DEFAULTS.read_text())["agents"]
    assert set(roster) == {name.value for name in AgentName}
    assert "socratic_questioner" not in roster


def test_corruption_produces_named_diff(tmp_path):
    data = yaml.safe_load(DEFAULTS.read_text())
    data["agents"]["triage"]["verdicts_block"] = ["MISSPELLED"]
    path = tmp_path / "petri.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(AssertionError) as error:
        assert_conforms(AgentName.TRIAGE, path)
    message = str(error.value)
    for token in (
        "triage",
        "missing-from-YAML",
        "LOW_VALUE_DEFER",
        "missing-from-enum",
        "MISSPELLED",
    ):
        assert token in message


def test_suite_does_not_import_content_loader():
    tree = ast.parse(Path(__file__).read_text())
    imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert "petri.agents.config" not in imports
