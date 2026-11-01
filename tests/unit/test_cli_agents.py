"""Cost-neutral roster inspection and validation through the public CLI."""

import json
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from typer.testing import CliRunner

from petri.agents.contract import AGENT_SPECS
from petri.cli import app

RUNNER = CliRunner()
DEFAULTS = Path(__file__).parents[2] / "petri/defaults"


@pytest.fixture
def v2_dish(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    destination = tmp_path / ".petri/defaults"
    destination.mkdir(parents=True)
    shutil.copyfile(DEFAULTS / "petri.yaml", destination / "petri.yaml")
    shutil.copyfile(DEFAULTS / "constitution.md", destination / "constitution.md")
    return destination / "petri.yaml"


def test_list_table_and_json(v2_dish):
    table = RUNNER.invoke(app, ["agents", "list"])
    assert table.exit_code == 0, table.output
    assert len(table.output.splitlines()) == len(AGENT_SPECS) + 1
    assert "investigator | research | blocking" in table.output
    assert "EVIDENCE_SUFFICIENT" in table.output
    result = RUNNER.invoke(app, ["agents", "list", "--json"])
    rows = json.loads(result.output)
    assert len(rows) == len(AGENT_SPECS)
    assert {row["name"] for row in rows} == set(AGENT_SPECS)


def test_check_clean_explicit_config(v2_dish):
    result = RUNNER.invoke(app, ["agents", "check", "--config", str(v2_dish)])
    assert result.exit_code == 0, result.output
    assert "13 roster members" in result.output


@pytest.mark.parametrize(
    "kind,key",
    [("unknown", "invented_agent"), ("verdict", "invented_verdict"), ("structure", "blocking")],
)
def test_check_invalid_content_names_key(v2_dish, kind, key):
    cfg = yaml.safe_load(v2_dish.read_text())
    if kind == "unknown":
        cfg["agents"][key] = cfg["agents"].pop("triage")
    elif kind == "verdict":
        cfg["agents"]["triage"]["verdicts_pass"] = [key]
    else:
        cfg["agents"]["triage"][key] = True
    v2_dish.write_text(yaml.safe_dump(cfg))
    result = RUNNER.invoke(app, ["agents", "check"])
    assert result.exit_code == 1
    assert key in result.output


@pytest.mark.parametrize("command", ["list", "check"])
def test_missing_dish(command, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = RUNNER.invoke(app, ["agents", command])
    assert result.exit_code == 3
    assert "No petri dish found" in result.output


@pytest.mark.parametrize("command", ["list", "check"])
def test_inspection_never_constructs_model_or_launches_process(command, v2_dish):
    with (
        patch("subprocess.Popen", side_effect=AssertionError("process")),
        patch("socket.socket.connect", side_effect=AssertionError("network")),
        patch("pydantic_ai.Agent", side_effect=AssertionError("agent")),
    ):
        result = RUNNER.invoke(app, ["agents", command])
    assert result.exit_code == 0, result.output
