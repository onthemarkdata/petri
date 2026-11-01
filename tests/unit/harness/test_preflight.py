"""Version checks use mocked command results and never call a model."""

import subprocess

import pytest
import typer
from typer.testing import CliRunner

from petri.cli.inspect import register
from petri.engine import preflight


@pytest.fixture
def versions(monkeypatch):
    monkeypatch.delenv("PETRI_PI_BIN", raising=False)
    monkeypatch.setattr(preflight.shutil, "which", lambda name: f"/bin/{name}")
    outputs = {"pi": (0, "0.73.1\n"), "node": (0, "v22.19.0\n")}

    def run(command, **kwargs):
        name = command[0].rsplit("/", 1)[-1]
        code, text = outputs[name]
        return subprocess.CompletedProcess(command, code, text, "")

    monkeypatch.setattr(preflight.subprocess, "run", run)
    return outputs


def test_pi_and_node_versions(versions):
    result = preflight.check_pi()
    assert result.passed
    assert "0.73.1" in result.message and "v22.19.0" in result.message


@pytest.mark.parametrize("version", ["v22.18.0", "v20.19.0", "garbage", "v23.0.0-rc.1"])
def test_node_floor(versions, version):
    versions["node"] = (0, version)
    result = preflight.check_pi()
    assert not result.passed
    assert "22.19.0" in result.message


@pytest.mark.parametrize("binary", ["pi", "node"])
def test_missing_binary(monkeypatch, versions, binary):
    monkeypatch.setattr(preflight.shutil, "which", lambda name: None if name == binary else f"/bin/{name}")
    result = preflight.check_pi()
    assert not result.passed
    assert "README.md#install-pi-and-nodejs" in result.message
    assert "claude-code" in result.message


def test_bad_exit(versions):
    versions["pi"] = (1, "failed")
    assert not preflight.check_pi().passed


def test_timeout(monkeypatch, versions):
    def timeout(command, **kwargs):
        raise subprocess.TimeoutExpired(command, 10)
    monkeypatch.setattr(preflight.subprocess, "run", timeout)
    assert not preflight.check_pi().passed


def test_override_binary(monkeypatch, versions):
    monkeypatch.setenv("PETRI_PI_BIN", "/custom/pi")
    assert preflight.check_pi().passed


def test_backend_checks(monkeypatch, versions):
    monkeypatch.setattr(preflight, "check_claude_cli", lambda: preflight.CheckResult("Claude", True, "version"))
    assert preflight.run_preflight()[-1].name == "Claude"
    assert preflight.run_preflight({"harness": {"backend": "pi"}})[-1].name == "pi CLI"
    assert len(preflight.run_preflight({"harness": {"backend": "provider"}})) == 1
    assert not preflight.run_preflight({"harness": {"backend": "unknown"}})[-1].passed


def test_inspect_selected_backend(monkeypatch, tmp_path, versions):
    monkeypatch.chdir(tmp_path)
    defaults = tmp_path / ".petri" / "defaults"
    defaults.mkdir(parents=True)
    (defaults / "petri.yaml").write_text("harness:\n  backend: pi\n")
    app = typer.Typer()
    register(app)
    result = CliRunner().invoke(app, [])
    assert result.exit_code == 0
    assert "0.73.1" in result.stdout and "v22.19.0" in result.stdout


def test_inspect_missing_pi_is_actionable(monkeypatch, tmp_path, versions):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(preflight.shutil, "which", lambda _: None)
    app = typer.Typer()
    register(app)
    result = CliRunner().invoke(app, [])
    assert result.exit_code == 1
    assert "README.md#install-pi-and-nodejs" in result.stdout


def test_init_selects_pi_and_checks_created_dish(monkeypatch, tmp_path, versions):
    from petri.cli.init import register as register_init
    from petri.config import load_dish_config
    from petri.harness.pi.model import PiModel
    from petri.harness.resolve import resolve_model

    app = typer.Typer()
    register_init(app)
    versions["node"] = (0, "v22.18.0")
    result = CliRunner().invoke(app, [str(tmp_path), "--no-questions"])
    assert result.exit_code == 0, result.output
    assert "requires Node.js 22.19.0+" in result.output
    config = load_dish_config(tmp_path / ".petri")
    assert config["harness"]["backend"] == "pi"
    assert isinstance(resolve_model(config), PiModel)


def test_repair_preserves_existing_harness(tmp_path):
    from petri.cli.init import create_petri_dish
    from petri.config import load_dish_config

    defaults = tmp_path / "defaults"
    defaults.mkdir()
    original = "name: existing\nharness:\n  backend: provider\n  model: test\n"
    (defaults / "petri.yaml").write_text(original)
    create_petri_dish(tmp_path, dish_name="replacement")
    assert (defaults / "petri.yaml").read_text() == original
    assert load_dish_config(tmp_path)["harness"]["backend"] == "provider"
