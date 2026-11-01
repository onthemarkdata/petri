"""Explicit local-only seed validation; never substitutes a hosted provider."""

import os
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from petri.cli import app
from petri.cli.init import create_petri_dish

GEMMA_MODEL = "huggingface.co/google/gemma-4-e2b-it-qat-q4_0-gguf:Q4_0"


def prepare_dish(tmp_path, harness):
    dish = tmp_path / ".petri"
    create_petri_dish(dish, dish_name="bridge-test")
    path = dish / "defaults" / "petri.yaml"
    config = yaml.safe_load(path.read_text())
    # Exercise the transition config: legacy model mapping plus explicit harness.
    assert isinstance(config["model"], dict)
    config["harness"] = harness
    config["max_decomposition_depth"] = 1
    config["max_nodes_per_layer"] = 2
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return dish


def seed_and_dry_run(dish):
    runner = CliRunner()
    seeded = runner.invoke(app, ["seed", "Water freezes at zero degrees Celsius at standard atmospheric pressure.",
                                 "--no-questions", "--colony", "water"])
    assert seeded.exit_code == 0, f"{seeded.output}\n{seeded.exception!r}"
    from petri.storage.database import list_cells
    from petri.storage.paths import dish_db_path

    assert len(list_cells(dish_db_path(dish), colony="bridge-test-water")) >= 2
    grown = runner.invoke(app, ["grow", "--dry-run"])
    assert grown.exit_code == 0, f"{grown.output}\n{grown.exception!r}"
    assert "Would process" in grown.output
    assert "bridge-test-water" in grown.output
    return seeded, grown


@pytest.mark.integration
@pytest.mark.parametrize("backend", ["pi", "provider"])
def test_local_gemma_seed_and_dry_run(tmp_path, monkeypatch, backend):
    if os.environ.get("PETRI_RUN_LOCAL_BRIDGE") != "1":
        pytest.skip("Set PETRI_RUN_LOCAL_BRIDGE=1 to call only local Docker Model Runner.")
    monkeypatch.chdir(tmp_path)
    if backend == "pi":
        binary = os.environ.get("PETRI_PI_BIN")
        if not binary or not Path(binary).is_file() or not os.environ.get("PI_CODING_AGENT_DIR"):
            pytest.skip("Set PETRI_PI_BIN and PI_CODING_AGENT_DIR for the local pi provider.")
        harness = {"backend": "pi", "provider": "petri-local", "model": GEMMA_MODEL, "tools": []}
    else:
        monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:12434/engines/llama.cpp/v1")
        monkeypatch.setenv("OPENAI_API_KEY", "local-no-auth")
        harness = {"backend": "provider", "model": f"openai-chat:{GEMMA_MODEL}"}
    dish = prepare_dish(tmp_path, harness)
    seed_and_dry_run(dish)
