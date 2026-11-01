"""Fresh DBOS lifecycles leave legacy dishes untouched and need no Unix queue."""

import ast
from dataclasses import replace
import importlib
import json
from pathlib import Path
import subprocess
import sys

from typer.testing import CliRunner

from petri.cli import app
from petri.execution.dbos_backend import DBOSExecutionBackend
from petri.storage import database
from petri.storage.paths import dish_db_path
from tests.pipeline_stubs import StubPhases
from tests.unit.test_durable_seed import SeedRunner


def test_fresh_lifecycle_preserves_a_neighboring_legacy_dish(tmp_path, monkeypatch):
    old = tmp_path / "legacy" / ".petri"
    old.mkdir(parents=True)
    (old / "queue.json").write_text('{"old": "greppable research"}\n')
    (old / "notes.md").write_text("An existing v1 claim\n")
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in old.iterdir()}
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    monkeypatch.chdir(fresh)
    monkeypatch.setattr("petri.engine.preflight.run_preflight", lambda _: [])
    runner = CliRunner()
    initialized = runner.invoke(app, ["init", "--no-questions"])
    assert initialized.exit_code == 0, initialized.output
    monkeypatch.setattr("petri.cli.seed.resolve_provider", lambda _: SeedRunner())
    seeded = runner.invoke(app, ["seed", "Physical units", "--colony", "units", "--no-questions"])
    assert seeded.exit_code == 0, (seeded.output, seeded.exception)
    directory = fresh / ".petri"
    path = dish_db_path(directory)
    phases = StubPhases()
    async def finish(value, result):
        with database._write(path) as conn:
            conn.execute("UPDATE cells SET status=? WHERE id=?", (result.status.upper(), value.cell_id))
    deps = replace(phases.dependencies(), finish=finish)
    class LocalBackend(DBOSExecutionBackend):
        def start(self, petri_dir, **options):
            return super().start(petri_dir, **options, dependencies=lambda _: deps)
    grow = importlib.import_module("petri.cli.grow")
    monkeypatch.setattr(grow, "resolve_backend", lambda _: LocalBackend())
    grown = runner.invoke(app, ["grow"])
    assert grown.exit_code == 0, (grown.output, grown.exception)
    checked = runner.invoke(app, ["check", "--json"])
    assert checked.exit_code == 0, checked.output
    assert len(json.loads(checked.output)) == 3
    stopped = runner.invoke(app, ["stop"])
    assert stopped.exit_code == 0 and "No active runs" in stopped.output
    assert not (directory / "queue.json").exists()
    assert all(row["status"] == "VALIDATED" for row in database.list_cells(path))
    assert {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in old.iterdir()} == before


def test_package_import_does_not_require_fcntl():
    script = '''import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == "fcntl":
        raise AssertionError("Unix queue lock imported")
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import petri
'''
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_no_live_imports_of_retired_runtime_modules():
    root = Path(__file__).parents[2]
    retired = {
        "petri.storage.queue", "petri.engine.processor", "petri.engine.grow_loop",
        "petri.engine.propagation", "petri.engine.load_balancer", "petri.execution.v1_engine",
    }
    for directory in (root / "petri", root / "tests"):
        for path in directory.rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom):
                    assert node.module not in retired, path
                elif isinstance(node, ast.Import):
                    assert not (retired & {alias.name for alias in node.names}), path
    source = (root / "petri/config.py").read_text()
    assert "lru_cache" not in source
    assert "MAX_ITERATIONS" not in source
    assert "LLM_INFERENCE_MODEL" not in source
