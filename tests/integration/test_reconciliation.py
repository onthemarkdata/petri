"""Domain snapshots cannot silently roll back checkpointed research."""

import hashlib
import importlib
import json
import sqlite3
from contextlib import closing
from dataclasses import replace

import pytest
import typer
from typer.testing import CliRunner

from petri.execution.dbos_backend import CellRunSpec
from petri.execution.pipeline import SyncConflict
from petri.models import Event
from petri.storage import database
from tests.unit import test_durable_scheduler

scheduler_fixture = test_durable_scheduler.scheduler_fixture


def backup(source, destination):
    with closing(sqlite3.connect(source)) as left, closing(sqlite3.connect(destination)) as right:
        left.backup(right)


def event_hash(path):
    events = database.read_events(path)
    return hashlib.sha256(json.dumps(events, sort_keys=True).encode()).hexdigest(), len(events)


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("restore", ["events", "status", "missing"])
async def test_startup_parks_rolled_back_events_or_nonterminal_success(
    scheduler_fixture,
    canonical_colony,
    restore,
    monkeypatch,
):
    directory, path, backend, dependencies, starts, _ = scheduler_fixture
    (directory / "petri.yaml").write_text("name: test-dish\nexecution:\n  backend: dbos\n")
    snapshot = directory / "older.sqlite"
    backup(path, snapshot)

    def with_event(spec):
        deps = dependencies(spec)

        async def research(value):
            result = await deps.research(value)
            database.append_event(
                path,
                Event(
                    id=f"{spec.cell_id}:receipt-test",
                    cell_id=spec.cell_id,
                    timestamp="2026-11-01T00:00:00Z",
                    type="verdict_issued",
                    agent="investigator",
                    iteration=0,
                    data={"verdict": "EVIDENCE_SUFFICIENT", "summary": "Committed evidence"},
                ),
            )
            return result

        return replace(deps, research=research)

    cell = canonical_colony["cell1"].id
    spec = CellRunSpec(cell, 1, "2026-11-01")
    await backend.start_async(directory, db_path=path, dependencies=with_event)
    handle = await backend.enqueue_cell(spec, level=2)
    await handle.get_result()
    await backend.close_async()
    if restore == "events":
        backup(snapshot, path)
    elif restore == "status":
        with database._write(path) as conn:
            conn.execute("UPDATE cells SET status='NEW' WHERE id=?", (cell,))
    else:
        with database._write(path) as conn:
            conn.execute("DELETE FROM edges WHERE to_cell=? OR from_cell=?", (cell, cell))
            conn.execute("DELETE FROM cells WHERE id=?", (cell,))
    before = event_hash(path)
    domain_before = database.get_cell(path, cell)
    await backend.start_async(directory, db_path=path, dependencies=with_event)
    try:
        parked = await backend.enqueue_cell(spec, level=2)
        result = await parked.get_result()
        assert isinstance(result, SyncConflict)
        expected = {
            "events": "domain events",
            "status": "SUCCESS/VALIDATED",
            "missing": "domain row is missing",
        }
        assert expected[restore] in result.reason
        assert len(starts) == 1
        assert event_hash(path) == before
        assert database.get_cell(path, cell) == domain_before
        assert backend.entries(directory)[cell]["queue_state"] == "sync_conflict"
    finally:
        await backend.close_async()
    module = importlib.import_module("petri.cli.check")
    monkeypatch.setattr(module, "find_petri_dir", lambda: directory)
    app = typer.Typer()
    module.register(app)
    output = CliRunner().invoke(app, [])
    assert output.exit_code == 0, (output.output, output.exception)
    assert "Sync conflict:" in output.output
    assert result.reason in output.output
    detail = CliRunner().invoke(app, ["--cell", cell, "--json"])
    assert detail.exit_code == 0, (detail.output, detail.exception)
    assert json.loads(detail.output)["sync_conflict_reason"] == result.reason
    assert event_hash(path) == before


@pytest.mark.integration
def test_older_snapshot_after_mid_colony_sigkill_parks_recovery(tmp_path):
    import os
    import signal
    import subprocess
    import sys
    import time
    from pathlib import Path

    from petri.storage.colonies import save_colony
    from tests.conftest import _build_canonical_colony

    colony = _build_canonical_colony()
    domain = tmp_path / "petri.sqlite"
    save_colony(domain, colony["graph"], colony["colony_model"])
    snapshot = tmp_path / "older.sqlite"
    backup(domain, snapshot)
    root = Path(__file__).resolve().parents[2]
    worker = root / "tests/helpers/dbos_recovery_worker.py"
    environment = {**os.environ, "PYTHONPATH": str(root)}
    command = [sys.executable, str(worker), str(tmp_path)]
    with (tmp_path / "worker.log").open("w") as log:
        process = subprocess.Popen(
            command + ["colony-start"], env=environment, stdout=log, stderr=log
        )
        try:
            deadline = time.monotonic() + 20
            while not (tmp_path / "in-research").exists():
                assert process.poll() is None, (tmp_path / "worker.log").read_text()
                assert time.monotonic() < deadline, (tmp_path / "worker.log").read_text()
                time.sleep(0.02)
            assert database.get_cell(domain, colony["cell1"].id)["status"] == "VALIDATED"
            process.kill()
            assert process.wait(timeout=5) == -signal.SIGKILL
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
    backup(snapshot, domain)
    before = event_hash(domain)
    (tmp_path / "release").write_text("resume")
    resumed = subprocess.run(
        command + ["colony-resume"], env=environment, capture_output=True, text=True, timeout=30
    )
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    from petri.execution.reconciliation import read_conflicts

    conflicts = read_conflicts(tmp_path / "dbos.sqlite")
    assert colony["cell1"].id in conflicts
    assert colony["cell2"].id in conflicts
    assert event_hash(domain) == before
    assert all(row["status"] == "NEW" for row in database.list_cells(domain))
    assert colony["cell1"].id in json.loads((tmp_path / "result.json").read_text())["not_eligible"]
