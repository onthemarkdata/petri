"""SIGKILL recovery with local fake phases and separate domain/checkpoint stores."""

import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

from petri.storage import database


@pytest.mark.integration
@pytest.mark.parametrize("content_event", [False, True])
def test_sigkill_resumes_checkpointed_cell_and_excludes_another_owner(tmp_path, content_event):
    root = Path(__file__).resolve().parents[2]
    worker = root / "tests/helpers/dbos_recovery_worker.py"
    environment = {**os.environ, "PYTHONPATH": str(root)}
    command = [sys.executable, str(worker), str(tmp_path)]
    prefix = "content-" if content_event else ""
    with (tmp_path / "worker.log").open("w") as log:
        process = subprocess.Popen(command + [prefix + "start"], env=environment, stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 20
            while not (tmp_path / "in-research").exists():
                assert process.poll() is None, (tmp_path / "worker.log").read_text()
                assert time.monotonic() < deadline, (tmp_path / "worker.log").read_text()
                time.sleep(0.02)
            probe = subprocess.run(
                command + ["probe"], env=environment, capture_output=True, text=True, timeout=20
            )
            assert probe.returncode != 0
            assert "Another process owns this dish" in probe.stderr
            os.kill(process.pid, signal.SIGKILL)
            assert process.wait(timeout=5) == -signal.SIGKILL
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
    (tmp_path / "release").write_text("resume")
    resumed = subprocess.run(
        command + [prefix + "resume"], env=environment, capture_output=True, text=True, timeout=30
    )
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert json.loads((tmp_path / "result.json").read_text())["status"] == "validated"
    with sqlite3.connect(tmp_path / "audit.sqlite") as conn:
        calls = dict(conn.execute("SELECT phase,COUNT(*) FROM calls GROUP BY phase"))
    assert calls == {"socratic": 1, "research": 2}
    events = [event for event in database.read_events(tmp_path / "petri.sqlite")
              if event["type"] in {"verdict_issued", "evidence_appended"}]
    assert len(events) == 2
    assert {event["id"] for event in events} == {
        "dish-colony-002-001:socratic",
        "dish-colony-002-001:research",
    }
    queued = database.read_events(tmp_path / "petri.sqlite", event_type="cell_queued")
    assert len(queued) == 1
    assert queued[0]["data"]["workflow_id"] == "dish-colony-002-001:r1"
    if content_event:
        assert database.get_cell(tmp_path / "petri.sqlite", "dish-colony-002-001")["evidence_md"] == "research"


@pytest.mark.integration
def test_sigkill_resumes_colony_without_reexecuting_validated_leaves(tmp_path):
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
            leaves = [
                row for row in database.list_cells(tmp_path / "petri.sqlite") if row["level"] == 2
            ]
            assert len(leaves) == 2
            assert all(row["status"] == "VALIDATED" for row in leaves)
            process.kill()
            assert process.wait(timeout=5) == -signal.SIGKILL
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
    (tmp_path / "release").write_text("resume")
    resumed = subprocess.run(
        command + ["colony-resume"], env=environment, capture_output=True, text=True, timeout=30
    )
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    result = json.loads((tmp_path / "result.json").read_text())
    assert len(result["completed"]) == 5
    assert result["not_eligible"] == []
    with sqlite3.connect(tmp_path / "audit.sqlite") as conn:
        for leaf in leaves:
            calls = dict(
                conn.execute(
                    "SELECT phase,COUNT(*) FROM calls WHERE cell_id=? GROUP BY phase", (leaf["id"],)
                )
            )
            assert calls == {"socratic": 1, "research": 1}
    assert all(
        row["status"] == "VALIDATED" for row in database.list_cells(tmp_path / "petri.sqlite")
    )
    assert len(database.read_events(tmp_path / "petri.sqlite", event_type="verdict_issued")) == 10
    queued = database.read_events(tmp_path / "petri.sqlite", event_type="cell_queued")
    assert len(queued) == 5
    assert {event["data"]["workflow_id"] for event in queued} == {
        f"{row['id']}:r1" for row in database.list_cells(tmp_path / "petri.sqlite")
    }
