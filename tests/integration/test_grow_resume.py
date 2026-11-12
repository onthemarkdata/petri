"""The grow CLI survives real process interruption without repeating completed steps."""

import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import time

import pytest
from dbos import DBOSClient
from filelock import FileLock

from petri.execution.dbos_backend import DBOSExecutionBackend
from petri.execution.pipeline import NeedsHuman
from petri.storage import database
from petri.storage.paths import dish_db_path


@pytest.mark.integration
@pytest.mark.parametrize("interruption", [
    signal.SIGKILL, signal.SIGINT, "graceful", "graceful_stalled", "graceful_parallel"
])
def test_grow_cli_resumes_completed_steps(tmp_path, interruption):
    root = Path(__file__).parents[2]
    command = [sys.executable, str(root / "tests/helpers/dbos_grow_worker.py"), str(tmp_path)]
    stalled = interruption == "graceful_stalled"
    parallel = interruption == "graceful_parallel"
    graceful = interruption in {"graceful", "graceful_stalled", "graceful_parallel"}
    if stalled:
        command.append("--park-sibling")
    if parallel:
        command.append("--parallel-stop")
    environment = {**os.environ, "PYTHONPATH": str(root)}
    before_resume_events = []
    failure_events = []
    with (tmp_path / "worker.log").open("w") as log:
        worker = subprocess.Popen(command, env=environment, stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 20
            while not (tmp_path / "in-research").exists():
                assert worker.poll() is None, (tmp_path / "worker.log").read_text()
                assert time.monotonic() < deadline, (tmp_path / "worker.log").read_text()
                time.sleep(0.02)
            if graceful:
                client = DBOSClient(system_database_url=f"sqlite:///{tmp_path / 'dbos.sqlite'}")
                try:
                    queued = {
                        run.workflow_id for run in client.list_workflows(name="petri.validate_cell")
                        if run.status == "ENQUEUED"
                    }
                    if parallel:
                        assert len(queued) == 1
                    assert DBOSExecutionBackend().request_stop(tmp_path) >= (4 if parallel else 2)
                    if parallel:
                        assert all(
                            run.status == "CANCELLED" for run in client.list_workflows(workflow_ids=list(queued))
                        )
                        assert sum(
                            run.status == "PENDING" for run in client.list_workflows(name="petri.validate_cell")
                        ) == 2
                finally:
                    client.destroy()
                (tmp_path / "release").write_text("finish the current phase")
            else:
                os.kill(worker.pid, interruption)
            code = worker.wait(timeout=15)
            assert code == (-signal.SIGKILL if interruption == signal.SIGKILL else 0), (tmp_path / "worker.log").read_text()
            if graceful:
                stopped = json.loads((tmp_path / "cli-result.json").read_text())
                assert "user_stop" in stopped["output"], stopped
                with FileLock(tmp_path / ".execution-owner.lock").acquire(timeout=0):
                    pass
                client = DBOSClient(system_database_url=f"sqlite:///{tmp_path / 'dbos.sqlite'}")
                try:
                    runs = client.list_workflows(name="petri.validate_cell")
                    assert all(run.status in {"SUCCESS", "CANCELLED"} for run in runs)
                    cancelled = [run for run in runs if run.status == "CANCELLED"]
                    assert len(cancelled) == (3 if parallel else 1)
                    with sqlite3.connect(tmp_path / "audit.sqlite") as conn:
                        for run in cancelled:
                            stopped_cell = run.workflow_id.rpartition(":r")[0]
                            current = [row[0] for row in conn.execute(
                                "SELECT phase FROM calls WHERE cell_id=?", (stopped_cell,)
                            )]
                            assert current == ([] if run.workflow_id in queued else [
                                "sync_check", "socratic", "research"
                            ])
                    if stalled:
                        parked = [run for run in runs if isinstance(run.output, NeedsHuman)]
                        assert len(parked) == 1
                        assert parked[0].output.reason == "Offline sibling needs review"
                finally:
                    client.destroy()
                before_resume_events = database.read_events(dish_db_path(tmp_path))
                failure_events = [
                    event for event in before_resume_events
                    if event["type"] == "agent_execution_failed"
                ]
                assert len(failure_events) == int(stalled)
                if stalled:
                    assert database.get_cell(
                        dish_db_path(tmp_path), failure_events[0]["cell_id"]
                    )["status"] == "STALLED"
                assert any(row["status"] == "NEW" for row in database.list_cells(dish_db_path(tmp_path)))
        finally:
            if worker.poll() is None:
                worker.kill()
                worker.wait(timeout=5)
    (tmp_path / "release").write_text("resume")
    resumed = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=30)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    result = json.loads((tmp_path / "cli-result.json").read_text())
    assert result["exit_code"] == int(stalled), result
    assert ("needs_human" if stalled else "all_terminal") in result["output"]
    rows = database.list_cells(dish_db_path(tmp_path))
    if stalled:
        assert sum(row["status"] == "STALLED" for row in rows) == 1
        assert sum(row["status"] == "NEW" for row in rows) == 1
        assert sum(row["status"] == "VALIDATED" for row in rows) == 3
    else:
        assert all(row["status"] == "VALIDATED" for row in rows)
    after_resume_events = database.read_events(dish_db_path(tmp_path))
    assert all(event in after_resume_events for event in before_resume_events)
    assert [event for event in after_resume_events if event["type"] == "agent_execution_failed"] == failure_events
    if stalled:
        client = DBOSClient(system_database_url=f"sqlite:///{tmp_path / 'dbos.sqlite'}")
        try:
            parked = client.list_workflows(workflow_ids=[failure_events[0]["cell_id"] + ":r1"])
            assert len(parked) == 1
            assert parked[0].status == "SUCCESS"
            assert isinstance(parked[0].output, NeedsHuman)
            assert parked[0].output.reason == "Offline sibling needs review"
        finally:
            client.destroy()
    with sqlite3.connect(tmp_path / "audit.sqlite") as conn:
        calls = list(conn.execute("SELECT cell_id,phase,COUNT(*) FROM calls GROUP BY cell_id,phase"))
    assert len(calls) == (24 if stalled else 42 if parallel else 35)
    stub_events = [event for event in after_resume_events if event["agent"] == "stub"]
    assert len(stub_events) == len(calls) + int(stalled)
    assert all(count == 1 for _, _, count in calls)
    assert not (tmp_path / "queue.json").exists()
    assert not (tmp_path / ".stop").exists()
