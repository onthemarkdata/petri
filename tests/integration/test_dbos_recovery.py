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
def test_sigkill_resumes_checkpointed_cell_and_excludes_another_owner(tmp_path):
    root = Path(__file__).resolve().parents[2]
    worker = root / "tests/helpers/dbos_recovery_worker.py"
    environment = {**os.environ, "PYTHONPATH": str(root)}
    command = [sys.executable, str(worker), str(tmp_path)]
    with (tmp_path / "worker.log").open("w") as log:
        process = subprocess.Popen(command + ["start"], env=environment, stdout=log, stderr=log)
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
        command + ["resume"], env=environment, capture_output=True, text=True, timeout=30
    )
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert json.loads((tmp_path / "result.json").read_text())["status"] == "validated"
    with sqlite3.connect(tmp_path / "audit.sqlite") as conn:
        calls = dict(conn.execute("SELECT phase,COUNT(*) FROM calls GROUP BY phase"))
    assert calls == {"socratic": 1, "research": 2}
    events = database.read_events(tmp_path / "petri.sqlite")
    assert len(events) == 2
    assert {event["id"] for event in events} == {"socratic", "research"}
