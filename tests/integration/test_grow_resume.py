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

from petri.storage import database
from petri.storage.paths import dish_db_path


@pytest.mark.integration
@pytest.mark.parametrize("interruption", [signal.SIGKILL, signal.SIGINT])
def test_grow_cli_resumes_completed_steps(tmp_path, interruption):
    root = Path(__file__).parents[2]
    command = [sys.executable, str(root / "tests/helpers/dbos_grow_worker.py"), str(tmp_path)]
    environment = {**os.environ, "PYTHONPATH": str(root)}
    with (tmp_path / "worker.log").open("w") as log:
        worker = subprocess.Popen(command, env=environment, stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 20
            while not (tmp_path / "in-research").exists():
                assert worker.poll() is None, (tmp_path / "worker.log").read_text()
                assert time.monotonic() < deadline, (tmp_path / "worker.log").read_text()
                time.sleep(0.02)
            os.kill(worker.pid, interruption)
            code = worker.wait(timeout=15)
            assert code == (-signal.SIGKILL if interruption == signal.SIGKILL else 0), (tmp_path / "worker.log").read_text()
        finally:
            if worker.poll() is None:
                worker.kill()
                worker.wait(timeout=5)
    (tmp_path / "release").write_text("resume")
    resumed = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=30)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    result = json.loads((tmp_path / "cli-result.json").read_text())
    assert result["exit_code"] == 0, result
    assert "all_terminal" in result["output"]
    assert all(row["status"] == "VALIDATED" for row in database.list_cells(dish_db_path(tmp_path)))
    with sqlite3.connect(tmp_path / "audit.sqlite") as conn:
        calls = list(conn.execute("SELECT cell_id,phase,COUNT(*) FROM calls GROUP BY cell_id,phase"))
    assert len(calls) == 35
    assert all(count == 1 for _, _, count in calls)
    assert not (tmp_path / "queue.json").exists()
    assert not (tmp_path / ".stop").exists()
