"""A dashboard seed survives owner death and resumes without another submission."""

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
from petri.storage.paths import dish_db_path

pytestmark = pytest.mark.integration


def test_dashboard_seed_recovers_after_sigkill(tmp_path):
    root = Path(__file__).parents[2]
    worker = root / "tests/helpers/dashboard_run_worker.py"
    environment = {**os.environ, "PYTHONPATH": str(root), "OTEL_SDK_DISABLED": "true"}
    with (tmp_path / "worker.log").open("w") as log:
        process = subprocess.Popen(
            [sys.executable, str(worker), str(tmp_path), "submit"],
            cwd=root,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 20
            while not (tmp_path / "blocked").exists() and time.monotonic() < deadline:
                assert process.poll() is None, (tmp_path / "worker.log").read_text()
                time.sleep(0.05)
            assert (tmp_path / "blocked").is_file(), (tmp_path / "worker.log").read_text()
            receipt = json.loads((tmp_path / "receipt.json").read_text())
            assert receipt["workflow_id"] == "dish-units"
            process.kill()
            assert process.wait(timeout=5) == -signal.SIGKILL
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
    (tmp_path / "release").touch()
    result = subprocess.run(
        [sys.executable, str(worker), str(tmp_path), "recover"],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads((tmp_path / "result.json").read_text())["result"]["complete"]
    with sqlite3.connect(tmp_path / "calls.sqlite") as connection:
        calls = [row[0] for row in connection.execute("SELECT name FROM calls")]
    assert calls.count("initial") == 1
    assert len(calls) == 3
    path = dish_db_path(tmp_path, "dish")
    assert len(database.list_cells(path)) == 3
    assert len(database.read_events(path, event_type="seed_started")) == 1
