"""A killed proposal reuses its journaled response and commits its graph once."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from petri.storage import database


@pytest.mark.integration
def test_sigkill_after_model_journal_does_not_repeat_the_call(tmp_path):
    root = Path(__file__).resolve().parents[2]
    command = [sys.executable, str(root / "tests/helpers/dbos_redecompose_worker.py"), str(tmp_path)]
    environment = {**os.environ, "PYTHONPATH": str(root)}
    with (tmp_path / "worker.log").open("w") as log:
        process = subprocess.Popen(command + ["start"], env=environment, stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 25
            while not (tmp_path / "response-recorded").exists():
                assert process.poll() is None, (tmp_path / "worker.log").read_text()
                assert time.monotonic() < deadline, (tmp_path / "worker.log").read_text()
                time.sleep(0.02)
            process.kill()
            assert process.wait(timeout=5) == -signal.SIGKILL
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
    resumed = subprocess.run(command + ["resume"], env=environment, capture_output=True, text=True, timeout=30)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["state"] == "proposed"
    assert len((tmp_path / "calls.jsonl").read_text().splitlines()) == 1
    path = tmp_path / "petri-dishes/dish/petri.sqlite"
    assert len(database.list_cells(path)) == 3
    assert len(database.read_edges(path)) == 2
    assert len(database.read_events(path, event_type="redecomposition_proposed")) == 1
