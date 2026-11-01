"""Field report #8: interrupted seed CLI calls retain completed decomposition."""

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
@pytest.mark.parametrize("interrupt", [signal.SIGKILL, signal.SIGINT])
def test_field_report_8_seed_cli_resumes_after_interruption(tmp_path, interrupt):
    root = Path(__file__).resolve().parents[2]
    worker = root / "tests/helpers/dbos_seed_worker.py"
    environment = {**os.environ, "PYTHONPATH": str(root)}
    command = [sys.executable, str(worker), str(tmp_path)]
    with (tmp_path / "worker.log").open("w") as log:
        process = subprocess.Popen(command + ["start"], env=environment, stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 20
            while not (tmp_path / "in-expansion").exists():
                assert process.poll() is None, (tmp_path / "worker.log").read_text()
                assert time.monotonic() < deadline, (tmp_path / "worker.log").read_text()
                time.sleep(0.02)
            os.kill(process.pid, interrupt)
            process.wait(timeout=15)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
    (tmp_path / "release").write_text("resume")
    resumed = subprocess.run(
        command + ["resume"], env=environment, capture_output=True, text=True, timeout=30
    )
    output = json.loads((tmp_path / "cli-result.json").read_text())
    assert resumed.returncode == 0, output
    assert "Execution backend: dbos" in output["output"]
    with sqlite3.connect(tmp_path / "audit.sqlite") as conn:
        calls = dict(conn.execute("SELECT name,COUNT(*) FROM calls GROUP BY name"))
    assert calls == {
        "initial": 1,
        "The meter is a defined unit": 1,
        "The second is a defined unit": 1,
    }
    domain = tmp_path / "petri-dishes" / "dish" / "petri.sqlite"
    assert len(database.list_cells(domain)) == 3
    assert len(database.read_edges(domain)) == 2
    assert len(database.read_events(domain, event_type="cell_expansion_completed")) == 3
    assert len(database.read_events(domain, event_type="colony_approved")) == 1
    assert len(database.read_events(domain, event_type="decomposition_completed")) == 1
