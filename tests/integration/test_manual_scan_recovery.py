"""Manual inspection leaves killed research workflows and checkpoints untouched."""

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from petri.storage import database
from petri.storage.paths import dish_db_path


@pytest.mark.integration
def test_scan_leaves_pending_research_checkpoint_bytes_unchanged(tmp_path):
    root = Path(__file__).resolve().parents[2]
    petri_dir = tmp_path / ".petri"
    environment = {**os.environ, "PYTHONPATH": str(root)}
    worker = root / "tests/helpers/manual_scan_worker.py"
    with (tmp_path / "worker.log").open("w") as log:
        process = subprocess.Popen(
            [sys.executable, str(worker), str(petri_dir)], env=environment, stdout=log, stderr=log
        )
        try:
            deadline = time.monotonic() + 15
            while not (petri_dir / "research-started").exists():
                assert process.poll() is None, (tmp_path / "worker.log").read_text()
                assert time.monotonic() < deadline, (tmp_path / "worker.log").read_text()
                time.sleep(0.02)
            process.kill()
            process.wait(timeout=5)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
    checkpoint_files = [
        petri_dir / name
        for name in ["dbos.sqlite", "dbos.sqlite-wal", "dbos.sqlite-shm"]
        if (petri_dir / name).exists()
    ]
    assert checkpoint_files
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in checkpoint_files}
    cell_before = database.get_cell(dish_db_path(petri_dir), "dish-colony-001-001")
    result = subprocess.run(
        [sys.executable, "-m", "petri", "scan"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "No inconsistencies found." in result.stdout
    assert {
        path: (path.read_bytes(), path.stat().st_mtime_ns) for path in checkpoint_files
    } == before
    assert database.get_cell(dish_db_path(petri_dir), "dish-colony-001-001") == cell_before
    events = database.read_events(dish_db_path(petri_dir), event_type="scan_completed")
    assert len(events) == 1 and events[0]["data"]["trigger"] == "manual"
