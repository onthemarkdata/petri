"""A detached CLI owner leaves dashboard reads and durable event polling available."""

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from petri.cli.init import create_petri_dish
from petri.dashboard.api import create_app
from petri.dashboard.stream_api import stream_messages
from petri.models import Event
from petri.storage import database
from petri.storage.paths import dish_db_path
from petri.storage.colonies import save_colony
from tests.conftest import _build_canonical_colony

pytestmark = pytest.mark.integration


def test_detached_cli_owner_preserves_dashboard_reads_and_sse(tmp_path):
    root = Path(__file__).parents[2]
    create_petri_dish(tmp_path, dish_name="test-dish")
    colony = _build_canonical_colony()
    save_colony(dish_db_path(tmp_path), colony["graph"], colony["colony_model"])
    environment = {**os.environ, "PYTHONPATH": str(root), "OTEL_SDK_DISABLED": "true"}
    with (tmp_path / "cli.log").open("w") as log:
        process = subprocess.Popen(
            [sys.executable, str(root / "tests/helpers/dbos_grow_worker.py"), str(tmp_path)],
            cwd=root,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 20
            while not (tmp_path / "in-research").exists() and time.monotonic() < deadline:
                assert process.poll() is None, (tmp_path / "cli.log").read_text()
                time.sleep(0.05)
            assert (tmp_path / "in-research").exists(), (tmp_path / "cli.log").read_text()
            app = create_app(tmp_path)
            with TestClient(app) as client:
                assert not app.state.execution_owner.started
                cells = client.get("/api/cells")
                assert cells.status_code == 200 and cells.json()
                response = client.post("/api/runs", json={"command": "grow"})
                assert response.status_code == 503
                assert "restart the dashboard" in response.json()["detail"]

                async def committed_progress():
                    stream = stream_messages(tmp_path)
                    try:
                        assert (await anext(stream))["event"] == "connected"
                        database.append_event(
                            dish_db_path(tmp_path),
                            Event(
                                id="detached-owner-event",
                                cell_id=cells.json()[0]["cell_id"],
                                timestamp="2026-11-01T00:00:00Z",
                                type="verdict_issued",
                                agent="investigator",
                                iteration=0,
                                data={
                                    "verdict": "EVIDENCE_SUFFICIENT",
                                    "summary": "Public fixture evidence",
                                },
                            ),
                        )
                        for _ in range(200):
                            event = await asyncio.wait_for(anext(stream), timeout=5)
                            assert event["event"] == "event_inserted"
                            if json.loads(event["data"])["id"] == "detached-owner-event":
                                break
                        else:
                            pytest.fail("The detached owner's committed event was not streamed.")
                    finally:
                        await stream.aclose()

                asyncio.run(committed_progress())
            (tmp_path / "release").touch()
            assert process.wait(timeout=20) == 0, (tmp_path / "cli.log").read_text()
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
    with TestClient(create_app(tmp_path)) as restarted:
        assert restarted.app.state.execution_owner.started
        assert restarted.get("/api/cells").status_code == 200
