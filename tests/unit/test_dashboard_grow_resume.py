"""Dashboard grow resumes selected cancelled rounds without replaying completed phases."""

import asyncio
from collections import Counter
from dataclasses import replace

from dbos import DBOS
from httpx import ASGITransport, AsyncClient
import pytest

from petri.cli.init import create_petri_dish
from petri.dashboard.api import create_app
from petri.execution.dbos_backend import DBOSExecutionBackend
from petri.graph.colony import ColonyGraph
from petri.models import Cell, Colony
from petri.storage import database
from petri.storage.colonies import save_colony
from petri.storage.paths import dish_db_path
from tests.pipeline_stubs import StubPhases


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_source", ["dashboard", "stop"])
async def test_dashboard_cancel_then_scoped_grow_resumes_only_selected_round(tmp_path, cancel_source):
    create_petri_dish(tmp_path, dish_name="dish")
    path = dish_db_path(tmp_path, "dish")
    ids = ["dish-alpha-000-000", "dish-alpha-000-001", "dish-beta-000-000"]
    for colony_id in ("dish-alpha", "dish-beta"):
        graph = ColonyGraph(colony_id)
        for cell_id in ids:
            if cell_id.rsplit("-", 2)[0] == colony_id:
                graph.add_cell(Cell(id=cell_id, colony_id=colony_id, claim_text=cell_id, level=0))
        save_colony(path, graph, Colony(
            id=colony_id, dish="dish", center_cell_id=graph.get_all_cells()[0].id,
            center_claim=colony_id,
        ))
    entered = asyncio.Event()
    release = asyncio.Event()
    phases = StubPhases()
    researching = set()

    async def research(value):
        researching.add(value.cell_id)
        if researching == set(ids):
            entered.set()
        await release.wait()
        return await phases.research(value)

    async def finish(value, result):
        with database.write_connection(path) as connection:
            connection.execute("UPDATE cells SET status=? WHERE id=?", (result.status.upper(), value.cell_id))

    dependencies = replace(phases.dependencies(), research=research, finish=finish)
    backend = DBOSExecutionBackend()
    await backend.start_async(tmp_path, db_path=path, dependencies=lambda _: dependencies)
    try:
        async with AsyncClient(transport=ASGITransport(app=create_app(
            tmp_path, execution_backend=backend)), base_url="http://localhost:8090") as client:
            submitted = await client.post("/api/runs", json={"command": "grow"})
            assert submitted.status_code == 202
            await asyncio.wait_for(entered.wait(), 10)
            first_id = submitted.json()["id"]
            if cancel_source == "dashboard":
                cancelled = await client.post("/api/runs/" + first_id + "/cancel")
                assert cancelled.json()["status"] == "CANCELLED"
            else:
                assert backend.request_stop(tmp_path, force=True) >= len(ids)
            release.set()
            async with asyncio.timeout(10):
                while backend.app._active_steps:
                    await asyncio.sleep(0.01)
            resubmitted = await client.post("/api/runs", json={
                "command": "grow", "colony": "alpha", "cell_ids": [ids[0]],
            })
            assert resubmitted.status_code == 202
            async with asyncio.timeout(10):
                while True:
                    status = await client.get(resubmitted.json()["status_url"])
                    if status.json()["status"] not in {"PENDING", "ENQUEUED"}:
                        break
                    await asyncio.sleep(0.05)
            assert status.json()["status"] == "SUCCESS", status.text
            assert database.get_cell(path, ids[0])["status"] == "VALIDATED"
            for cell_id in ids[1:]:
                assert database.get_cell(path, cell_id)["status"] == "NEW"
                assert (await DBOS.get_workflow_status_async(cell_id + ":r1")).status == "CANCELLED"
            assert (await DBOS.get_workflow_status_async(first_id)).status == "CANCELLED"
            if cancel_source == "stop":
                from petri.execution.control import cancellation_mode

                system = tmp_path / "dbos.sqlite"
                assert cancellation_mode(system, ids[0] + ":r1") is None
                assert all(cancellation_mode(system, cell_id + ":r1") == "force" for cell_id in ids[1:])
            socratic_calls = Counter(value.cell_id for phase, value in phases.calls if phase == "socratic")
            assert socratic_calls == Counter({cell_id: 1 for cell_id in ids})
            assert len(database.read_events(path, cell_id=ids[0], event_type="cell_queued")) == 1
    finally:
        release.set()
        await backend.close_async()
