"""CLI grow filters preserve cancelled work outside the requested scope."""

import asyncio
from collections import Counter
from dataclasses import replace

from dbos import DBOSClient
import pytest
from typer.testing import CliRunner

from petri.cli import app
from petri.cli.init import create_petri_dish
from petri.execution.control import cancellation_mode
from petri.execution.dbos_backend import DBOSExecutionBackend
from petri.execution.run_controls import RunRequest
from petri.graph.colony import ColonyGraph
from petri.models import Cell, Colony
from petri.storage import database
from petri.storage.colonies import save_colony
from petri.storage.paths import dish_db_path
from tests.pipeline_stubs import StubPhases


@pytest.mark.parametrize("scope", ["colony", "cell", "both"])
def test_cli_scoped_resume_preserves_unselected_cancellation(tmp_path, monkeypatch, scope):
    directory = tmp_path / ".petri"
    create_petri_dish(directory, dish_name="dish")
    path = dish_db_path(directory, "dish")
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
    phases = StubPhases()

    async def finish(value, result):
        with database.write_connection(path) as connection:
            connection.execute("UPDATE cells SET status=? WHERE id=?", (result.status.upper(), value.cell_id))

    backend = DBOSExecutionBackend()

    async def prepare_cancelled_runs():
        entered, release = asyncio.Event(), asyncio.Event()
        researching = set()

        async def research(value):
            researching.add(value.cell_id)
            if researching == set(ids):
                entered.set()
            await release.wait()
            return await phases.research(value)

        dependencies = replace(phases.dependencies(), research=research, finish=finish)
        await backend.start_async(directory, db_path=path, dependencies=lambda _: dependencies)
        try:
            await backend.submit_run(RunRequest(command="grow"))
            await asyncio.wait_for(entered.wait(), 10)
            assert backend.request_stop(directory, force=True) >= len(ids)
            release.set()
            async with asyncio.timeout(10):
                while backend.app._active_steps:
                    await asyncio.sleep(0.01)
        finally:
            release.set()
            await backend.close_async()

    asyncio.run(prepare_cancelled_runs())
    start = backend.start
    dependencies = replace(phases.dependencies(), finish=finish)
    monkeypatch.setattr(backend, "start", lambda directory, **options: start(
        directory, **options, dependencies=lambda _: dependencies))
    monkeypatch.setattr("petri.cli.grow.resolve_backend", lambda _: backend)
    monkeypatch.chdir(tmp_path)
    arguments = ["grow"]
    if scope in {"colony", "both"}:
        arguments += ["--colony", "alpha"]
    if scope in {"cell", "both"}:
        arguments += ["--cell", ids[0]]
    result = CliRunner().invoke(app, arguments)
    assert result.exit_code in {0, 1}, (result.output, result.exception)
    selected = set(ids[:2] if scope == "colony" else ids[:1])
    client = DBOSClient(system_database_url=f"sqlite:///{directory / 'dbos.sqlite'}")
    try:
        runs = {run.workflow_id: run.status for run in client.list_workflows(name="petri.validate_cell")}
    finally:
        client.destroy()
    for cell_id in ids:
        chosen = cell_id in selected
        assert database.get_cell(path, cell_id)["status"] == ("VALIDATED" if chosen else "NEW")
        assert runs[cell_id + ":r1"] == ("SUCCESS" if chosen else "CANCELLED")
        assert cancellation_mode(directory / "dbos.sqlite", cell_id + ":r1") == (None if chosen else "force")
    assert Counter(value.cell_id for phase, value in phases.calls if phase == "socratic") == Counter({key: 1 for key in ids})
