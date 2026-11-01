"""Scoped grow outcomes count selected cells and retain selected failures."""

from dataclasses import replace

import pytest
from typer.testing import CliRunner

from petri.cli import app
from petri.cli.init import create_petri_dish
from petri.execution.dbos_backend import DBOSExecutionBackend
from petri.graph.colony import ColonyGraph
from petri.models import Cell, Colony
from petri.storage import database
from petri.storage.colonies import save_colony
from petri.storage.edge_registry import EdgeRecord, append
from petri.storage.paths import dish_db_path
from tests.pipeline_stubs import StubPhases


@pytest.mark.parametrize("scope", ["colony", "cell"])
@pytest.mark.parametrize("outcome", ["success", "stalled", "blocked"])
def test_scoped_grow_reports_only_selected_outcome(tmp_path, monkeypatch, scope, outcome):
    directory = tmp_path / ".petri"
    create_petri_dish(directory, dish_name="dish")
    path = dish_db_path(directory, "dish")
    ids = ["dish-alpha-000-000", "dish-beta-000-000"]
    for cell_id in ids:
        colony_id = cell_id.rsplit("-", 2)[0]
        graph = ColonyGraph(colony_id)
        graph.add_cell(Cell(id=cell_id, colony_id=colony_id, claim_text=cell_id, level=0))
        save_colony(path, graph, Colony(
            id=colony_id, dish="dish", center_cell_id=cell_id, center_claim=cell_id,
        ))
    if outcome == "blocked":
        append(path, EdgeRecord(from_cell=ids[0], to_cell=ids[1], edge_type="cross_colony",
                                created_at="2026-01-01T00:00:00Z"))
    phases = StubPhases(park_phase="research" if outcome == "stalled" else None)

    async def finish(value, result):
        with database.write_connection(path) as connection:
            connection.execute("UPDATE cells SET status=? WHERE id=?", (
                getattr(result, "cell_status", result.status.upper()), value.cell_id,
            ))

    dependencies = replace(phases.dependencies(), finish=finish)
    backend = DBOSExecutionBackend()
    start = backend.start
    monkeypatch.setattr(backend, "start", lambda directory, **options: start(
        directory, **options, dependencies=lambda _: dependencies))
    monkeypatch.setattr("petri.cli.grow.resolve_backend", lambda _: backend)
    monkeypatch.chdir(tmp_path)
    arguments = ["grow", "--colony", "alpha"] if scope == "colony" else ["grow", "--cell", ids[0]]
    result = CliRunner().invoke(app, arguments)
    assert result.exit_code == (0 if outcome == "success" else 1), (result.output, result.exception)
    reason, final_states = {
        "success": ("all_terminal", "done: 1"),
        "stalled": ("needs_human", "needs_human: 1"),
        "blocked": ("not_eligible", "queued: 1"),
    }[outcome]
    assert f"Grow loop finished: {reason}" in result.output
    assert f"Final queue: {final_states}" in result.output
    assert database.get_cell(path, ids[1])["status"] == "NEW"
    assert all(value.cell_id == ids[0] for _, value in phases.calls)
