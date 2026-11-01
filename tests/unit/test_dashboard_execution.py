"""Dashboard execution reads share a backend contract and need no queue file."""

import ast
from pathlib import Path

from fastapi.testclient import TestClient

from petri.dashboard.api import create_app
from petri.execution.dbos_backend import DBOSExecutionBackend
from petri.storage import database
from petri.storage.paths import dish_db_path


def test_queue_and_stats_read_sqlite_without_legacy_artifacts(seeded_petri_dir):
    petri_dir = seeded_petri_dir["petri_dir"]
    client = TestClient(create_app(petri_dir, execution_backend=DBOSExecutionBackend()))
    entries = client.get("/api/queue").json()
    expected = sorted(row["id"] for row in database.list_cells(dish_db_path(petri_dir)))
    assert sorted(row["cell_id"] for row in entries) == expected
    assert {row["queue_state"] for row in entries} == {"queued"}
    assert client.get("/api/stats").json()["queue_size"] == 5
    assert not (petri_dir / "queue.json").exists()
    assert not (petri_dir / "dbos.sqlite").exists()
    cell_id = entries[0]["cell_id"]
    row = database.get_cell(dish_db_path(petri_dir), cell_id)
    database.put_cell(dish_db_path(petri_dir), {**row, "status": "VALIDATED"})
    assert next(row for row in client.get("/api/queue").json() if row["cell_id"] == cell_id)["queue_state"] == "done"


def test_dashboard_has_no_queue_store_imports():
    root = Path(__file__).parents[2] / "petri" / "dashboard"
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                assert node.module != "petri.storage.queue", path
                if node.module == "petri.storage":
                    assert all(alias.name != "queue" for alias in node.names), path
            elif isinstance(node, ast.Import):
                assert all(alias.name != "petri.storage.queue" for alias in node.names), path


def test_cell_reads_include_explicit_dish_identity(seeded_petri_dir):
    petri_dir = seeded_petri_dir['petri_dir']
    client = TestClient(create_app(petri_dir,execution_backend=DBOSExecutionBackend()))
    cells = client.get('/api/cells').json()
    assert cells and all(cell['dish_id'] == seeded_petri_dir['dish_id'] for cell in cells)
    detail = client.get('/api/cell/'+cells[0]['cell_id']).json()
    assert detail['dish_id'] == seeded_petri_dir['dish_id']
    assert detail['colony_id'] == cells[0]['colony_id']
    assert not (petri_dir/'dbos.sqlite').exists()
