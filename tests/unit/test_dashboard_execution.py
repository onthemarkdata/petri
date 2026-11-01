"""Dashboard execution reads share a backend contract and need no queue file."""

import ast
from pathlib import Path

from fastapi.testclient import TestClient

from petri.dashboard.api import create_app
from petri.execution.dbos_backend import DBOSExecutionBackend
from petri.execution.v1_engine import V1ExecutionBackend
from petri.storage import database, queue
from petri.storage.paths import dish_db_path


def test_queue_and_stats_match_between_backends(seeded_petri_dir):
    petri_dir = seeded_petri_dir["petri_dir"]
    path = petri_dir / "queue.json"
    legacy = {"version": 1, "entries": {
        row["id"]: {"cell_id": row["id"], "queue_state": "queued", "iteration": 0}
        for row in database.list_cells(dish_db_path(petri_dir))
    }}
    queue.save_queue(path, legacy)
    v1 = TestClient(create_app(petri_dir, execution_backend=V1ExecutionBackend()))
    before = v1.get("/api/queue").json()
    stats = v1.get("/api/stats").json()
    path.unlink()
    dbos = TestClient(create_app(petri_dir, execution_backend=DBOSExecutionBackend()))
    after = dbos.get("/api/queue").json()
    def projection(rows):
        return sorted(
            (row["cell_id"], row["queue_state"], row["iteration"]) for row in rows
        )
    assert projection(after) == projection(before)
    assert dbos.get("/api/stats").json() == stats
    assert not path.exists()
    assert not (petri_dir / "dbos.sqlite").exists()
    cell_id = after[0]["cell_id"]
    row = database.get_cell(dish_db_path(petri_dir), cell_id)
    database.put_cell(dish_db_path(petri_dir), {**row, "status": "VALIDATED"})
    assert next(row for row in dbos.get("/api/queue").json() if row["cell_id"] == cell_id)["queue_state"] == "done"


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
