"""Dish paths and CLI reads use the domain database."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import json

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from petri.cli import app
from petri.cli.init import create_petri_dish
from petri.dashboard.api import create_app
from petri.models import Colony
from petri.storage import database
from petri.storage.colonies import load_colony, reserve_colony
from petri.storage.paths import dish_db_path, dish_dir
from petri.storage.seed_guard import ColonyExistsError


def test_init_creates_current_database_and_empty_exports(tmp_path):
    petri = tmp_path / ".petri"
    create_petri_dish(petri, dish_name="research-dish")
    assert dish_dir(petri).name == "research-dish"
    with closing(database.connect(dish_db_path(petri))) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == database.SCHEMA_VERSION
    assert list((dish_dir(petri) / "exports").iterdir()) == []


@pytest.mark.parametrize("name", ["", ".", "..", "../escape", "x/y", "x\\y"])
def test_dish_path_rejects_path_components(tmp_path, name):
    with pytest.raises(ValueError):
        dish_dir(tmp_path, name)


def test_cli_and_dashboard_ignore_legacy_directories(seeded_petri_dir, monkeypatch):
    env = seeded_petri_dir
    monkeypatch.setattr("petri.graph.colony.deserialize_colony", lambda *args: pytest.fail("Legacy read"))
    stale = env["petri_dir"] / "petri-dishes" / "old-colony"
    stale.mkdir()
    (stale / "colony.json").write_text("invalid legacy data")
    checked = CliRunner().invoke(app, ["check", "--json"])
    assert checked.exit_code == 0, checked.output
    assert len(json.loads(checked.output)) == 5
    graphed = CliRunner().invoke(app, ["graph", "--format", "dot"])
    assert graphed.exit_code == 0 and "digraph" in graphed.output
    client = TestClient(create_app(env["petri_dir"]))
    assert len(client.get("/api/cells").json()) == 5
    assert [dish["id"] for dish in client.get("/api/dishes").json()] == [env["dish_id"]]
    graph, colony = load_colony(dish_db_path(env["petri_dir"]), env["colony_id"])
    assert len(graph.get_edges()) == 5
    assert colony.center_cell_id == env["center"].id


def test_seed_reservation_has_one_concurrent_winner(tmp_path):
    path = tmp_path / "dish" / "petri.sqlite"
    colony = Colony(id="dish-colony", dish="dish", center_claim="Claim",
                    center_cell_id="dish-colony-000-000", created_at="2026-11-01T00:00:00Z")

    def reserve(_):
        try:
            reserve_colony(path, colony)
            return True
        except ColonyExistsError:
            return False

    with ThreadPoolExecutor(max_workers=8) as executor:
        assert sum(executor.map(reserve, range(16))) == 1


def test_connect_and_stop_use_rows(seeded_petri_dir):
    from petri.graph.colony import ColonyGraph
    from petri.models import Cell
    from petri.storage.colonies import save_colony

    env = seeded_petri_dir
    db = dish_db_path(env["petri_dir"])
    other_id = f"{env['dish_id']}-other"
    other_cell = Cell(id=f"{other_id}-000-000", colony_id=other_id, claim_text="Other claim", level=0)
    graph = ColonyGraph(other_id)
    graph.add_cell(other_cell)
    save_colony(db, graph, Colony(id=other_id, dish=env["dish_id"], center_claim=other_cell.claim_text,
                                 center_cell_id=other_cell.id))
    linked = CliRunner().invoke(app, ["connect", env["cell1"].id, other_cell.id])
    assert linked.exit_code == 0, linked.output
    assert any(edge["to_cell"] == other_cell.id for edge in database.read_edges(db))
    config_path = env["petri_dir"] / "defaults" / "petri.yaml"
    config_path.write_text(config_path.read_text() + "execution:\n  backend: dbos\n")
    before = database.get_cell(db, other_cell.id)
    stopped = CliRunner().invoke(app, ["stop"])
    assert stopped.exit_code == 0, stopped.output
    assert "No active runs" in stopped.output
    assert database.get_cell(db, other_cell.id) == before
    assert database.read_events(db, cell_id=other_cell.id) == []
    assert not list(dish_dir(env["petri_dir"]).rglob("metadata.json"))
