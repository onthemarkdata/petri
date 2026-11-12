"""Seeding must not overwrite previously created research."""

import hashlib
import time
from contextlib import closing
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from petri.cli import app
from petri.dashboard.api import create_app
from petri.graph.colony import ColonyGraph
from petri.models import Cell, Colony
from petri.storage import database
from petri.storage.colonies import existing_colony_message, reserve_colony, save_colony
from petri.storage.paths import dish_db_path
from petri.storage.seed_guard import ColonyExistsError, require_empty_colony
from tests.conftest import TypedFakeProvider as FakeProvider


def snapshot(path):
    return {
        str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in path.rglob("*") if p.is_file()
    }



@pytest.fixture
def neighboring_colony(seeded_petri_dir):
    env = seeded_petri_dir
    colony_id = f"{env['dish_id']}-neighbor"
    center_id = f"{colony_id}-000-000"
    graph = ColonyGraph(colony_id=colony_id)
    graph.add_cell(Cell(id=center_id, colony_id=colony_id, claim_text="Another claim", level=0))
    save_colony(
        dish_db_path(env["petri_dir"]), graph,
        Colony(id=colony_id, dish=env["dish_id"], center_cell_id=center_id, center_claim="Another claim"),
    )
    return env


def test_cli_preserves_existing_colony(neighboring_colony, monkeypatch):
    env = neighboring_colony
    before = snapshot(env["petri_dir"])
    provider = FakeProvider()
    monkeypatch.setattr("petri.cli.seed.resolve_provider", lambda _: provider)
    result = CliRunner().invoke(app, [
        "seed", "New claim", "--colony", env["colony_name"], "--no-questions",
    ])
    assert result.exit_code == 1, result.output
    assert f"Colony '{env['colony_id']}' already exists with 5 stored cells." in result.output
    assert str(dish_db_path(env["petri_dir"]).resolve()) in result.output
    assert f"seed --resume --colony '{env['colony_name']}'" in result.output
    assert snapshot(env["petri_dir"]) == before
    assert provider.decompose_calls == []


@pytest.mark.parametrize("same_claim", [False, True])
def test_dashboard_preserves_existing_colony(neighboring_colony, monkeypatch, same_claim):
    env = neighboring_colony
    def unexpected_provider(self, spec):
        pytest.fail("Existing colonies must be rejected before resolving inference.")

    monkeypatch.setattr("petri.execution.dbos_backend.DBOSExecutionBackend._default_seed_runner", unexpected_provider)
    with TestClient(create_app(env["petri_dir"])) as client:
        before = {name: digest for name, digest in snapshot(env["petri_dir"]).items()
                  if not name.endswith(("-wal", "-shm"))}
        claim = env["colony_model"].center_claim if same_claim else "New"
        result = client.post("/api/seed", json={"claim": claim, "colony": env["colony_name"]})
        assert result.status_code == 409
        detail = result.json()["detail"]
        assert f"Colony '{env['colony_id']}' already exists with 5 stored cells." in detail
        assert str(dish_db_path(env["petri_dir"]).resolve()) in detail
        assert "regeneration" in detail
        after = {name: digest for name, digest in snapshot(env["petri_dir"]).items()
                 if not name.endswith(("-wal", "-shm"))}
        assert after == before


@pytest.mark.parametrize("cell_count", [0, 2])
@pytest.mark.parametrize("supplied_connection", [False, True])
def test_reservation_conflict_keeps_stored_cells_and_name(
    tmp_path, cell_count, supplied_connection
):
    path = tmp_path / "dish" / "petri.sqlite"
    colony = Colony(
        id="dish-existing", dish="dish", center_cell_id="dish-existing-000-000",
        center_claim="Existing claim",
    )
    reserve_colony(path, colony)
    for seq in range(cell_count):
        database.put_cell(path, {
            "id": f"dish-existing-001-{seq:03d}", "colony": colony.id, "level": 1,
            "seq": seq, "claim_text": "Stored premise", "created_at": "2026-11-01",
        })
    database.put_cell(path, {
        "id": "dish-other-000-000", "colony": "dish-other", "level": 0, "seq": 0,
        "claim_text": "Unrelated premise", "created_at": "2026-11-01",
    })
    before = snapshot(path.parent)

    with closing(database.connect(path)) as connection:
        if supplied_connection:
            connection.execute("BEGIN IMMEDIATE")
        with pytest.raises(ColonyExistsError) as raised:
            reserve_colony(path, colony, connection=connection if supplied_connection else None)
        message = str(raised.value)
        assert f"Colony '{colony.id}' already exists with {cell_count} stored cells." in message
        assert str(path.resolve()) in message
        assert connection.in_transaction == supplied_connection
        assert connection.execute("SELECT count(*) FROM colonies WHERE id=?", (colony.id,)).fetchone()[0] == 1
        assert connection.execute("SELECT count(*) FROM cells WHERE colony=?", (colony.id,)).fetchone()[0] == cell_count
        if supplied_connection:
            connection.rollback()
    assert snapshot(path.parent) == before


def test_reservation_diagnostic_reads_uncommitted_cells(tmp_path, monkeypatch):
    path = tmp_path / "petri.sqlite"
    colony = Colony(
        id="dish-existing", dish="dish", center_cell_id="dish-existing-000-000",
        center_claim="Existing claim",
    )
    reserve_colony(path, colony)
    with closing(database.connect(path)) as connection:
        connection.execute("BEGIN IMMEDIATE")
        database.put_cell(path, {
            "id": colony.center_cell_id, "colony": colony.id, "level": 0, "seq": 0,
            "claim_text": colony.center_claim, "created_at": "2026-11-01",
        }, connection=connection)

        def unexpected_connection(*args, **kwargs):
            pytest.fail("The caller's transaction must supply the diagnostic count.")

        monkeypatch.setattr(database, "connect", unexpected_connection)
        with pytest.raises(ColonyExistsError, match="with 1 stored cell[.]"):
            reserve_colony(path, colony, connection=connection)
        assert connection.in_transaction
        assert connection.execute("SELECT count(*) FROM cells").fetchone()[0] == 1
        connection.rollback()
        assert connection.execute("SELECT count(*) FROM cells").fetchone()[0] == 0
        assert connection.execute("SELECT count(*) FROM colonies").fetchone()[0] == 1


def test_conflict_diagnostic_uses_read_only_connection(neighboring_colony, monkeypatch):
    env = neighboring_colony
    path = dish_db_path(env["petri_dir"])
    before = snapshot(env["petri_dir"])

    def unexpected_writer(*args, **kwargs):
        pytest.fail("Formatting a refusal must not open or migrate a writer.")

    monkeypatch.setattr(database, "connect", unexpected_writer)
    assert existing_colony_message(path, env["colony_id"]) == (
        f"Colony '{env['colony_id']}' already exists with 5 stored cells. "
        f"Dish database: {path.resolve()}."
    )
    assert snapshot(env["petri_dir"]) == before


def test_conflict_diagnostic_counts_committed_wal_rows(tmp_path, monkeypatch):
    path = tmp_path / "petri.sqlite"
    colony = Colony(
        id="dish-wal", dish="dish", center_cell_id="dish-wal-000-000",
        center_claim="Existing claim",
    )
    reserve_colony(path, colony)
    with closing(database.connect(path)) as writer:
        writer.execute("PRAGMA wal_autocheckpoint=0")
        writer.execute("BEGIN IMMEDIATE")
        for colony_id, cell_id in [
            (colony.id, "dish-wal-001-001"), ("dish-other", "dish-other-000-000"),
        ]:
            database.put_cell(path, {
                "id": cell_id, "colony": colony_id, "level": 1, "seq": 1,
                "claim_text": "A committed WAL premise", "created_at": "2026-11-01",
            }, connection=writer)
        writer.commit()
        wal = path.with_name(path.name + "-wal")
        assert b"dish-wal-001-001" in wal.read_bytes()
        assert b"dish-wal-001-001" not in path.read_bytes()
        before = (path.read_bytes(), wal.read_bytes())

        def unexpected_writer(*args, **kwargs):
            pytest.fail("Counting committed WAL rows must not run migrations.")

        monkeypatch.setattr(database, "connect", unexpected_writer)
        assert existing_colony_message(path, colony.id) == (
            f"Colony '{colony.id}' already exists with 1 stored cell. "
            f"Dish database: {path.resolve()}."
        )
        assert (path.read_bytes(), wal.read_bytes()) == before


@pytest.mark.parametrize("surface", ["cli", "dashboard"])
def test_cell_free_directory_can_be_reused(seeded_petri_dir, monkeypatch, surface):
    env = seeded_petri_dir
    target = env["petri_dir"] / "petri-dishes" / "fresh"
    target.mkdir()
    provider = FakeProvider()
    monkeypatch.setattr("petri.cli.seed.resolve_provider", lambda _: provider)
    monkeypatch.setattr("petri.cli._bootstrap.resolve_provider", lambda _: provider)
    if surface == "cli":
        result = CliRunner().invoke(app, [
            "seed", "A new claim", "--colony", "fresh", "--no-questions",
        ])
        assert result.exit_code == 0, result.output
    else:
        monkeypatch.setattr("petri.execution.dbos_backend.DBOSExecutionBackend._default_seed_runner",
                            lambda self, spec: provider)
        with TestClient(create_app(env["petri_dir"])) as client:
            result = client.post("/api/seed", json={"claim": "A new claim", "colony": "fresh"})
            assert result.status_code == 202, result.text
            for _ in range(150):
                status = client.get(result.json()["status_url"])
                if status.json()["status"] not in {"PENDING", "ENQUEUED"}:
                    break
                time.sleep(0.05)
            assert status.json()["status"] == "SUCCESS", status.text
    from petri.storage.database import list_cells
    from petri.storage.paths import dish_db_path

    assert len(list_cells(dish_db_path(env["petri_dir"]), colony=f"{env['dish_id']}-fresh")) == 3
    assert not list(target.iterdir())


def test_incomplete_colony_is_protected(tmp_path):
    cell = tmp_path / "partial" / "cell"
    cell.mkdir(parents=True)
    (cell / "metadata.json").write_text("incomplete")
    with pytest.raises(ColonyExistsError, match="1 cell"):
        require_empty_colony(cell.parent)
    assert (cell / "metadata.json").read_text() == "incomplete"


def test_symlink_destination_is_rejected(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ColonyExistsError, match="symbolic link"):
        require_empty_colony(link)


def test_regeneration_replaces_only_current_run(seeded_petri_dir, monkeypatch):
    from click.testing import _NamedTextIOWrapper

    env = seeded_petri_dir
    before = snapshot(env["colony_path"])
    provider = FakeProvider()
    provider.questions_response = []
    actions = iter(["Regenerate with guidance", "Accept"])
    monkeypatch.setattr("petri.cli.seed.resolve_provider", lambda _: provider)
    monkeypatch.setattr(_NamedTextIOWrapper, "isatty", lambda _: True)
    monkeypatch.setattr("questionary.select", lambda *a, **kw: SimpleNamespace(ask=lambda: next(actions)))
    monkeypatch.setattr("questionary.text", lambda *a, **kw: SimpleNamespace(ask=lambda: "Focus on evidence"))
    result = CliRunner().invoke(app, ["seed", "New claim", "--colony", "fresh"])
    assert result.exit_code == 0, result.output
    assert len(provider.decompose_calls) == 2
    assert provider.decompose_calls[-1]["guidance"] == "Focus on evidence"
    assert snapshot(env["colony_path"]) == before
    from petri.storage import database
    from petri.storage.paths import dish_db_path

    path = dish_db_path(env["petri_dir"])
    assert len(database.read_events(path, event_type="cell_created")) == 4


@pytest.mark.parametrize("interrupt_replacement", [False, True])
def test_abort_then_fresh_seed_reuses_name_without_replaying_old_output(
    seeded_petri_dir, monkeypatch, interrupt_replacement
):
    from click.testing import _NamedTextIOWrapper
    from petri.storage import database
    from petri.storage.colonies import colony_exists, load_colony
    from petri.storage.paths import dish_db_path

    env = seeded_petri_dir
    path = dish_db_path(env["petri_dir"])
    colony_id = f"{env['dish_id']}-fresh"
    unrelated = database.list_cells(path, colony=env["colony_id"])
    first = FakeProvider()
    second = FakeProvider()
    first.questions_response = second.questions_response = []
    second.decompose_response = {
        "nodes": [
            {"claim_text": "A replacement meter premise"},
            {"claim_text": "A replacement second premise", "relation": "limits"},
        ]
    }
    selected = [first]
    actions = iter(["Abort", "Accept"])
    monkeypatch.setattr("petri.cli.seed.resolve_provider", lambda _: selected[0])
    monkeypatch.setattr(_NamedTextIOWrapper, "isatty", lambda _: True)
    monkeypatch.setattr(
        "questionary.select", lambda *a, **kw: SimpleNamespace(ask=lambda: next(actions))
    )
    runner = CliRunner()
    aborted = runner.invoke(app, ["seed", "Original units claim", "--colony", "fresh"])
    assert aborted.exit_code == 1 and "Decomposition rejected" in aborted.output, aborted.output
    assert not database.list_cells(path, colony=colony_id)
    assert not colony_exists(path, colony_id)
    history = {event["id"]: event for event in database.read_events(path)}
    assert history

    selected[0] = second
    expand = second.expand_why
    if interrupt_replacement:
        def fail_expansion(*args, **kwargs):
            raise RuntimeError("Interrupted replacement draft")
        monkeypatch.setattr(second, "expand_why", fail_expansion)
    accepted = runner.invoke(app, ["seed", "Replacement units claim", "--colony", "fresh"])
    if interrupt_replacement:
        assert accepted.exit_code == 2 and "Interrupted replacement" in accepted.output
        monkeypatch.setattr(second, "expand_why", expand)
        accepted = runner.invoke(app, ["seed", "--resume", "--colony", "fresh", "--no-questions"])
    assert accepted.exit_code == 0, (accepted.output, accepted.exception)
    graph, colony = load_colony(path, colony_id)
    assert colony.center_claim == "Replacement units claim"
    assert {cell.claim_text for cell in graph.get_all_cells() if cell.level} == {
        "A replacement meter premise", "A replacement second premise"
    }
    assert len(first.decompose_calls) == len(second.decompose_calls) == 1
    events = database.read_events(path)
    by_id = {event["id"]: event for event in events}
    assert all(by_id[key] == event for key, event in history.items())
    assert len([event for event in events if event["type"] == "decomposition_started"]) == 2
    assert len([event for event in events if event["type"] == "decomposition_completed"]) == 2
    assert len([event for event in events if event["type"] == "cell_created"]) == 4
    assert len([event for event in events if event["type"] == "colony_approved"]) == 1
    assert database.list_cells(path, colony=env["colony_id"]) == unrelated
