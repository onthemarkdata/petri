"""Backup snapshots remain consistent while domain writers are active."""

from contextlib import closing

from typer.testing import CliRunner

from petri.cli import app
from petri.query.connection import enumerate_dishes
from petri.storage import database
from petri.storage.backup import list_snapshots, snapshot_before_destructive


def seed(petri_dir, dish="dish"):
    path = petri_dir / "petri-dishes" / dish / "petri.sqlite"
    for index in range(2):
        database.put_cell(path, {"id": f"{dish}-colony-001-00{index}", "colony": f"{dish}-colony",
            "level": 1, "seq": index, "claim_text": "A test claim", "created_at": "2026-01-01"})
    database.add_edge(path, from_cell=f"{dish}-colony-001-000", to_cell=f"{dish}-colony-001-001",
                      edge_type="intra_colony", created_at="2026-01-01")
    with closing(database.connect(path)) as connection:
        connection.execute("INSERT INTO events VALUES ('event','cell','search_executed','agent',1,'{}','2026-01-01')")
    defaults = petri_dir / "defaults"
    defaults.mkdir(exist_ok=True)
    (defaults / "petri.yaml").write_text("name: dish\n")
    return path


def rows(path):
    with closing(database.connect(path)) as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        return {table: [tuple(row) for row in connection.execute(f"SELECT * FROM {table}")]
                for table in ("cells", "edges", "events")}


def test_snapshot_matches_rows_and_defaults(tmp_path):
    path = seed(tmp_path)
    snapshot = snapshot_before_destructive(tmp_path, "dish")
    assert rows(path) == rows(snapshot / "petri.sqlite")
    assert (snapshot / "defaults/petri.yaml").read_bytes() == (tmp_path / "defaults/petri.yaml").read_bytes()
    assert [dish.dish_id for dish in enumerate_dishes(tmp_path)] == ["dish"]


def test_snapshot_includes_only_committed_rows(tmp_path):
    path = seed(tmp_path)
    original = rows(path)
    with closing(database.connect(path)) as writer:
        writer.execute("BEGIN IMMEDIATE")
        writer.execute("UPDATE cells SET claim_text='uncommitted'")
        snapshot = snapshot_before_destructive(tmp_path, "dish")
        assert rows(snapshot / "petri.sqlite") == original
        writer.rollback()


def test_list_and_cli_order(tmp_path, monkeypatch):
    from petri.cli import backup as module

    seed(tmp_path)
    first = snapshot_before_destructive(tmp_path, "dish")
    second = snapshot_before_destructive(tmp_path, "dish")
    snapshots = list_snapshots(tmp_path)
    assert [snapshot.path for snapshot in snapshots] == [second, first]
    assert all(snapshot.cell_count == 2 and snapshot.size_bytes > 0 for snapshot in snapshots)
    monkeypatch.setattr(module, "find_petri_dir", lambda: tmp_path)
    result = CliRunner().invoke(app, ["backup", "--list"])
    assert result.exit_code == 0
    assert result.output.index(second.name) < result.output.index(first.name)
    assert "2 cells" in result.output and "bytes" in result.output
    result = CliRunner().invoke(app, ["backup", "--dish", "dish"])
    assert result.exit_code == 0 and "Snapshot saved" in result.output


def test_restore_round_trip_guard_siblings_and_sidecars(tmp_path):
    import pytest
    from petri.storage.backup import restore_snapshot

    path = seed(tmp_path)
    sibling = seed(tmp_path, "other")
    sibling_before = sibling.read_bytes(), sibling.stat().st_mtime_ns
    expected = rows(path)
    snapshot = snapshot_before_destructive(tmp_path, "dish")
    with closing(database.connect(path)) as connection:
        connection.execute("INSERT INTO events VALUES ('later','cell','search_executed','agent',1,'{}','2026-01-02')")
    with pytest.raises(ValueError, match="--force"):
        restore_snapshot(tmp_path, snapshot.name)
    for suffix in ("-wal", "-shm"):
        from pathlib import Path
        Path(str(path) + suffix).write_bytes(b"stale")
    assert restore_snapshot(tmp_path, snapshot.name, force=True) == path
    assert rows(path) == expected
    assert (sibling.read_bytes(), sibling.stat().st_mtime_ns) == sibling_before


def test_restore_cli_warns_about_dbos(tmp_path, monkeypatch):
    from petri.cli import backup as module

    seed(tmp_path)
    snapshot = snapshot_before_destructive(tmp_path, "dish")
    (tmp_path / "dbos.sqlite").touch()
    monkeypatch.setattr(module, "find_petri_dir", lambda: tmp_path)
    result = CliRunner().invoke(app, ["backup", "--restore", snapshot.name, "--force"])
    assert result.exit_code == 0, result.output
    assert "WARNING: DBOS" in result.output and "desynchronize" in result.output


def test_restore_rejects_active_execution_and_path_escape(tmp_path):
    import pytest
    from filelock import FileLock
    from petri.storage.backup import restore_snapshot

    seed(tmp_path)
    snapshot = snapshot_before_destructive(tmp_path, "dish")
    with FileLock(tmp_path / ".execution-owner.lock"):
        with pytest.raises(RuntimeError, match="Stop grow"):
            restore_snapshot(tmp_path, snapshot.name, force=True)
    with pytest.raises(ValueError, match="snapshot name"):
        restore_snapshot(tmp_path, "../elsewhere", force=True)
