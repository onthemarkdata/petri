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


def defaults_bytes(petri_dir):
    return {str(path.relative_to(petri_dir/'defaults')):path.read_bytes()
            for path in (petri_dir/'defaults').rglob('*') if path.is_file()}


def test_restore_replaces_defaults_exactly_and_keeps_snapshot(tmp_path):
    from petri.storage.backup import restore_snapshot

    seed(tmp_path)
    defaults = tmp_path/'defaults'
    (defaults/'nested').mkdir()
    (defaults/'nested/original.txt').write_text('Snapshot content')
    expected = defaults_bytes(tmp_path)
    snapshot = snapshot_before_destructive(tmp_path,'dish')
    (defaults/'nested/original.txt').write_text('Changed after snapshot')
    (defaults/'later.txt').write_text('Created after snapshot')
    restore_snapshot(tmp_path,snapshot.name,force=True)
    assert defaults_bytes(tmp_path)==expected
    assert (snapshot/'defaults/nested/original.txt').read_text()=='Snapshot content'
    assert not list(tmp_path.glob('.restore-*'))


def test_restore_without_snapshotted_defaults_leaves_live_defaults(tmp_path):
    import shutil
    from petri.storage.backup import restore_snapshot

    seed(tmp_path)
    snapshot = snapshot_before_destructive(tmp_path,'dish')
    shutil.rmtree(snapshot/'defaults')
    (tmp_path/'defaults/later.txt').write_text('Keep live defaults')
    expected = defaults_bytes(tmp_path)
    restore_snapshot(tmp_path,snapshot.name,force=True)
    assert defaults_bytes(tmp_path)==expected


def test_defaults_staging_failure_preserves_live_database_and_defaults(tmp_path,monkeypatch):
    import shutil
    import pytest
    from petri.storage.backup import restore_snapshot

    path = seed(tmp_path)
    snapshot = snapshot_before_destructive(tmp_path,'dish')
    with closing(database.connect(path)) as connection:
        connection.execute("UPDATE cells SET claim_text='Post-snapshot content'")
    (tmp_path/'defaults/later.txt').write_text('Live content')
    expected_rows,expected_defaults = rows(path),defaults_bytes(tmp_path)
    def fail_copy(*args,**kwargs):
        raise OSError('Fixture copy failure')
    monkeypatch.setattr(shutil,'copytree',fail_copy)
    with pytest.raises(OSError,match='copy failure'):
        restore_snapshot(tmp_path,snapshot.name,force=True)
    assert rows(path)==expected_rows and defaults_bytes(tmp_path)==expected_defaults
    assert not list(tmp_path.glob('.restore-*'))


def test_defaults_publish_failure_rolls_back_database_sidecars_and_defaults(tmp_path,monkeypatch):
    from pathlib import Path
    import pytest
    from petri.storage.backup import restore_snapshot

    path = seed(tmp_path)
    snapshot = snapshot_before_destructive(tmp_path,'dish')
    with closing(database.connect(path)) as connection:
        connection.execute("UPDATE cells SET claim_text='Post-snapshot content'")
    expected_db = path.read_bytes()
    for suffix in ['-wal','-shm']:
        Path(str(path)+suffix).write_bytes(b'Preserve old sidecar')
    (tmp_path/'defaults/later.txt').write_text('Live content')
    expected_defaults = defaults_bytes(tmp_path)
    replace = Path.replace
    def fail_publish(source,target):
        if source.name=='defaults' and source.parent.name.startswith('.restore-'):
            raise OSError('Fixture publish failure')
        return replace(source,target)
    monkeypatch.setattr(Path,'replace',fail_publish)
    with pytest.raises(OSError,match='publish failure'):
        restore_snapshot(tmp_path,snapshot.name,force=True)
    assert path.read_bytes()==expected_db and defaults_bytes(tmp_path)==expected_defaults
    assert all(Path(str(path)+suffix).read_bytes()==b'Preserve old sidecar' for suffix in ['-wal','-shm'])
    assert not list(tmp_path.glob('.restore-*'))
