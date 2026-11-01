"""Domain-schema and migration guarantees for the explicit v2 storage seam."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from contextlib import closing
from datetime import UTC, datetime

import pytest

from petri.models import Event
from petri.storage import database, event_log

NOW = datetime.now(UTC).isoformat()


def cell(key="dish-colony-0-0", seq=0):
    return {
        "id": key,
        "colony": "dish-colony",
        "level": 0,
        "seq": seq,
        "claim_text": "A test claim",
        "created_at": NOW,
    }


def event(key="test-event", **overrides):
    return Event.model_validate(
        {
            "id": key,
            "cell_id": "dish-colony-0-0",
            "timestamp": NOW,
            "type": "search_executed",
            "agent": "researcher",
            "iteration": 1,
            "data": {"query": "test", "sources_found": 1},
            **overrides,
        }
    )


def test_connect_pragmas_and_reopen(tmp_path):
    path = tmp_path / "dish" / "petri.sqlite"
    with closing(database.connect(path)) as conn:
        assert path.is_file()
        for pragma, expected in [
            ("journal_mode", "wal"),
            ("foreign_keys", 1),
            ("recursive_triggers", 1),
            ("busy_timeout", 30000),
            ("user_version", database.SCHEMA_VERSION),
        ]:
            assert conn.execute(f"PRAGMA {pragma}").fetchone()[0] == expected
        trace = []
        conn.set_trace_callback(trace.append)
        database.migrate(conn)
        assert not any("CREATE" in statement or "BEGIN" in statement for statement in trace)
    with closing(database.connect(path)) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == database.SCHEMA_VERSION


def test_future_version_fails_loudly(tmp_path):
    path = tmp_path / "petri.sqlite"
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA user_version = 999")
    with pytest.raises(database.SchemaVersionError, match="999.*supported version 1"):
        database.connect(path)
    with sqlite3.connect(path) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 999


def test_failing_migration_rolls_back_ddl_and_version(tmp_path):
    with sqlite3.connect(tmp_path / "test.sqlite", isolation_level=None) as conn:
        first = "CREATE TABLE first_table(id TEXT);"
        database.migrate(conn, [first])
        with pytest.raises(sqlite3.OperationalError, match="no such table"):
            database.migrate(
                conn, [first, "CREATE TABLE second_table(id TEXT); INSERT INTO absent VALUES (1);"]
            )
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 1
        assert (
            conn.execute("SELECT name FROM sqlite_master WHERE name='second_table'").fetchone()
            is None
        )
        assert conn.execute("SELECT name FROM sqlite_master WHERE name='first_table'").fetchone()
        assert not conn.in_transaction


def test_fixture_dag_round_trips(tmp_path):
    path = tmp_path / "petri.sqlite"
    first, second = cell(), cell("dish-colony-0-1", 1)
    database.put_cell(path, first)
    database.put_cell(path, second)
    assert database.add_edge(
        path,
        from_cell=first["id"],
        to_cell=second["id"],
        edge_type="intra_colony",
        reason="requires evidence",
        created_at=NOW,
    )
    assert not database.add_edge(
        path, from_cell=first["id"], to_cell=second["id"], edge_type="intra_colony", created_at=NOW
    )
    written, inserted = database.append_event(path, event())
    assert inserted
    assert database.get_cell(path, first["id"])["claim_text"] == first["claim_text"]
    assert [row["id"] for row in database.list_cells(path)] == [first["id"], second["id"]]
    assert database.list_cells(path, colony="missing") == []
    assert database.read_edges(path)[0]["reason"] == "requires evidence"
    assert database.read_dependents(path, second["id"])[0]["dependent_cell_id"] == first["id"]
    assert database.read_dependents(path, first["id"]) == []
    assert database.read_events(path) == [written.model_dump(mode="json")]
    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        database.add_edge(
            path, from_cell=first["id"], to_cell="missing", edge_type="intra_colony", created_at=NOW
        )


def test_duplicate_event_returns_original_and_append_only_guards(tmp_path):
    path = tmp_path / "petri.sqlite"
    original, inserted = database.append_event(path, event())
    assert inserted
    duplicate, inserted = database.append_event(
        path, event(timestamp="later", data={"query": "different", "sources_found": 2})
    )
    assert not inserted and duplicate == original
    assert len(database.read_events(path)) == 1
    with closing(database.connect(path)) as conn:
        for statement in [
            "UPDATE events SET agent='other'",
            "DELETE FROM events",
            "INSERT OR REPLACE INTO events SELECT * FROM events",
        ]:
            with pytest.raises(sqlite3.IntegrityError, match="append-only"):
                conn.execute(statement)
        assert conn.execute("SELECT count(*) FROM events").fetchone()[0] == 1


@pytest.mark.parametrize("created_at", [None, ""])
def test_cell_timestamp_constraint(tmp_path, created_at):
    with pytest.raises(sqlite3.IntegrityError):
        database.put_cell(tmp_path / "petri.sqlite", {**cell(), "created_at": created_at})


def test_stable_identity_across_processes():
    command = "from petri.storage.database import stable_event_id; print(stable_event_id('cell', 'type', 2))"
    values = [
        subprocess.check_output([sys.executable, "-c", command], text=True).strip()
        for _ in range(2)
    ]
    assert values == [database.stable_event_id("cell", "type", 2)] * 2
    assert database.stable_event_id("cell", "type", 2) != database.stable_event_id(
        "cell", "type", "2"
    )
    assert database.stable_event_id("cell", "a:b", "c") != database.stable_event_id(
        "cell", "a", "b:c"
    )


def test_sqlite_wrappers_scope_validation_filters_and_legacy(tmp_path):
    path = tmp_path / "petri.sqlite"
    target = database.SQLiteEventLog(path, "cell-a")
    written = event_log.append_event(
        target,
        "cell-a",
        "search_executed",
        "researcher",
        1,
        {"query": "sqlite", "sources_found": 1},
    )
    event_log.append_event(
        database.SQLiteEventLog(path),
        "cell-b",
        "search_executed",
        "other",
        2,
        {"query": "other", "sources_found": 2},
    )
    assert event_log.load_events(target) == [written.model_dump(mode="json")]
    assert len(event_log.load_events(database.SQLiteEventLog(path))) == 2
    assert event_log.query_events(target, cell_id="cell-b") == []
    assert (
        len(
            event_log.query_events(
                target, event_type="search_executed", agent="researcher", iteration=1, since=NOW
            )
        )
        == 1
    )
    assert event_log.query_events(target, agent="other") == []
    with pytest.raises(ValueError, match="scope"):
        event_log.append_event(
            target, "cell-b", "search_executed", "a", 1, {"query": "x", "sources_found": 0}
        )
    legacy = tmp_path / "events.jsonl"
    event_log.append_event(
        legacy, "cell-a", "search_executed", "a", 1, {"query": "x", "sources_found": 0}
    )
    assert json.loads(legacy.read_text())["cell_id"] == "cell-a"


def test_invalid_event_data_is_rejected_before_writing(tmp_path):
    with pytest.raises(ValueError):
        database.append_event(tmp_path / "petri.sqlite", event(data={"sources_found": -1}))
    assert not (tmp_path / "petri.sqlite").exists()
