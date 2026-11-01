"""Deterministic event identities survive retries and independent processes."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing

import pytest

from petri.models import validate_event_data
from petri.storage import database, event_log

COORDINATES = {
    "round_number": 2,
    "node": "research",
    "agent": "researcher",
    "iteration": 3,
    "event_type": "source_reviewed",
    "sequence": 4,
}


def test_identity_is_stable_across_independent_processes():
    script = (
        "import json; from petri.storage.event_log import derive_event_id; "
        f"print(derive_event_id('dish-colony-1-1', **json.loads({json.dumps(COORDINATES)!r})))"
    )
    values = [
        subprocess.check_output(
            [sys.executable, "-c", script],
            text=True,
            env={**os.environ, "PYTHONHASHSEED": str(seed)},
        ).strip()
        for seed in (1, 827)
    ]
    assert values == [event_log.derive_event_id("dish-colony-1-1", **COORDINATES)] * 2


@pytest.mark.parametrize(
    "field,value",
    [
        ("round_number", 3),
        ("node", "validate"),
        ("agent", "reviewer"),
        ("iteration", 4),
        ("event_type", "search_executed"),
        ("sequence", 5),
    ],
)
def test_every_coordinate_distinguishes_events(field, value):
    original = event_log.derive_event_id("cell", **COORDINATES)
    assert event_log.derive_event_id("cell", **{**COORDINATES, field: value}) != original
    assert event_log.derive_event_id("other-cell", **COORDINATES) != original


@pytest.mark.parametrize(
    "field,value",
    [
        ("round_number", -1),
        ("round_number", True),
        ("iteration", 1.5),
        ("sequence", -1),
        ("node", ""),
        ("agent", None),
    ],
)
def test_invalid_coordinates_are_rejected(field, value):
    with pytest.raises(ValueError, match=field):
        event_log.derive_event_id("cell", **{**COORDINATES, field: value})


def test_retry_preserves_original_envelope_and_payload(tmp_path, monkeypatch):
    target = database.SQLiteEventLog(tmp_path / "petri.sqlite", "cell")

    def append(data, **identity):
        return event_log.append_event(
            target,
            "cell",
            "search_executed",
            "researcher",
            1,
            data,
            round_number=2,
            node="research",
            **identity,
        )

    # A SQLite append must not call the log reader or maintain an application presence index.
    monkeypatch.setattr(event_log, "load_events", lambda *_: pytest.fail("Unexpected log scan"))
    first = append({"query": "original", "sources_found": 1})
    again = append({"query": "changed on retry", "sources_found": 2})
    assert again == first
    assert set(first.model_dump()) == {
        "id",
        "cell_id",
        "timestamp",
        "type",
        "agent",
        "iteration",
        "data",
    }
    assert validate_event_data(first.type.value, first.data) == first.data
    distinct = append({"query": "original", "sources_found": 1}, sequence=1)
    assert distinct.id != first.id
    assert len(database.read_events(target.db_path)) == 2


def test_fifty_concurrent_retries_write_one_row(tmp_path):
    path = tmp_path / "petri.sqlite"
    target = database.SQLiteEventLog(path, "cell")

    def append(_):
        return event_log.append_event(
            target,
            "cell",
            "source_reviewed",
            "researcher",
            1,
            {"url": "https://example.org/source"},
            round_number=3,
            node="research",
            sequence=9,
        )

    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(append, range(50)))
    assert len({result.id for result in results}) == 1
    assert len({result.timestamp for result in results}) == 1
    events = database.read_events(path)
    assert len(events) == 1
    assert validate_event_data(events[0]["type"], events[0]["data"]) == events[0]["data"]
    with closing(database.connect(path)) as conn:
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def test_independent_processes_replay_one_event(tmp_path):
    path = tmp_path / "petri.sqlite"
    with closing(database.connect(path)):
        pass
    script = (
        "from pathlib import Path; from petri.storage.database import SQLiteEventLog; "
        "from petri.storage.event_log import append_event; "
        f"target=SQLiteEventLog(Path({str(path)!r}), 'cell'); "
        "[append_event(target,'cell','search_executed','researcher',1,"
        "{'query':'shared','sources_found':1},round_number=1,node='research',sequence=2) "
        "for _ in range(5)]"
    )
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", script], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        for _ in range(10)
    ]
    for process in processes:
        _, stderr = process.communicate(timeout=30)
        assert process.returncode == 0, stderr.decode()
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT count(*) FROM events").fetchone()[0] == 1
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def test_legacy_path_does_not_claim_execution_idempotency(tmp_path):
    with pytest.raises(ValueError, match="SQLiteEventLog"):
        event_log.append_event(
            tmp_path / "events.jsonl",
            "cell",
            "search_executed",
            "agent",
            1,
            {"query": "x", "sources_found": 0},
            node="research",
        )
