"""Transitional legacy event-file index; v2 reads use petri.query directly."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

# ── Schema ────────────────────────────────────────────────────────────────

_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS events (
    id          TEXT PRIMARY KEY,
    cell_id     TEXT NOT NULL,
    timestamp   TEXT NOT NULL,
    type        TEXT NOT NULL,
    agent       TEXT NOT NULL,
    iteration   INTEGER NOT NULL DEFAULT 0,
    data        TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_events_cell_id ON events(cell_id);
CREATE INDEX IF NOT EXISTS idx_events_cell_iteration ON events(cell_id, iteration);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(type);
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
"""

_INSERT_SQL = (
    "INSERT OR IGNORE INTO events "
    "(id, cell_id, timestamp, type, agent, iteration, data) "
    "VALUES (?, ?, ?, ?, ?, ?, ?)"
)


# ── Public API ────────────────────────────────────────────────────────────


def init_db(db_path: Path) -> None:
    """Create the SQLite database and schema (idempotent)."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.executescript(_SCHEMA_SQL)
    conn.close()


def rebuild_sqlite(petri_dir: Path, db_path: Path) -> int:
    """Populate the legacy index directly from per-cell files without a rollup."""
    init_db(db_path)
    return incremental_sync(petri_dir, db_path, {})


def incremental_sync(petri_dir: Path, db_path: Path, file_offsets: dict[str, int]) -> int:
    """Tail legacy per-cell files while the remaining dashboard readers migrate."""
    inserted = 0
    with sqlite3.connect(str(db_path), timeout=10) as connection:
        for path in sorted((petri_dir / "petri-dishes").rglob("events.jsonl")):
            offset = file_offsets.get(str(path), 0)
            if path.stat().st_size <= offset:
                continue
            with path.open() as source:
                source.seek(offset)
                lines = source.readlines()
                file_offsets[str(path)] = source.tell()
            for line in lines:
                try:
                    event = json.loads(line)
                    cursor = connection.execute(
                        _INSERT_SQL,
                        (
                            event["id"],
                            event["cell_id"],
                            event["timestamp"],
                            event["type"],
                            event["agent"],
                            event.get("iteration", 0),
                            json.dumps(event.get("data", {})),
                        ),
                    )
                    inserted += cursor.rowcount
                except (json.JSONDecodeError, KeyError):
                    continue
    return inserted
