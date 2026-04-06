"""JSONL rollup + SQLite rebuild for the Petri dashboard.

The SQLite database is a disposable read index rebuilt from the
append-only JSONL event files.  ``rebuild_sqlite`` performs a full
cold-start rebuild; ``incremental_sync`` tails combined.jsonl for
new events without re-reading the entire file.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from petri.event_log import rollup_to_combined

# ── Schema ────────────────────────────────────────────────────────────────

_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS events (
    id          TEXT PRIMARY KEY,
    node_id     TEXT NOT NULL,
    timestamp   TEXT NOT NULL,
    type        TEXT NOT NULL,
    agent       TEXT NOT NULL,
    iteration   INTEGER NOT NULL DEFAULT 0,
    data        TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_events_node_id ON events(node_id);
CREATE INDEX IF NOT EXISTS idx_events_node_iteration ON events(node_id, iteration);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(type);
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
"""

_INSERT_SQL = (
    "INSERT OR IGNORE INTO events "
    "(id, node_id, timestamp, type, agent, iteration, data) "
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
    """Full cold-start rebuild of the SQLite read index.

    1. Rolls up per-node JSONL files into ``combined.jsonl``.
    2. Creates the database (if needed).
    3. Reads every line of ``combined.jsonl`` and inserts into SQLite.
    4. Returns the number of events inserted.
    """
    # Step 1: rollup
    combined_path = rollup_to_combined(petri_dir)

    # Step 2: init
    init_db(db_path)

    # Step 3: ingest
    conn = sqlite3.connect(str(db_path), timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")

    inserted = 0
    if combined_path.exists():
        for line in combined_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                evt = json.loads(line)
                conn.execute(
                    _INSERT_SQL,
                    (
                        evt["id"],
                        evt["node_id"],
                        evt["timestamp"],
                        evt["type"],
                        evt["agent"],
                        evt.get("iteration", 0),
                        json.dumps(evt.get("data", {})),
                    ),
                )
                inserted += 1
            except (json.JSONDecodeError, KeyError):
                pass  # skip malformed lines

    conn.commit()
    conn.close()
    return inserted


def incremental_sync(
    petri_dir: Path,
    db_path: Path,
    file_offsets: dict[str, int],
) -> int:
    """Tail ``combined.jsonl`` for new events since the last sync.

    Reads only the bytes appended since the stored offset, parses them,
    and inserts into SQLite.  Updates *file_offsets* in place.

    Returns the number of new events inserted.
    """
    combined_path = petri_dir / "combined.jsonl"
    path_key = str(combined_path)

    if not combined_path.exists():
        return 0

    try:
        current_size = combined_path.stat().st_size
    except OSError:
        return 0

    last_offset = file_offsets.get(path_key, 0)
    if current_size <= last_offset:
        return 0

    # Read only the new bytes
    with open(combined_path) as f:
        f.seek(last_offset)
        new_content = f.read()
        file_offsets[path_key] = f.tell()

    conn = sqlite3.connect(str(db_path), timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")

    inserted = 0
    for line in new_content.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            evt = json.loads(line)
            conn.execute(
                _INSERT_SQL,
                (
                    evt["id"],
                    evt["node_id"],
                    evt["timestamp"],
                    evt["type"],
                    evt["agent"],
                    evt.get("iteration", 0),
                    json.dumps(evt.get("data", {})),
                ),
            )
            inserted += 1
        except (json.JSONDecodeError, KeyError):
            pass

    conn.commit()
    conn.close()
    return inserted
