"""SQLite domain storage, separate from durable-execution checkpoints.

Callers opt into v2 event storage with ``SQLiteEventLog``. Existing JSONL callers
remain supported until the dish and engine paths migrate.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import closing, contextmanager
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

from petri.models import Event, validate_event_data

MIGRATIONS = (
    "001_domain.sql", "002_colonies.sql", "003_telemetry.sql", "004_seed_checkpoints.sql",
    "005_cell_content.sql", "006_analytics.sql", "007_edge_tombstones.sql",
    "008_lifecycle.sql", "009_retire_seed_state.sql",
)

SCHEMA_VERSION = len(MIGRATIONS)
BUSY_TIMEOUT_MS = 30_000


class SchemaVersionError(RuntimeError):
    """The database requires a newer Petri schema than this build supports."""


@dataclass(frozen=True)
class SQLiteEventLog:
    """Select a dish database, optionally scoped to one cell for reads."""

    db_path: Path
    cell_id: str | None = None


def _statements(script: str) -> Iterator[str]:
    """Split complete SQLite statements, preserving trigger bodies."""
    pending = ""
    for character in script:
        pending += character
        if character == ";" and sqlite3.complete_statement(pending):
            yield pending
            pending = ""
    if pending.strip():
        raise ValueError("Migration ends with an incomplete SQL statement")


def migrate(conn: sqlite3.Connection, scripts: Sequence[str] | None = None) -> None:
    """Apply each new script and its version marker in one transaction.

    ``scripts`` is an ordered sequence of SQL source strings. The default loads
    the packaged migration files. A future version fails without downgrading.
    """
    if conn.in_transaction:
        raise ValueError("Migrations require a connection without an active transaction")
    if scripts is None:
        source = files("petri.storage.migrations")
        scripts = tuple(source.joinpath(name).read_text(encoding="utf-8") for name in MIGRATIONS)
    current = conn.execute("PRAGMA user_version").fetchone()[0]
    if current > len(scripts):
        raise SchemaVersionError(
            f"Database schema version {current} is newer than supported version {len(scripts)}"
        )
    for version, script in enumerate(scripts, 1):
        if version <= current:
            continue
        conn.execute("BEGIN IMMEDIATE")
        try:
            # Another opener may have migrated while this connection waited for the lock.
            current = conn.execute("PRAGMA user_version").fetchone()[0]
            if current > len(scripts):
                raise SchemaVersionError(
                    f"Database schema version {current} is newer than supported version {len(scripts)}"
                )
            if current < version:
                for statement in _statements(script):
                    conn.execute(statement)
                conn.execute(f"PRAGMA user_version = {version}")
            conn.commit()
        except BaseException:
            conn.rollback()
            raise


def _enable_wal(conn: sqlite3.Connection) -> None:
    """Retry startup lock upgrades within one busy-timeout budget.

    SQLite can return BUSY immediately when journal-mode setup races another
    opener; that lock upgrade does not always invoke its busy handler. Bound
    both SQLite's wait and our retry delay by the same monotonic deadline.
    """
    deadline = time.monotonic() + BUSY_TIMEOUT_MS / 1000
    delay = 0.005
    try:
        while True:
            remaining = max(0.0, deadline - time.monotonic())
            conn.execute(f"PRAGMA busy_timeout = {int(remaining * 1000)}")
            try:
                mode = conn.execute("PRAGMA journal_mode = WAL").fetchone()[0]
                if mode.lower() != "wal":
                    raise sqlite3.OperationalError(f"SQLite did not enable WAL mode: {mode}")
                return
            except sqlite3.OperationalError as error:
                code = getattr(error, "sqlite_errorcode", 0) & 0xFF
                remaining = deadline - time.monotonic()
                if code not in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED) or remaining <= 0:
                    raise
                time.sleep(min(delay, remaining))
                delay = min(delay * 2, 0.05)
    finally:
        conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")


def connect(db_path: Path) -> sqlite3.Connection:
    """Open and migrate a domain database; the caller must close the connection.

    Connections use explicit transactions. WAL permits readers during writes;
    the busy timeout permits writers to wait for the existing writer.
    """
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=BUSY_TIMEOUT_MS / 1000, isolation_level=None)
    conn.row_factory = sqlite3.Row
    try:
        _enable_wal(conn)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA recursive_triggers = ON")
        migrate(conn)
        return conn
    except BaseException:
        conn.close()
        raise


@contextmanager
def _write(db_path: Path) -> Iterator[sqlite3.Connection]:
    with closing(connect(db_path)) as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise


@contextmanager
def write_connection(db_path: Path, connection: sqlite3.Connection | None = None) -> Iterator[sqlite3.Connection]:
    """Own a writer transaction, or join an active caller-owned transaction.

    A supplied connection is never committed, rolled back, migrated, or closed.
    The caller must use a writer transaction and roll it back if an operation fails.
    """
    if connection is not None:
        if not connection.in_transaction:
            raise ValueError('A supplied connection requires an active writer transaction')
        yield connection
        return
    with _write(db_path) as owned:
        yield owned


def stable_event_id(cell_id: str, *identity: str | int) -> str:
    """Hash a versioned JSON array; field boundaries and types remain distinct."""
    if (
        not isinstance(cell_id, str)
        or not cell_id
        or any(type(value) not in (str, int) for value in identity)
    ):
        raise ValueError("Event identity requires a cell ID and string/integer components")
    encoded = json.dumps(
        ["petri-event-v1", cell_id, *identity],
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return f"{cell_id}-{hashlib.sha256(encoded).hexdigest()}"


def _event(row: sqlite3.Row) -> Event:
    return Event(
        id=row["id"],
        cell_id=row["cell_id"],
        timestamp=row["created_at"],
        type=row["event_type"],
        agent=row["agent"],
        iteration=row["iteration"],
        data=json.loads(row["payload"]),
    )


def append_event(db_path: Path, event: Event, *, connection: sqlite3.Connection | None = None) -> tuple[Event, bool]:
    """Insert an immutable event; a duplicate returns the stored event and False."""
    data = validate_event_data(event.type.value, event.data)
    if not event.id or not event.cell_id or not event.timestamp:
        raise ValueError("Event ID, cell ID, and timestamp must be nonempty")
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    with write_connection(db_path,connection) as conn:
        cursor = conn.execute(
            "INSERT OR IGNORE INTO events "
            "(id,cell_id,event_type,agent,iteration,payload,created_at) VALUES (?,?,?,?,?,?,?)",
            (
                event.id,
                event.cell_id,
                event.type.value,
                event.agent,
                event.iteration,
                payload,
                event.timestamp,
            ),
        )
        inserted = cursor.rowcount == 1
        row = conn.execute("SELECT * FROM events WHERE id = ?", (event.id,)).fetchone()
        if row is None:
            raise ValueError("Event was rejected by a database constraint")
        return _event(row), inserted


def read_events(
    db_path: Path,
    *,
    cell_id: str | None = None,
    iteration: int | None = None,
    event_type: str | None = None,
    agent: str | None = None,
    since: str | None = None,
) -> list[dict]:
    """Read event envelopes in append order, with SQL filters combined by AND."""
    predicates = []
    values = []
    for column, value in (
        ("cell_id", cell_id),
        ("iteration", iteration),
        ("event_type", event_type),
        ("agent", agent),
    ):
        if value is not None:
            predicates.append(f"{column} = ?")
            values.append(value)
    if since is not None:
        predicates.append("created_at >= ?")
        values.append(since)
    query = "SELECT * FROM events"
    if predicates:
        query += " WHERE " + " AND ".join(predicates)
    query += " ORDER BY rowid"
    with closing(connect(db_path)) as conn:
        return [_event(row).model_dump(mode="json") for row in conn.execute(query, values)]


def put_cell(db_path: Path, cell: Mapping[str, Any], *, connection: sqlite3.Connection | None = None) -> None:
    """Insert or update a cell record; dependencies live in the edges table."""
    record = {"status": "NEW", "relation": None, "is_atomic": False, "bedrock_reason": None, **cell}
    columns = (
        "id",
        "colony",
        "level",
        "seq",
        "claim_text",
        "status",
        "relation",
        "is_atomic",
        "bedrock_reason",
        "created_at",
    )
    with write_connection(db_path,connection) as conn:
        conn.execute(
            f"INSERT INTO cells ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)}) "
            "ON CONFLICT(id) DO UPDATE SET " + ",".join(f"{c}=excluded.{c}" for c in columns[1:]),
            tuple(record[c] for c in columns),
        )


def get_cell(db_path: Path, cell_id: str) -> dict | None:
    with closing(connect(db_path)) as conn:
        row = conn.execute("SELECT * FROM active_cells WHERE id = ?", (cell_id,)).fetchone()
        return dict(row) if row is not None else None


def list_cells(db_path: Path, *, colony: str | None = None) -> list[dict]:
    with closing(connect(db_path)) as conn:
        rows = conn.execute(
            "SELECT * FROM active_cells WHERE (? IS NULL OR colony = ?) ORDER BY colony,level,seq,id",
            (colony, colony),
        )
        return [dict(row) for row in rows]


def add_edge(
    db_path: Path,
    *,
    from_cell: str,
    to_cell: str,
    edge_type: str,
    created_at: str,
    reason: str = "",
) -> bool:
    """Compatibility entry point; the edge registry owns all edge inserts."""
    from petri.storage.edge_registry import EdgeRecord, append

    return append(db_path, EdgeRecord(from_cell=from_cell, to_cell=to_cell,
                  edge_type=edge_type, reason=reason, created_at=created_at))


def read_edges(db_path: Path) -> list[dict]:
    with closing(connect(db_path)) as conn:
        return [
            dict(row)
            for row in conn.execute("SELECT * FROM active_edges ORDER BY from_cell,to_cell,edge_type")
        ]


def read_dependents(db_path: Path, cell_id: str) -> list[dict]:
    with closing(connect(db_path)) as conn:
        return [
            dict(row)
            for row in conn.execute(
                "SELECT * FROM dependents WHERE cell_id = ? ORDER BY dependent_cell_id,edge_type",
                (cell_id,),
            )
        ]
