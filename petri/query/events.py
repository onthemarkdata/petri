"""Event filters and append-order cursors over the immutable domain event table."""

import json
import sqlite3
from pathlib import Path
from typing import Any, TypedDict

from petri.query.connection import has_table, read_connection


class EventRow(TypedDict):
    id: str
    cell_id: str
    timestamp: str
    type: str
    agent: str
    iteration: int
    data: dict[str, Any]


class CursorEvent(EventRow):
    rowid: int


def _event(row: sqlite3.Row) -> EventRow:
    return EventRow(
        id=row["id"],
        cell_id=row["cell_id"],
        timestamp=row["created_at"],
        type=row["event_type"],
        agent=row["agent"],
        iteration=row["iteration"],
        data=json.loads(row["payload"]),
    )


def _limit(limit: int | None) -> None:
    if limit is not None and (type(limit) is not int or not 1 <= limit <= 5000):
        raise ValueError("limit must be between 1 and 5000, or None.")


def query_events(
    db_path: Path,
    *,
    cell_id: str | None = None,
    iteration: int | None = None,
    event_type: str | None = None,
    agent: str | None = None,
    since: str | None = None,
    limit: int | None = 500,
) -> list[EventRow]:
    """Match dashboard filters, ordered by timestamp with append order breaking ties."""
    _limit(limit)
    predicates, values = [], []
    for column, value in (
        ("cell_id", cell_id),
        ("iteration", iteration),
        ("event_type", event_type),
        ("agent", agent),
    ):
        if value is not None:
            predicates.append(f"{column}=?")
            values.append(value)
    if since is not None:
        predicates.append("created_at>=?")
        values.append(since)
    sql = "SELECT * FROM events"
    if predicates:
        sql += " WHERE " + " AND ".join(predicates)
    sql += " ORDER BY created_at,rowid"
    if limit is not None:
        sql += " LIMIT ?"
        values.append(limit)
    with read_connection(db_path) as connection:
        if connection is None or not has_table(connection, "events"):
            return []
        return [_event(row) for row in connection.execute(sql, values)]


def events_since(db_path: Path, watermark: int, *, limit: int = 500) -> list[CursorEvent]:
    """Return rows strictly after a per-database rowid watermark, in append order."""
    if type(watermark) is not int or watermark < 0:
        raise ValueError("watermark must be a nonnegative rowid.")
    _limit(limit)
    with read_connection(db_path) as connection:
        if connection is None or not has_table(connection, "events"):
            return []
        return [
            CursorEvent(**_event(row), rowid=row["append_order"])
            for row in connection.execute(
                "SELECT rowid AS append_order,* FROM events WHERE rowid>? ORDER BY rowid LIMIT ?",
                (watermark, limit),
            )
        ]


def event_stats(db_path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "total_events": 0,
        "cells_with_events": 0,
        "events_by_type": [],
        "top_cells": [],
    }
    with read_connection(db_path) as connection:
        if connection is None or not has_table(connection, "events"):
            return result
        row = connection.execute("SELECT COUNT(*),COUNT(DISTINCT cell_id) FROM events").fetchone()
        result["total_events"], result["cells_with_events"] = row
        result["events_by_type"] = [
            {"type": row["event_type"], "count": row["count"]}
            for row in connection.execute(
                "SELECT event_type,COUNT(*) AS count FROM events GROUP BY event_type ORDER BY count DESC,event_type"
            )
        ]
        result["top_cells"] = [
            {"cell_id": row["cell_id"], "event_count": row["count"]}
            for row in connection.execute(
                "SELECT cell_id,COUNT(*) AS count FROM events GROUP BY cell_id ORDER BY count DESC,cell_id LIMIT 10"
            )
        ]
    return result


def get_verdicts(
    db_path: Path,
    *,
    cell_id: str | None = None,
    iteration: int | None = None,
    agent: str | None = None,
):
    """Return the existing typed verdict contract without scanning event files."""
    from petri.models import Verdict

    return [
        Verdict(
            cell_id=row["cell_id"],
            agent=row["agent"],
            iteration=row["iteration"],
            verdict=row["data"].get("verdict", ""),
            summary=row["data"].get("summary", ""),
        )
        for row in query_events(
            db_path,
            cell_id=cell_id,
            iteration=iteration,
            agent=agent,
            event_type="verdict_issued",
            limit=None,
        )
    ]


def get_sources(db_path: Path, *, cell_id: str | None = None) -> list[EventRow]:
    """Return the first source-review event per nonempty URL in append order."""
    with read_connection(db_path) as connection:
        if connection is None or not has_table(connection, "events"):
            return []
        return [
            _event(row)
            for row in connection.execute(
                "SELECT * FROM events WHERE rowid IN ("
                "SELECT MIN(rowid) FROM events WHERE event_type='source_reviewed' "
                "AND (? IS NULL OR cell_id=?) AND COALESCE(json_extract(payload,'$.url'),'')<>'' "
                "GROUP BY json_extract(payload,'$.url')) ORDER BY rowid",
                (cell_id, cell_id),
            )
        ]


read_events = query_events
