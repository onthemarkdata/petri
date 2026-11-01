"""Read-only research-health views over committed domain events."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, TypedDict, cast

from petri.query.connection import has_table, read_connection


class BlockingPattern(TypedDict):
    colony: str
    level: int
    agent: str
    blocked_verdicts: int
    blocked_cells: int
    cell_ids: list[str]
    last_blocked_at: str


class CellVelocity(TypedDict):
    cell_id: str
    colony: str
    level: int
    first_event_at: str
    last_event_at: str
    converged_at: str | None
    seconds_to_convergence: float | None


class StalledCell(TypedDict):
    cell_id: str
    colony: str
    level: int
    first_event_at: str
    last_event_at: str


class ConvergenceRate(TypedDict):
    colony: str
    total_cells: int
    converged_cells: int
    convergence_rate: float


def _view(db_path: Path, view: str, where: str = '', values: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    with read_connection(db_path) as connection:
        if connection is None or not has_table(connection, view):
            return []
        return [dict(row) for row in connection.execute('SELECT * FROM '+view+where,values)]


def blocking_verdict_patterns(db_path: Path) -> list[BlockingPattern]:
    rows = _view(db_path,'blocking_verdict_patterns',' ORDER BY blocked_verdicts DESC,colony,level,agent')
    for row in rows:
        row['cell_ids'] = sorted(json.loads(row['cell_ids']))
    return cast(list[BlockingPattern],rows)


def cell_velocity(db_path: Path) -> list[CellVelocity]:
    return cast(list[CellVelocity],_view(db_path,'cell_velocity',' ORDER BY colony,cell_id'))


def stalled_cells(db_path: Path, *, window_hours: float = 24, now: datetime | None = None) -> list[StalledCell]:
    if window_hours <= 0:
        raise ValueError('window_hours must be positive.')
    cutoff = ((now or datetime.now(timezone.utc))-timedelta(hours=window_hours)).isoformat()
    return cast(list[StalledCell],_view(db_path,'stalled_cells',
        ' WHERE julianday(last_event_at)<=julianday(?) ORDER BY last_event_at,cell_id',(cutoff,)))


def convergence_rates(db_path: Path) -> list[ConvergenceRate]:
    return cast(list[ConvergenceRate],_view(db_path,'colony_convergence_rates',' ORDER BY colony'))
