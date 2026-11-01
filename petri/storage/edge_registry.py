"""Dish-wide dependency queries and append-only edge additions and tombstones."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from typing import Any
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from petri.storage import database


class EdgeRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    from_cell: str = Field(min_length=1)
    to_cell: str = Field(min_length=1)
    edge_type: Literal["intra_colony", "cross_colony"] = "intra_colony"
    reason: str = ""
    created_at: str = Field(min_length=1)

    @model_validator(mode="after")
    def reject_self_reference(self):
        if self.from_cell == self.to_cell:
            raise ValueError("A cell cannot reference itself")
        return self


class EdgeCycleError(ValueError):
    """An edge would close a cycle in the dish-wide dependency graph."""


class EdgeTombstone(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    from_cell: str
    to_cell: str
    edge_type: str
    edge_created_at: str
    reason: str
    created_at: str


class DishGraph:
    """Compose all colonies; duplicate records represent one logical dependency."""

    def __init__(self, cells: Iterable[Mapping[str, Any]], edges: Iterable[EdgeRecord]):
        self.cells = {str(cell['id']): dict(cell) for cell in cells}
        self.edges = tuple(sorted({(edge.from_cell,edge.to_cell,edge.edge_type):edge for edge in edges}.values(),
                                  key=lambda edge:(edge.from_cell,edge.to_cell,edge.edge_type)))
        self.adjacency: dict[str,set[str]] = {cell:set() for cell in self.cells}
        self.reverse_adjacency: dict[str,set[str]] = {cell:set() for cell in self.cells}
        for edge in self.edges:
            if edge.from_cell not in self.cells or edge.to_cell not in self.cells:
                raise ValueError(f'Dish edge references a missing cell: {edge.from_cell} -> {edge.to_cell}')
            self.adjacency[edge.from_cell].add(edge.to_cell)
            self.reverse_adjacency[edge.to_cell].add(edge.from_cell)

    def dependencies(self, cell_id: str) -> list[str]:
        return sorted(self.adjacency.get(cell_id,set()))

    def dependents(self, cell_id: str) -> list[str]:
        return sorted(self.reverse_adjacency.get(cell_id,set()))

    def reachable(self, cell_id: str, *, reverse: bool = False) -> list[str]:
        adjacency = self.reverse_adjacency if reverse else self.adjacency
        seen: set[str] = set()
        pending = list(adjacency.get(cell_id,set()))
        while pending:
            current = pending.pop()
            if current in seen or current == cell_id:
                continue
            seen.add(current)
            pending.extend(adjacency.get(current,set())-seen)
        return sorted(seen)

    def has_cycle_with_edge(self, from_cell: str, to_cell: str) -> bool:
        return from_cell == to_cell or from_cell in self.reachable(to_cell)

    @classmethod
    def from_connection(cls, connection: sqlite3.Connection) -> DishGraph:
        from petri.query.connection import has_table
        cell_table = 'active_cells' if has_table(connection,'active_cells') else 'cells'
        cells = [dict(row) for row in connection.execute(f'SELECT * FROM {cell_table}')]
        table = 'active_edges' if has_table(connection,'active_edges') else 'edges'
        edges = [EdgeRecord.model_validate(dict(row)) for row in connection.execute(
            f'SELECT * FROM {table} ORDER BY from_cell,to_cell,edge_type,created_at')]
        return cls(cells,edges)

    @classmethod
    def read(cls, db_path: Path) -> DishGraph:
        if not db_path.is_file():
            return cls([],[])
        from petri.query.connection import read_connection
        with read_connection(db_path) as connection:
            assert connection is not None
            return cls.from_connection(connection)


def append(db_path: Path, edge: EdgeRecord, *, connection: sqlite3.Connection | None = None) -> bool:
    """Append an active version once; reject cycles under the same writer lock."""
    with database.write_connection(db_path,connection) as conn:
        graph = DishGraph.from_connection(conn)
        if edge.from_cell not in graph.cells or edge.to_cell not in graph.cells:
            raise sqlite3.IntegrityError('FOREIGN KEY constraint failed: edge endpoints must exist in the dish')
        if any(item.from_cell == edge.from_cell and item.to_cell == edge.to_cell and item.edge_type == edge.edge_type for item in graph.edges):
            return False
        if graph.has_cycle_with_edge(edge.from_cell,edge.to_cell):
            raise EdgeCycleError(f'Adding edge {edge.from_cell} -> {edge.to_cell} would create a dish-level cycle')
        result = conn.execute(
            'INSERT INTO edges VALUES (?,?,?,?,?) ON CONFLICT(from_cell,to_cell,edge_type,created_at) DO NOTHING',
            (edge.from_cell,edge.to_cell,edge.edge_type,edge.reason,edge.created_at),
        )
        return result.rowcount == 1


def read(db_path: Path) -> list[EdgeRecord]:
    """Read active dependency records in a stable order."""
    if not db_path.is_file():
        return []
    from petri.query.cells import get_edges
    return [EdgeRecord.model_validate(row) for row in get_edges(db_path)]


def remove(db_path: Path, *, from_cell: str, to_cell: str, edge_type: str | None = None,
           reason: str = '', created_at: str | None = None) -> list[EdgeTombstone]:
    """Tombstone every active version; earlier edge rows remain unchanged."""
    now = created_at or datetime.now(timezone.utc).isoformat()
    result = []
    with database._write(db_path) as conn:
        rows = conn.execute(
            'SELECT * FROM active_edges WHERE from_cell=? AND to_cell=? '
            'AND (? IS NULL OR edge_type=?)',
            (from_cell,to_cell,edge_type,edge_type),
        ).fetchall()
        for row in rows:
            tombstone = EdgeTombstone(from_cell=row['from_cell'],
                to_cell=row['to_cell'],edge_type=row['edge_type'],edge_created_at=row['created_at'],reason=reason,created_at=now)
            conn.execute('INSERT INTO edge_tombstones VALUES (?,?,?,?,?,?)',tuple(tombstone.model_dump().values()))
            result.append(tombstone)
    return result


def read_tombstones(db_path: Path) -> list[EdgeTombstone]:
    from petri.query.connection import has_table, read_connection
    with read_connection(db_path) as connection:
        if connection is None or not has_table(connection,'edge_tombstones'):
            return []
        return [EdgeTombstone.model_validate(dict(row)) for row in connection.execute('SELECT * FROM edge_tombstones ORDER BY created_at,from_cell,to_cell,edge_type,edge_created_at')]


def retire_cell(db_path: Path, cell_id: str, *, reason: str = 'Removed from colony topology',
                connection: sqlite3.Connection | None = None) -> bool:
    """Retire a cell and incident edges atomically, preserving historical references."""
    now = datetime.now(timezone.utc).isoformat()
    with database.write_connection(db_path,connection) as writer:
        if writer.execute('SELECT 1 FROM active_cells WHERE id=?',(cell_id,)).fetchone() is None:
            return False
        for row in writer.execute('SELECT * FROM active_edges WHERE from_cell=? OR to_cell=?',(cell_id,cell_id)).fetchall():
            writer.execute('INSERT OR IGNORE INTO edge_tombstones VALUES (?,?,?,?,?,?)',
                (row['from_cell'],row['to_cell'],row['edge_type'],row['created_at'],reason,now))
        writer.execute('INSERT INTO cell_tombstones VALUES (?,?,?)',(cell_id,reason,now))
    return True
