"""Read one consistent domain snapshot for deterministic text export."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from petri.query.connection import has_table, read_connection
from petri.query.events import CursorEvent, _event


@dataclass(frozen=True)
class ExportSnapshot:
    cells: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    events: list[CursorEvent]
    watermark: int


def export_snapshot(db_path: Path) -> ExportSnapshot:
    with read_connection(db_path) as connection:
        if connection is None:
            raise FileNotFoundError(f"Dish database does not exist: {db_path}")
        cell_table = "active_cells" if has_table(connection,"active_cells") else "cells"
        cells = [dict(row) for row in connection.execute(f"SELECT * FROM {cell_table} ORDER BY colony,level,seq,id")]
        edge_table = "active_edges" if has_table(connection, "active_edges") else "edges"
        edges = [dict(row) for row in connection.execute(f"SELECT * FROM {edge_table} ORDER BY from_cell,to_cell,edge_type")]
        events = [CursorEvent(**_event(row), rowid=row["append_order"]) for row in connection.execute(
            "SELECT rowid AS append_order,* FROM events ORDER BY rowid")]
        return ExportSnapshot(cells, edges, events, events[-1]["rowid"] if events else 0)
