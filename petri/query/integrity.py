"""Read raw domain rows so malformed payloads remain visible to the scanner."""

from pathlib import Path
from typing import Any

from petri.query.connection import has_table, read_connection


def integrity_rows(db_path: Path) -> tuple[list[dict[str, Any]], ...]:
    with read_connection(db_path) as connection:
        if connection is None:
            return [], [], []
        cells = "active_cells" if has_table(connection, "active_cells") else "cells"
        edges = "active_edges" if has_table(connection, "active_edges") else "edges"
        return tuple(
            [dict(row) for row in connection.execute(f"SELECT * FROM {table} ORDER BY {order}")]
            for table, order in ((cells, "id"), (edges, "from_cell,to_cell,edge_type"), ("events", "rowid"))
        )
