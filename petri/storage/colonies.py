"""Persist and assemble colonies through the dish database."""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from petri.graph.colony import ColonyGraph
from petri.models import Cell, Colony, parse_key
from petri.storage.edge_registry import EdgeRecord, append, retire_cell
from petri.storage import database

logger = logging.getLogger(__name__)


def colony_exists(db_path: Path, colony_id: str) -> bool:
    if not db_path.is_file():
        return False
    with closing(database.connect(db_path)) as conn:
        return conn.execute(
            "SELECT 1 FROM colonies WHERE id=? UNION ALL SELECT 1 FROM cells WHERE colony=? LIMIT 1",
            (colony_id, colony_id),
        ).fetchone() is not None


def reserve_colony(db_path: Path, colony: Colony) -> None:
    """Claim a seed name atomically so concurrent runs cannot overwrite it."""
    from petri.storage.seed_guard import ColonyExistsError

    with database._write(db_path) as conn:
        if conn.execute("SELECT 1 FROM cells WHERE colony=? LIMIT 1", (colony.id,)).fetchone():
            raise ColonyExistsError(f"Colony '{colony.id}' already exists")
        try:
            conn.execute("INSERT INTO colonies VALUES (?,?,?,?,?,?)", (
                colony.id, colony.dish, colony.center_cell_id, colony.center_claim,
                json.dumps(colony.clarifications), colony.created_at or datetime.now(timezone.utc).isoformat(),
            ))
        except sqlite3.IntegrityError as error:
            raise ColonyExistsError(f"Colony '{colony.id}' already exists") from error


def save_colony(db_path: Path, graph: ColonyGraph, colony: Colony, *,
                connection: sqlite3.Connection | None = None) -> None:
    """Save topology atomically, or join the caller's active writer transaction."""
    now = datetime.now(timezone.utc).isoformat()
    with database.write_connection(db_path,connection) as writer:
        for cell in graph.get_all_cells():
            if not cell.created_at:
                cell.created_at = now
            record = cell.model_dump(mode='json')
            record['colony'] = cell.colony_id
            record['seq'] = parse_key(cell.id,colony.dish,strict=True)['seq']
            database.put_cell(db_path,record,connection=writer)
        writer.execute(
            'INSERT INTO colonies VALUES (?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET '
            'center_claim=excluded.center_claim, clarifications=excluded.clarifications',
            (colony.id,colony.dish,colony.center_cell_id,colony.center_claim,
             json.dumps(colony.clarifications),colony.created_at or now),
        )
        for cell_id in graph.removed_cell_ids:
            retire_cell(db_path,cell_id,connection=writer)
        for edge in graph.get_edge_records():
            append(db_path,edge,connection=writer)


def load_colony(db_path: Path, colony: str) -> tuple[ColonyGraph, Colony]:
    """Assemble one colony from rows, preserving outgoing cross-colony edges."""
    rows = database.list_cells(db_path, colony=colony)
    if not rows:
        raise KeyError(f"Colony not found: {colony}")
    graph = ColonyGraph(colony_id=colony)
    for row in rows:
        graph.add_cell(Cell.model_validate({**row, "colony_id": row["colony"]}))
    own_ids = {row["id"] for row in rows}
    all_ids = {row['id'] for row in database.list_cells(db_path)}
    for row in database.read_edges(db_path):
        if row['from_cell'] in own_ids or row['to_cell'] in own_ids:
            edge = EdgeRecord.model_validate(row)
            if edge.from_cell not in all_ids or edge.to_cell not in all_ids:
                raise ValueError(f'Edge references a missing dish cell: {edge.from_cell} -> {edge.to_cell}')
            graph.add_edge(edge)
    for cell in graph.get_all_cells():
        cell.dependencies = sorted(graph.get_dependencies(cell.id))
        cell.dependents = sorted(graph.get_dependents(cell.id))
    center = min(graph.get_all_cells(), key=lambda cell: (cell.level, cell.id))
    with closing(database.connect(db_path)) as conn:
        metadata = conn.execute("SELECT * FROM colonies WHERE id = ?", (colony,)).fetchone()
    result = Colony(
        id=colony, dish=metadata["dish"] if metadata else db_path.parent.name,
        center_claim=center.claim_text, center_cell_id=center.id,
        clarifications=json.loads(metadata["clarifications"]) if metadata else [],
        created_at=metadata["created_at"] if metadata else center.created_at,
    )
    return graph, result


def list_colonies(db_path: Path) -> list[tuple[ColonyGraph, Colony]]:
    """List database colonies without scanning legacy directories."""
    if not db_path.is_file():
        return []
    names = sorted({row["colony"] for row in database.list_cells(db_path)})
    result = []
    for name in names:
        try:
            result.append(load_colony(db_path,name))
        except (ValueError,KeyError,sqlite3.Error) as error:
            logger.error('Cannot load dish %s colony %s: %s',db_path.parent.name,name,error)
    return result


def clear_unapproved_colony(db_path: Path, colony_id: str, *, connection: sqlite3.Connection | None = None) -> None:
    """Discard this seed's draft rows while retaining its event history."""
    with database.write_connection(db_path, connection) as conn:
        approved = conn.execute(
            "SELECT 1 FROM events WHERE event_type='colony_approved' AND cell_id IN "
            "(SELECT id FROM cells WHERE colony=?) LIMIT 1", (colony_id,),
        ).fetchone()
        if approved:
            raise ValueError("An approved colony cannot be discarded")
        external = conn.execute(
            "SELECT 1 FROM edges WHERE to_cell IN (SELECT id FROM cells WHERE colony=?) "
            "AND from_cell NOT IN (SELECT id FROM cells WHERE colony=?) LIMIT 1",
            (colony_id, colony_id),
        ).fetchone()
        if external:
            raise ValueError("Another colony references this draft")
        conn.execute("DELETE FROM edges WHERE from_cell IN (SELECT id FROM cells WHERE colony=?)", (colony_id,))
        conn.execute("DELETE FROM cells WHERE colony=?", (colony_id,))
