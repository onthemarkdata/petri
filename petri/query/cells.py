"""Typed cell and edge reads without graph assembly or per-cell file access."""

import json
from pathlib import Path
from typing import Any, TypedDict, cast

from petri.query.connection import has_table, read_connection


class EdgeRow(TypedDict):
    from_cell: str
    to_cell: str
    edge_type: str
    reason: str
    created_at: str


class CellRow(TypedDict):
    cell_id: str
    colony_id: str
    claim_text: str
    level: int
    seq: int
    status: str
    dependencies: list[str]
    dependents: list[str]
    relation: str | None
    is_atomic: bool
    bedrock_reason: str | None
    created_at: str
    evidence_md: str
    summary_md: str


def list_cells(
    db_path: Path,
    *,
    colony: str | None = None,
    status: str | None = None,
    level: int | None = None,
    cell_id: str | None = None,
) -> list[CellRow]:
    with read_connection(db_path) as connection:
        if connection is None or not has_table(connection, "cells"):
            return []
        sql = (
            "SELECT c.*,"
            "(SELECT json_group_array(to_cell) FROM (SELECT DISTINCT to_cell FROM edges WHERE from_cell=c.id ORDER BY to_cell)) AS dependencies,"
            "(SELECT json_group_array(dependent_cell_id) FROM (SELECT DISTINCT dependent_cell_id FROM dependents WHERE cell_id=c.id ORDER BY dependent_cell_id)) AS dependents "
            "FROM cells c WHERE (? IS NULL OR c.colony=?) AND (? IS NULL OR c.status=?) "
            "AND (? IS NULL OR c.level=?) AND (? IS NULL OR c.id=?) ORDER BY c.colony,c.level,c.seq,c.id"
        )
        result = []
        for row in connection.execute(
            sql, (colony, colony, status, status, level, level, cell_id, cell_id)
        ):
            record = dict(row)
            result.append(
                CellRow(
                    cell_id=record["id"],
                    colony_id=record["colony"],
                    claim_text=record["claim_text"],
                    level=record["level"],
                    seq=record["seq"],
                    status=record["status"],
                    dependencies=json.loads(record["dependencies"]),
                    dependents=json.loads(record["dependents"]),
                    relation=record["relation"],
                    is_atomic=bool(record["is_atomic"]),
                    bedrock_reason=record["bedrock_reason"],
                    created_at=record["created_at"],
                    evidence_md=record.get("evidence_md", ""),
                    summary_md=record.get("summary_md", ""),
                )
            )
        return result


def get_cell(db_path: Path, cell_id: str) -> CellRow | None:
    rows = list_cells(db_path, cell_id=cell_id)
    return rows[0] if rows else None


def get_edges(
    db_path: Path, *, cell_id: str | None = None, colony: str | None = None
) -> list[EdgeRow]:
    with read_connection(db_path) as connection:
        if connection is None or not has_table(connection, "edges"):
            return []
        rows = connection.execute(
            "SELECT e.* FROM edges e WHERE (? IS NULL OR e.from_cell=? OR e.to_cell=?) "
            "AND (? IS NULL OR EXISTS (SELECT 1 FROM cells c WHERE c.colony=? AND c.id IN (e.from_cell,e.to_cell))) "
            "ORDER BY e.from_cell,e.to_cell,e.edge_type",
            (cell_id, cell_id, cell_id, colony, colony),
        )
        return [cast(EdgeRow, dict(row)) for row in rows]


def list_colonies(db_path: Path) -> list[dict[str, Any]]:
    with read_connection(db_path) as connection:
        if connection is None or not has_table(connection, "colonies"):
            return []
        result = []
        for row in connection.execute("SELECT * FROM colonies ORDER BY id"):
            item = dict(row)
            item["clarifications"] = json.loads(item["clarifications"])
            result.append(item)
        return result


read_edges = get_edges
