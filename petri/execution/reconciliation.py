"""Park execution/domain divergence without changing domain rows or events."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dbos import DBOSClient

from petri.storage import database


@dataclass(frozen=True)
class DomainReceipt:
    cell_id: str
    event_count: int
    digest: str
    status: str
    row_digest: str = ""


def _digest(events):
    return hashlib.sha256(
        json.dumps(events, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _project_content(db_path: Path, cell_id: str, event_count: int) -> dict[str, str]:
    """Rebuild the trigger-maintained fields at an append-order event boundary."""
    with closing(database.connect(db_path)) as conn:
        row = conn.execute(
            "WITH prefix AS (SELECT rowid AS position,event_type,agent,"
            "json_extract(payload,'$.summary') AS summary FROM events "
            "WHERE cell_id=? ORDER BY rowid LIMIT ?) "
            "SELECT COALESCE((SELECT group_concat(summary,char(10)||char(10)) "
            "FROM (SELECT summary FROM prefix WHERE event_type='evidence_appended' "
            "AND COALESCE(summary,'')<>'' ORDER BY position)),''), "
            "COALESCE((SELECT CAST(summary AS TEXT) FROM prefix "
            "WHERE event_type='verdict_issued' AND agent='cell_lead' "
            "ORDER BY position DESC LIMIT 1),'')",
            (cell_id, event_count),
        ).fetchone()
    return {"evidence_md": row[0], "summary_md": row[1]}


def capture_receipt(db_path: Path, cell_id: str) -> DomainReceipt | None:
    row = database.get_cell(db_path, cell_id)
    if row is None:
        return None
    events = database.read_events(db_path, cell_id=cell_id)
    return DomainReceipt(
        cell_id,
        len(events),
        _digest(events),
        row["status"],
        _digest({key: value for key, value in row.items() if key != "status"}),
    )


def read_conflicts(system_path: Path) -> dict[str, dict]:
    if not system_path.is_file():
        return {}
    with closing(sqlite3.connect(system_path)) as conn:
        if not conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='petri_sync_conflicts'"
        ).fetchone():
            return {}
        return {
            row[0]: {"cell_id": row[0], "workflow_id": row[1], "reason": row[2]}
            for row in conn.execute("SELECT cell_id,workflow_id,reason FROM petri_sync_conflicts")
        }


def park_conflict(system_path: Path, cell_id: str, workflow_id: str, reason: str) -> None:
    with closing(sqlite3.connect(system_path, timeout=30)) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS petri_sync_conflicts ("
            "cell_id TEXT PRIMARY KEY, workflow_id TEXT NOT NULL, reason TEXT NOT NULL)"
        )
        conn.execute(
            "INSERT OR IGNORE INTO petri_sync_conflicts VALUES (?,?,?)",
            (cell_id, workflow_id, reason),
        )
        conn.commit()


def reconcile(db_path: Path, system_path: Path) -> dict[str, dict]:
    """Check stored receipts before DBOS starts automatic recovery."""
    if not system_path.is_file():
        return {}
    with closing(sqlite3.connect(system_path)) as conn:
        if not conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='workflow_status'"
        ).fetchone():
            return read_conflicts(system_path)
    client = DBOSClient(system_database_url=f"sqlite:///{system_path.resolve()}")
    try:
        latest: dict[str, tuple[int, Any]] = {}
        for run in client.list_workflows(name="petri.validate_cell", load_output=True):
            cell_id, marker, round_text = run.workflow_id.rpartition(":r")
            if marker and round_text.isdigit():
                if cell_id not in latest or int(round_text) > latest[cell_id][0]:
                    latest[cell_id] = (int(round_text), run)
        for cell_id, (round_number, run) in latest.items():
            reason = None
            row = database.get_cell(db_path, cell_id) if db_path.is_file() else None
            if row is None:
                reason = "The execution database has a cell round, but its domain row is missing"
            else:
                receipts = []
                for step in client.list_workflow_steps(run.workflow_id):
                    output = step["output"]
                    receipt = (
                        output
                        if isinstance(output, DomainReceipt)
                        else getattr(output, "domain_receipt", None)
                    )
                    if receipt is not None:
                        receipts.append(receipt)
                if receipts:
                    receipt = max(receipts, key=lambda item: item.event_count)
                    events = database.read_events(db_path, cell_id=cell_id)
                    row_digest = getattr(receipt, "row_digest", "")
                    if len(events) < receipt.event_count:
                        reason = (
                            f"Checkpoint requires {receipt.event_count} domain events; "
                            f"the domain database contains {len(events)}"
                        )
                    elif _digest(events[: receipt.event_count]) != receipt.digest:
                        reason = (
                            "Domain events differ from the checkpoint's committed event history"
                        )
                    elif row_digest:
                        current_content = _project_content(db_path, cell_id, len(events))
                        if any(row[key] != value for key, value in current_content.items()):
                            reason = "Domain cell content differs from its committed event history"
                        else:
                            checkpoint_row = {
                                key: value for key, value in row.items() if key != "status"
                            }
                            checkpoint_row.update(
                                _project_content(db_path, cell_id, receipt.event_count)
                            )
                            if row_digest != _digest(checkpoint_row):
                                reason = (
                                    "Domain cell fields differ from the checkpoint's committed cell"
                                )
                with closing(database.connect(db_path)) as conn:
                    planned = conn.execute(
                        "SELECT MAX(round_number) FROM execution_round_requests WHERE cell_id = ?",
                        (cell_id,),
                    ).fetchone()[0]
                reopening = (
                    planned is not None and planned > round_number and row["status"] == "NEW"
                )
                if reason is None and run.status in {"PENDING", "ENQUEUED", "DELAYED"}:
                    dependencies = [
                        edge["to_cell"]
                        for edge in database.read_edges(db_path)
                        if edge["from_cell"] == cell_id
                    ]
                    if any(
                        (database.get_cell(db_path, dependency) or {}).get("status") != "VALIDATED"
                        for dependency in dependencies
                    ):
                        reason = (
                            "Pending workflow has unvalidated dependencies in the domain database"
                        )
                if reason is None and run.status == "SUCCESS" and not reopening:
                    outcome = run.output
                    if getattr(outcome, "status", None) == "sync_conflict":
                        reason = getattr(
                            outcome, "reason", "Execution is parked for reconciliation"
                        )
                    else:
                        expected = (
                            getattr(outcome, "cell_status", None)
                            or getattr(outcome, "status", "").upper()
                        )
                        if expected and row["status"] != expected:
                            reason = (
                                f"Workflow reports SUCCESS/{expected}, but the domain cell "
                                f"has status {row['status']}"
                            )
            if reason:
                park_conflict(system_path, cell_id, run.workflow_id, reason)
        return read_conflicts(system_path)
    finally:
        client.destroy()
