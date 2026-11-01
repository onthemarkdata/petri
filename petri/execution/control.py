"""Persist user cancellation intent beside the DBOS system database."""

from contextlib import closing
from pathlib import Path
import sqlite3


def cancellation_mode(system_path: Path, workflow_id: str) -> str | None:
    if not system_path.is_file():
        return None
    with closing(sqlite3.connect(system_path, timeout=30)) as conn:
        if not conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name='petri_cancel_requests'"
        ).fetchone():
            return None
        row = conn.execute(
            "SELECT mode FROM petri_cancel_requests WHERE workflow_id=?", (workflow_id,)
        ).fetchone()
        return row[0] if row else None


def record_cancellation(system_path: Path, workflow_ids: list[str], *, force: bool) -> None:
    with closing(sqlite3.connect(system_path, timeout=30)) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS petri_cancel_requests "
            "(workflow_id TEXT PRIMARY KEY, mode TEXT NOT NULL)"
        )
        conn.executemany(
            "INSERT INTO petri_cancel_requests VALUES (?,?) "
            "ON CONFLICT(workflow_id) DO UPDATE SET mode=excluded.mode",
            [(workflow_id, "force" if force else "graceful") for workflow_id in workflow_ids],
        )
        conn.commit()


def clear_cancellation(system_path: Path, *, workflow_ids: list[str] | None = None) -> None:
    if not system_path.is_file():
        return
    with closing(sqlite3.connect(system_path, timeout=30)) as conn:
        if conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name='petri_cancel_requests'"
        ).fetchone():
            if workflow_ids is None:
                conn.execute("DELETE FROM petri_cancel_requests")
            else:
                conn.executemany(
                    "DELETE FROM petri_cancel_requests WHERE workflow_id=?",
                    [(workflow_id,) for workflow_id in workflow_ids],
                )
            conn.commit()


def has_cancellation(system_path: Path) -> bool:
    if not system_path.is_file():
        return False
    with closing(sqlite3.connect(system_path, timeout=30)) as conn:
        if not conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name='petri_cancel_requests'"
        ).fetchone():
            return False
        return conn.execute("SELECT 1 FROM petri_cancel_requests LIMIT 1").fetchone() is not None


def cancel_at_boundary(system_path: Path, workflow_id: str) -> None:
    """Cancel only after the preceding step has committed its checkpoint."""
    if cancellation_mode(system_path, workflow_id) is None:
        return
    from dbos import DBOSClient
    from dbos._error import DBOSWorkflowCancelledError

    client = DBOSClient(system_database_url=f"sqlite:///{system_path.resolve()}")
    try:
        client.cancel_workflow(workflow_id)
    finally:
        client.destroy()
    raise DBOSWorkflowCancelledError("Stop requested by user at a step boundary")
