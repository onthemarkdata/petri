"""Consistent domain snapshots, separate from durable execution state."""

from __future__ import annotations

import json
import shutil
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from petri.query.connection import read_connection
from petri.storage.paths import dish_db_path


@dataclass(frozen=True)
class Snapshot:
    name: str
    dish_id: str
    created_at: str
    cell_count: int
    size_bytes: int
    path: Path


def snapshot_before_destructive(petri_dir: Path, dish_id: str) -> Path:
    """Copy committed domain rows with VACUUM INTO before a destructive write."""
    source = dish_db_path(petri_dir, dish_id)
    if not source.is_file():
        raise FileNotFoundError(f"Dish database does not exist: {source}")
    now = datetime.now(UTC)
    target = Path(petri_dir) / "backups" / f"{dish_id}-{now:%Y%m%dT%H%M%S%fZ}-{uuid4().hex[:8]}"
    target.mkdir(parents=True)
    try:
        with sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True, timeout=30) as connection:
            connection.execute("VACUUM INTO ?", (str(target / "petri.sqlite"),))
        defaults = Path(petri_dir) / "defaults"
        if defaults.is_dir():
            shutil.copytree(defaults, target / "defaults")
        (target / "snapshot.json").write_text(json.dumps({
            "format_version": 1, "dish_id": dish_id, "created_at": now.isoformat(),
        }, sort_keys=True) + "\n", encoding="utf-8")
        return target
    except (sqlite3.Error, OSError) as error:
        shutil.rmtree(target)
        raise RuntimeError(
            f"Could not snapshot dish '{dish_id}': {error}. "
            "Check that the backup disk has room for the database and defaults."
        ) from error


def list_snapshots(petri_dir: Path, dish_id: str | None = None) -> list[Snapshot]:
    """List completed snapshots newest first, reading counts from each snapshot."""
    root = Path(petri_dir) / "backups"
    if not root.is_dir():
        return []
    result = []
    for directory in root.iterdir():
        if not directory.is_dir() or directory.is_symlink():
            continue
        metadata = directory / "snapshot.json"
        if not metadata.is_file():
            continue
        details = json.loads(metadata.read_text(encoding="utf-8"))
        if dish_id is not None and details["dish_id"] != dish_id:
            continue
        path = directory / "petri.sqlite"
        with read_connection(path) as connection:
            if connection is None:
                continue
            count = connection.execute("SELECT COUNT(*) FROM cells").fetchone()[0]
        result.append(Snapshot(directory.name, details["dish_id"], details["created_at"],
                               count, path.stat().st_size, directory))
    return sorted(result, key=lambda snapshot: (snapshot.created_at, snapshot.name), reverse=True)


def restore_snapshot(petri_dir: Path, name: str, *, force: bool = False) -> Path:
    """Restore one validated snapshot, refusing overwrites and active execution."""
    from filelock import FileLock, Timeout

    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError("Use a snapshot name from petri backup --list")
    snapshot = Path(petri_dir) / "backups" / name
    if snapshot.is_symlink():
        raise ValueError("A snapshot must be a local directory, not a symlink")
    metadata = json.loads((snapshot / "snapshot.json").read_text(encoding="utf-8"))
    target = dish_db_path(petri_dir, metadata["dish_id"])
    source = snapshot / "petri.sqlite"
    if source.is_symlink():
        raise ValueError("A snapshot database must not be a symlink")
    with read_connection(source) as connection:
        if connection is None or connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("Snapshot database failed its integrity check")
    if target.exists() and not force:
        raise ValueError(f"Refusing to overwrite {target}; use --force to restore this snapshot")
    owner = FileLock(Path(petri_dir) / ".execution-owner.lock")
    try:
        owner.acquire(timeout=0)
    except Timeout as error:
        raise RuntimeError("Execution is active. Stop grow before restoring this dish.") from error
    staging = target.with_name(f".restore-{uuid4().hex}.sqlite")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, staging)
        # The owner lock prevents DBOS from reopening the old generation here.
        for suffix in ("-wal", "-shm"):
            Path(str(target) + suffix).unlink(missing_ok=True)
        staging.replace(target)
        defaults = snapshot / "defaults"
        if defaults.is_dir():
            shutil.copytree(defaults, Path(petri_dir) / "defaults", dirs_exist_ok=True)
    finally:
        staging.unlink(missing_ok=True)
        owner.release()
    return target
