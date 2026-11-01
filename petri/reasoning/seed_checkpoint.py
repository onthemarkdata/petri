"""Seed checkpoint storage and event-based recovery classification."""

import json
import os
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

from petri.reasoning.seed_graph import SeedState
from petri.storage import database


class SeedCheckpointStore:
    """Own checkpoint persistence; a corrupt checkpoint falls back to events.

    SQLite is the default. An explicit directory selects the atomic file driver
    for environments that keep checkpoints separately from the domain database.
    """

    def __init__(self, db_path: Path, *, file_dir: Path | None = None):
        self.db_path = Path(db_path)
        self.file_dir = file_dir

    def _file(self, colony_id: str) -> Path:
        import hashlib

        assert self.file_dir is not None
        return self.file_dir / (hashlib.sha256(colony_id.encode()).hexdigest() + ".json")

    def save(self, state: SeedState) -> None:
        value = state.model_dump_json()
        if self.file_dir is not None:
            self.file_dir.mkdir(parents=True, exist_ok=True)
            path = self._file(state.colony_id)
            temporary = path.with_suffix(".tmp")
            with temporary.open("w", encoding="utf-8") as output:
                output.write(value)
                output.flush()
                os.fsync(output.fileno())
            temporary.replace(path)
            return
        with database._write(self.db_path) as connection:
            connection.execute(
                "INSERT INTO seed_checkpoints VALUES (?,?,?) ON CONFLICT(colony) "
                "DO UPDATE SET state=excluded.state,updated_at=excluded.updated_at",
                (state.colony_id, value, datetime.now(UTC).isoformat()),
            )

    def load(self, colony_id: str, attempt: int) -> SeedState | None:
        if self.file_dir is not None:
            try:
                value = self._file(colony_id).read_text(encoding="utf-8")
            except FileNotFoundError:
                return None
        else:
            with closing(database.connect(self.db_path)) as connection:
                row = connection.execute(
                    "SELECT state FROM seed_checkpoints WHERE colony=?", (colony_id,),
                ).fetchone()
            if row is None:
                return None
            value = row[0]
        try:
            state = SeedState.model_validate_json(value)
        except (ValidationError, ValueError, json.JSONDecodeError):
            return None
        return state if state.colony_id == colony_id and state.attempt == attempt else None

    def delete(self, colony_id: str) -> None:
        if self.file_dir is not None:
            self._file(colony_id).unlink(missing_ok=True)
        else:
            with database._write(self.db_path) as connection:
                connection.execute("DELETE FROM seed_checkpoints WHERE colony=?", (colony_id,))


def current_attempt(db_path: Path, center_id: str) -> int:
    """Regeneration starts a new event namespace; resume retains the last one."""
    return len(database.read_events(db_path, cell_id=center_id, event_type="decomposition_started"))


def classify_cells(db_path: Path, colony_id: str, attempt: int) -> dict[str, Literal["pending", "expanded", "atomic"]]:
    """Classify persisted cells using only their current seed attempt's events."""
    result: dict[str, Literal["pending", "expanded", "atomic"]] = {
        cell["id"]: "pending" for cell in database.list_cells(db_path, colony=colony_id)
    }
    for event in database.read_events(db_path):
        if event["cell_id"] not in result or event["data"].get("attempt") != attempt:
            continue
        if event["type"] == "cell_expansion_completed":
            result[event["cell_id"]] = "expanded"
        elif event["type"] == "cell_marked_atomic":
            result[event["cell_id"]] = "atomic"
    return result
