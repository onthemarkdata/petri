"""Classify seed work and regeneration attempts from immutable domain events."""

from pathlib import Path
from typing import Literal

from petri.storage import database


def current_attempt(db_path: Path, center_id: str) -> int:
    """Regeneration starts a new event namespace; resume retains the last one."""
    return len(database.read_events(db_path, cell_id=center_id, event_type="decomposition_started"))


def classify_cells(
    db_path: Path, colony_id: str, attempt: int
) -> dict[str, Literal["pending", "expanded", "atomic"]]:
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
