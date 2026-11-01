"""Classify seed work and regeneration attempts from immutable domain events."""

from pathlib import Path
from typing import Literal

from petri.storage import database
from petri.reasoning.seed_graph import SeedState
from petri.storage.event_log import append_event


def current_attempt(db_path: Path, center_id: str) -> int:
    """Regeneration starts a new event namespace; resume retains the last one."""
    return len(database.read_events(db_path, cell_id=center_id, event_type="decomposition_started"))


def record_completion(db_path: Path, center_id: str, *, attempt: int, cell_count: int) -> None:
    """Record completion once per seed attempt, including replay after a crash."""
    append_event(
        database.SQLiteEventLog(db_path),
        center_id,
        "decomposition_completed",
        "decomposition_lead",
        0,
        {"attempt": attempt, "cell_count": cell_count},
        round_number=attempt,
        node="seed-completion",
    )


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
    center = next(
        (cell for cell in database.list_cells(db_path, colony=colony_id) if cell["level"] == 0),
        None,
    )
    edited = latest_edited_state(db_path, center["id"], attempt) if center else None
    if edited is not None:
        return {cell.id: "atomic" if cell.is_atomic else "expanded" for cell in edited.cells}
    return result


def latest_edited_state(db_path: Path, center_id: str, attempt: int) -> SeedState | None:
    """A committed human edit supersedes earlier model-output events."""
    events = database.read_events(db_path, cell_id=center_id, event_type="decomposition_edited")
    for event in reversed(events):
        if event["data"].get("attempt") == attempt:
            state = SeedState.model_validate(event["data"]["state"])
            if (
                not state.complete
                or state.attempt != attempt
                or not any(cell.id == center_id for cell in state.cells)
            ):
                raise ValueError("Recorded seed edit has an invalid completed state")
            return state
    return None
