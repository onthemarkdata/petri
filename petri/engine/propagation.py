"""Evidence re-entry and propagation engine for the Petri framework.

When new evidence arrives, this module handles re-opening settled cells and
walking the dependency graph to flag dependents that may need re-evaluation.
Propagation is conservative: it flags cells but does NOT automatically re-open
them -- the user decides which dependents to re-queue.
"""

from __future__ import annotations

from pathlib import Path

from petri.storage.dish import resolve_dish_id as _get_dish_id
from petri.graph.colony import ColonyGraph
from petri.storage.event_log import append_event
from petri.models import CellStatus
from petri.storage.database import SQLiteEventLog, get_cell, put_cell
from petri.storage.paths import dish_db_path
from petri.storage.queue import add_to_queue, load_queue


# ── Re-openable statuses ────────────────────────────────────────────────

_REOPENABLE = frozenset({
    CellStatus.VALIDATED,
    CellStatus.DISPROVEN,
    CellStatus.DEFER_OPEN,
})


# ── Helpers ─────────────────────────────────────────────────────────────


# ── Core Functions ──────────────────────────────────────────────────────


def reopen_cell(
    petri_dir: Path,
    cell_id: str,
    trigger: str,
    colony_graph: ColonyGraph | None = None,
) -> dict:
    """Re-open a validated/disproven cell for re-evaluation.

    1. Load the cell from the dish database.
    2. Verify cell is in a re-openable status (VALIDATED, DISPROVEN, DEFER_OPEN).
    3. Set the row status to NEW while preserving prior evidence.
    4. Log ``cell_reopened`` event with trigger description and prior status.
    5. Add cell back to queue.
    6. Return ``{"cell_id": ..., "prior_status": ..., "new_status": "NEW"}``.

    Raises ``ValueError`` if cell is not in a re-openable status.
    """
    dish_id = _get_dish_id(petri_dir)
    db_path = dish_db_path(petri_dir, dish_id)
    metadata = get_cell(db_path, cell_id)
    if metadata is None:
        raise FileNotFoundError(f"Cell not found: {cell_id}")

    prior_status_str = metadata.get("status", "")
    try:
        prior_status = CellStatus(prior_status_str)
    except ValueError:
        raise ValueError(
            f"Cell {cell_id} has unrecognised status '{prior_status_str}'"
        )

    if prior_status not in _REOPENABLE:
        raise ValueError(
            f"Cell {cell_id} is in status '{prior_status.value}' which is not "
            f"re-openable. Must be one of: "
            f"{', '.join(s.value for s in _REOPENABLE)}"
        )

    # Update status to NEW (append-only: evidence stays, only status changes)
    metadata["status"] = CellStatus.NEW.value
    put_cell(db_path, metadata)

    # Log the cell_reopened event
    events_path = SQLiteEventLog(db_path, cell_id)
    append_event(
        events_path=events_path,
        cell_id=cell_id,
        event_type="cell_reopened",
        agent="propagation_engine",
        iteration=0,
        data={
            "trigger": trigger,
            "prior_status": prior_status.value,
        },
    )

    # Add cell back to queue (remove first if already present)
    queue_path = petri_dir / "queue.json"
    queue = load_queue(queue_path)
    if cell_id in queue.get("entries", {}):
        from petri.storage.queue import remove_from_queue
        remove_from_queue(queue_path, cell_id)
    add_to_queue(queue_path, cell_id)

    return {
        "cell_id": cell_id,
        "prior_status": prior_status.value,
        "new_status": CellStatus.NEW.value,
    }


def propagate_upward(
    petri_dir: Path,
    cell_id: str,
    colony_graph: ColonyGraph,
    dish_id: str,
) -> list[str]:
    """Walk the dependency graph from a re-opened cell toward the colony center.

    Finds all dependents (cells that depend on the re-opened cell) and flags
    them with a ``propagation_triggered`` event. Recursively propagates to
    their dependents as well.

    Propagation flags dependents but does NOT automatically re-open them.
    The user decides whether to re-queue.

    Returns a deduplicated list of all flagged cell IDs.
    """
    flagged: list[str] = []
    visited: set[str] = set()
    stack = [cell_id]

    while stack:
        current = stack.pop()
        dependents = colony_graph.get_dependents(current)
        for dep_id in dependents:
            if dep_id in visited:
                continue
            visited.add(dep_id)
            flagged.append(dep_id)

            # Log propagation_triggered event for this dependent
            events_path = SQLiteEventLog(dish_db_path(petri_dir, dish_id), dep_id)
            append_event(
                events_path=events_path,
                cell_id=dep_id,
                event_type="propagation_triggered",
                agent="propagation_engine",
                iteration=0,
                data={
                    "reopened_cell_id": cell_id,
                    "flagged_dependents": [dep_id],
                },
            )

            # Continue propagation to this dependent's dependents
            stack.append(dep_id)

    return flagged


def get_impact_report(
    petri_dir: Path,
    cell_id: str,
    colony_graph: ColonyGraph,
    dish_id: str,
) -> dict:
    """Generate a report of which cells would be affected by re-opening a cell.

    Walks the dependency graph upward (toward the colony center) to find all
    transitive dependents. Does not modify any state -- read-only analysis.

    Returns::

        {
            "reopened_cell": cell_id,
            "affected_cells": [
                {"cell_id": "...", "level": int, "status": str, "claim_text": str}
            ],
            "total_affected": int,
        }
    """
    affected: list[dict] = []
    visited: set[str] = set()
    stack = [cell_id]

    while stack:
        current = stack.pop()
        dependents = colony_graph.get_dependents(current)
        for dep_id in dependents:
            if dep_id in visited:
                continue
            visited.add(dep_id)

            # Read cell info from the graph
            try:
                cell = colony_graph.get_cell(dep_id)
                affected.append({
                    "cell_id": dep_id,
                    "level": cell.level,
                    "status": cell.status.value
                    if isinstance(cell.status, CellStatus)
                    else str(cell.status),
                    "claim_text": cell.claim_text,
                })
            except KeyError:
                # Cell not in this graph (cross-colony reference); skip
                continue

            stack.append(dep_id)

    # Sort by level ascending (deepest dependencies first)
    affected.sort(key=lambda a: (a["level"], a["cell_id"]))

    return {
        "reopened_cell": cell_id,
        "affected_cells": affected,
        "total_affected": len(affected),
    }
