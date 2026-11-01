"""Report and flag transitive dependents without reopening or enqueueing them."""

from pathlib import Path

from petri.graph.colony import ColonyGraph
from petri.models import CellStatus
from petri.storage.database import SQLiteEventLog
from petri.storage.event_log import append_event
from petri.storage.paths import dish_db_path


def propagate_upward(
    petri_dir: Path,
    cell_id: str,
    colony_graph: ColonyGraph,
    dish_id: str,
    request_id: str = "feed",
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
                node=f"propagate:{request_id}",
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
