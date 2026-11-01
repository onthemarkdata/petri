"""Report and flag transitive dish dependents without reopening or enqueueing them."""

from pathlib import Path

from petri.graph.colony import ColonyGraph
from petri.storage.database import SQLiteEventLog
from petri.storage.edge_registry import DishGraph
from petri.storage.event_log import append_event
from petri.storage.paths import dish_db_path


def propagate_upward(
    petri_dir: Path,
    cell_id: str,
    colony_graph: ColonyGraph,
    dish_id: str,
    request_id: str = "feed",
) -> list[str]:
    """Flag every transitive dependent in the persisted dish graph.

    Flags preserve cell statuses and do not enqueue validation. The colony
    argument remains compatible with callers; persisted dish edges determine
    reachability across colony boundaries.
    """
    db_path = dish_db_path(petri_dir, dish_id)
    flagged = DishGraph.read(db_path).reachable(cell_id, reverse=True)
    for dep_id in flagged:
        append_event(
            events_path=SQLiteEventLog(db_path, dep_id),
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
    return flagged


def get_impact_report(
    petri_dir: Path,
    cell_id: str,
    colony_graph: ColonyGraph,
    dish_id: str,
) -> dict:
    """Read transitive impact across the dish without changing stored state."""
    graph = DishGraph.read(dish_db_path(petri_dir, dish_id))
    affected = [
        {
            "cell_id": dep_id,
            "level": graph.cells[dep_id]["level"],
            "status": graph.cells[dep_id]["status"],
            "claim_text": graph.cells[dep_id]["claim_text"],
        }
        for dep_id in graph.reachable(cell_id, reverse=True)
    ]
    affected.sort(key=lambda row: (row["level"], row["cell_id"]))
    return {
        "reopened_cell": cell_id,
        "affected_cells": affected,
        "total_affected": len(affected),
    }
