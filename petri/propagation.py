"""Evidence re-entry and propagation engine for the Petri framework.

When new evidence arrives, this module handles re-opening settled nodes and
walking the dependency graph to flag dependents that may need re-evaluation.
Propagation is conservative: it flags nodes but does NOT automatically re-open
them -- the user decides which dependents to re-queue.
"""

from __future__ import annotations

import json
from pathlib import Path

from petri.colony import ColonyGraph, deserialize_colony
from petri.event_log import append_event
from petri.models import NodeStatus
from petri.queue import add_to_queue, load_queue


# ── Re-openable statuses ────────────────────────────────────────────────

_REOPENABLE = frozenset({
    NodeStatus.VALIDATED,
    NodeStatus.DISPROVEN,
    NodeStatus.DEFER_OPEN,
})


# ── Helpers ─────────────────────────────────────────────────────────────


def _node_dir_for(petri_dir: Path, node_id: str, dish_id: str) -> Path:
    """Derive the filesystem directory for a node from its composite key.

    Uses the ``node_paths`` mapping stored in ``colony.json`` when available.
    Falls back to scanning for the node's ``metadata.json`` via rglob.
    """
    parts = node_id.split("-")
    # Colony slug is everything between dish_id and the level-seq suffix.
    prefix = "-".join(parts[:-2])  # "mydish-mycolony"
    if prefix.startswith(dish_id + "-"):
        colony_slug = prefix[len(dish_id) + 1:]
    else:
        colony_slug = prefix
    colony_dir = petri_dir / "petri-dishes" / colony_slug

    # Try node_paths from colony.json first
    colony_json = colony_dir / "colony.json"
    if colony_json.exists():
        colony_data = json.loads(colony_json.read_text(encoding="utf-8"))
        node_paths = colony_data.get("node_paths", {})
        if node_id in node_paths:
            return colony_dir / node_paths[node_id]

    # Fallback: scan for the node's metadata.json
    for metadata_path in colony_dir.rglob("metadata.json"):
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        if meta.get("id") == node_id:
            return metadata_path.parent

    # Last resort: old flat layout
    seq_str = parts[-1]
    level_str = parts[-2]
    return colony_dir / f"{level_str}-{seq_str}"


def _load_node_metadata(node_dir: Path) -> dict:
    """Read and return the metadata.json for a node directory."""
    metadata_path = node_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"No metadata.json at {metadata_path}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _save_node_metadata(node_dir: Path, metadata: dict) -> None:
    """Write metadata.json for a node directory."""
    metadata_path = node_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _get_dish_id(petri_dir: Path) -> str:
    """Get the dish ID from config or derive from directory name."""
    config_path = petri_dir / "petri.yaml"
    if config_path.exists():
        for line in config_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("name:"):
                return line.split(":", 1)[1].strip()
    return petri_dir.parent.name


# ── Core Functions ──────────────────────────────────────────────────────


def reopen_node(
    petri_dir: Path,
    node_id: str,
    trigger: str,
    colony_graph: ColonyGraph | None = None,
) -> dict:
    """Re-open a validated/disproven node for re-evaluation.

    1. Load node metadata from filesystem.
    2. Verify node is in a re-openable status (VALIDATED, DISPROVEN, DEFER_OPEN).
    3. Update node status to NEW in metadata.json (all prior evidence preserved).
    4. Log ``node_reopened`` event with trigger description and prior status.
    5. Add node back to queue.
    6. Return ``{"node_id": ..., "prior_status": ..., "new_status": "NEW"}``.

    Raises ``ValueError`` if node is not in a re-openable status.
    """
    dish_id = _get_dish_id(petri_dir)
    node_dir = _node_dir_for(petri_dir, node_id, dish_id)
    metadata = _load_node_metadata(node_dir)

    prior_status_str = metadata.get("status", "")
    try:
        prior_status = NodeStatus(prior_status_str)
    except ValueError:
        raise ValueError(
            f"Node {node_id} has unrecognised status '{prior_status_str}'"
        )

    if prior_status not in _REOPENABLE:
        raise ValueError(
            f"Node {node_id} is in status '{prior_status.value}' which is not "
            f"re-openable. Must be one of: "
            f"{', '.join(s.value for s in _REOPENABLE)}"
        )

    # Update status to NEW (append-only: evidence stays, only status changes)
    metadata["status"] = NodeStatus.NEW.value
    _save_node_metadata(node_dir, metadata)

    # Log the node_reopened event
    events_path = node_dir / "events.jsonl"
    append_event(
        events_path=events_path,
        node_id=node_id,
        event_type="node_reopened",
        agent="propagation_engine",
        iteration=0,
        data={
            "trigger": trigger,
            "prior_status": prior_status.value,
        },
    )

    # Add node back to queue (remove first if already present)
    queue_path = petri_dir / "queue.json"
    queue = load_queue(queue_path)
    if node_id in queue.get("entries", {}):
        from petri.queue import remove_from_queue
        remove_from_queue(queue_path, node_id)
    add_to_queue(queue_path, node_id)

    return {
        "node_id": node_id,
        "prior_status": prior_status.value,
        "new_status": NodeStatus.NEW.value,
    }


def propagate_upward(
    petri_dir: Path,
    node_id: str,
    colony_graph: ColonyGraph,
    dish_id: str,
) -> list[str]:
    """Walk the dependency graph from a re-opened node toward the colony center.

    Finds all dependents (nodes that depend on the re-opened node) and flags
    them with a ``propagation_triggered`` event. Recursively propagates to
    their dependents as well.

    Propagation flags dependents but does NOT automatically re-open them.
    The user decides whether to re-queue.

    Returns a deduplicated list of all flagged node IDs.
    """
    flagged: list[str] = []
    visited: set[str] = set()
    stack = [node_id]

    while stack:
        current = stack.pop()
        dependents = colony_graph.get_dependents(current)
        for dep_id in dependents:
            if dep_id in visited:
                continue
            visited.add(dep_id)
            flagged.append(dep_id)

            # Log propagation_triggered event for this dependent
            dep_dir = _node_dir_for(petri_dir, dep_id, dish_id)
            events_path = dep_dir / "events.jsonl"
            append_event(
                events_path=events_path,
                node_id=dep_id,
                event_type="propagation_triggered",
                agent="propagation_engine",
                iteration=0,
                data={
                    "reopened_node_id": node_id,
                    "flagged_dependents": [dep_id],
                },
            )

            # Continue propagation to this dependent's dependents
            stack.append(dep_id)

    return flagged


def get_impact_report(
    petri_dir: Path,
    node_id: str,
    colony_graph: ColonyGraph,
    dish_id: str,
) -> dict:
    """Generate a report of which nodes would be affected by re-opening a node.

    Walks the dependency graph upward (toward the colony center) to find all
    transitive dependents. Does not modify any state -- read-only analysis.

    Returns::

        {
            "reopened_node": node_id,
            "affected_nodes": [
                {"node_id": "...", "level": int, "status": str, "claim_text": str}
            ],
            "total_affected": int,
        }
    """
    affected: list[dict] = []
    visited: set[str] = set()
    stack = [node_id]

    while stack:
        current = stack.pop()
        dependents = colony_graph.get_dependents(current)
        for dep_id in dependents:
            if dep_id in visited:
                continue
            visited.add(dep_id)

            # Read node info from the graph
            try:
                node = colony_graph.get_node(dep_id)
                affected.append({
                    "node_id": dep_id,
                    "level": node.level,
                    "status": node.status.value
                    if isinstance(node.status, NodeStatus)
                    else str(node.status),
                    "claim_text": node.claim_text,
                })
            except KeyError:
                # Node not in this graph (cross-colony reference); skip
                continue

            stack.append(dep_id)

    # Sort by level ascending (deepest dependencies first)
    affected.sort(key=lambda a: (a["level"], a["node_id"]))

    return {
        "reopened_node": node_id,
        "affected_nodes": affected,
        "total_affected": len(affected),
    }
