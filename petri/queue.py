"""Workflow queue manager for the Petri validation pipeline.

Manages ``queue.json`` -- a file-locked JSON store that tracks where each node
is in the 13-state validation state machine.  The queue stores NO research
data; verdicts and events live in the event log (see ``petri.event_log``).

Agents never write the queue file directly; they call the functions in this
module which enforce valid transitions and serialise concurrent access via
``fcntl`` file locking.
"""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

if sys.platform == "win32":
    raise ImportError(
        "petri.queue requires Unix file locking (fcntl). Windows is not supported."
    )
import fcntl

from petri.models import QueueEntry, QueueState

# ── Valid State Transitions ─────────────────────────────────────────────

VALID_TRANSITIONS: dict[str, list[str]] = {
    "queued": ["socratic_active"],
    "socratic_active": ["research_active", "stalled"],
    "research_active": ["critique_active", "stalled"],
    "critique_active": ["mediating", "stalled"],
    "mediating": ["converged", "stalled", "research_active"],  # research_active = iterate
    "converged": ["red_team_active", "deferred_open", "deferred_closed"],
    "stalled": ["needs_human", "queued"],
    "needs_human": ["queued"],
    "red_team_active": ["evaluating", "deferred_open"],
    "evaluating": ["done", "deferred_open", "deferred_closed"],
    "done": ["queued"],  # re-entry
    "deferred_open": ["queued"],
    "deferred_closed": [],  # terminal
    "sync_conflict": ["queued"],
}

VALID_STATES: list[str] = list(VALID_TRANSITIONS.keys())

_RESUMABLE_STATES: list[str] = [
    "queued",
    "socratic_active",
    "research_active",
    "critique_active",
    "mediating",
]


# ── Helpers ─────────────────────────────────────────────────────────────


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def _queue_lock(queue_path: Path) -> Iterator[None]:
    """Exclusive file lock for concurrent queue access.

    Multiple agents may call queue functions simultaneously from different
    processes.  ``fcntl.flock`` serialises access at the OS level.
    """
    lock_path = queue_path.with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = open(lock_path, "w")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()


def _empty_queue() -> dict:
    return {"version": 1, "last_updated": None, "entries": {}}


# ── Core I/O ────────────────────────────────────────────────────────────


def load_queue(queue_path: Path) -> dict:
    """Load queue from JSON file. Return empty structure if file doesn't exist."""
    if not queue_path.exists():
        return _empty_queue()
    with open(queue_path) as f:
        return json.load(f)


def save_queue(queue_path: Path, queue: dict) -> None:
    """Save queue with last_updated timestamp.

    Writes to a temporary file then renames atomically so concurrent
    readers never see a truncated or partial file.
    """
    import os
    import tempfile

    queue["last_updated"] = _now()
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=queue_path.parent, suffix=".tmp", prefix=".queue-"
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(queue, f, indent=2)
            f.write("\n")
        os.rename(tmp_path, queue_path)
    except BaseException:
        os.unlink(tmp_path)
        raise


# ── Queue Operations ────────────────────────────────────────────────────


def add_to_queue(
    queue_path: Path,
    node_id: str,
    entered_at: str | None = None,
) -> QueueEntry:
    """Add a node to the queue.

    Validates via Pydantic ``QueueEntry`` model.  Raises ``ValueError`` if the
    node is already present.  Uses file lock.
    """
    with _queue_lock(queue_path):
        queue = load_queue(queue_path)

        if node_id in queue["entries"]:
            existing_state = queue["entries"][node_id].get("queue_state", "unknown")
            raise ValueError(
                f"Node {node_id} is already in the queue (state: {existing_state})"
            )

        now = _now()
        entry = QueueEntry(
            node_id=node_id,
            entered_at=entered_at or now,
            last_activity=now,
        )

        queue["entries"][node_id] = entry.model_dump()
        save_queue(queue_path, queue)

    return entry


def update_state(queue_path: Path, node_id: str, new_state: str) -> None:
    """Transition queue state.

    Validates against ``VALID_TRANSITIONS`` -- raises ``ValueError`` if the
    transition is not allowed.  Uses file lock.  Updates ``last_activity``.
    """
    if new_state not in VALID_STATES:
        raise ValueError(
            f"Invalid state '{new_state}'. Valid states: {', '.join(VALID_STATES)}"
        )

    with _queue_lock(queue_path):
        queue = load_queue(queue_path)

        if node_id not in queue["entries"]:
            raise ValueError(f"Node {node_id} is not in the queue")

        entry = queue["entries"][node_id]
        old_state = entry["queue_state"]
        allowed = VALID_TRANSITIONS.get(old_state, [])

        if new_state not in allowed:
            allowed_str = ", ".join(allowed) if allowed else "(none -- terminal state)"
            raise ValueError(
                f"Invalid transition '{old_state}' -> '{new_state}'. "
                f"Allowed from '{old_state}': {allowed_str}"
            )

        entry["queue_state"] = new_state
        entry["last_activity"] = _now()
        save_queue(queue_path, queue)


def set_weakest_link(queue_path: Path, node_id: str, weakest_link: str) -> None:
    """Set the weakest link for a node. Uses lock."""
    with _queue_lock(queue_path):
        queue = load_queue(queue_path)

        if node_id not in queue["entries"]:
            raise ValueError(f"Node {node_id} is not in the queue")

        queue["entries"][node_id]["weakest_link"] = weakest_link
        queue["entries"][node_id]["last_activity"] = _now()
        save_queue(queue_path, queue)


def set_focused_directive(queue_path: Path, node_id: str, directive: str) -> None:
    """Set focused directive for a node. Uses lock."""
    with _queue_lock(queue_path):
        queue = load_queue(queue_path)

        if node_id not in queue["entries"]:
            raise ValueError(f"Node {node_id} is not in the queue")

        queue["entries"][node_id]["focused_directive"] = directive
        queue["entries"][node_id]["last_activity"] = _now()
        save_queue(queue_path, queue)


def set_iteration(queue_path: Path, node_id: str, iteration: int) -> None:
    """Set iteration counter for a node. Uses lock."""
    with _queue_lock(queue_path):
        queue = load_queue(queue_path)

        if node_id not in queue["entries"]:
            raise ValueError(f"Node {node_id} is not in the queue")

        queue["entries"][node_id]["iteration"] = iteration
        queue["entries"][node_id]["last_activity"] = _now()
        save_queue(queue_path, queue)


def new_cycle(queue_path: Path, node_id: str) -> None:
    """Advance to new cycle.

    Increments iteration, sets ``cycle_start_iteration`` to the new value,
    clears ``weakest_link`` and ``focused_directive``.  Uses lock.
    """
    with _queue_lock(queue_path):
        queue = load_queue(queue_path)

        if node_id not in queue["entries"]:
            raise ValueError(f"Node {node_id} is not in the queue")

        entry = queue["entries"][node_id]
        new_iter = entry["iteration"] + 1
        entry["iteration"] = new_iter
        entry["cycle_start_iteration"] = new_iter
        entry["weakest_link"] = None
        entry["focused_directive"] = None
        entry["last_activity"] = _now()
        save_queue(queue_path, queue)


def get_next(queue_path: Path) -> dict | None:
    """Get the next node to process.

    Returns the first entry in a resumable state (queued, socratic_active,
    research_active, critique_active, mediating).  No lock needed (read-only).
    """
    queue = load_queue(queue_path)

    for _node_id, entry in queue["entries"].items():
        if entry.get("queue_state") in _RESUMABLE_STATES:
            return entry

    return None


def list_queue(queue_path: Path) -> list[dict]:
    """Return all queue entries as a list of dicts. No lock needed (read-only)."""
    queue = load_queue(queue_path)
    return list(queue["entries"].values())


def remove_from_queue(queue_path: Path, node_id: str) -> None:
    """Remove a node from the queue. Uses lock."""
    with _queue_lock(queue_path):
        queue = load_queue(queue_path)

        if node_id not in queue["entries"]:
            raise ValueError(f"Node {node_id} is not in the queue")

        del queue["entries"][node_id]
        save_queue(queue_path, queue)


def sync_check(queue_path: Path, nodes_dir: Path) -> dict:
    """Compare queue state against node metadata files.

    Reads each active queue entry and compares its ``queue_state`` against the
    ``status`` field in the node's ``metadata.json`` file.  Returns a dict with:

    - **synced** (``bool``): True if all entries are consistent.
    - **conflicts** (``list[dict]``): Entries with discrepancies.
    - **reconciled** (``list[dict]``): Entries that were auto-reconciled.

    Discrepancies are flagged as ``sync_conflict``.  Uses lock.
    """
    with _queue_lock(queue_path):
        queue = load_queue(queue_path)
        conflicts: list[dict] = []
        reconciled: list[dict] = []

        for node_id, entry in queue["entries"].items():
            # Skip terminal states -- their metadata may diverge legitimately.
            if entry["queue_state"] in ("done", "deferred_closed"):
                continue

            # Derive node directory from composite key.
            # Key format: {dish}-{colony}-{level}-{seq}
            # Filesystem: nodes_dir/{colony}/{level}-{seq}/metadata.json
            parts = node_id.rsplit("-", 2)
            if len(parts) < 3:
                conflicts.append(
                    {
                        "node_id": node_id,
                        "issue": "invalid_key_format",
                        "details": "Cannot parse composite key into colony/level/seq",
                    }
                )
                continue

            # parts[-2] is level (3 digits), parts[-1] is seq (3 digits)
            # Everything before that is the colony prefix (dish-colony)
            colony_prefix = parts[0]
            level_str = parts[1]
            seq_str = parts[2]

            metadata_path = (
                nodes_dir / colony_prefix / f"{level_str}-{seq_str}" / "metadata.json"
            )

            if not metadata_path.exists():
                conflicts.append(
                    {
                        "node_id": node_id,
                        "issue": "metadata_not_found",
                        "details": f"No metadata.json at {metadata_path}",
                    }
                )
                entry["queue_state"] = "sync_conflict"
                entry["last_activity"] = _now()
                continue

            try:
                with open(metadata_path) as f:
                    metadata = json.load(f)
            except (json.JSONDecodeError, OSError) as exc:
                conflicts.append(
                    {
                        "node_id": node_id,
                        "issue": "metadata_unreadable",
                        "details": str(exc),
                    }
                )
                entry["queue_state"] = "sync_conflict"
                entry["last_activity"] = _now()
                continue

            file_status = metadata.get("status")
            if file_status is None:
                continue  # No status in metadata -- nothing to compare.

            # Map terminal NodeStatus values to queue expectations.
            terminal_statuses = {"VALIDATED", "DISPROVEN", "DEFER_CLOSED"}
            if file_status in terminal_statuses and entry["queue_state"] != "done":
                reconciled.append(
                    {
                        "node_id": node_id,
                        "old_queue_state": entry["queue_state"],
                        "new_queue_state": "done",
                        "file_status": file_status,
                    }
                )
                entry["queue_state"] = "done"
                entry["last_activity"] = _now()

        if conflicts or reconciled:
            save_queue(queue_path, queue)

    return {
        "synced": len(conflicts) == 0 and len(reconciled) == 0,
        "conflicts": conflicts,
        "reconciled": reconciled,
    }
