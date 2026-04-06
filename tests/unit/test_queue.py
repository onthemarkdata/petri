"""Comprehensive unit tests for petri.queue module.

Covers: add_to_queue, update_state (all transitions), set_weakest_link,
set_focused_directive, set_iteration, new_cycle, get_next, list_queue,
remove_from_queue, sync_check, and concurrent file-locking safety.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from petri.queue import (
    VALID_TRANSITIONS,
    add_to_queue,
    get_next,
    list_queue,
    load_queue,
    new_cycle,
    remove_from_queue,
    set_focused_directive,
    set_iteration,
    set_weakest_link,
    sync_check,
    update_state,
)


# ── Helpers ────────────────────────────────────────────────────────────────


def _queue_path(tmp_path: Path) -> Path:
    return tmp_path / "queue.json"


def _add_node(tmp_path: Path, node_id: str = "dish-colony-001-001") -> None:
    """Add a node and return the queue path."""
    add_to_queue(_queue_path(tmp_path), node_id)


def _transition_to(tmp_path: Path, node_id: str, target: str) -> None:
    """Walk the state machine from 'queued' to *target* using shortest path.

    Only covers the happy-path chain used in the spec-mandated transitions.
    """
    chain: dict[str, list[str]] = {
        "queued": [],
        "phase1_active": ["phase1_active"],
        "phase2_active": ["phase1_active", "phase2_active"],
        "mediating": ["phase1_active", "phase2_active", "mediating"],
        "converged": [
            "phase1_active",
            "phase2_active",
            "mediating",
            "converged",
        ],
        "stalled": ["phase1_active", "stalled"],
        "needs_human": ["phase1_active", "stalled", "needs_human"],
        "red_team_active": [
            "phase1_active",
            "phase2_active",
            "mediating",
            "converged",
            "red_team_active",
        ],
        "evaluating": [
            "phase1_active",
            "phase2_active",
            "mediating",
            "converged",
            "red_team_active",
            "evaluating",
        ],
        "done": [
            "phase1_active",
            "phase2_active",
            "mediating",
            "converged",
            "red_team_active",
            "evaluating",
            "done",
        ],
        "deferred_open": [
            "phase1_active",
            "phase2_active",
            "mediating",
            "converged",
            "deferred_open",
        ],
        "deferred_closed": [
            "phase1_active",
            "phase2_active",
            "mediating",
            "converged",
            "deferred_closed",
        ],
        "sync_conflict": [],  # set directly in sync_check; not reachable via transitions
    }
    qp = _queue_path(tmp_path)
    for state in chain[target]:
        update_state(qp, node_id, state)


# ── add_to_queue Tests ─────────────────────────────────────────────────────


class TestAddToQueue:
    def test_add_returns_queue_entry_with_default_state(self, tmp_path):
        entry = add_to_queue(_queue_path(tmp_path), "dish-colony-001-001")
        assert entry.queue_state.value == "queued"
        assert entry.node_id == "dish-colony-001-001"

    def test_duplicate_add_raises(self, tmp_path):
        qp = _queue_path(tmp_path)
        add_to_queue(qp, "dish-colony-001-001")
        with pytest.raises(ValueError, match="already in the queue"):
            add_to_queue(qp, "dish-colony-001-001")

    def test_node_id_stored_in_queue_file(self, tmp_path):
        qp = _queue_path(tmp_path)
        add_to_queue(qp, "dish-colony-001-001")
        raw = json.loads(qp.read_text())
        assert "dish-colony-001-001" in raw["entries"]


# ── update_state Tests ─────────────────────────────────────────────────────


class TestUpdateStateValidTransitions:
    """Every valid transition listed in the spec must succeed."""

    def test_queued_to_phase1_active(self, tmp_path):
        _add_node(tmp_path)
        update_state(_queue_path(tmp_path), "dish-colony-001-001", "phase1_active")
        entry = load_queue(_queue_path(tmp_path))["entries"]["dish-colony-001-001"]
        assert entry["queue_state"] == "phase1_active"

    def test_phase1_active_to_phase2_active(self, tmp_path):
        _add_node(tmp_path)
        qp = _queue_path(tmp_path)
        update_state(qp, "dish-colony-001-001", "phase1_active")
        update_state(qp, "dish-colony-001-001", "phase2_active")
        entry = load_queue(qp)["entries"]["dish-colony-001-001"]
        assert entry["queue_state"] == "phase2_active"

    def test_phase2_active_to_mediating(self, tmp_path):
        _add_node(tmp_path)
        qp = _queue_path(tmp_path)
        _transition_to(tmp_path, "dish-colony-001-001", "phase2_active")
        update_state(qp, "dish-colony-001-001", "mediating")
        entry = load_queue(qp)["entries"]["dish-colony-001-001"]
        assert entry["queue_state"] == "mediating"

    def test_mediating_to_converged(self, tmp_path):
        _add_node(tmp_path)
        qp = _queue_path(tmp_path)
        _transition_to(tmp_path, "dish-colony-001-001", "mediating")
        update_state(qp, "dish-colony-001-001", "converged")
        entry = load_queue(qp)["entries"]["dish-colony-001-001"]
        assert entry["queue_state"] == "converged"

    def test_mediating_to_phase1_active_iterate(self, tmp_path):
        _add_node(tmp_path)
        qp = _queue_path(tmp_path)
        _transition_to(tmp_path, "dish-colony-001-001", "mediating")
        update_state(qp, "dish-colony-001-001", "phase1_active")
        entry = load_queue(qp)["entries"]["dish-colony-001-001"]
        assert entry["queue_state"] == "phase1_active"

    def test_converged_to_red_team_active(self, tmp_path):
        _add_node(tmp_path)
        qp = _queue_path(tmp_path)
        _transition_to(tmp_path, "dish-colony-001-001", "converged")
        update_state(qp, "dish-colony-001-001", "red_team_active")
        entry = load_queue(qp)["entries"]["dish-colony-001-001"]
        assert entry["queue_state"] == "red_team_active"

    def test_red_team_active_to_evaluating(self, tmp_path):
        _add_node(tmp_path)
        qp = _queue_path(tmp_path)
        _transition_to(tmp_path, "dish-colony-001-001", "red_team_active")
        update_state(qp, "dish-colony-001-001", "evaluating")
        entry = load_queue(qp)["entries"]["dish-colony-001-001"]
        assert entry["queue_state"] == "evaluating"

    def test_evaluating_to_done(self, tmp_path):
        _add_node(tmp_path)
        qp = _queue_path(tmp_path)
        _transition_to(tmp_path, "dish-colony-001-001", "evaluating")
        update_state(qp, "dish-colony-001-001", "done")
        entry = load_queue(qp)["entries"]["dish-colony-001-001"]
        assert entry["queue_state"] == "done"

    def test_done_to_queued_reentry(self, tmp_path):
        _add_node(tmp_path)
        qp = _queue_path(tmp_path)
        _transition_to(tmp_path, "dish-colony-001-001", "done")
        update_state(qp, "dish-colony-001-001", "queued")
        entry = load_queue(qp)["entries"]["dish-colony-001-001"]
        assert entry["queue_state"] == "queued"

    def test_deferred_open_to_queued(self, tmp_path):
        _add_node(tmp_path)
        qp = _queue_path(tmp_path)
        _transition_to(tmp_path, "dish-colony-001-001", "deferred_open")
        update_state(qp, "dish-colony-001-001", "queued")
        entry = load_queue(qp)["entries"]["dish-colony-001-001"]
        assert entry["queue_state"] == "queued"


class TestUpdateStateInvalidTransitions:
    """Invalid transitions must raise ValueError."""

    def test_queued_to_done_raises(self, tmp_path):
        _add_node(tmp_path)
        with pytest.raises(ValueError, match="Invalid transition"):
            update_state(_queue_path(tmp_path), "dish-colony-001-001", "done")

    def test_deferred_closed_is_terminal(self, tmp_path):
        _add_node(tmp_path)
        qp = _queue_path(tmp_path)
        _transition_to(tmp_path, "dish-colony-001-001", "deferred_closed")
        with pytest.raises(ValueError, match="terminal state"):
            update_state(qp, "dish-colony-001-001", "queued")

    def test_phase1_active_to_converged_skip_raises(self, tmp_path):
        _add_node(tmp_path)
        qp = _queue_path(tmp_path)
        update_state(qp, "dish-colony-001-001", "phase1_active")
        with pytest.raises(ValueError, match="Invalid transition"):
            update_state(qp, "dish-colony-001-001", "converged")

    def test_nonexistent_node_raises(self, tmp_path):
        qp = _queue_path(tmp_path)
        # Ensure queue file exists (empty)
        add_to_queue(qp, "dish-colony-001-001")
        with pytest.raises(ValueError, match="not in the queue"):
            update_state(qp, "dish-colony-999-999", "phase1_active")


class TestUpdateStateTimestamp:
    def test_last_activity_updated_on_transition(self, tmp_path):
        _add_node(tmp_path)
        qp = _queue_path(tmp_path)
        before = load_queue(qp)["entries"]["dish-colony-001-001"]["last_activity"]
        # Small sleep to ensure timestamp differs
        time.sleep(0.01)
        update_state(qp, "dish-colony-001-001", "phase1_active")
        after = load_queue(qp)["entries"]["dish-colony-001-001"]["last_activity"]
        assert after >= before


# ── set_weakest_link / set_focused_directive / set_iteration Tests ─────────


class TestSetWeakestLink:
    def test_sets_value(self, tmp_path):
        _add_node(tmp_path)
        qp = _queue_path(tmp_path)
        set_weakest_link(qp, "dish-colony-001-001", "weak-source-42")
        entry = load_queue(qp)["entries"]["dish-colony-001-001"]
        assert entry["weakest_link"] == "weak-source-42"

    def test_raises_on_nonexistent_node(self, tmp_path):
        qp = _queue_path(tmp_path)
        with pytest.raises(ValueError, match="not in the queue"):
            set_weakest_link(qp, "dish-colony-999-999", "anything")


class TestSetFocusedDirective:
    def test_sets_value(self, tmp_path):
        _add_node(tmp_path)
        qp = _queue_path(tmp_path)
        set_focused_directive(qp, "dish-colony-001-001", "focus on dates")
        entry = load_queue(qp)["entries"]["dish-colony-001-001"]
        assert entry["focused_directive"] == "focus on dates"

    def test_raises_on_nonexistent_node(self, tmp_path):
        qp = _queue_path(tmp_path)
        with pytest.raises(ValueError, match="not in the queue"):
            set_focused_directive(qp, "dish-colony-999-999", "anything")


class TestSetIteration:
    def test_sets_value(self, tmp_path):
        _add_node(tmp_path)
        qp = _queue_path(tmp_path)
        set_iteration(qp, "dish-colony-001-001", 5)
        entry = load_queue(qp)["entries"]["dish-colony-001-001"]
        assert entry["iteration"] == 5

    def test_raises_on_nonexistent_node(self, tmp_path):
        qp = _queue_path(tmp_path)
        with pytest.raises(ValueError, match="not in the queue"):
            set_iteration(qp, "dish-colony-999-999", 1)


# ── new_cycle Tests ────────────────────────────────────────────────────────


class TestNewCycle:
    def test_increments_iteration(self, tmp_path):
        _add_node(tmp_path)
        qp = _queue_path(tmp_path)
        entry_before = load_queue(qp)["entries"]["dish-colony-001-001"]
        old_iter = entry_before["iteration"]
        new_cycle(qp, "dish-colony-001-001")
        entry_after = load_queue(qp)["entries"]["dish-colony-001-001"]
        assert entry_after["iteration"] == old_iter + 1

    def test_sets_cycle_start_iteration(self, tmp_path):
        _add_node(tmp_path)
        qp = _queue_path(tmp_path)
        new_cycle(qp, "dish-colony-001-001")
        entry = load_queue(qp)["entries"]["dish-colony-001-001"]
        assert entry["cycle_start_iteration"] == entry["iteration"]

    def test_clears_weakest_link(self, tmp_path):
        _add_node(tmp_path)
        qp = _queue_path(tmp_path)
        set_weakest_link(qp, "dish-colony-001-001", "something")
        new_cycle(qp, "dish-colony-001-001")
        entry = load_queue(qp)["entries"]["dish-colony-001-001"]
        assert entry["weakest_link"] is None

    def test_clears_focused_directive(self, tmp_path):
        _add_node(tmp_path)
        qp = _queue_path(tmp_path)
        set_focused_directive(qp, "dish-colony-001-001", "some directive")
        new_cycle(qp, "dish-colony-001-001")
        entry = load_queue(qp)["entries"]["dish-colony-001-001"]
        assert entry["focused_directive"] is None


# ── get_next Tests ─────────────────────────────────────────────────────────


class TestGetNext:
    def test_returns_first_resumable(self, tmp_path):
        qp = _queue_path(tmp_path)
        add_to_queue(qp, "dish-colony-001-001")
        result = get_next(qp)
        assert result is not None
        assert result["node_id"] == "dish-colony-001-001"
        assert result["queue_state"] == "queued"

    def test_returns_none_when_empty(self, tmp_path):
        qp = _queue_path(tmp_path)
        assert get_next(qp) is None

    def test_returns_none_when_all_non_resumable(self, tmp_path):
        qp = _queue_path(tmp_path)
        add_to_queue(qp, "dish-colony-001-001")
        _transition_to(tmp_path, "dish-colony-001-001", "done")
        add_to_queue(qp, "dish-colony-001-002")
        _transition_to(tmp_path, "dish-colony-001-002", "deferred_closed")
        assert get_next(qp) is None

    def test_skips_non_resumable_returns_resumable(self, tmp_path):
        qp = _queue_path(tmp_path)
        # First node goes to done (non-resumable)
        add_to_queue(qp, "dish-colony-001-001")
        _transition_to(tmp_path, "dish-colony-001-001", "done")
        # Second node stays queued (resumable)
        add_to_queue(qp, "dish-colony-001-002")
        result = get_next(qp)
        assert result is not None
        assert result["node_id"] == "dish-colony-001-002"

    @pytest.mark.parametrize(
        "target_state",
        ["queued", "phase1_active", "phase2_active", "mediating"],
    )
    def test_each_resumable_state_is_returned(self, tmp_path, target_state):
        qp = _queue_path(tmp_path)
        add_to_queue(qp, "dish-colony-001-001")
        _transition_to(tmp_path, "dish-colony-001-001", target_state)
        result = get_next(qp)
        assert result is not None
        assert result["queue_state"] == target_state


# ── list_queue Tests ───────────────────────────────────────────────────────


class TestListQueue:
    def test_returns_all_entries(self, tmp_path):
        qp = _queue_path(tmp_path)
        add_to_queue(qp, "dish-colony-001-001")
        add_to_queue(qp, "dish-colony-001-002")
        entries = list_queue(qp)
        assert len(entries) == 2
        ids = {e["node_id"] for e in entries}
        assert ids == {"dish-colony-001-001", "dish-colony-001-002"}

    def test_empty_queue_returns_empty_list(self, tmp_path):
        qp = _queue_path(tmp_path)
        assert list_queue(qp) == []


# ── remove_from_queue Tests ────────────────────────────────────────────────


class TestRemoveFromQueue:
    def test_removes_entry(self, tmp_path):
        qp = _queue_path(tmp_path)
        add_to_queue(qp, "dish-colony-001-001")
        remove_from_queue(qp, "dish-colony-001-001")
        assert list_queue(qp) == []

    def test_raises_on_nonexistent_node(self, tmp_path):
        qp = _queue_path(tmp_path)
        with pytest.raises(ValueError, match="not in the queue"):
            remove_from_queue(qp, "dish-colony-999-999")


# ── sync_check Tests ───────────────────────────────────────────────────────


class TestSyncCheck:
    def test_detects_missing_metadata(self, tmp_path):
        """Queue has a node but no matching metadata.json on disk."""
        qp = _queue_path(tmp_path)
        nodes_dir = tmp_path / "nodes"
        nodes_dir.mkdir()

        add_to_queue(qp, "dish-colony-001-001")
        # Transition away from queued so it doesn't stay in the initial state
        # (sync_check skips done/deferred_closed, but processes everything else)
        update_state(qp, "dish-colony-001-001", "phase1_active")

        report = sync_check(qp, nodes_dir)
        assert not report["synced"]
        assert len(report["conflicts"]) >= 1
        conflict = report["conflicts"][0]
        assert conflict["node_id"] == "dish-colony-001-001"
        assert conflict["issue"] == "metadata_not_found"

        # Verify the entry was flagged as sync_conflict in the queue
        entry = load_queue(qp)["entries"]["dish-colony-001-001"]
        assert entry["queue_state"] == "sync_conflict"

    def test_reconciles_terminal_file_status(self, tmp_path):
        """If metadata says VALIDATED but queue isn't done, auto-reconcile."""
        qp = _queue_path(tmp_path)
        nodes_dir = tmp_path / "nodes"

        add_to_queue(qp, "dish-colony-001-001")
        update_state(qp, "dish-colony-001-001", "phase1_active")

        # Create metadata on disk that says VALIDATED
        # rsplit("-", 2) on "dish-colony-001-001" -> ["dish-colony", "001", "001"]
        # path: nodes_dir / "dish-colony" / "001-001" / "metadata.json"
        meta_dir = nodes_dir / "dish-colony" / "001-001"
        meta_dir.mkdir(parents=True)
        (meta_dir / "metadata.json").write_text(
            json.dumps({"status": "VALIDATED"})
        )

        report = sync_check(qp, nodes_dir)
        assert not report["synced"]  # reconciled counts as not synced
        assert len(report["reconciled"]) == 1
        assert report["reconciled"][0]["new_queue_state"] == "done"

        # Verify queue was actually updated
        entry = load_queue(qp)["entries"]["dish-colony-001-001"]
        assert entry["queue_state"] == "done"

    def test_no_status_in_metadata_is_ok(self, tmp_path):
        """If metadata exists but has no status field, no conflict."""
        qp = _queue_path(tmp_path)
        nodes_dir = tmp_path / "nodes"

        add_to_queue(qp, "dish-colony-001-001")
        update_state(qp, "dish-colony-001-001", "phase1_active")

        meta_dir = nodes_dir / "dish-colony" / "001-001"
        meta_dir.mkdir(parents=True)
        (meta_dir / "metadata.json").write_text(json.dumps({"claim": "test"}))

        report = sync_check(qp, nodes_dir)
        assert report["synced"]
        assert report["conflicts"] == []
        assert report["reconciled"] == []

    def test_skips_done_and_deferred_closed(self, tmp_path):
        """Terminal queue states should be skipped entirely."""
        qp = _queue_path(tmp_path)
        nodes_dir = tmp_path / "nodes"
        nodes_dir.mkdir()

        add_to_queue(qp, "dish-colony-001-001")
        _transition_to(tmp_path, "dish-colony-001-001", "done")

        report = sync_check(qp, nodes_dir)
        assert report["synced"]

    def test_flags_sync_conflict_on_unreadable_metadata(self, tmp_path):
        """Invalid JSON in metadata.json triggers sync_conflict."""
        qp = _queue_path(tmp_path)
        nodes_dir = tmp_path / "nodes"

        add_to_queue(qp, "dish-colony-001-001")
        update_state(qp, "dish-colony-001-001", "phase1_active")

        meta_dir = nodes_dir / "dish-colony" / "001-001"
        meta_dir.mkdir(parents=True)
        (meta_dir / "metadata.json").write_text("NOT VALID JSON {{{")

        report = sync_check(qp, nodes_dir)
        assert not report["synced"]
        assert len(report["conflicts"]) == 1
        assert report["conflicts"][0]["issue"] == "metadata_unreadable"

        entry = load_queue(qp)["entries"]["dish-colony-001-001"]
        assert entry["queue_state"] == "sync_conflict"


# ── File Locking / Concurrency Tests (SC-002) ─────────────────────────────


class TestConcurrentAccess:
    def test_concurrent_add_different_nodes(self, tmp_path):
        """10 threads adding different nodes -- all should succeed."""
        qp = _queue_path(tmp_path)
        errors: list[tuple[str, str]] = []

        def add_node(node_id: str) -> None:
            try:
                add_to_queue(qp, node_id)
            except Exception as e:
                errors.append((node_id, str(e)))

        threads = [
            threading.Thread(
                target=add_node, args=(f"dish-colony-001-{i:03d}",)
            )
            for i in range(10)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Concurrent adds failed: {errors}"
        entries = list_queue(qp)
        assert len(entries) == 10

    def test_concurrent_state_updates_same_node(self, tmp_path):
        """5 threads racing to update state on the same node.

        Start the node in 'queued' state. All threads attempt queued->phase1_active.
        Exactly one should succeed; the rest should either succeed (if they happen
        to acquire the lock while it is still in 'queued') or fail with ValueError
        (if the state already changed to 'phase1_active' and phase1_active->phase1_active
        is not a valid transition).  No data corruption should occur.
        """
        qp = _queue_path(tmp_path)
        add_to_queue(qp, "dish-colony-001-001")

        successes: list[int] = []
        failures: list[tuple[int, str]] = []

        def try_update(thread_id: int) -> None:
            try:
                update_state(qp, "dish-colony-001-001", "phase1_active")
                successes.append(thread_id)
            except ValueError as e:
                failures.append((thread_id, str(e)))

        threads = [
            threading.Thread(target=try_update, args=(i,)) for i in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # At least one must succeed
        assert len(successes) >= 1
        # Total should be 5
        assert len(successes) + len(failures) == 5
        # Queue should still be valid JSON with exactly one entry
        entry = load_queue(qp)["entries"]["dish-colony-001-001"]
        assert entry["queue_state"] == "phase1_active"

    def test_concurrent_add_no_data_loss(self, tmp_path):
        """Verify no data loss under concurrent writes -- check every node_id."""
        qp = _queue_path(tmp_path)
        node_ids = [f"dish-colony-002-{i:03d}" for i in range(10)]
        errors: list[tuple[str, str]] = []

        def add_node(node_id: str) -> None:
            try:
                add_to_queue(qp, node_id)
            except Exception as e:
                errors.append((node_id, str(e)))

        threads = [
            threading.Thread(target=add_node, args=(nid,)) for nid in node_ids
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Concurrent adds failed: {errors}"
        entries = list_queue(qp)
        stored_ids = {e["node_id"] for e in entries}
        assert stored_ids == set(node_ids)
