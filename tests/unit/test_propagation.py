"""Unit tests for petri/propagation.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from petri.graph.colony import ColonyGraph, serialize_colony
from petri.storage.event_log import append_event, load_events
from petri.models import CellStatus, Colony, build_cell_key
from petri.engine.propagation import get_impact_report, propagate_upward, reopen_cell
from petri.storage.queue import add_to_queue, load_queue

from tests.conftest import make_cell, make_edge


def _resolve_cell_dir(petri_dir: Path, colony_name: str, cell_id: str) -> Path:
    """Resolve a cell's directory using colony.json cell_paths."""
    colony_root = petri_dir / "petri-dishes" / colony_name
    colony_data = json.loads((colony_root / "colony.json").read_text())
    rel = colony_data["cell_paths"][cell_id]
    return colony_root / rel


# ── reopen_cell ─────────────────────────────────────────────────────────


class TestReopenCell:
    def test_reopens_validated_cell(self, petri_env_with_colony):
        env = petri_env_with_colony
        result = reopen_cell(
            env["petri_dir"], env["cell2"].id, trigger="New paper published"
        )

        assert result["cell_id"] == env["cell2"].id
        assert result["prior_status"] == "VALIDATED"
        assert result["new_status"] == "NEW"

    def test_updates_metadata_to_new(self, petri_env_with_colony):
        env = petri_env_with_colony
        reopen_cell(env["petri_dir"], env["cell2"].id, trigger="test")

        cell_path = _resolve_cell_dir(
            env["petri_dir"], env["colony_name"], env["cell2"].id
        )
        metadata = json.loads((cell_path / "metadata.json").read_text())
        assert metadata["status"] == "NEW"

    def test_logs_cell_reopened_event(self, petri_env_with_colony):
        env = petri_env_with_colony
        reopen_cell(
            env["petri_dir"], env["cell2"].id, trigger="Contradicting evidence"
        )

        cell_path = _resolve_cell_dir(
            env["petri_dir"], env["colony_name"], env["cell2"].id
        )
        events = load_events(cell_path / "events.jsonl")
        reopened_events = [event for event in events if event["type"] == "cell_reopened"]
        assert len(reopened_events) == 1
        assert reopened_events[0]["data"]["trigger"] == "Contradicting evidence"
        assert reopened_events[0]["data"]["prior_status"] == "VALIDATED"

    def test_adds_cell_back_to_queue(self, petri_env_with_colony):
        env = petri_env_with_colony
        reopen_cell(env["petri_dir"], env["cell2"].id, trigger="test")

        queue = load_queue(env["petri_dir"] / "queue.json")
        assert env["cell2"].id in queue["entries"]
        assert queue["entries"][env["cell2"].id]["queue_state"] == "queued"

    def test_preserves_prior_evidence(self, petri_env_with_colony):
        """Re-opening must NOT delete existing events (append-only)."""
        env = petri_env_with_colony
        cell_path = _resolve_cell_dir(
            env["petri_dir"], env["colony_name"], env["cell2"].id
        )
        events_path = cell_path / "events.jsonl"

        # Add some prior evidence events
        append_event(
            events_path,
            cell_id=env["cell2"].id,
            event_type="source_reviewed",
            agent="investigator",
            iteration=1,
            data={"url": "https://example.com", "title": "Prior evidence"},
        )
        append_event(
            events_path,
            cell_id=env["cell2"].id,
            event_type="verdict_issued",
            agent="investigator",
            iteration=1,
            data={"verdict": "EVIDENCE_SUFFICIENT", "summary": "Looks good"},
        )

        events_before = load_events(events_path)
        count_before = len(events_before)

        reopen_cell(env["petri_dir"], env["cell2"].id, trigger="new data")

        events_after = load_events(events_path)
        # Should have all prior events PLUS the cell_reopened event
        assert len(events_after) == count_before + 1
        # Prior events still present
        types = [event["type"] for event in events_after]
        assert "source_reviewed" in types
        assert "verdict_issued" in types
        assert "cell_reopened" in types

    def test_reopens_disproven_cell(self, petri_env_with_colony):
        env = petri_env_with_colony
        cell_path = _resolve_cell_dir(
            env["petri_dir"], env["colony_name"], env["premise1"].id
        )
        metadata = json.loads((cell_path / "metadata.json").read_text())
        metadata["status"] = "DISPROVEN"
        (cell_path / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )

        result = reopen_cell(env["petri_dir"], env["premise1"].id, trigger="test")
        assert result["prior_status"] == "DISPROVEN"
        assert result["new_status"] == "NEW"

    def test_reopens_defer_open_cell(self, petri_env_with_colony):
        env = petri_env_with_colony
        cell_path = _resolve_cell_dir(
            env["petri_dir"], env["colony_name"], env["premise1"].id
        )
        metadata = json.loads((cell_path / "metadata.json").read_text())
        metadata["status"] = "DEFER_OPEN"
        (cell_path / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )

        result = reopen_cell(env["petri_dir"], env["premise1"].id, trigger="test")
        assert result["prior_status"] == "DEFER_OPEN"
        assert result["new_status"] == "NEW"

    def test_rejects_new_status(self, petri_env_with_colony):
        env = petri_env_with_colony
        with pytest.raises(ValueError, match="not re-openable"):
            reopen_cell(env["petri_dir"], env["center"].id, trigger="test")

    def test_rejects_research_status(self, petri_env_with_colony):
        env = petri_env_with_colony
        cell_path = _resolve_cell_dir(
            env["petri_dir"], env["colony_name"], env["cell2"].id
        )
        metadata = json.loads((cell_path / "metadata.json").read_text())
        metadata["status"] = "RESEARCH"
        (cell_path / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )

        with pytest.raises(ValueError, match="not re-openable"):
            reopen_cell(env["petri_dir"], env["cell2"].id, trigger="test")

    def test_rejects_stalled_status(self, petri_env_with_colony):
        env = petri_env_with_colony
        cell_path = _resolve_cell_dir(
            env["petri_dir"], env["colony_name"], env["cell2"].id
        )
        metadata = json.loads((cell_path / "metadata.json").read_text())
        metadata["status"] = "STALLED"
        (cell_path / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )

        with pytest.raises(ValueError, match="not re-openable"):
            reopen_cell(env["petri_dir"], env["cell2"].id, trigger="test")

    def test_handles_cell_already_in_queue(self, petri_env_with_colony):
        """If cell is already queued, reopen should still succeed."""
        env = petri_env_with_colony
        queue_path = env["petri_dir"] / "queue.json"
        add_to_queue(queue_path, env["cell2"].id)

        # Should not raise — removes old entry and re-adds
        result = reopen_cell(env["petri_dir"], env["cell2"].id, trigger="test")
        assert result["new_status"] == "NEW"


# ── propagate_upward ────────────────────────────────────────────────────


class TestPropagateUpward:
    def test_flags_direct_dependents(self, petri_env_with_colony):
        env = petri_env_with_colony
        flagged = propagate_upward(
            env["petri_dir"], env["cell2"].id, env["graph"], env["dish_id"]
        )

        # cell2 is depended on by premise1 and premise2
        assert env["premise1"].id in flagged
        assert env["premise2"].id in flagged

    def test_propagates_transitively_to_center(self, petri_env_with_colony):
        env = petri_env_with_colony
        flagged = propagate_upward(
            env["petri_dir"], env["cell2"].id, env["graph"], env["dish_id"]
        )

        # Should propagate all the way up to center
        assert env["center"].id in flagged

    def test_full_chain_length(self, petri_env_with_colony):
        env = petri_env_with_colony
        flagged = propagate_upward(
            env["petri_dir"], env["cell2"].id, env["graph"], env["dish_id"]
        )

        # cell2 -> premise1, premise2 -> center = 3 flagged cells
        assert len(flagged) == 3

    def test_logs_propagation_events(self, petri_env_with_colony):
        env = petri_env_with_colony
        propagate_upward(
            env["petri_dir"], env["cell2"].id, env["graph"], env["dish_id"]
        )

        # Check that each flagged cell has a propagation_triggered event
        for cell_id in [env["premise1"].id, env["premise2"].id, env["center"].id]:
            cell_path = _resolve_cell_dir(
                env["petri_dir"], env["colony_name"], cell_id
            )
            events = load_events(cell_path / "events.jsonl")
            prop_events = [
                event for event in events if event["type"] == "propagation_triggered"
            ]
            assert len(prop_events) >= 1, f"No propagation event for {cell_id}"
            assert prop_events[0]["data"]["reopened_cell_id"] == env["cell2"].id

    def test_deduplicates_flagged_cells(self, petri_env_with_colony):
        env = petri_env_with_colony
        flagged = propagate_upward(
            env["petri_dir"], env["cell2"].id, env["graph"], env["dish_id"]
        )

        # No duplicates
        assert len(flagged) == len(set(flagged))

    def test_does_not_flag_reopened_cell_itself(self, petri_env_with_colony):
        env = petri_env_with_colony
        flagged = propagate_upward(
            env["petri_dir"], env["cell2"].id, env["graph"], env["dish_id"]
        )

        assert env["cell2"].id not in flagged

    def test_empty_propagation_for_center(self, petri_env_with_colony):
        """Center has no dependents, so propagation should return empty."""
        env = petri_env_with_colony
        flagged = propagate_upward(
            env["petri_dir"], env["center"].id, env["graph"], env["dish_id"]
        )
        assert flagged == []

    def test_linear_chain_propagation(self, tmp_path):
        """Test propagation through a simple linear chain: A -> B -> C."""
        petri_dir = tmp_path / ".petri"
        petri_dir.mkdir()
        (petri_dir / "petri-dishes").mkdir()
        (petri_dir / "petri.yaml").write_text("name: d\n")
        queue_data = {"version": 1, "last_updated": None, "entries": {}}
        (petri_dir / "queue.json").write_text(json.dumps(queue_data) + "\n")

        graph = ColonyGraph(colony_id="d-c")
        cell_a = make_cell("d", "c", 0, 0, "A", status=CellStatus.NEW, dependencies=["d-c-001-001"])
        cell_b = make_cell("d", "c", 1, 1, "B", status=CellStatus.VALIDATED, dependencies=["d-c-002-002"])
        cell_c = make_cell("d", "c", 2, 2, "C", status=CellStatus.VALIDATED)

        for cell in [cell_a, cell_b, cell_c]:
            graph.add_cell(cell)
        graph.add_edge(make_edge(cell_a.id, cell_b.id))
        graph.add_edge(make_edge(cell_b.id, cell_c.id))

        colony_model = Colony(
            id="d-c", dish="d", center_claim="A",
            center_cell_id=cell_a.id, created_at="2026-01-01T00:00:00Z",
        )
        serialize_colony(graph, colony_model, petri_dir / "petri-dishes" / "c")

        flagged = propagate_upward(petri_dir, cell_c.id, graph, "d")
        assert cell_b.id in flagged
        assert cell_a.id in flagged
        assert len(flagged) == 2


# ── get_impact_report ───────────────────────────────────────────────────


class TestGetImpactReport:
    def test_reports_all_affected_cells(self, petri_env_with_colony):
        env = petri_env_with_colony
        report = get_impact_report(
            env["petri_dir"], env["cell2"].id, env["graph"], env["dish_id"]
        )

        assert report["reopened_cell"] == env["cell2"].id
        assert report["total_affected"] == 3

        affected_ids = {affected["cell_id"] for affected in report["affected_cells"]}
        assert env["premise1"].id in affected_ids
        assert env["premise2"].id in affected_ids
        assert env["center"].id in affected_ids

    def test_includes_cell_metadata(self, petri_env_with_colony):
        env = petri_env_with_colony
        report = get_impact_report(
            env["petri_dir"], env["cell2"].id, env["graph"], env["dish_id"]
        )

        for affected in report["affected_cells"]:
            assert "cell_id" in affected
            assert "level" in affected
            assert "status" in affected
            assert "claim_text" in affected

    def test_sorted_by_level(self, petri_env_with_colony):
        env = petri_env_with_colony
        report = get_impact_report(
            env["petri_dir"], env["cell2"].id, env["graph"], env["dish_id"]
        )

        levels = [affected["level"] for affected in report["affected_cells"]]
        assert levels == sorted(levels)

    def test_center_has_no_affected(self, petri_env_with_colony):
        env = petri_env_with_colony
        report = get_impact_report(
            env["petri_dir"], env["center"].id, env["graph"], env["dish_id"]
        )
        assert report["total_affected"] == 0
        assert report["affected_cells"] == []

    def test_is_read_only(self, petri_env_with_colony):
        """Impact report must not modify any state."""
        env = petri_env_with_colony
        cell_path = _resolve_cell_dir(
            env["petri_dir"], env["colony_name"], env["cell2"].id
        )
        metadata_before = (cell_path / "metadata.json").read_text()
        events_before = (cell_path / "events.jsonl").read_text()
        queue_before = (env["petri_dir"] / "queue.json").read_text()

        get_impact_report(
            env["petri_dir"], env["cell2"].id, env["graph"], env["dish_id"]
        )

        # Nothing changed
        assert (cell_path / "metadata.json").read_text() == metadata_before
        assert (cell_path / "events.jsonl").read_text() == events_before
        assert (env["petri_dir"] / "queue.json").read_text() == queue_before
