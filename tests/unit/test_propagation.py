"""Unit tests for petri/propagation.py."""

from __future__ import annotations

import json
from pathlib import Path


from petri.storage import database
from petri.graph.colony import ColonyGraph
from petri.storage.event_log import load_events
from petri.storage.database import SQLiteEventLog, get_cell
from petri.storage.paths import dish_db_path
from petri.storage.colonies import save_colony
from petri.models import CellStatus, Colony
from petri.lifecycle.propagation import get_impact_report, propagate_upward

from tests.conftest import make_cell, make_edge


def _resolve_cell_dir(petri_dir: Path, colony_name: str, cell_id: str) -> SQLiteEventLog:
    return SQLiteEventLog(dish_db_path(petri_dir), cell_id)


# ── reopen_cell ─────────────────────────────────────────────────────────


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
            events = load_events(cell_path)
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
        save_colony(dish_db_path(petri_dir), graph, colony_model)

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
        metadata_before = get_cell(cell_path.db_path, cell_path.cell_id)
        events_before = load_events(cell_path)
        cells_before = database.list_cells(dish_db_path(env["petri_dir"]))

        get_impact_report(
            env["petri_dir"], env["cell2"].id, env["graph"], env["dish_id"]
        )

        # Nothing changed
        assert get_cell(cell_path.db_path, cell_path.cell_id) == metadata_before
        assert load_events(cell_path) == events_before
        assert database.list_cells(dish_db_path(env["petri_dir"])) == cells_before
