"""Comprehensive unit tests for petri/propagation.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from petri.colony import ColonyGraph, serialize_colony
from petri.event_log import append_event, load_events
from petri.models import Colony, Edge, Node, NodeStatus, build_node_key
from petri.propagation import get_impact_report, propagate_upward, reopen_node
from petri.queue import add_to_queue, load_queue


def _resolve_node_dir(petri_dir: Path, colony_name: str, node_id: str) -> Path:
    """Resolve a node's directory using colony.json node_paths."""
    colony_dir = petri_dir / "petri-dishes" / colony_name
    colony_data = json.loads((colony_dir / "colony.json").read_text())
    rel = colony_data["node_paths"][node_id]
    return colony_dir / rel


# ── Helpers / fixtures ────────────────────────────────────────────────────


def _node(
    dish: str,
    colony: str,
    level: int,
    seq: int,
    claim: str = "",
    status: NodeStatus = NodeStatus.NEW,
    dependencies: list[str] | None = None,
    dependents: list[str] | None = None,
) -> Node:
    nid = build_node_key(dish, colony, level, seq)
    colony_id = f"{dish}-{colony}"
    return Node(
        id=nid,
        colony_id=colony_id,
        claim_text=claim or f"claim-{level}-{seq}",
        level=level,
        status=status,
        dependencies=dependencies or [],
        dependents=dependents or [],
    )


def _edge(from_id: str, to_id: str) -> Edge:
    return Edge(from_node=from_id, to_node=to_id, edge_type="intra_colony")


@pytest.fixture
def petri_env(tmp_path):
    """Create a full .petri environment with a colony for propagation tests.

    Colony structure:
        center (L0, 000-000) depends on premise1 (L1, 001-001) and premise2 (L1, 001-002)
        premise1 depends on cell1 (L2, 002-003)
        premise2 depends on cell1 (L2, 002-003) — shared dependency

    All cell/premise nodes start as VALIDATED so they can be re-opened.
    Center starts as NEW.
    """
    petri_dir = tmp_path / ".petri"
    petri_dir.mkdir()
    (petri_dir / "petri-dishes").mkdir()

    # petri.yaml
    (petri_dir / "petri.yaml").write_text(
        "name: test-dish\nmodel:\n  name: gemma-3-4b-it\n  provider: local\n"
        "harness: claude-code\nmax_iterations: 3\n"
    )

    # queue.json
    queue_data = {"version": 1, "last_updated": None, "entries": {}}
    (petri_dir / "queue.json").write_text(json.dumps(queue_data, indent=2) + "\n")

    # Build colony
    dish_id = "test-dish"
    colony_name = "colony"
    colony_id = f"{dish_id}-{colony_name}"

    center = _node(
        dish_id, colony_name, 0, 0, "Central thesis",
        status=NodeStatus.NEW,
        dependencies=[
            build_node_key(dish_id, colony_name, 1, 1),
            build_node_key(dish_id, colony_name, 1, 2),
        ],
    )
    premise1 = _node(
        dish_id, colony_name, 1, 1, "First premise",
        status=NodeStatus.VALIDATED,
        dependencies=[build_node_key(dish_id, colony_name, 2, 3)],
    )
    premise2 = _node(
        dish_id, colony_name, 1, 2, "Second premise",
        status=NodeStatus.VALIDATED,
        dependencies=[build_node_key(dish_id, colony_name, 2, 3)],
    )
    cell1 = _node(
        dish_id, colony_name, 2, 3, "Cell node",
        status=NodeStatus.VALIDATED,
    )

    graph = ColonyGraph(colony_id=colony_id)
    for n in [center, premise1, premise2, cell1]:
        graph.add_node(n)

    graph.add_edge(_edge(center.id, premise1.id))
    graph.add_edge(_edge(center.id, premise2.id))
    graph.add_edge(_edge(premise1.id, cell1.id))
    graph.add_edge(_edge(premise2.id, cell1.id))

    colony_model = Colony(
        id=colony_id,
        dish=dish_id,
        center_claim="Central thesis",
        center_node_id=center.id,
        created_at="2026-01-01T00:00:00Z",
    )

    colony_path = petri_dir / "petri-dishes" / colony_name
    serialize_colony(graph, colony_model, colony_path)

    return {
        "tmp_path": tmp_path,
        "petri_dir": petri_dir,
        "dish_id": dish_id,
        "colony_name": colony_name,
        "colony_id": colony_id,
        "graph": graph,
        "center": center,
        "premise1": premise1,
        "premise2": premise2,
        "cell1": cell1,
    }


# ── reopen_node ─────────────────────────────────────────────────────────


class TestReopenNode:
    def test_reopens_validated_node(self, petri_env):
        env = petri_env
        result = reopen_node(
            env["petri_dir"], env["cell1"].id, trigger="New paper published"
        )

        assert result["node_id"] == env["cell1"].id
        assert result["prior_status"] == "VALIDATED"
        assert result["new_status"] == "NEW"

    def test_updates_metadata_to_new(self, petri_env):
        env = petri_env
        reopen_node(env["petri_dir"], env["cell1"].id, trigger="test")

        # Read metadata back
        node_dir = _resolve_node_dir(
            env["petri_dir"], env["colony_name"], env["cell1"].id
        )
        metadata = json.loads((node_dir / "metadata.json").read_text())
        assert metadata["status"] == "NEW"

    def test_logs_node_reopened_event(self, petri_env):
        env = petri_env
        reopen_node(
            env["petri_dir"], env["cell1"].id, trigger="Contradicting evidence"
        )

        node_dir = _resolve_node_dir(
            env["petri_dir"], env["colony_name"], env["cell1"].id
        )
        events = load_events(node_dir / "events.jsonl")
        reopened_events = [e for e in events if e["type"] == "node_reopened"]
        assert len(reopened_events) == 1
        assert reopened_events[0]["data"]["trigger"] == "Contradicting evidence"
        assert reopened_events[0]["data"]["prior_status"] == "VALIDATED"

    def test_adds_node_back_to_queue(self, petri_env):
        env = petri_env
        reopen_node(env["petri_dir"], env["cell1"].id, trigger="test")

        queue = load_queue(env["petri_dir"] / "queue.json")
        assert env["cell1"].id in queue["entries"]
        assert queue["entries"][env["cell1"].id]["queue_state"] == "queued"

    def test_preserves_prior_evidence(self, petri_env):
        """Re-opening must NOT delete existing events (append-only)."""
        env = petri_env
        node_dir = _resolve_node_dir(
            env["petri_dir"], env["colony_name"], env["cell1"].id
        )
        events_path = node_dir / "events.jsonl"

        # Add some prior evidence events
        append_event(
            events_path,
            node_id=env["cell1"].id,
            event_type="source_reviewed",
            agent="investigator",
            iteration=1,
            data={"url": "https://example.com", "title": "Prior evidence"},
        )
        append_event(
            events_path,
            node_id=env["cell1"].id,
            event_type="verdict_issued",
            agent="investigator",
            iteration=1,
            data={"verdict": "EVIDENCE_SUFFICIENT", "summary": "Looks good"},
        )

        events_before = load_events(events_path)
        count_before = len(events_before)

        reopen_node(env["petri_dir"], env["cell1"].id, trigger="new data")

        events_after = load_events(events_path)
        # Should have all prior events PLUS the node_reopened event
        assert len(events_after) == count_before + 1
        # Prior events still present
        types = [e["type"] for e in events_after]
        assert "source_reviewed" in types
        assert "verdict_issued" in types
        assert "node_reopened" in types

    def test_reopens_disproven_node(self, petri_env):
        env = petri_env
        # Change premise1 to DISPROVEN
        node_dir = _resolve_node_dir(
            env["petri_dir"], env["colony_name"], env["premise1"].id
        )
        metadata = json.loads((node_dir / "metadata.json").read_text())
        metadata["status"] = "DISPROVEN"
        (node_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )

        result = reopen_node(env["petri_dir"], env["premise1"].id, trigger="test")
        assert result["prior_status"] == "DISPROVEN"
        assert result["new_status"] == "NEW"

    def test_reopens_defer_open_node(self, petri_env):
        env = petri_env
        node_dir = _resolve_node_dir(
            env["petri_dir"], env["colony_name"], env["premise1"].id
        )
        metadata = json.loads((node_dir / "metadata.json").read_text())
        metadata["status"] = "DEFER_OPEN"
        (node_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )

        result = reopen_node(env["petri_dir"], env["premise1"].id, trigger="test")
        assert result["prior_status"] == "DEFER_OPEN"
        assert result["new_status"] == "NEW"

    def test_rejects_new_status(self, petri_env):
        env = petri_env
        with pytest.raises(ValueError, match="not re-openable"):
            reopen_node(env["petri_dir"], env["center"].id, trigger="test")

    def test_rejects_research_status(self, petri_env):
        env = petri_env
        # Change cell to RESEARCH
        node_dir = _resolve_node_dir(
            env["petri_dir"], env["colony_name"], env["cell1"].id
        )
        metadata = json.loads((node_dir / "metadata.json").read_text())
        metadata["status"] = "RESEARCH"
        (node_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )

        with pytest.raises(ValueError, match="not re-openable"):
            reopen_node(env["petri_dir"], env["cell1"].id, trigger="test")

    def test_rejects_stalled_status(self, petri_env):
        env = petri_env
        node_dir = _resolve_node_dir(
            env["petri_dir"], env["colony_name"], env["cell1"].id
        )
        metadata = json.loads((node_dir / "metadata.json").read_text())
        metadata["status"] = "STALLED"
        (node_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )

        with pytest.raises(ValueError, match="not re-openable"):
            reopen_node(env["petri_dir"], env["cell1"].id, trigger="test")

    def test_handles_node_already_in_queue(self, petri_env):
        """If node is already queued, reopen should still succeed."""
        env = petri_env
        queue_path = env["petri_dir"] / "queue.json"
        add_to_queue(queue_path, env["cell1"].id)

        # Should not raise — removes old entry and re-adds
        result = reopen_node(env["petri_dir"], env["cell1"].id, trigger="test")
        assert result["new_status"] == "NEW"


# ── propagate_upward ────────────────────────────────────────────────────


class TestPropagateUpward:
    def test_flags_direct_dependents(self, petri_env):
        env = petri_env
        flagged = propagate_upward(
            env["petri_dir"], env["cell1"].id, env["graph"], env["dish_id"]
        )

        # cell1 is depended on by premise1 and premise2
        assert env["premise1"].id in flagged
        assert env["premise2"].id in flagged

    def test_propagates_transitively_to_center(self, petri_env):
        env = petri_env
        flagged = propagate_upward(
            env["petri_dir"], env["cell1"].id, env["graph"], env["dish_id"]
        )

        # Should propagate all the way up to center
        assert env["center"].id in flagged

    def test_full_chain_length(self, petri_env):
        env = petri_env
        flagged = propagate_upward(
            env["petri_dir"], env["cell1"].id, env["graph"], env["dish_id"]
        )

        # cell1 -> premise1, premise2 -> center = 3 flagged nodes
        assert len(flagged) == 3

    def test_logs_propagation_events(self, petri_env):
        env = petri_env
        propagate_upward(
            env["petri_dir"], env["cell1"].id, env["graph"], env["dish_id"]
        )

        # Check that each flagged node has a propagation_triggered event
        for node_id in [env["premise1"].id, env["premise2"].id, env["center"].id]:
            node_dir = _resolve_node_dir(
                env["petri_dir"], env["colony_name"], node_id
            )
            events = load_events(node_dir / "events.jsonl")
            prop_events = [
                e for e in events if e["type"] == "propagation_triggered"
            ]
            assert len(prop_events) >= 1, f"No propagation event for {node_id}"
            assert prop_events[0]["data"]["reopened_node_id"] == env["cell1"].id

    def test_deduplicates_flagged_nodes(self, petri_env):
        env = petri_env
        flagged = propagate_upward(
            env["petri_dir"], env["cell1"].id, env["graph"], env["dish_id"]
        )

        # No duplicates
        assert len(flagged) == len(set(flagged))

    def test_does_not_flag_reopened_node_itself(self, petri_env):
        env = petri_env
        flagged = propagate_upward(
            env["petri_dir"], env["cell1"].id, env["graph"], env["dish_id"]
        )

        assert env["cell1"].id not in flagged

    def test_empty_propagation_for_center(self, petri_env):
        """Center has no dependents, so propagation should return empty."""
        env = petri_env
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
        a = _node("d", "c", 0, 0, "A", status=NodeStatus.NEW, dependencies=["d-c-001-001"])
        b = _node("d", "c", 1, 1, "B", status=NodeStatus.VALIDATED, dependencies=["d-c-002-002"])
        c = _node("d", "c", 2, 2, "C", status=NodeStatus.VALIDATED)

        for n in [a, b, c]:
            graph.add_node(n)
        graph.add_edge(_edge(a.id, b.id))
        graph.add_edge(_edge(b.id, c.id))

        colony_model = Colony(
            id="d-c", dish="d", center_claim="A",
            center_node_id=a.id, created_at="2026-01-01T00:00:00Z",
        )
        serialize_colony(graph, colony_model, petri_dir / "petri-dishes" / "c")

        flagged = propagate_upward(petri_dir, c.id, graph, "d")
        # C -> B -> A
        assert b.id in flagged
        assert a.id in flagged
        assert len(flagged) == 2


# ── get_impact_report ───────────────────────────────────────────────────


class TestGetImpactReport:
    def test_reports_all_affected_nodes(self, petri_env):
        env = petri_env
        report = get_impact_report(
            env["petri_dir"], env["cell1"].id, env["graph"], env["dish_id"]
        )

        assert report["reopened_node"] == env["cell1"].id
        assert report["total_affected"] == 3

        affected_ids = {a["node_id"] for a in report["affected_nodes"]}
        assert env["premise1"].id in affected_ids
        assert env["premise2"].id in affected_ids
        assert env["center"].id in affected_ids

    def test_includes_node_metadata(self, petri_env):
        env = petri_env
        report = get_impact_report(
            env["petri_dir"], env["cell1"].id, env["graph"], env["dish_id"]
        )

        for affected in report["affected_nodes"]:
            assert "node_id" in affected
            assert "level" in affected
            assert "status" in affected
            assert "claim_text" in affected

    def test_sorted_by_level(self, petri_env):
        env = petri_env
        report = get_impact_report(
            env["petri_dir"], env["cell1"].id, env["graph"], env["dish_id"]
        )

        levels = [a["level"] for a in report["affected_nodes"]]
        assert levels == sorted(levels)

    def test_center_has_no_affected(self, petri_env):
        env = petri_env
        report = get_impact_report(
            env["petri_dir"], env["center"].id, env["graph"], env["dish_id"]
        )
        assert report["total_affected"] == 0
        assert report["affected_nodes"] == []

    def test_is_read_only(self, petri_env):
        """Impact report must not modify any state."""
        env = petri_env
        # Take a snapshot of metadata before
        node_dir = _resolve_node_dir(
            env["petri_dir"], env["colony_name"], env["cell1"].id
        )
        metadata_before = (node_dir / "metadata.json").read_text()
        events_before = (node_dir / "events.jsonl").read_text()
        queue_before = (env["petri_dir"] / "queue.json").read_text()

        get_impact_report(
            env["petri_dir"], env["cell1"].id, env["graph"], env["dish_id"]
        )

        # Nothing changed
        assert (node_dir / "metadata.json").read_text() == metadata_before
        assert (node_dir / "events.jsonl").read_text() == events_before
        assert (env["petri_dir"] / "queue.json").read_text() == queue_before
