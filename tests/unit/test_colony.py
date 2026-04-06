"""Comprehensive unit tests for petri/colony.py — ColonyGraph and serialization."""

from __future__ import annotations

import json

import pytest

from petri.colony import ColonyGraph, deserialize_colony, serialize_colony
from petri.models import Colony, Edge, Node, NodeStatus, build_node_key


# ── Helpers / fixtures ────────────────────────────────────────────────────


def _node(
    dish: str,
    colony: str,
    level: int,
    seq: int,
    claim: str = "",
    status: NodeStatus = NodeStatus.NEW,
    dependencies: list[str] | None = None,
) -> Node:
    """Shorthand factory for building a test Node."""
    nid = build_node_key(dish, colony, level, seq)
    colony_id = f"{dish}-{colony}"
    return Node(
        id=nid,
        colony_id=colony_id,
        claim_text=claim or f"claim-{level}-{seq}",
        level=level,
        status=status,
        dependencies=dependencies or [],
    )


def _edge(from_id: str, to_id: str, edge_type: str = "intra_colony") -> Edge:
    return Edge(from_node=from_id, to_node=to_id, edge_type=edge_type)


@pytest.fixture
def colony_graph() -> ColonyGraph:
    """Return an empty ColonyGraph with a standard colony id."""
    return ColonyGraph(colony_id="test-dish-colony")


@pytest.fixture
def realistic_graph() -> ColonyGraph:
    """Build the reference DAG described in the spec.

    Level 0: center (thesis)
      Level 1: premise1, premise2
        Level 2: sub1 (dep of premise1), sub2 (dep of premise1 AND premise2)
    """
    g = ColonyGraph(colony_id="test-dish-colony")

    center = _node("test-dish", "colony", 0, 0, "Central thesis")
    premise1 = _node("test-dish", "colony", 1, 1, "First premise")
    premise2 = _node("test-dish", "colony", 1, 2, "Second premise")
    sub1 = _node("test-dish", "colony", 2, 3, "Sub-premise of P1")
    sub2 = _node("test-dish", "colony", 2, 4, "Shared sub-premise")

    for n in [center, premise1, premise2, sub1, sub2]:
        g.add_node(n)

    # center depends on premise1, premise2
    g.add_edge(_edge(center.id, premise1.id))
    g.add_edge(_edge(center.id, premise2.id))
    # premise1 depends on sub1, sub2
    g.add_edge(_edge(premise1.id, sub1.id))
    g.add_edge(_edge(premise1.id, sub2.id))
    # premise2 depends on sub2 (shared)
    g.add_edge(_edge(premise2.id, sub2.id))

    return g


# ── ColonyGraph Basic Operations ─────────────────────────────────────────


class TestAddNode:
    def test_add_and_get(self, colony_graph: ColonyGraph) -> None:
        node = _node("test-dish", "colony", 0, 0)
        colony_graph.add_node(node)
        assert colony_graph.get_node(node.id) is node

    def test_duplicate_raises(self, colony_graph: ColonyGraph) -> None:
        node = _node("test-dish", "colony", 0, 0)
        colony_graph.add_node(node)
        with pytest.raises(ValueError, match="already exists"):
            colony_graph.add_node(node)


class TestRemoveNode:
    def test_remove_clears_node_and_edges(self, colony_graph: ColonyGraph) -> None:
        a = _node("test-dish", "colony", 0, 0)
        b = _node("test-dish", "colony", 1, 1)
        colony_graph.add_node(a)
        colony_graph.add_node(b)
        colony_graph.add_edge(_edge(a.id, b.id))

        colony_graph.remove_node(b.id)

        assert colony_graph.get_edges() == []
        with pytest.raises(KeyError):
            colony_graph.get_node(b.id)
        # a still exists
        assert colony_graph.get_node(a.id) is a

    def test_remove_missing_raises(self, colony_graph: ColonyGraph) -> None:
        with pytest.raises(KeyError, match="not found"):
            colony_graph.remove_node("nonexistent")


class TestGetNodes:
    def test_sorted_by_level_then_id(self, realistic_graph: ColonyGraph) -> None:
        nodes = realistic_graph.get_nodes()
        keys = [n.id for n in nodes]
        # Level 0 first, then level 1 (sorted by id), then level 2
        assert keys == [
            "test-dish-colony-000-000",
            "test-dish-colony-001-001",
            "test-dish-colony-001-002",
            "test-dish-colony-002-003",
            "test-dish-colony-002-004",
        ]


class TestGetEdges:
    def test_returns_all_edges(self, realistic_graph: ColonyGraph) -> None:
        edges = realistic_graph.get_edges()
        assert len(edges) == 5


# ── Edge Operations ──────────────────────────────────────────────────────


class TestAddEdge:
    def test_creates_directed_dependency(self, colony_graph: ColonyGraph) -> None:
        a = _node("test-dish", "colony", 0, 0)
        b = _node("test-dish", "colony", 1, 1)
        colony_graph.add_node(a)
        colony_graph.add_node(b)
        colony_graph.add_edge(_edge(a.id, b.id))

        assert colony_graph.get_dependencies(a.id) == [b.id]
        assert colony_graph.get_dependents(b.id) == [a.id]

    def test_short_cycle_raises(self, colony_graph: ColonyGraph) -> None:
        a = _node("test-dish", "colony", 0, 0)
        b = _node("test-dish", "colony", 1, 1)
        colony_graph.add_node(a)
        colony_graph.add_node(b)
        colony_graph.add_edge(_edge(a.id, b.id))

        with pytest.raises(ValueError, match="cycle"):
            colony_graph.add_edge(_edge(b.id, a.id))

    def test_long_cycle_raises(self, colony_graph: ColonyGraph) -> None:
        a = _node("test-dish", "colony", 0, 0)
        b = _node("test-dish", "colony", 1, 1)
        c = _node("test-dish", "colony", 2, 2)
        colony_graph.add_node(a)
        colony_graph.add_node(b)
        colony_graph.add_node(c)
        colony_graph.add_edge(_edge(a.id, b.id))
        colony_graph.add_edge(_edge(b.id, c.id))

        with pytest.raises(ValueError, match="cycle"):
            colony_graph.add_edge(_edge(c.id, a.id))

    def test_self_loop_raises(self, colony_graph: ColonyGraph) -> None:
        a = _node("test-dish", "colony", 0, 0)
        colony_graph.add_node(a)

        with pytest.raises(ValueError, match="cycle"):
            colony_graph.add_edge(_edge(a.id, a.id))

    def test_valid_acyclic_accepted(self, colony_graph: ColonyGraph) -> None:
        a = _node("test-dish", "colony", 0, 0)
        b = _node("test-dish", "colony", 1, 1)
        c = _node("test-dish", "colony", 2, 2)
        colony_graph.add_node(a)
        colony_graph.add_node(b)
        colony_graph.add_node(c)

        # Diamond: a -> b, a -> c, b -> c  (no cycles)
        colony_graph.add_edge(_edge(a.id, b.id))
        colony_graph.add_edge(_edge(a.id, c.id))
        colony_graph.add_edge(_edge(b.id, c.id))

        assert len(colony_graph.get_edges()) == 3


# ── Level Computation ────────────────────────────────────────────────────


class TestComputeLevels:
    def test_center_is_level_zero(self, realistic_graph: ColonyGraph) -> None:
        levels = realistic_graph.compute_levels("test-dish-colony-000-000")
        assert levels["test-dish-colony-000-000"] == 0

    def test_direct_deps_are_level_one(self, realistic_graph: ColonyGraph) -> None:
        levels = realistic_graph.compute_levels("test-dish-colony-000-000")
        assert levels["test-dish-colony-001-001"] == 1
        assert levels["test-dish-colony-001-002"] == 1

    def test_transitive_deps_are_level_two(self, realistic_graph: ColonyGraph) -> None:
        levels = realistic_graph.compute_levels("test-dish-colony-000-000")
        assert levels["test-dish-colony-002-003"] == 2
        assert levels["test-dish-colony-002-004"] == 2

    def test_three_plus_levels(self, colony_graph: ColonyGraph) -> None:
        nodes = [_node("test-dish", "colony", i, i) for i in range(4)]
        for n in nodes:
            colony_graph.add_node(n)
        # Chain: 0 -> 1 -> 2 -> 3
        for i in range(3):
            colony_graph.add_edge(_edge(nodes[i].id, nodes[i + 1].id))

        levels = colony_graph.compute_levels(nodes[0].id)
        assert levels[nodes[0].id] == 0
        assert levels[nodes[1].id] == 1
        assert levels[nodes[2].id] == 2
        assert levels[nodes[3].id] == 3

    def test_diamond_dependency(self, realistic_graph: ColonyGraph) -> None:
        """sub2 is reachable through both premise1 and premise2.

        BFS should assign level 2 because it is first reached at depth 2
        regardless of path.
        """
        levels = realistic_graph.compute_levels("test-dish-colony-000-000")
        assert levels["test-dish-colony-002-004"] == 2


# ── Cell Nodes ───────────────────────────────────────────────────────────


class TestCellNodes:
    def test_cells_have_no_dependencies(self, realistic_graph: ColonyGraph) -> None:
        cells = realistic_graph.get_cell_nodes()
        cell_ids = {n.id for n in cells}
        assert cell_ids == {
            "test-dish-colony-002-003",
            "test-dish-colony-002-004",
        }

    def test_non_cells_excluded(self, realistic_graph: ColonyGraph) -> None:
        cells = realistic_graph.get_cell_nodes()
        cell_ids = {n.id for n in cells}
        # center, premise1, premise2 all have deps
        assert "test-dish-colony-000-000" not in cell_ids
        assert "test-dish-colony-001-001" not in cell_ids
        assert "test-dish-colony-001-002" not in cell_ids

    def test_center_only_colony(self, colony_graph: ColonyGraph) -> None:
        center = _node("test-dish", "colony", 0, 0)
        colony_graph.add_node(center)
        cells = colony_graph.get_cell_nodes()
        assert [n.id for n in cells] == [center.id]


# ── Eligible for Validation ──────────────────────────────────────────────


class TestEligibleForValidation:
    def test_cell_new_is_eligible(self, realistic_graph: ColonyGraph) -> None:
        statuses = {n.id: NodeStatus.NEW for n in realistic_graph.get_nodes()}
        eligible = realistic_graph.get_eligible_for_validation(statuses)
        eligible_ids = {n.id for n in eligible}
        # Only the two cell nodes qualify
        assert eligible_ids == {
            "test-dish-colony-002-003",
            "test-dish-colony-002-004",
        }

    def test_all_deps_validated_makes_parent_eligible(
        self, realistic_graph: ColonyGraph
    ) -> None:
        statuses = {n.id: NodeStatus.NEW for n in realistic_graph.get_nodes()}
        # Validate both sub-premises
        statuses["test-dish-colony-002-003"] = NodeStatus.VALIDATED
        statuses["test-dish-colony-002-004"] = NodeStatus.VALIDATED

        eligible = realistic_graph.get_eligible_for_validation(statuses)
        eligible_ids = {n.id for n in eligible}
        # premise1 and premise2 should now be eligible
        assert "test-dish-colony-001-001" in eligible_ids
        assert "test-dish-colony-001-002" in eligible_ids

    def test_unresolved_deps_blocks(self, realistic_graph: ColonyGraph) -> None:
        statuses = {n.id: NodeStatus.NEW for n in realistic_graph.get_nodes()}
        # Only validate one sub-premise
        statuses["test-dish-colony-002-003"] = NodeStatus.VALIDATED

        eligible = realistic_graph.get_eligible_for_validation(statuses)
        eligible_ids = {n.id for n in eligible}
        # premise1 still has sub2 unresolved, premise2 has sub2 unresolved
        assert "test-dish-colony-001-001" not in eligible_ids
        assert "test-dish-colony-001-002" not in eligible_ids

    def test_non_new_excluded(self, realistic_graph: ColonyGraph) -> None:
        statuses = {n.id: NodeStatus.VALIDATED for n in realistic_graph.get_nodes()}
        eligible = realistic_graph.get_eligible_for_validation(statuses)
        assert eligible == []


# ── Shared Premises ──────────────────────────────────────────────────────


class TestSharedPremises:
    def test_matching_claims_returns_pairs(self) -> None:
        g1 = ColonyGraph(colony_id="dish-c1")
        g2 = ColonyGraph(colony_id="dish-c2")

        n1 = _node("dish", "c1", 1, 1, "Shared claim")
        n2 = _node("dish", "c2", 1, 1, "Shared claim")

        g1.add_node(n1)
        g2.add_node(n2)

        pairs = g1.find_shared_premises(g2)
        assert pairs == [(n1.id, n2.id)]

    def test_no_matches_returns_empty(self) -> None:
        g1 = ColonyGraph(colony_id="dish-c1")
        g2 = ColonyGraph(colony_id="dish-c2")

        g1.add_node(_node("dish", "c1", 1, 1, "Alpha"))
        g2.add_node(_node("dish", "c2", 1, 1, "Beta"))

        assert g1.find_shared_premises(g2) == []


# ── DAG Validation ───────────────────────────────────────────────────────


class TestDAGValidation:
    def test_valid_dag(self, realistic_graph: ColonyGraph) -> None:
        assert realistic_graph.validate_dag() is True

    def test_has_cycle_with_edge_detects_would_be_cycle(
        self, colony_graph: ColonyGraph
    ) -> None:
        a = _node("test-dish", "colony", 0, 0)
        b = _node("test-dish", "colony", 1, 1)
        colony_graph.add_node(a)
        colony_graph.add_node(b)
        colony_graph.add_edge(_edge(a.id, b.id))

        # b -> a would close a cycle
        assert colony_graph.has_cycle_with_edge(b.id, a.id) is True
        # a -> b already exists; adding it as a check is benign but not a cycle
        # because has_cycle_with_edge only checks reachability, not duplicate edges
        # The direction c -> a where c is new would not form a cycle:
        c = _node("test-dish", "colony", 2, 2)
        colony_graph.add_node(c)
        assert colony_graph.has_cycle_with_edge(c.id, a.id) is False


# ── Serialization / Deserialization ──────────────────────────────────────


class TestSerialization:
    @pytest.fixture
    def colony_model(self) -> Colony:
        return Colony(
            id="test-dish-colony",
            dish="test-dish",
            center_claim="Central thesis",
            center_node_id="test-dish-colony-000-000",
            created_at="2026-01-01T00:00:00Z",
        )

    @pytest.fixture
    def graph_with_deps(self) -> ColonyGraph:
        """Build realistic graph whose nodes carry dependency lists.

        serialize reads the graph; deserialize rebuilds edges from
        node.dependencies, so the dependency lists must be populated.
        """
        g = ColonyGraph(colony_id="test-dish-colony")

        center = _node(
            "test-dish", "colony", 0, 0, "Central thesis",
            dependencies=["test-dish-colony-001-001", "test-dish-colony-001-002"],
        )
        p1 = _node(
            "test-dish", "colony", 1, 1, "First premise",
            dependencies=["test-dish-colony-002-003", "test-dish-colony-002-004"],
        )
        p2 = _node(
            "test-dish", "colony", 1, 2, "Second premise",
            dependencies=["test-dish-colony-002-004"],
        )
        sub1 = _node("test-dish", "colony", 2, 3, "Sub-premise of P1")
        sub2 = _node("test-dish", "colony", 2, 4, "Shared sub-premise")

        for n in [center, p1, p2, sub1, sub2]:
            g.add_node(n)

        g.add_edge(_edge(center.id, p1.id))
        g.add_edge(_edge(center.id, p2.id))
        g.add_edge(_edge(p1.id, sub1.id))
        g.add_edge(_edge(p1.id, sub2.id))
        g.add_edge(_edge(p2.id, sub2.id))

        return g

    def test_creates_expected_structure(
        self,
        tmp_path,
        graph_with_deps: ColonyGraph,
        colony_model: Colony,
    ) -> None:
        base = tmp_path / "colony"
        serialize_colony(graph_with_deps, colony_model, base)

        assert (base / "colony.json").exists()

        # The new layout nests nodes under level dirs; verify all 5 nodes
        # have the required files by scanning with rglob.
        metadata_files = sorted(base.rglob("metadata.json"))
        assert len(metadata_files) == 5
        for mf in metadata_files:
            node_dir = mf.parent
            assert (node_dir / "events.jsonl").exists()
            assert (node_dir / "evidence.md").exists()

        # node_paths in colony.json maps every node id to its relative dir
        colony_data = json.loads((base / "colony.json").read_text())
        assert len(colony_data["node_paths"]) == 5

    def test_node_gets_metadata_events_evidence(
        self,
        tmp_path,
        graph_with_deps: ColonyGraph,
        colony_model: Colony,
    ) -> None:
        base = tmp_path / "colony"
        serialize_colony(graph_with_deps, colony_model, base)

        # Resolve the center node's directory via node_paths
        colony_data = json.loads((base / "colony.json").read_text())
        center_rel = colony_data["node_paths"]["test-dish-colony-000-000"]
        center_dir = base / center_rel

        meta = json.loads((center_dir / "metadata.json").read_text())
        assert meta["id"] == "test-dish-colony-000-000"
        assert meta["claim_text"] == "Central thesis"

        events = (center_dir / "events.jsonl").read_text()
        assert events == ""

        evidence = (center_dir / "evidence.md").read_text()
        assert "Central thesis" in evidence

    def test_deserialize_reconstructs_graph(
        self,
        tmp_path,
        graph_with_deps: ColonyGraph,
        colony_model: Colony,
    ) -> None:
        base = tmp_path / "colony"
        serialize_colony(graph_with_deps, colony_model, base)

        loaded_graph, loaded_colony = deserialize_colony(base, "test-dish")

        assert loaded_colony.id == colony_model.id
        assert loaded_colony.center_claim == colony_model.center_claim

        loaded_ids = {n.id for n in loaded_graph.get_nodes()}
        original_ids = {n.id for n in graph_with_deps.get_nodes()}
        assert loaded_ids == original_ids

    def test_roundtrip_preserves_edges(
        self,
        tmp_path,
        graph_with_deps: ColonyGraph,
        colony_model: Colony,
    ) -> None:
        base = tmp_path / "colony"
        serialize_colony(graph_with_deps, colony_model, base)
        loaded_graph, _ = deserialize_colony(base, "test-dish")

        original_edge_pairs = sorted(
            (e.from_node, e.to_node) for e in graph_with_deps.get_edges()
        )
        loaded_edge_pairs = sorted(
            (e.from_node, e.to_node) for e in loaded_graph.get_edges()
        )
        assert loaded_edge_pairs == original_edge_pairs
