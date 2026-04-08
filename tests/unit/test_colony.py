"""Unit tests for petri/colony.py — ColonyGraph and serialization."""

from __future__ import annotations

import json

import pytest

from petri.graph.colony import ColonyGraph, deserialize_colony, serialize_colony
from petri.models import CellStatus, Colony

from tests.conftest import make_cell, make_edge


# ── Local fixtures ───────────────────────────────────────────────────────


@pytest.fixture
def colony_graph() -> ColonyGraph:
    """Return an empty ColonyGraph with a standard colony id."""
    return ColonyGraph(colony_id="test-dish-colony")


# ── ColonyGraph Basic Operations ─────────────────────────────────────────


class TestAddCell:
    def test_add_and_get(self, colony_graph: ColonyGraph) -> None:
        cell = make_cell("test-dish", "colony", 0, 0)
        colony_graph.add_cell(cell)
        assert colony_graph.get_cell(cell.id) is cell

    def test_duplicate_raises(self, colony_graph: ColonyGraph) -> None:
        cell = make_cell("test-dish", "colony", 0, 0)
        colony_graph.add_cell(cell)
        with pytest.raises(ValueError, match="already exists"):
            colony_graph.add_cell(cell)


class TestRemoveCell:
    def test_remove_clears_cell_and_edges(self, colony_graph: ColonyGraph) -> None:
        cell_a = make_cell("test-dish", "colony", 0, 0)
        cell_b = make_cell("test-dish", "colony", 1, 1)
        colony_graph.add_cell(cell_a)
        colony_graph.add_cell(cell_b)
        colony_graph.add_edge(make_edge(cell_a.id, cell_b.id))

        colony_graph.remove_cell(cell_b.id)

        assert colony_graph.get_edges() == []
        with pytest.raises(KeyError):
            colony_graph.get_cell(cell_b.id)
        assert colony_graph.get_cell(cell_a.id) is cell_a

    def test_remove_missing_raises(self, colony_graph: ColonyGraph) -> None:
        with pytest.raises(KeyError, match="not found"):
            colony_graph.remove_cell("nonexistent")


class TestGetAllCells:
    def test_sorted_by_level_then_id(self, canonical_colony) -> None:
        graph = canonical_colony["graph"]
        cells = graph.get_all_cells()
        cell_ids = [cell.id for cell in cells]
        assert cell_ids == [
            "test-dish-colony-000-000",
            "test-dish-colony-001-001",
            "test-dish-colony-001-002",
            "test-dish-colony-002-003",
            "test-dish-colony-002-004",
        ]


class TestGetEdges:
    def test_returns_all_edges(self, canonical_colony) -> None:
        assert len(canonical_colony["graph"].get_edges()) == 5


# ── Edge Operations ──────────────────────────────────────────────────────


class TestAddEdge:
    def test_creates_directed_dependency(self, colony_graph: ColonyGraph) -> None:
        cell_a = make_cell("test-dish", "colony", 0, 0)
        cell_b = make_cell("test-dish", "colony", 1, 1)
        colony_graph.add_cell(cell_a)
        colony_graph.add_cell(cell_b)
        colony_graph.add_edge(make_edge(cell_a.id, cell_b.id))

        assert colony_graph.get_dependencies(cell_a.id) == [cell_b.id]
        assert colony_graph.get_dependents(cell_b.id) == [cell_a.id]

    def test_short_cycle_raises(self, colony_graph: ColonyGraph) -> None:
        cell_a = make_cell("test-dish", "colony", 0, 0)
        cell_b = make_cell("test-dish", "colony", 1, 1)
        colony_graph.add_cell(cell_a)
        colony_graph.add_cell(cell_b)
        colony_graph.add_edge(make_edge(cell_a.id, cell_b.id))

        with pytest.raises(ValueError, match="cycle"):
            colony_graph.add_edge(make_edge(cell_b.id, cell_a.id))

    def test_long_cycle_raises(self, colony_graph: ColonyGraph) -> None:
        cell_a = make_cell("test-dish", "colony", 0, 0)
        cell_b = make_cell("test-dish", "colony", 1, 1)
        cell_c = make_cell("test-dish", "colony", 2, 2)
        colony_graph.add_cell(cell_a)
        colony_graph.add_cell(cell_b)
        colony_graph.add_cell(cell_c)
        colony_graph.add_edge(make_edge(cell_a.id, cell_b.id))
        colony_graph.add_edge(make_edge(cell_b.id, cell_c.id))

        with pytest.raises(ValueError, match="cycle"):
            colony_graph.add_edge(make_edge(cell_c.id, cell_a.id))

    def test_self_loop_raises(self, colony_graph: ColonyGraph) -> None:
        cell_a = make_cell("test-dish", "colony", 0, 0)
        colony_graph.add_cell(cell_a)

        with pytest.raises(ValueError, match="cycle"):
            colony_graph.add_edge(make_edge(cell_a.id, cell_a.id))

    def test_valid_acyclic_accepted(self, colony_graph: ColonyGraph) -> None:
        cell_a = make_cell("test-dish", "colony", 0, 0)
        cell_b = make_cell("test-dish", "colony", 1, 1)
        cell_c = make_cell("test-dish", "colony", 2, 2)
        colony_graph.add_cell(cell_a)
        colony_graph.add_cell(cell_b)
        colony_graph.add_cell(cell_c)

        colony_graph.add_edge(make_edge(cell_a.id, cell_b.id))
        colony_graph.add_edge(make_edge(cell_a.id, cell_c.id))
        colony_graph.add_edge(make_edge(cell_b.id, cell_c.id))

        assert len(colony_graph.get_edges()) == 3


# ── Level Computation ────────────────────────────────────────────────────


class TestComputeLevels:
    def test_center_is_level_zero(self, canonical_colony) -> None:
        graph = canonical_colony["graph"]
        levels = graph.compute_levels(canonical_colony["center"].id)
        assert levels[canonical_colony["center"].id] == 0

    def test_direct_deps_are_level_one(self, canonical_colony) -> None:
        graph = canonical_colony["graph"]
        levels = graph.compute_levels(canonical_colony["center"].id)
        assert levels[canonical_colony["premise1"].id] == 1
        assert levels[canonical_colony["premise2"].id] == 1

    def test_transitive_deps_are_level_two(self, canonical_colony) -> None:
        graph = canonical_colony["graph"]
        levels = graph.compute_levels(canonical_colony["center"].id)
        assert levels[canonical_colony["cell1"].id] == 2
        assert levels[canonical_colony["cell2"].id] == 2

    def test_three_plus_levels(self, colony_graph: ColonyGraph) -> None:
        cells = [make_cell("test-dish", "colony", idx, idx) for idx in range(4)]
        for cell in cells:
            colony_graph.add_cell(cell)
        for idx in range(3):
            colony_graph.add_edge(make_edge(cells[idx].id, cells[idx + 1].id))

        levels = colony_graph.compute_levels(cells[0].id)
        assert levels[cells[0].id] == 0
        assert levels[cells[1].id] == 1
        assert levels[cells[2].id] == 2
        assert levels[cells[3].id] == 3

    def test_diamond_dependency(self, canonical_colony) -> None:
        """sub2 is reachable through both premise1 and premise2.
        BFS should assign level 2 regardless of path.
        """
        graph = canonical_colony["graph"]
        levels = graph.compute_levels(canonical_colony["center"].id)
        assert levels[canonical_colony["cell2"].id] == 2


# ── Leaf Cells ───────────────────────────────────────────────────────────


class TestLeafCells:
    def test_leaves_have_no_dependencies(self, canonical_colony) -> None:
        cells = canonical_colony["graph"].get_cells()
        cell_ids = {cell.id for cell in cells}
        assert cell_ids == {
            canonical_colony["cell1"].id,
            canonical_colony["cell2"].id,
        }

    def test_non_leaves_excluded(self, canonical_colony) -> None:
        cells = canonical_colony["graph"].get_cells()
        cell_ids = {cell.id for cell in cells}
        assert canonical_colony["center"].id not in cell_ids
        assert canonical_colony["premise1"].id not in cell_ids
        assert canonical_colony["premise2"].id not in cell_ids

    def test_center_only_colony(self, colony_graph: ColonyGraph) -> None:
        center = make_cell("test-dish", "colony", 0, 0)
        colony_graph.add_cell(center)
        cells = colony_graph.get_cells()
        assert [cell.id for cell in cells] == [center.id]


# ── Eligible for Validation ──────────────────────────────────────────────


class TestEligibleForValidation:
    def test_leaf_new_is_eligible(self, canonical_colony) -> None:
        graph = canonical_colony["graph"]
        statuses = {cell.id: CellStatus.NEW for cell in graph.get_all_cells()}
        eligible = graph.get_eligible_for_validation(statuses)
        eligible_ids = {cell.id for cell in eligible}
        assert eligible_ids == {
            canonical_colony["cell1"].id,
            canonical_colony["cell2"].id,
        }

    def test_all_deps_validated_makes_parent_eligible(self, canonical_colony) -> None:
        graph = canonical_colony["graph"]
        statuses = {cell.id: CellStatus.NEW for cell in graph.get_all_cells()}
        statuses[canonical_colony["cell1"].id] = CellStatus.VALIDATED
        statuses[canonical_colony["cell2"].id] = CellStatus.VALIDATED

        eligible = graph.get_eligible_for_validation(statuses)
        eligible_ids = {cell.id for cell in eligible}
        assert canonical_colony["premise1"].id in eligible_ids
        assert canonical_colony["premise2"].id in eligible_ids

    def test_unresolved_deps_blocks(self, canonical_colony) -> None:
        graph = canonical_colony["graph"]
        statuses = {cell.id: CellStatus.NEW for cell in graph.get_all_cells()}
        statuses[canonical_colony["cell1"].id] = CellStatus.VALIDATED

        eligible = graph.get_eligible_for_validation(statuses)
        eligible_ids = {cell.id for cell in eligible}
        assert canonical_colony["premise1"].id not in eligible_ids
        assert canonical_colony["premise2"].id not in eligible_ids

    def test_non_new_excluded(self, canonical_colony) -> None:
        graph = canonical_colony["graph"]
        statuses = {cell.id: CellStatus.VALIDATED for cell in graph.get_all_cells()}
        eligible = graph.get_eligible_for_validation(statuses)
        assert eligible == []


# ── Shared Premises ──────────────────────────────────────────────────────


class TestSharedPremises:
    def test_matching_claims_returns_pairs(self) -> None:
        graph1 = ColonyGraph(colony_id="dish-c1")
        graph2 = ColonyGraph(colony_id="dish-c2")

        cell1 = make_cell("dish", "c1", 1, 1, "Shared claim")
        cell2 = make_cell("dish", "c2", 1, 1, "Shared claim")

        graph1.add_cell(cell1)
        graph2.add_cell(cell2)

        pairs = graph1.find_shared_premises(graph2)
        assert pairs == [(cell1.id, cell2.id)]

    def test_no_matches_returns_empty(self) -> None:
        graph1 = ColonyGraph(colony_id="dish-c1")
        graph2 = ColonyGraph(colony_id="dish-c2")

        graph1.add_cell(make_cell("dish", "c1", 1, 1, "Alpha"))
        graph2.add_cell(make_cell("dish", "c2", 1, 1, "Beta"))

        assert graph1.find_shared_premises(graph2) == []


# ── DAG Validation ───────────────────────────────────────────────────────


class TestDAGValidation:
    def test_valid_dag(self, canonical_colony) -> None:
        assert canonical_colony["graph"].validate_dag() is True

    def test_has_cycle_with_edge_detects_would_be_cycle(
        self, colony_graph: ColonyGraph
    ) -> None:
        cell_a = make_cell("test-dish", "colony", 0, 0)
        cell_b = make_cell("test-dish", "colony", 1, 1)
        colony_graph.add_cell(cell_a)
        colony_graph.add_cell(cell_b)
        colony_graph.add_edge(make_edge(cell_a.id, cell_b.id))

        assert colony_graph.has_cycle_with_edge(cell_b.id, cell_a.id) is True

        cell_c = make_cell("test-dish", "colony", 2, 2)
        colony_graph.add_cell(cell_c)
        assert colony_graph.has_cycle_with_edge(cell_c.id, cell_a.id) is False


# ── Serialization / Deserialization ──────────────────────────────────────


class TestSerialization:
    @pytest.fixture
    def colony_model(self) -> Colony:
        return Colony(
            id="test-dish-colony",
            dish="test-dish",
            center_claim="Central thesis",
            center_cell_id="test-dish-colony-000-000",
            created_at="2026-01-01T00:00:00Z",
        )

    @pytest.fixture
    def graph_with_deps(self) -> ColonyGraph:
        """Build graph with explicit dependency lists on cells (needed for roundtrip)."""
        graph = ColonyGraph(colony_id="test-dish-colony")

        center = make_cell(
            "test-dish", "colony", 0, 0, "Central thesis",
            dependencies=["test-dish-colony-001-001", "test-dish-colony-001-002"],
        )
        premise1 = make_cell(
            "test-dish", "colony", 1, 1, "First premise",
            dependencies=["test-dish-colony-002-003", "test-dish-colony-002-004"],
        )
        premise2 = make_cell(
            "test-dish", "colony", 1, 2, "Second premise",
            dependencies=["test-dish-colony-002-004"],
        )
        cell1 = make_cell("test-dish", "colony", 2, 3, "Leaf premise of P1")
        cell2 = make_cell("test-dish", "colony", 2, 4, "Shared leaf premise")

        for cell in [center, premise1, premise2, cell1, cell2]:
            graph.add_cell(cell)

        graph.add_edge(make_edge(center.id, premise1.id))
        graph.add_edge(make_edge(center.id, premise2.id))
        graph.add_edge(make_edge(premise1.id, cell1.id))
        graph.add_edge(make_edge(premise1.id, cell2.id))
        graph.add_edge(make_edge(premise2.id, cell2.id))

        return graph

    def test_creates_expected_structure(
        self, tmp_path, graph_with_deps, colony_model,
    ) -> None:
        base = tmp_path / "colony"
        serialize_colony(graph_with_deps, colony_model, base)

        assert (base / "colony.json").exists()

        metadata_files = sorted(base.rglob("metadata.json"))
        assert len(metadata_files) == 5
        for metadata_file in metadata_files:
            cell_dir = metadata_file.parent
            assert (cell_dir / "events.jsonl").exists()
            assert (cell_dir / "evidence.md").exists()

        colony_data = json.loads((base / "colony.json").read_text())
        assert len(colony_data["cell_paths"]) == 5

    def test_cell_gets_metadata_events_evidence(
        self, tmp_path, graph_with_deps, colony_model,
    ) -> None:
        base = tmp_path / "colony"
        serialize_colony(graph_with_deps, colony_model, base)

        colony_data = json.loads((base / "colony.json").read_text())
        center_rel = colony_data["cell_paths"]["test-dish-colony-000-000"]
        center_dir = base / center_rel

        meta = json.loads((center_dir / "metadata.json").read_text())
        assert meta["id"] == "test-dish-colony-000-000"
        assert meta["claim_text"] == "Central thesis"

        events = (center_dir / "events.jsonl").read_text()
        assert events == ""

        evidence = (center_dir / "evidence.md").read_text()
        assert "Central thesis" in evidence

    def test_deserialize_reconstructs_graph(
        self, tmp_path, graph_with_deps, colony_model,
    ) -> None:
        base = tmp_path / "colony"
        serialize_colony(graph_with_deps, colony_model, base)

        loaded_graph, loaded_colony = deserialize_colony(base, "test-dish")

        assert loaded_colony.id == colony_model.id
        assert loaded_colony.center_claim == colony_model.center_claim

        loaded_ids = {cell.id for cell in loaded_graph.get_all_cells()}
        original_ids = {cell.id for cell in graph_with_deps.get_all_cells()}
        assert loaded_ids == original_ids

    def test_roundtrip_preserves_edges(
        self, tmp_path, graph_with_deps, colony_model,
    ) -> None:
        base = tmp_path / "colony"
        serialize_colony(graph_with_deps, colony_model, base)
        loaded_graph, _ = deserialize_colony(base, "test-dish")

        original_edge_pairs = sorted(
            (edge.from_cell, edge.to_cell) for edge in graph_with_deps.get_edges()
        )
        loaded_edge_pairs = sorted(
            (edge.from_cell, edge.to_cell) for edge in loaded_graph.get_edges()
        )
        assert loaded_edge_pairs == original_edge_pairs
