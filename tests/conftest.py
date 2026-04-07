"""Shared test fixtures for Petri test suite."""

import json
import shutil
from pathlib import Path

import pytest

from petri.colony import ColonyGraph, serialize_colony
from petri.config import LLM_INFERENCE_MODEL, MAX_CONCURRENT, MAX_ITERATIONS
from petri.models import Colony, Edge, Node, NodeStatus, Verdict, build_node_key


# ── Canonical Constants ──────────────────────────────────────────────────
# Single source of truth for test IDs used across all test files.

CANONICAL_DISH_ID = "test-dish"
CANONICAL_COLONY_NAME = "colony"
CANONICAL_COLONY_ID = "test-dish-colony"
CANONICAL_NODE_IDS = {
    "center": "test-dish-colony-000-000",
    "premise1": "test-dish-colony-001-001",
    "premise2": "test-dish-colony-001-002",
    "cell1": "test-dish-colony-002-003",
    "cell2": "test-dish-colony-002-004",
}


# ── Shared Helpers ────────────────────────────────────────────────────────


def make_node(
    dish: str,
    colony: str,
    level: int,
    seq: int,
    claim: str = "",
    status: NodeStatus = NodeStatus.NEW,
    dependencies: list[str] | None = None,
    dependents: list[str] | None = None,
) -> Node:
    """Build a test Node from composite-key parts."""
    node_id = build_node_key(dish, colony, level, seq)
    colony_id = f"{dish}-{colony}"
    return Node(
        id=node_id,
        colony_id=colony_id,
        claim_text=claim or f"claim-{level}-{seq}",
        level=level,
        status=status,
        dependencies=dependencies or [],
        dependents=dependents or [],
    )


def make_edge(from_id: str, to_id: str, edge_type: str = "intra_colony") -> Edge:
    """Build a test Edge."""
    return Edge(from_node=from_id, to_node=to_id, edge_type=edge_type)


def make_verdict(
    agent: str,
    verdict: str,
    node_id: str = CANONICAL_NODE_IDS["center"],
) -> Verdict:
    """Build a test Verdict using canonical node ID by default."""
    return Verdict(
        node_id=node_id, agent=agent, iteration=0, verdict=verdict, summary=""
    )


def make_event(
    *,
    node_id: str = CANONICAL_NODE_IDS["premise1"],
    event_id: str | None = None,
    event_type: str = "verdict_issued",
    agent: str = "analyst",
    iteration: int = 1,
    timestamp: str = "2026-01-15T10:00:00+00:00",
    data: dict | None = None,
) -> dict:
    """Build a raw event dict (as it would appear serialised in JSONL).

    Defaults to canonical node IDs from the shared colony fixture.
    """
    if event_id is None:
        event_id = f"{node_id}-aabbccdd"
    if data is None:
        data = {"verdict": "VALIDATED", "summary": "Looks good"}
    return {
        "id": event_id,
        "node_id": node_id,
        "timestamp": timestamp,
        "type": event_type,
        "agent": agent,
        "iteration": iteration,
        "data": data,
    }


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_petri_dir(tmp_path):
    """Create a temporary .petri/ directory with default structure."""
    petri_dir = tmp_path / ".petri"
    petri_dir.mkdir()
    (petri_dir / "petri-dishes").mkdir()
    (petri_dir / "defaults").mkdir()

    # Default config
    config = {
        "name": "test-dish",
        "model": {"name": LLM_INFERENCE_MODEL, "provider": "local"},
        "harness": "claude-code",
        "max_iterations": MAX_ITERATIONS,
    }
    (petri_dir / "petri.yaml").write_text(
        "# Petri configuration\n"
        f"name: {config['name']}\n"
        f"model:\n  name: {config['model']['name']}\n  provider: {config['model']['provider']}\n"
        f"harness: {config['harness']}\n"
        f"max_iterations: {config['max_iterations']}\n"
    )

    # Empty queue
    queue = {"version": 1, "last_updated": None, "entries": {}}
    (petri_dir / "queue.json").write_text(json.dumps(queue, indent=2) + "\n")

    return tmp_path


@pytest.fixture
def petri_defaults_dir():
    """Return the path to the package defaults directory."""
    return Path(__file__).parent.parent / "petri" / "defaults"


@pytest.fixture
def petri_env(tmp_path):
    """Set up a .petri/ directory with real defaults from the package.

    Used by scanner, adapter, and propagation tests. Includes:
    - petri.yaml config
    - queue.json (empty)
    - defaults/ copied from petri/defaults/
    - petri-dishes/ directory
    """
    petri_dir = tmp_path / ".petri"
    petri_dir.mkdir()
    (petri_dir / "petri-dishes").mkdir()

    # Copy real defaults (agents.yaml, debates.yaml, constitution, etc.)
    src_defaults = Path(__file__).parent.parent / "petri" / "defaults"
    dst_defaults = petri_dir / "defaults"
    shutil.copytree(src_defaults, dst_defaults)

    # Write petri.yaml
    (petri_dir / "petri.yaml").write_text(
        "name: test-dish\n"
        "model:\n"
        f"  name: {LLM_INFERENCE_MODEL}\n"
        "  provider: local\n"
        "harness: claude-code\n"
        f"max_iterations: {MAX_ITERATIONS}\n"
        f"max_concurrent: {MAX_CONCURRENT}\n"
    )

    # Empty queue
    queue = {"version": 1, "last_updated": None, "entries": {}}
    (petri_dir / "queue.json").write_text(json.dumps(queue, indent=2) + "\n")

    return tmp_path


def _build_canonical_colony(
    cell_status: NodeStatus = NodeStatus.NEW,
    premise_status: NodeStatus = NodeStatus.NEW,
    center_status: NodeStatus = NodeStatus.NEW,
) -> dict:
    """Build the canonical 5-node diamond DAG.

    Structure:
        center (L0, 000-000) depends on premise1, premise2
        premise1 (L1, 001-001) depends on cell1, cell2
        premise2 (L1, 001-002) depends on cell2 (shared = diamond)
        cell1 (L2, 002-003): cell node
        cell2 (L2, 002-004): cell node (shared dependency)
    """
    dish_id = CANONICAL_DISH_ID
    colony_name = CANONICAL_COLONY_NAME
    colony_id = CANONICAL_COLONY_ID

    center = make_node(
        dish_id, colony_name, 0, 0, "Central thesis",
        status=center_status,
        dependencies=[
            build_node_key(dish_id, colony_name, 1, 1),
            build_node_key(dish_id, colony_name, 1, 2),
        ],
    )
    premise1 = make_node(
        dish_id, colony_name, 1, 1, "First premise",
        status=premise_status,
        dependencies=[
            build_node_key(dish_id, colony_name, 2, 3),
            build_node_key(dish_id, colony_name, 2, 4),
        ],
    )
    premise2 = make_node(
        dish_id, colony_name, 1, 2, "Second premise",
        status=premise_status,
        dependencies=[
            build_node_key(dish_id, colony_name, 2, 4),
        ],
    )
    cell1 = make_node(
        dish_id, colony_name, 2, 3, "Cell premise of P1",
        status=cell_status,
    )
    cell2 = make_node(
        dish_id, colony_name, 2, 4, "Shared cell premise",
        status=cell_status,
    )

    graph = ColonyGraph(colony_id=colony_id)
    for node in [center, premise1, premise2, cell1, cell2]:
        graph.add_node(node)

    graph.add_edge(make_edge(center.id, premise1.id))
    graph.add_edge(make_edge(center.id, premise2.id))
    graph.add_edge(make_edge(premise1.id, cell1.id))
    graph.add_edge(make_edge(premise1.id, cell2.id))
    graph.add_edge(make_edge(premise2.id, cell2.id))

    colony_model = Colony(
        id=colony_id,
        dish=dish_id,
        center_claim="Central thesis",
        center_node_id=center.id,
        created_at="2026-01-01T00:00:00Z",
    )

    return {
        "dish_id": dish_id,
        "colony_name": colony_name,
        "colony_id": colony_id,
        "graph": graph,
        "colony_model": colony_model,
        "center": center,
        "premise1": premise1,
        "premise2": premise2,
        "cell1": cell1,
        "cell2": cell2,
    }


@pytest.fixture
def canonical_colony():
    """Build the canonical 5-node diamond DAG (all nodes NEW)."""
    return _build_canonical_colony()


@pytest.fixture
def canonical_colony_validated_cells():
    """Build the canonical diamond DAG with cells and premises VALIDATED.

    Used by propagation and dashboard tests where nodes need to be
    re-openable (VALIDATED -> NEW).
    """
    return _build_canonical_colony(
        cell_status=NodeStatus.VALIDATED,
        premise_status=NodeStatus.VALIDATED,
        center_status=NodeStatus.NEW,
    )


@pytest.fixture
def petri_env_with_colony(tmp_path, canonical_colony_validated_cells):
    """Full .petri/ environment with the canonical colony serialized to disk.

    Combines petri_env setup with colony serialization. Used by propagation,
    dashboard, and other tests that need on-disk colony state.
    """
    colony = canonical_colony_validated_cells
    petri_dir = tmp_path / ".petri"
    petri_dir.mkdir()
    (petri_dir / "petri-dishes").mkdir()

    (petri_dir / "petri.yaml").write_text(
        "name: test-dish\n"
        "model:\n"
        f"  name: {LLM_INFERENCE_MODEL}\n"
        "  provider: local\n"
        "harness: claude-code\n"
        f"max_iterations: {MAX_ITERATIONS}\n"
    )

    queue_data = {"version": 1, "last_updated": None, "entries": {}}
    (petri_dir / "queue.json").write_text(json.dumps(queue_data, indent=2) + "\n")

    colony_path = petri_dir / "petri-dishes" / colony["colony_name"]
    serialize_colony(colony["graph"], colony["colony_model"], colony_path)

    return {
        "tmp_path": tmp_path,
        "petri_dir": petri_dir,
        "colony_path": colony_path,
        "dish_id": colony["dish_id"],
        "colony_name": colony["colony_name"],
        "colony_id": colony["colony_id"],
        "graph": colony["graph"],
        "colony_model": colony["colony_model"],
        "center": colony["center"],
        "premise1": colony["premise1"],
        "premise2": colony["premise2"],
        "cell1": colony["cell1"],
        "cell2": colony["cell2"],
    }
