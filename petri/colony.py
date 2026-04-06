"""Colony graph module for DAG operations.

Manages the directed acyclic graph structure of a colony. A colony is a
connected DAG of nodes rooted at a colony center (level 0). Edges represent
logical dependencies: Edge(from_node=A, to_node=B) means A depends on B.
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

from petri.models import Colony, Edge, Node, NodeStatus, build_node_key


class ColonyGraph:
    """Mutable graph structure that manages nodes and directed dependency edges.

    Nodes are keyed by composite ID. Edges are directed: from_node depends on
    to_node. The adjacency dict (_adj) maps a node to the set of nodes that
    depend on it (forward/dependents). The reverse adjacency dict (_rev) maps a
    node to the set of nodes it depends on (dependencies).
    """

    def __init__(self, colony_id: str = "") -> None:
        self.colony_id: str = colony_id
        self._nodes: dict[str, Node] = {}
        self._edges: list[Edge] = []
        self._adj: dict[str, set[str]] = {}  # node_id -> dependents
        self._rev: dict[str, set[str]] = {}  # node_id -> dependencies

    # ── Node operations ──────────────────────────────────────────────

    def add_node(self, node: Node) -> None:
        """Add a node to the graph.

        Raises ValueError if a node with the same ID already exists.
        """
        if node.id in self._nodes:
            raise ValueError(f"Node already exists: {node.id}")
        self._nodes[node.id] = node
        self._adj.setdefault(node.id, set())
        self._rev.setdefault(node.id, set())

    def remove_node(self, node_id: str) -> None:
        """Remove a node and all edges that reference it.

        Raises KeyError if the node is not found.
        """
        if node_id not in self._nodes:
            raise KeyError(f"Node not found: {node_id}")

        # Remove edges that reference this node
        self._edges = [
            e
            for e in self._edges
            if e.from_node != node_id and e.to_node != node_id
        ]

        # Clean up adjacency references
        for dep_id in self._adj.get(node_id, set()):
            self._rev.get(dep_id, set()).discard(node_id)
        for dep_id in self._rev.get(node_id, set()):
            self._adj.get(dep_id, set()).discard(node_id)

        self._adj.pop(node_id, None)
        self._rev.pop(node_id, None)
        del self._nodes[node_id]

    def get_node(self, node_id: str) -> Node:
        """Get a node by its composite ID.

        Raises KeyError if not found.
        """
        if node_id not in self._nodes:
            raise KeyError(f"Node not found: {node_id}")
        return self._nodes[node_id]

    def get_nodes(self) -> list[Node]:
        """Return all nodes sorted by level then seq."""
        return sorted(self._nodes.values(), key=lambda n: (n.level, n.id))

    def get_edges(self) -> list[Edge]:
        """Return all edges."""
        return list(self._edges)

    # ── Edge operations ──────────────────────────────────────────────

    def add_edge(self, edge: Edge) -> None:
        """Add a directed dependency edge.

        Edge(from_node=A, to_node=B) means A depends on B. Before adding,
        checks that adding this edge would not create a cycle.

        Raises ValueError if the edge would create a cycle.
        """
        if self.has_cycle_with_edge(edge.from_node, edge.to_node):
            raise ValueError(
                f"Adding edge {edge.from_node} -> {edge.to_node} would create a cycle"
            )

        self._edges.append(edge)
        self._adj.setdefault(edge.to_node, set()).add(edge.from_node)
        self._rev.setdefault(edge.from_node, set()).add(edge.to_node)

    def has_cycle_with_edge(self, from_node: str, to_node: str) -> bool:
        """Check if adding an edge from from_node to to_node would create a cycle.

        Uses DFS from to_node following _rev edges (dependency direction) to
        see if from_node is reachable. If it is, adding the edge would close
        a cycle.
        """
        # Edge means from_node depends on to_node.
        # A cycle would exist if to_node already (transitively) depends on
        # from_node. We check by walking _rev from to_node.
        if from_node == to_node:
            return True

        visited: set[str] = set()
        stack: list[str] = [to_node]

        while stack:
            current = stack.pop()
            if current == from_node:
                return True
            if current in visited:
                continue
            visited.add(current)
            for dep in self._rev.get(current, set()):
                if dep not in visited:
                    stack.append(dep)

        return False

    def validate_dag(self) -> bool:
        """Verify the graph contains no cycles.

        Uses Kahn's algorithm (topological sort via in-degree counting).
        Returns True if the graph is a valid DAG, False otherwise.
        """
        # Compute in-degrees (number of dependencies for each node)
        in_degree: dict[str, int] = {nid: 0 for nid in self._nodes}
        for nid in self._nodes:
            in_degree[nid] = len(self._rev.get(nid, set()))

        # Start with nodes that have no dependencies
        queue: deque[str] = deque(
            nid for nid, deg in in_degree.items() if deg == 0
        )
        processed = 0

        while queue:
            current = queue.popleft()
            processed += 1
            for dependent in self._adj.get(current, set()):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        return processed == len(self._nodes)

    # ── Level computation ────────────────────────────────────────────

    def compute_levels(self, center_id: str) -> dict[str, int]:
        """Compute node levels via BFS from the colony center.

        The center node is level 0. Levels increase outward following the
        dependency direction: the center depends on level-1 nodes, level-1
        nodes depend on level-2 nodes, etc.

        BFS follows _rev edges from the center outward because
        _rev[node] = the set of nodes that node depends on.

        Returns a dict mapping node_id to its computed level.
        """
        levels: dict[str, int] = {center_id: 0}
        queue: deque[str] = deque([center_id])

        while queue:
            current = queue.popleft()
            current_level = levels[current]
            for dep in self._rev.get(current, set()):
                if dep not in levels:
                    levels[dep] = current_level + 1
                    queue.append(dep)

        return levels

    # ── Dependency queries ───────────────────────────────────────────

    def get_dependencies(self, node_id: str) -> list[str]:
        """Return composite keys of nodes this node depends on."""
        return sorted(self._rev.get(node_id, set()))

    def get_dependents(self, node_id: str) -> list[str]:
        """Return composite keys of nodes that depend on this node."""
        return sorted(self._adj.get(node_id, set()))

    def get_cell_nodes(self) -> list[Node]:
        """Return cells — nodes that have no dependencies.

        Cells are the most granular claims in the colony, at the deepest
        levels of the DAG. They must be validated first (bottom-up).
        """
        return [
            self._nodes[nid]
            for nid in sorted(self._nodes)
            if not self._rev.get(nid, set())
        ]

    # Keep old name as alias for backwards compatibility in tests
    get_leaf_nodes = get_cell_nodes

    def get_eligible_for_validation(
        self, nodes_status: dict[str, NodeStatus]
    ) -> list[Node]:
        """Return nodes eligible to enter the validation queue.

        A node is eligible if it has NEW status AND either:
        - It is a cell (no dependencies), OR
        - ALL of its dependencies have VALIDATED status.
        """
        eligible: list[Node] = []
        for nid, node in sorted(self._nodes.items()):
            status = nodes_status.get(nid, node.status)
            if status != NodeStatus.NEW:
                continue

            deps = self._rev.get(nid, set())
            if not deps:
                # Cell node with NEW status
                eligible.append(node)
            elif all(
                nodes_status.get(d, NodeStatus.NEW) == NodeStatus.VALIDATED
                for d in deps
            ):
                # All dependencies validated
                eligible.append(node)

        return eligible

    # ── Cross-colony queries ─────────────────────────────────────────

    def find_shared_premises(
        self, other: ColonyGraph
    ) -> list[tuple[str, str]]:
        """Find nodes in this colony that share claim_text with another colony.

        Returns a list of (this_node_id, other_node_id) pairs where the
        claim_text matches exactly.
        """
        # Index the other colony's claims for efficient lookup
        other_claims: dict[str, list[str]] = {}
        for node in other._nodes.values():
            other_claims.setdefault(node.claim_text, []).append(node.id)

        pairs: list[tuple[str, str]] = []
        for node in self._nodes.values():
            for other_id in other_claims.get(node.claim_text, []):
                pairs.append((node.id, other_id))

        return sorted(pairs)


# ── Serialization ────────────────────────────────────────────────────────


def serialize_colony(
    graph: ColonyGraph, colony: Colony, base_path: Path
) -> None:
    """Save a colony to the filesystem.

    Creates the following structure at base_path:
        colony.json
        {level:03d}-{claim-slug}/
            {seq:03d}-{claim-slug}/
                metadata.json
                events.jsonl
                evidence.md
    """
    from petri.models import claim_to_slug

    base_path.mkdir(parents=True, exist_ok=True)

    # Group nodes by level for level directory naming
    by_level: dict[int, list] = {}
    for node in graph.get_nodes():
        by_level.setdefault(node.level, []).append(node)

    # Build node_paths mapping
    node_paths: dict[str, str] = {}

    for node in graph.get_nodes():
        parts = node.id.split("-")
        seq_str = parts[-1]
        level_str = parts[-2]

        # Level directory: use first node at this level for the level slug
        level_nodes = by_level[node.level]
        level_slug = claim_to_slug(level_nodes[0].claim_text)
        level_dir_name = f"{level_str}-{level_slug}"

        # Node directory: seq + claim slug
        node_slug = claim_to_slug(node.claim_text)
        node_dir_name = f"{seq_str}-{node_slug}"

        rel_path = f"{level_dir_name}/{node_dir_name}"
        node_paths[node.id] = rel_path

        node_dir = base_path / rel_path
        node_dir.mkdir(parents=True, exist_ok=True)

        # metadata.json
        (node_dir / "metadata.json").write_text(
            json.dumps(node.model_dump(), indent=2, default=str) + "\n",
            encoding="utf-8",
        )

        # events.jsonl (empty)
        events_path = node_dir / "events.jsonl"
        if not events_path.exists():
            events_path.touch()

        # evidence.md
        (node_dir / "evidence.md").write_text(
            f"# {node.id}\n\n"
            f"**Claim:** {node.claim_text}\n\n"
            f"**Status:** {NodeStatus.NEW.value}\n",
            encoding="utf-8",
        )

    # Write colony metadata with node_paths
    colony.node_paths = node_paths
    (base_path / "colony.json").write_text(
        json.dumps(colony.model_dump(), indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def deserialize_colony(
    base_path: Path, dish_id: str
) -> tuple[ColonyGraph, Colony]:
    """Load a colony from the filesystem.

    Reads colony.json for the Colony model, then scans subdirectories for
    node metadata.json files. Reconstructs the full graph with nodes and
    edges.

    Returns a (ColonyGraph, Colony) tuple.
    """
    # Read colony metadata
    colony_path = base_path / "colony.json"
    colony_data = json.loads(colony_path.read_text(encoding="utf-8"))
    colony = Colony.model_validate(colony_data)

    graph = ColonyGraph(colony_id=colony.id)

    # Scan for node metadata (supports both flat and nested layouts)
    nodes: list[Node] = []
    for metadata_path in sorted(base_path.rglob("metadata.json")):
        node_data = json.loads(metadata_path.read_text(encoding="utf-8"))
        node = Node.model_validate(node_data)
        nodes.append(node)

    # Add nodes first
    for node in nodes:
        graph.add_node(node)

    # Reconstruct edges from node dependency lists
    seen_edges: set[tuple[str, str]] = set()
    for node in nodes:
        for dep_id in node.dependencies:
            edge_pair = (node.id, dep_id)
            if edge_pair not in seen_edges:
                seen_edges.add(edge_pair)
                # Determine edge type based on colony membership
                if dep_id.startswith(colony.id):
                    edge_type = "intra_colony"
                else:
                    edge_type = "cross_colony"
                edge = Edge(
                    from_node=node.id,
                    to_node=dep_id,
                    edge_type=edge_type,
                )
                # Only add if both nodes are in the graph
                if dep_id in graph._nodes:
                    graph.add_edge(edge)

    return graph, colony
