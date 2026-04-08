"""Colony graph module for DAG operations.

Manages the directed acyclic graph structure of a colony. A colony is a
connected DAG of cells rooted at a colony center (level 0). Edges represent
logical dependencies: Edge(from_cell=A, to_cell=B) means A depends on B.
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

from petri.models import Cell, CellStatus, Colony, Edge


class ColonyGraph:
    """Mutable graph structure that manages cells and directed dependency edges.

    Cells are keyed by composite ID. Edges are directed: from_cell depends on
    to_cell. The adjacency dict (_adj) maps a cell to the set of cells that
    depend on it (forward/dependents). The reverse adjacency dict (_rev) maps a
    cell to the set of cells it depends on (dependencies).
    """

    def __init__(self, colony_id: str = "") -> None:
        self.colony_id: str = colony_id
        self._cells: dict[str, Cell] = {}
        self._edges: list[Edge] = []
        self._adj: dict[str, set[str]] = {}  # cell_id -> dependents
        self._rev: dict[str, set[str]] = {}  # cell_id -> dependencies

    # ── Cell operations ──────────────────────────────────────────────

    def add_cell(self, cell: Cell) -> None:
        """Add a cell to the graph.

        Raises ValueError if a cell with the same ID already exists.
        """
        if cell.id in self._cells:
            raise ValueError(f"Cell already exists: {cell.id}")
        self._cells[cell.id] = cell
        self._adj.setdefault(cell.id, set())
        self._rev.setdefault(cell.id, set())

    def remove_cell(self, cell_id: str) -> None:
        """Remove a cell and all edges that reference it.

        Raises KeyError if the cell is not found.
        """
        if cell_id not in self._cells:
            raise KeyError(f"Cell not found: {cell_id}")

        # Remove edges that reference this cell
        self._edges = [
            edge
            for edge in self._edges
            if edge.from_cell != cell_id and edge.to_cell != cell_id
        ]

        # Clean up adjacency references
        for dep_id in self._adj.get(cell_id, set()):
            self._rev.get(dep_id, set()).discard(cell_id)
        for dep_id in self._rev.get(cell_id, set()):
            self._adj.get(dep_id, set()).discard(cell_id)

        self._adj.pop(cell_id, None)
        self._rev.pop(cell_id, None)
        del self._cells[cell_id]

    def get_cell(self, cell_id: str) -> Cell:
        """Get a cell by its composite ID.

        Raises KeyError if not found.
        """
        if cell_id not in self._cells:
            raise KeyError(f"Cell not found: {cell_id}")
        return self._cells[cell_id]

    def get_all_cells(self) -> list[Cell]:
        """Return all cells sorted by level then seq."""
        return sorted(self._cells.values(), key=lambda cell: (cell.level, cell.id))

    def get_edges(self) -> list[Edge]:
        """Return all edges."""
        return list(self._edges)

    # ── Edge operations ──────────────────────────────────────────────

    def add_edge(self, edge: Edge) -> None:
        """Add a directed dependency edge.

        Edge(from_cell=A, to_cell=B) means A depends on B. Before adding,
        checks that adding this edge would not create a cycle.

        Raises ValueError if the edge would create a cycle.
        """
        if self.has_cycle_with_edge(edge.from_cell, edge.to_cell):
            raise ValueError(
                f"Adding edge {edge.from_cell} -> {edge.to_cell} would create a cycle"
            )

        self._edges.append(edge)
        self._adj.setdefault(edge.to_cell, set()).add(edge.from_cell)
        self._rev.setdefault(edge.from_cell, set()).add(edge.to_cell)

    def has_cycle_with_edge(self, from_cell: str, to_cell: str) -> bool:
        """Check if adding an edge from from_cell to to_cell would create a cycle.

        Uses DFS from to_cell following _rev edges (dependency direction) to
        see if from_cell is reachable. If it is, adding the edge would close
        a cycle.
        """
        # Edge means from_cell depends on to_cell.
        # A cycle would exist if to_cell already (transitively) depends on
        # from_cell. We check by walking _rev from to_cell.
        if from_cell == to_cell:
            return True

        visited: set[str] = set()
        stack: list[str] = [to_cell]

        while stack:
            current = stack.pop()
            if current == from_cell:
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
        # Compute in-degrees (number of dependencies for each cell)
        in_degree: dict[str, int] = {cell_id: 0 for cell_id in self._cells}
        for cell_id in self._cells:
            in_degree[cell_id] = len(self._rev.get(cell_id, set()))

        # Start with cells that have no dependencies
        queue: deque[str] = deque(
            cell_id for cell_id, deg in in_degree.items() if deg == 0
        )
        processed = 0

        while queue:
            current = queue.popleft()
            processed += 1
            for dependent in self._adj.get(current, set()):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        return processed == len(self._cells)

    # ── Level computation ────────────────────────────────────────────

    def compute_levels(self, center_id: str) -> dict[str, int]:
        """Compute cell levels via BFS from the colony center.

        The center cell is level 0. Levels increase outward following the
        dependency direction: the center depends on level-1 cells, level-1
        cells depend on level-2 cells, etc.

        BFS follows _rev edges from the center outward because
        _rev[cell] = the set of cells that cell depends on.

        Returns a dict mapping cell_id to its computed level.
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

    def get_dependencies(self, cell_id: str) -> list[str]:
        """Return composite keys of cells this cell depends on."""
        return sorted(self._rev.get(cell_id, set()))

    def get_dependents(self, cell_id: str) -> list[str]:
        """Return composite keys of cells that depend on this cell."""
        return sorted(self._adj.get(cell_id, set()))

    def get_cells(self) -> list[Cell]:
        """Return cells that have no dependencies.

        These are the most granular claims in the colony, at the deepest
        levels of the DAG. They must be validated first (bottom-up).
        """
        return [
            self._cells[cell_id]
            for cell_id in sorted(self._cells)
            if not self._rev.get(cell_id, set())
        ]

    def get_eligible_for_validation(
        self, cells_status: dict[str, CellStatus]
    ) -> list[Cell]:
        """Return cells eligible to enter the validation queue.

        A cell is eligible if it has NEW status AND either:
        - It has no dependencies, OR
        - ALL of its dependencies have VALIDATED status.
        """
        eligible: list[Cell] = []
        for cell_id, cell in sorted(self._cells.items()):
            status = cells_status.get(cell_id, cell.status)
            if status != CellStatus.NEW:
                continue

            deps = self._rev.get(cell_id, set())
            if not deps:
                # Cell with NEW status and no dependencies
                eligible.append(cell)
            elif all(
                cells_status.get(dependency, CellStatus.NEW) == CellStatus.VALIDATED
                for dependency in deps
            ):
                # All dependencies validated
                eligible.append(cell)

        return eligible

    # ── Cross-colony queries ─────────────────────────────────────────

    def find_shared_premises(
        self, other: ColonyGraph
    ) -> list[tuple[str, str]]:
        """Find cells in this colony that share claim_text with another colony.

        Returns a list of (this_cell_id, other_cell_id) pairs where the
        claim_text matches exactly.
        """
        # Index the other colony's claims for efficient lookup
        other_claims: dict[str, list[str]] = {}
        for cell in other._cells.values():
            other_claims.setdefault(cell.claim_text, []).append(cell.id)

        pairs: list[tuple[str, str]] = []
        for cell in self._cells.values():
            for other_id in other_claims.get(cell.claim_text, []):
                pairs.append((cell.id, other_id))

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

    # Group cells by level for level directory naming
    by_level: dict[int, list] = {}
    for cell in graph.get_all_cells():
        by_level.setdefault(cell.level, []).append(cell)

    # Build cell_paths mapping
    cell_paths: dict[str, str] = {}

    for cell in graph.get_all_cells():
        parts = cell.id.split("-")
        seq_str = parts[-1]
        level_str = parts[-2]

        # Level directory: use first cell at this level for the level slug
        level_cells = by_level[cell.level]
        level_slug = claim_to_slug(level_cells[0].claim_text)
        level_dir_name = f"{level_str}-{level_slug}"

        # Cell directory: seq + claim slug
        cell_slug = claim_to_slug(cell.claim_text)
        cell_dir_name = f"{seq_str}-{cell_slug}"

        rel_path = f"{level_dir_name}/{cell_dir_name}"
        cell_paths[cell.id] = rel_path

        cell_dir_path = base_path / rel_path
        cell_dir_path.mkdir(parents=True, exist_ok=True)

        # metadata.json
        (cell_dir_path / "metadata.json").write_text(
            json.dumps(cell.model_dump(), indent=2, default=str) + "\n",
            encoding="utf-8",
        )

        # events.jsonl (empty)
        events_path = cell_dir_path / "events.jsonl"
        if not events_path.exists():
            events_path.touch()

        # evidence.md
        (cell_dir_path / "evidence.md").write_text(
            f"# {cell.id}\n\n"
            f"**Claim:** {cell.claim_text}\n\n"
            f"**Status:** {CellStatus.NEW.value}\n",
            encoding="utf-8",
        )

    # Write colony metadata with cell_paths
    colony.cell_paths = cell_paths
    (base_path / "colony.json").write_text(
        json.dumps(colony.model_dump(), indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def deserialize_colony(
    base_path: Path, dish_id: str
) -> tuple[ColonyGraph, Colony]:
    """Load a colony from the filesystem.

    Reads colony.json for the Colony model, then scans subdirectories for
    cell metadata.json files. Reconstructs the full graph with cells and
    edges.

    Returns a (ColonyGraph, Colony) tuple.
    """
    # Read colony metadata
    colony_path = base_path / "colony.json"
    colony_data = json.loads(colony_path.read_text(encoding="utf-8"))
    colony = Colony.model_validate(colony_data)

    graph = ColonyGraph(colony_id=colony.id)

    # Scan for cell metadata (supports both flat and nested layouts)
    cells: list[Cell] = []
    for metadata_path in sorted(base_path.rglob("metadata.json")):
        cell_data = json.loads(metadata_path.read_text(encoding="utf-8"))
        cell = Cell.model_validate(cell_data)
        cells.append(cell)

    # Add cells first
    for cell in cells:
        graph.add_cell(cell)

    # Reconstruct edges from cell dependency lists
    seen_edges: set[tuple[str, str]] = set()
    for cell in cells:
        for dep_id in cell.dependencies:
            edge_pair = (cell.id, dep_id)
            if edge_pair not in seen_edges:
                seen_edges.add(edge_pair)
                # Determine edge type based on colony membership
                if dep_id.startswith(colony.id):
                    edge_type = "intra_colony"
                else:
                    edge_type = "cross_colony"
                edge = Edge(
                    from_cell=cell.id,
                    to_cell=dep_id,
                    edge_type=edge_type,
                )
                # Only add if both cells are in the graph
                if dep_id in graph._cells:
                    graph.add_edge(edge)

    return graph, colony
