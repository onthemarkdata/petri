"""Read-only dish composition and dependency-intelligence queries."""

from pathlib import Path
from typing import Any

from petri.analysis.edge_intelligence import EdgeIntelligenceReport, analyze_edges
from petri.query.connection import enumerate_dishes
from petri.storage.edge_registry import DishGraph, EdgeRecord


def dish_graph(petri_dir: Path) -> DishGraph:
    cells: list[dict[str, Any]] = []
    edges: list[EdgeRecord] = []
    for dish in enumerate_dishes(petri_dir):
        graph = DishGraph.read(dish.db_path)
        cells.extend(graph.cells.values())
        edges.extend(graph.edges)
    return DishGraph(cells,edges)


def get_intelligence(petri_dir: Path) -> EdgeIntelligenceReport:
    return analyze_edges(dish_graph(petri_dir))
