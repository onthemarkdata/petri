"""Pure dependency metrics and validation targets for a composed dish graph."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict


class Graph(Protocol):
    @property
    def cells(self) -> Mapping[str,Mapping[str,Any]]: ...

    def dependencies(self, cell_id: str) -> list[str]: ...
    def dependents(self, cell_id: str) -> list[str]: ...
    def reachable(self, cell_id: str, *, reverse: bool = False) -> list[str]: ...


class CellIntelligence(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')
    cell_id: str
    colony_id: str
    level: int
    importance: int
    cross_colony_importance: int
    longest_dependency_chain: int
    transitive_dependents: int
    fragility: int
    depending_colonies: tuple[str,...]


class EdgeIntelligenceReport(BaseModel):
    model_config = ConfigDict(extra='forbid')
    cells: dict[str,CellIntelligence]
    convergence_points: list[str]
    importance_ranking: list[str]
    fragility_ranking: list[str]


def importance(graph: Graph, cell_id: str) -> tuple[int,int]:
    """Count direct dependents, breaking out those in another colony."""
    colony = graph.cells[cell_id]['colony']
    dependents = set(graph.dependents(cell_id))
    return len(dependents),sum(graph.cells[parent]['colony'] != colony for parent in dependents)


def _longest(graph: Graph, cell_id: str, *, reverse: bool, cache: dict[str,int], active: set[str]) -> int:
    if cell_id in cache:
        return cache[cell_id]
    if cell_id in active:
        raise ValueError('Dependency metrics require an acyclic dish graph.')
    active.add(cell_id)
    neighbors = graph.dependents(cell_id) if reverse else graph.dependencies(cell_id)
    length = max((_longest(graph,item,reverse=reverse,cache=cache,active=active)+1 for item in neighbors),default=0)
    active.remove(cell_id)
    cache[cell_id] = length
    return length


def fragility(graph: Graph, cell_id: str) -> tuple[int,int,int]:
    """Return chain length, transitive-dependent count, and their sum."""
    chain = _longest(graph,cell_id,reverse=False,cache={},active=set()) + _longest(graph,cell_id,reverse=True,cache={},active=set())
    dependents = len(graph.reachable(cell_id,reverse=True))
    return chain,dependents,chain+dependents


def analyze_edges(graph: Graph) -> EdgeIntelligenceReport:
    cells = {}
    down: dict[str,int] = {}
    up: dict[str,int] = {}
    for cell_id,cell in sorted(graph.cells.items()):
        total,cross = importance(graph,cell_id)
        chain = _longest(graph,cell_id,reverse=False,cache=down,active=set()) + _longest(graph,cell_id,reverse=True,cache=up,active=set())
        dependent_count = len(graph.reachable(cell_id,reverse=True))
        colonies = tuple(sorted({str(graph.cells[parent]['colony']) for parent in graph.dependents(cell_id)}))
        cells[cell_id] = CellIntelligence(cell_id=cell_id,colony_id=str(cell['colony']),level=int(cell['level']),
            importance=total,cross_colony_importance=cross,longest_dependency_chain=chain,
            transitive_dependents=dependent_count,fragility=chain+dependent_count,depending_colonies=colonies)
    convergence = [cell_id for cell_id,item in cells.items() if len(item.depending_colonies)>=2]
    convergence.sort(key=lambda cell_id:(-len(cells[cell_id].depending_colonies),-cells[cell_id].importance,-cells[cell_id].fragility,cells[cell_id].level,cell_id))
    return EdgeIntelligenceReport(cells=cells,convergence_points=convergence,
        importance_ranking=sorted(cells,key=lambda cell_id:(-cells[cell_id].importance,cell_id)),
        fragility_ranking=sorted(cells,key=lambda cell_id:(-cells[cell_id].fragility,cell_id)))


def convergence_points(graph: Graph) -> list[CellIntelligence]:
    report = analyze_edges(graph)
    return [report.cells[cell_id] for cell_id in report.convergence_points]
