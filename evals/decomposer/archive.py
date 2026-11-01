"""Validate a separately supplied archive without publishing or rewriting it."""

from collections import Counter, defaultdict, deque
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from petri.reasoning.cell_search import search
from petri.storage import database


class ArchiveClaim(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str = Field(min_length=1)
    layer: int = Field(ge=0)
    title: str = Field(min_length=1)
    status: str
    evidence_file: str | None = None


class ArchiveGraph(BaseModel):
    claims: list[ArchiveClaim]
    edges: list[tuple[str, str]]

    @model_validator(mode="after")
    def validate_graph(self):
        ids = {claim.id for claim in self.claims}
        if len(ids) != len(self.claims):
            raise ValueError("Archive claim IDs must be unique")
        if len(set(self.edges)) != len(self.edges):
            raise ValueError("Archive edges must be unique")
        degree = Counter({identifier: 0 for identifier in ids})
        adjacent = defaultdict(list)
        for parent, child in self.edges:
            if parent not in ids or child not in ids:
                raise ValueError("Archive edge refers to an unknown claim")
            degree[child] += 1
            adjacent[parent].append(child)
        pending = deque(identifier for identifier, count in degree.items() if count == 0)
        visited = 0
        while pending:
            parent = pending.popleft()
            visited += 1
            for child in adjacent[parent]:
                degree[child] -= 1
                if degree[child] == 0:
                    pending.append(child)
        if visited != len(ids):
            raise ValueError("Archive graph must be acyclic")
        return self

    @property
    def counts(self) -> tuple[int, int, int]:
        layers = {claim.id: claim.layer for claim in self.claims}
        return (len(self.claims), len(self.edges),
                sum(layers[parent] != layers[child] for parent, child in self.edges))


def load_archive(path: Path, *, expected_counts: tuple[int, int, int] | None = (88, 128, 60)) -> ArchiveGraph:
    graph = ArchiveGraph.model_validate_json(path.read_text())
    if expected_counts is not None and graph.counts != expected_counts:
        raise ValueError(f"Archive counts {graph.counts} differ from expected {expected_counts}")
    return graph


def retrieval_report(graph: ArchiveGraph, db_path: Path, *, top_k: int = 5) -> dict[str, float | int]:
    """Measure exact-title lookup and retrieval of the graph's linked claims."""
    for claim in graph.claims:
        database.put_cell(db_path, {"id": claim.id, "colony": "archive", "level": claim.layer,
                          "seq": 1, "claim_text": claim.title, "status": "NEW",
                          "created_at": "2026-11-01T00:00:00Z"})
    links = defaultdict(set)
    for parent, child in graph.edges:
        links[parent].add(child)
    exact = hits = returned = expected = queries = 0
    for claim in graph.claims:
        matches = search(db_path, claim.title, top_k + 1)
        exact += bool(matches and matches[0].id == claim.id)
        if not links[claim.id]:
            continue
        found = {cell.id for cell in [cell for cell in matches if cell.id != claim.id][:top_k]}
        hits += len(found & links[claim.id])
        returned += len(found)
        expected += len(links[claim.id])
        queries += 1
    return {"linked_query_count": queries, "precision": hits / returned if returned else 0.0,
            "recall": hits / expected if expected else 0.0,
            "exact_title_recall": exact / len(graph.claims) if graph.claims else 0.0}
