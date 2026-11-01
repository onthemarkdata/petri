"""Build a cell graph from validated agent outputs with explicit persistence."""

from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from petri.models import Cell, ClarifyingQuestion, DecompositionResult, Edge, build_cell_key, parse_key
from petri.reasoning.decomposition_agents import DecompositionInput
from petri.reasoning.decomposition_schemas import InitialDecomposition, PremiseCandidate, WhyExpansion
from petri.reasoning.decomposition_settings import DecompositionSettings
from petri.storage import database

Progress = Callable[[str], None] | None
OnCellCreated = Callable[[Cell, list[Edge]], None] | None


class DecompositionRunner(Protocol):
    def initial_decomposition(self, request: DecompositionInput, on_progress: Progress = None) -> InitialDecomposition: ...
    def expand_why(self, request: DecompositionInput, on_progress: Progress = None) -> WhyExpansion: ...


def decompose_typed(
    claim: str, clarifications: list[ClarifyingQuestion], dish_id: str, colony_name: str,
    runner: DecompositionRunner, *, settings: DecompositionSettings, guidance: str = "",
    db_path: Path | None = None, center: Cell | None = None,
    on_progress: Progress = None, on_cell_created: OnCellCreated = None,
) -> DecompositionResult:
    """Persist valid cells incrementally; failed calls never become atomic stops."""
    colony_id = f"{dish_id}-{colony_name}"
    created_at = datetime.now(UTC).isoformat()
    center = center or Cell(id=build_cell_key(dish_id, colony_name, 0, 0), colony_id=colony_id,
                            claim_text=claim, level=0, created_at=created_at)
    if not center.created_at:
        center.created_at = created_at
    cells = [center]
    edges: list[Edge] = []
    counts = {0: 1}

    def persist(cell: Cell) -> None:
        if db_path is not None:
            database.put_cell(db_path, {
                "id": cell.id, "colony": cell.colony_id, "level": cell.level,
                "seq": parse_key(cell.id)["seq"], "claim_text": cell.claim_text,
                "status": cell.status.value, "relation": cell.relation, "is_atomic": cell.is_atomic,
                "bedrock_reason": cell.bedrock_reason, "created_at": cell.created_at,
            })

    def add_children(parent: Cell, candidates: list[PremiseCandidate]) -> list[Cell]:
        level = parent.level + 1
        children = []
        for candidate in candidates:
            if counts.get(level, 0) >= settings.per_layer:
                break
            seq = counts.get(level, 0) + 1
            cell = Cell(id=build_cell_key(dish_id, colony_name, level, seq), colony_id=colony_id,
                        claim_text=candidate.claim_text, level=level, relation=candidate.relation,
                        created_at=datetime.now(UTC).isoformat(), dependents=[parent.id])
            edge = Edge(from_cell=parent.id, to_cell=cell.id)
            parent.dependencies.append(cell.id)
            cells.append(cell)
            edges.append(edge)
            children.append(cell)
            counts[level] = seq
            persist(cell)
            if db_path is not None:
                database.add_edge(db_path, from_cell=edge.from_cell, to_cell=edge.to_cell,
                                  edge_type=edge.edge_type, created_at=cell.created_at)
            if on_cell_created:
                on_cell_created(cell, [edge])
        return children

    persist(center)
    initial = runner.initial_decomposition(DecompositionInput(
        claim=claim, clarifications=[question.model_dump() for question in clarifications],
        guidance=guidance, max_premises=settings.per_layer,
    ), on_progress)
    queue = deque(add_children(center, initial.nodes))
    while queue:
        parent = queue.popleft()
        next_level = parent.level + 1
        if parent.level >= settings.depth or counts.get(next_level, 0) >= settings.per_layer:
            continue
        expansion = runner.expand_why(DecompositionInput(
            claim=parent.claim_text, parent_level=parent.level, parent_seq=parse_key(parent.id)["seq"],
            max_premises=settings.per_layer - counts.get(next_level, 0),
        ), on_progress)
        parent.is_atomic = expansion.is_atomic
        parent.bedrock_reason = expansion.bedrock_reason
        persist(parent)
        if not expansion.is_atomic:
            queue.extend(add_children(parent, expansion.sub_premises))
    return DecompositionResult(cells=cells, edges=edges, colony_name=colony_name, center_claim=claim)
