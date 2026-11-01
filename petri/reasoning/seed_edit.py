"""Validate and journal edits to an unapproved seed draft."""

from collections import Counter
from collections.abc import Callable
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from petri.models import DecompositionResult, Edge, parse_key
from petri.reasoning.decomposition_agents import DUPLICATE_THRESHOLD, parent_overlap
from petri.reasoning.decomposition_schemas import (
    InitialDecomposition,
    PremiseCandidate,
    ReferencePremise,
    WhyExpansion,
)
from petri.reasoning.decomposition_settings import DecompositionSettings
from petri.reasoning.seed_events import current_attempt
from petri.reasoning.seed_graph import SeedState
from petri.storage import database, edge_registry, event_log
from petri.storage.colonies import clear_unapproved_colony


class EditablePremise(PremiseCandidate):
    id: str
    is_atomic: bool = Field(default=False, strict=True)
    bedrock_reason: str | None = None


class DraftEdit(BaseModel):
    model_config = ConfigDict(extra="forbid")
    premises: list[EditablePremise]


def dump_draft(result: DecompositionResult) -> str:
    """Keep stable IDs visible; omitting a row drops that draft premise."""
    return yaml.safe_dump(
        {
            "premises": [
                {
                    "id": cell.id,
                    "claim_text": cell.claim_text,
                    "relation": cell.relation,
                    "is_atomic": cell.is_atomic,
                    "bedrock_reason": cell.bedrock_reason,
                }
                for cell in result.cells
                if cell.level > 0
            ]
        },
        sort_keys=False,
        allow_unicode=True,
    )


def validate_draft(
    text: str, original: DecompositionResult, settings: DecompositionSettings
) -> DecompositionResult:
    """Use agent schemas, reject parent restatements, and apply runtime caps."""
    document = DraftEdit.model_validate(yaml.safe_load(text))
    original_cells = {cell.id: cell for cell in original.cells}
    edits = {entry.id: entry for entry in document.premises}
    if len(edits) != len(document.premises):
        raise ValueError("Each premise ID must appear once.")
    if any(key not in original_cells or original_cells[key].level == 0 for key in edits):
        raise ValueError(
            "Edit existing draft premise IDs; the root and reference targets cannot be edited."
        )
    for entry in edits.values():
        WhyExpansion(
            is_atomic=entry.is_atomic, bedrock_reason=entry.bedrock_reason, sub_premises=[]
        )
    root = next(cell.model_copy(deep=True) for cell in original.cells if cell.level == 0)
    cells = [root]
    edges: list[Edge] = []
    counts: Counter[int] = Counter()
    pending = [root]
    while pending:
        parent = pending.pop(0)
        candidates: list[tuple[Edge, PremiseCandidate | ReferencePremise]] = []
        if parent.is_atomic or parent.level >= settings.depth:
            continue
        for edge in original.edges:
            if edge.from_cell != parent.id:
                continue
            if edge.edge_type == "cross_colony":
                candidates.append(
                    (edge, ReferencePremise(existing_cell_id=edge.to_cell, reason=edge.reason))
                )
            elif edge.to_cell in edits:
                entry = edits[edge.to_cell]
                if parent_overlap(parent.claim_text, entry.claim_text) >= DUPLICATE_THRESHOLD:
                    raise ValueError(
                        f"Premise {entry.id} restates its parent; use a distinct claim."
                    )
                candidates.append(
                    (edge, PremiseCandidate(claim_text=entry.claim_text, relation=entry.relation))
                )
        values = [value for _, value in candidates]
        if parent.level == 0:
            InitialDecomposition(nodes=values)
            cap = settings.level_one_cap
        else:
            WhyExpansion(is_atomic=False, sub_premises=values)
            if len(values) >= 2 and not any(
                isinstance(value, PremiseCandidate) and value.relation != "supports"
                for value in values
            ):
                raise ValueError(f"Expansion {parent.id} needs a limits or rebuts premise.")
            cap = settings.per_parent_cap
        cap = min(cap, settings.per_layer - counts[parent.level + 1])
        selected = candidates[:cap]
        if (
            parent.level > 0
            and len(selected) >= 2
            and not any(
                isinstance(value, PremiseCandidate) and value.relation != "supports"
                for _, value in selected
            )
        ):
            counter = next(
                (
                    item
                    for item in candidates[cap:]
                    if isinstance(item[1], PremiseCandidate) and item[1].relation != "supports"
                ),
                None,
            )
            if counter:
                selected[-1] = counter
        for edge, value in selected:
            edges.append(edge.model_copy(deep=True))
            if isinstance(value, ReferencePremise):
                continue
            entry = edits[edge.to_cell]
            child = original_cells[entry.id].model_copy(
                update=entry.model_dump(exclude={"id"}), deep=True
            )
            cells.append(child)
            counts[child.level] += 1
            pending.append(child)
    for cell in cells:
        cell.dependencies = [edge.to_cell for edge in edges if edge.from_cell == cell.id]
        cell.dependents = [edge.from_cell for edge in edges if edge.to_cell == cell.id]
    return original.model_copy(update={"cells": cells, "edges": edges})


def materialize_edited_state(path: Path, state: SeedState) -> None:
    """Replay a journaled edit without consulting older model responses."""
    with database._write(path) as connection:
        clear_unapproved_colony(path, state.colony_id, connection=connection)
        for cell in state.cells:
            database.put_cell(path, {
                **cell.model_dump(mode="json"), "colony": cell.colony_id,
                "seq": parse_key(cell.id)["seq"],
            }, connection=connection)
        for edge in state.edges:
            edge_registry.append(path, edge_registry.EdgeRecord(
                **edge.model_dump(), created_at=state.cells[0].created_at), connection=connection)


def save_draft_edit(
    path: Path,
    original: DecompositionResult,
    edited: DecompositionResult,
    settings: DecompositionSettings,
    original_text: str,
    edited_text: str,
    before_write: Callable[[], object],
) -> None:
    """Record the entire validated edit before replacing any draft rows."""
    center = next(cell for cell in edited.cells if cell.level == 0)
    if database.read_events(path, cell_id=center.id, event_type="colony_approved"):
        raise ValueError("Only an unapproved seed draft can be edited.")
    attempt = current_attempt(path, center.id)
    revision = (
        len(database.read_events(path, cell_id=center.id, event_type="decomposition_edited")) + 1
    )
    state = SeedState(
        colony_id=center.colony_id,
        attempt=attempt,
        claim=edited.center_claim,
        settings=settings,
        cells=edited.cells,
        edges=edited.edges,
        pending=[],
    )
    before_write()
    event_log.append_event(
        database.SQLiteEventLog(path),
        center.id,
        "decomposition_edited",
        "human",
        0,
        {
            "attempt": attempt,
            "original_text": original_text,
            "edited_text": edited_text,
            "state": state.model_dump(mode="json"),
        },
        node=f"seed-edit:{attempt}",
        sequence=revision,
    )
    materialize_edited_state(path, state)
