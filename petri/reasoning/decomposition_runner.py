"""Build a cell graph from validated agent outputs with explicit persistence."""

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from petri.models import Cell, ClarifyingQuestion, DecompositionResult, Edge, build_cell_key, parse_key
from petri.reasoning.decomposition_agents import DecompositionInput
from petri.reasoning.decomposition_schemas import InitialDecomposition, PremiseCandidate, WhyExpansion
from petri.reasoning.decomposition_settings import DecompositionSettings
from petri.storage import database, event_log
from petri.reasoning.seed_checkpoint import SeedCheckpointStore, current_attempt
from petri.reasoning.seed_graph import ExpansionNode, InitialNode, SeedState

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
    resume: bool = False, checkpoint_store: SeedCheckpointStore | None = None,
) -> DecompositionResult:
    """Persist valid cells incrementally; failed calls never become atomic stops."""
    colony_id = f"{dish_id}-{colony_name}"
    created_at = datetime.now(UTC).isoformat()
    center = center or Cell(id=build_cell_key(dish_id, colony_name, 0, 0), colony_id=colony_id,
                            claim_text=claim, level=0, created_at=created_at)
    if not center.created_at:
        center.created_at = created_at
    attempt = current_attempt(db_path, center.id) if db_path else 0
    store = checkpoint_store or (SeedCheckpointStore(db_path) if db_path else None)
    state = store.load(colony_id, attempt) if resume and store else None
    rebuilding = state is None
    if state is None:
        center.dependencies = []
        center.dependents = []
        state = SeedState(colony_id=colony_id, attempt=attempt, claim=claim,
                          guidance=guidance, settings=settings, cells=[center])
    elif state.claim != claim:
        raise ValueError("Resume claim differs from the stored seed claim")
    settings, guidance = state.settings, state.guidance
    cells, edges = state.cells, state.edges
    counts: dict[int, int] = {}
    for cell in cells:
        counts[cell.level] = max(counts.get(cell.level, 0), parse_key(cell.id)["seq"])
    counts[0] = 1
    existing = {row["id"]: row for row in database.list_cells(db_path, colony=colony_id)} if db_path else {}
    outputs = {}
    if resume and db_path:
        for event in database.read_events(db_path):
            data = event["data"]
            if event["type"] == "cell_expansion_completed" and data.get("attempt") == attempt:
                if event["cell_id"] in existing and "output" in data:
                    outputs[event["cell_id"]] = data["output"]
                    if rebuilding and event["cell_id"] == center.id:
                        settings = DecompositionSettings.model_validate(data["settings"])
                        guidance = data.get("guidance", "")
                        state.settings, state.guidance = settings, guidance

    def journal(parent: Cell, output: InitialDecomposition | WhyExpansion) -> None:
        if db_path is None:
            return
        payload = {"attempt": attempt, "output": output.model_dump(mode="json"),
                   "settings": settings.model_dump(), "guidance": guidance}
        event_log.append_event(database.SQLiteEventLog(db_path), parent.id,
            "cell_expansion_completed", "decomposition_lead", 0, payload,
            node=f"seed-expansion:{attempt}")
        if isinstance(output, WhyExpansion) and output.is_atomic:
            event_log.append_event(database.SQLiteEventLog(db_path), parent.id,
                "cell_marked_atomic", "five_whys", 0,
                {"attempt": attempt, "bedrock_reason": output.bedrock_reason},
                node=f"seed-expansion:{attempt}")

    def checkpoint() -> None:
        if store:
            store.save(state)

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
                        created_at=existing.get(build_cell_key(dish_id, colony_name, level, seq), {}).get("created_at", datetime.now(UTC).isoformat()), dependents=[parent.id])
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
            if on_cell_created and cell.id not in existing:
                on_cell_created(cell, [edge])
        return children

    persist(cells[0])
    checkpoint()
    while state.pending:
        node = state.pending[0]
        if isinstance(node, InitialNode):
            initial = (InitialDecomposition.model_validate(outputs[cells[0].id])
                       if cells[0].id in outputs else runner.initial_decomposition(DecompositionInput(
                           claim=claim, clarifications=[question.model_dump() for question in clarifications],
                           guidance=guidance,
                       ), on_progress))
            journal(cells[0], initial)
            children = add_children(cells[0], initial.nodes[:settings.level_one_cap])
            state.pending = [ExpansionNode(cell_id=child.id) for child in children]
            checkpoint()
            continue
        parent = next(cell for cell in cells if cell.id == node.cell_id)
        next_level = parent.level + 1
        if parent.level >= settings.depth or counts.get(next_level, 0) >= settings.per_layer:
            state.pending.pop(0)
            checkpoint()
            continue
        def warn(reason: str, claim_text: str, threshold: float) -> None:
            if on_progress:
                on_progress(reason + "\n")
            if db_path is not None:
                event_log.append_event(database.SQLiteEventLog(db_path), parent.id,
                    "decomposition_warning", "five_whys", 0,
                    {"reason": reason, "claim_text": claim_text, "threshold": threshold},
                    node=f"parent-restatement:{attempt}", sequence=warning_count[0])
                warning_count[0] += 1

        warning_count = [0]
        expansion = (WhyExpansion.model_validate(outputs[parent.id]) if parent.id in outputs
                     else runner.expand_why(DecompositionInput(
                         claim=parent.claim_text, parent_level=parent.level,
                         parent_seq=parse_key(parent.id)["seq"], on_warning=warn,
                     ), on_progress))
        # Save validated model output before materialization so a crash during
        # cell writes can replay the output without requesting it again.
        journal(parent, expansion)
        parent.is_atomic = expansion.is_atomic
        parent.bedrock_reason = expansion.bedrock_reason
        persist(parent)
        children = []
        if not expansion.is_atomic:
            limit = min(settings.per_parent_cap, settings.per_layer - counts.get(next_level, 0))
            candidates = expansion.sub_premises[:limit]
            if len(candidates) >= 2 and all(child.relation == "supports" for child in candidates):
                counterargument = next((child for child in expansion.sub_premises[limit:]
                                        if child.relation != "supports"), None)
                if counterargument is not None:
                    candidates[-1] = counterargument
            children = add_children(parent, candidates)
        state.pending.pop(0)
        state.pending.extend(ExpansionNode(cell_id=child.id) for child in children)
        checkpoint()
    return DecompositionResult(cells=cells, edges=edges, colony_name=colony_name, center_claim=claim)
