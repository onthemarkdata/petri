"""Reserve bounded proposal capacity and atomically materialize proposed claims."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from petri.graph.colony import ColonyGraph
from petri.lifecycle.config import LifecycleConfig
from petri.lifecycle.triggers import RedecompositionTrigger
from petri.models import Cell, CellStatus, DecompositionResult, Edge, Event, EventType, build_cell_key, parse_key
from petri.reasoning.decomposition_schemas import PremiseCandidate, WhyExpansion
from petri.storage import database
from petri.storage.colonies import load_colony, save_colony


class Proposal(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: str
    parent_cell_id: str
    generation: int = Field(ge=1)
    state: str
    capacity: int = Field(ge=0)
    trigger: RedecompositionTrigger
    child_ids: list[str] = Field(default_factory=list)
    reason: str = ""


def _proposal(row: sqlite3.Row) -> Proposal:
    return Proposal(id=row["id"], parent_cell_id=row["parent_cell_id"], generation=row["generation"],
        state=row["state"], capacity=row["capacity"], trigger=json.loads(row["trigger_json"]),
        child_ids=json.loads(row["child_ids"]), reason=row["reason"])


def get_proposal(db_path: Path, proposal_id: str) -> Proposal:
    with closing(database.connect(db_path)) as conn:
        row = conn.execute("SELECT * FROM redecomposition_proposals WHERE id=?", (proposal_id,)).fetchone()
        if row is None:
            raise KeyError(f"Proposal not found: {proposal_id}")
        return _proposal(row)


def proposal_identity(trigger: RedecompositionTrigger) -> str:
    parts = (trigger.cell_id, "redecompose", trigger.generation + 1)
    return database.stable_event_id(*parts, trigger.source_event_id) if trigger.source_event_id else database.stable_event_id(*parts)


def _event(conn: sqlite3.Connection, db_path: Path, proposal: Proposal, kind: str, data: dict) -> None:
    database.append_event(db_path, Event(
        id=database.stable_event_id(proposal.id, kind), cell_id=proposal.parent_cell_id,
        timestamp=datetime.now(UTC).isoformat(), type=EventType(kind), agent="decomposition_lead",
        iteration=0, data=data,
    ), connection=conn)


def _halt(conn: sqlite3.Connection, db_path: Path, proposal: Proposal, reason: str) -> Proposal:
    conn.execute("UPDATE redecomposition_proposals SET state='halted',capacity=0,reason=? WHERE id=?", (reason, proposal.id))
    _event(conn, db_path, proposal, "redecomposition_halted", {
        "proposal_id": proposal.id, "generation": proposal.generation, "reason": reason,
    })
    return proposal.model_copy(update={"state": "halted", "capacity": 0, "reason": reason})


def reserve_proposal(db_path: Path, trigger: RedecompositionTrigger, config: LifecycleConfig) -> Proposal:
    """Reserve the maximum new-cell count before inference; replay keeps the same reservation."""
    generation = trigger.generation + 1
    proposal_id = proposal_identity(trigger)
    with database.write_connection(db_path) as conn:
        row = conn.execute("SELECT * FROM redecomposition_proposals WHERE id=?", (proposal_id,)).fetchone()
        if row is not None:
            return _proposal(row)
        parent = conn.execute("SELECT * FROM active_cells WHERE id=?", (trigger.cell_id,)).fetchone()
        if parent is None:
            raise ValueError(f"Cannot re-decompose a missing cell: {trigger.cell_id}")
        used = conn.execute("SELECT COALESCE(SUM(capacity),0) FROM redecomposition_proposals WHERE state<>'halted'").fetchone()[0]
        capacity = min(config.per_parent_cap, max(0, config.dish_proposal_budget - used))
        proposal = Proposal(id=proposal_id, parent_cell_id=trigger.cell_id, generation=generation,
            state="reserved", capacity=capacity, trigger=trigger)
        conn.execute("INSERT INTO redecomposition_proposals "
            "(id,parent_cell_id,generation,state,capacity,trigger_json,created_at) VALUES (?,?,?,?,?,?,?)",
            (proposal.id, trigger.cell_id, generation, "reserved", capacity, trigger.model_dump_json(), datetime.now(UTC).isoformat()))
        reason = ""
        if generation > config.max_rounds_per_cell:
            reason = "The cell has reached its re-decomposition round limit."
        elif config.stop_at_bedrock and parent["is_atomic"]:
            reason = "The triggering cell is marked atomic."
        elif capacity == 0:
            reason = "The dish proposal budget is exhausted."
        return _halt(conn, db_path, proposal, reason) if reason else proposal


def journal_output(db_path: Path, proposal_id: str, output: WhyExpansion) -> None:
    """Keep the first validated response so a restarted materialization never reruns inference."""
    with database.write_connection(db_path) as conn:
        conn.execute("UPDATE redecomposition_proposals SET output_json=? WHERE id=? AND output_json IS NULL AND state='reserved'",
            (output.model_dump_json(), proposal_id))


def recorded_output(db_path: Path, proposal_id: str) -> WhyExpansion | None:
    with closing(database.connect(db_path)) as conn:
        value = conn.execute("SELECT output_json FROM redecomposition_proposals WHERE id=?", (proposal_id,)).fetchone()
        return WhyExpansion.model_validate_json(value[0]) if value and value[0] else None


def materialize_proposal(db_path: Path, proposal_id: str) -> Proposal:
    """Publish the graph and proposal event in one transaction; proposed cells stay deferred."""
    with database.write_connection(db_path) as conn:
        row = conn.execute("SELECT * FROM redecomposition_proposals WHERE id=?", (proposal_id,)).fetchone()
        if row is None:
            raise KeyError(proposal_id)
        proposal = _proposal(row)
        if proposal.state != "reserved":
            return proposal
        if row["output_json"] is None:
            raise ValueError("A proposal needs a recorded model output before materialization")
        output = WhyExpansion.model_validate_json(row["output_json"])
        if output.is_atomic or not output.sub_premises:
            return _halt(conn, db_path, proposal, output.bedrock_reason or "The decomposer produced no sub-claims.")
        parent_row = conn.execute("SELECT * FROM active_cells WHERE id=?", (proposal.parent_cell_id,)).fetchone()
        if parent_row is None:
            return _halt(conn, db_path, proposal, "The triggering cell was removed before materialization.")
        graph, colony = load_colony(db_path, parent_row["colony"])
        parent = graph.get_cell(proposal.parent_cell_id)
        candidates = list(output.sub_premises)
        if proposal.trigger.trigger_verdict == "CRITICAL_FLAW_FOUND":
            candidates.sort(key=lambda item: not (isinstance(item, PremiseCandidate) and item.relation in {"limits", "rebuts"}))
        candidates = candidates[:proposal.capacity]
        level = parent.level + 1
        seq = conn.execute("SELECT COALESCE(MAX(seq),-1)+1 FROM cells WHERE colony=? AND level=?", (colony.id, level)).fetchone()[0]
        slug = parse_key(parent.id, colony.dish, strict=True)["colony"]
        children: list[Cell] = []
        edges: list[Edge] = []
        for candidate in candidates:
            if isinstance(candidate, PremiseCandidate):
                child = Cell(id=build_cell_key(colony.dish, slug, level, seq), colony_id=colony.id,
                    level=level, claim_text=candidate.claim_text, status=CellStatus.DEFER_CLOSED,
                    relation=candidate.relation, created_at=datetime.now(UTC).isoformat())
                seq += 1
                graph.add_cell(child)
                children.append(child)
                edge = Edge(from_cell=parent.id, to_cell=child.id, edge_type="intra_colony")
            else:
                edge = Edge(from_cell=parent.id, to_cell=candidate.existing_cell_id,
                    edge_type="cross_colony", reason=candidate.reason)
            graph.add_edge(edge)
            edges.append(edge)
        _apply_levels(graph, colony.center_cell_id)
        DecompositionResult(colony_name=colony.id, cells=children, edges=edges)
        save_colony(db_path, graph, colony, connection=conn)
        child_ids = [cell.id for cell in children]
        guard_status = "truncated" if len(output.sub_premises) > proposal.capacity else "within_limits"
        conn.execute("UPDATE redecomposition_proposals SET state='proposed',capacity=?,child_ids=?,reason=? WHERE id=?",
            (len(children), json.dumps(child_ids), guard_status, proposal.id))
        _event(conn, db_path, proposal, "redecomposition_proposed", {
            "proposal_id": proposal.id, "parent_cell_id": parent.id, "generation": proposal.generation,
            "trigger_verdict": proposal.trigger.trigger_verdict, "child_cell_ids": child_ids, "guard_status": guard_status,
        })
        return proposal.model_copy(update={"state": "proposed", "capacity": len(children), "child_ids": child_ids, "reason": guard_status})


def _apply_levels(graph: ColonyGraph, center: str) -> None:
    own = {cell.id: cell for cell in graph.get_all_cells()}
    for cell_id, level in graph.compute_levels(center).items():
        if cell_id in own:
            own[cell_id].level = level
