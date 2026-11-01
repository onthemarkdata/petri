"""Durable bottom-up scheduling over the colony graph stored in SQLite."""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from contextlib import closing

from dbos import DBOS

from petri.execution.dbos_app import active_app
from petri.execution.dbos_backend import CellRunSpec, _bounded
from petri.execution.models import ReopenRequest
from petri.execution.pipeline import SyncConflict
from petri.execution.priorities import enqueue_with_priority, prepare_priority_pass
from petri.models import CellStatus, parse_key
from petri.storage import database
from petri.storage.colonies import load_colony
from petri.storage.event_log import append_event


@dataclass(frozen=True)
class ColonyRunSpec:
    colony_id: str
    round_number: int
    today_iso: str
    max_iterations: int = 5
    cell_ids: tuple[str, ...] = ()

    def __post_init__(self):
        if (
            not self.colony_id
            or self.round_number < 0
            or self.max_iterations < 1
            or not self.today_iso
        ):
            raise ValueError("Provide a colony, nonnegative round, positive budget, and date")

    @property
    def workflow_id(self):
        scope = hashlib.sha256("\n".join(sorted(set(self.cell_ids))).encode()).hexdigest()[:16]
        suffix = f":{scope}" if self.cell_ids else ""
        return f"{self.colony_id}:grow:r{self.round_number}{suffix}"


@dataclass(frozen=True)
class ColonyRunResult:
    colony_id: str
    completed: tuple[str, ...]
    not_eligible: tuple[str, ...]




def prepare_rounds(db_path):
    with database._write(db_path) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS execution_round_requests ("
            "cell_id TEXT NOT NULL, request_id TEXT NOT NULL, round_number INTEGER NOT NULL, "
            "prior_status TEXT NOT NULL, trigger TEXT NOT NULL, "
            "PRIMARY KEY(cell_id,request_id), UNIQUE(cell_id,round_number))"
        )


def current_round(conn, cell_id):
    rounds = [
        row[0]
        for row in conn.execute(
            "SELECT round_number FROM execution_round_requests WHERE cell_id = ?", (cell_id,)
        )
    ]
    prefix = f"{cell_id}:r"
    for row in conn.execute(
        "SELECT workflow_id FROM execution_admissions WHERE workflow_id >= ? AND workflow_id < ?",
        (prefix, prefix + "\uffff"),
    ):
        if row[0].startswith(prefix) and row[0][len(prefix) :].isdigit():
            rounds.append(int(row[0][len(prefix) :]))
    return max(rounds, default=1)


@DBOS.step(name="petri.eligible_cells")
def eligible_cells(spec: ColonyRunSpec) -> list[tuple[CellRunSpec, int]]:
    path = active_app().db_path
    graph, _ = load_colony(path, spec.colony_id)
    prepare_priority_pass(active_app(),spec.colony_id)
    statuses = {row["id"]:CellStatus(row["status"]) for row in database.list_cells(path)}
    with closing(database.connect(path)) as conn:
        return _bounded(
            [
                (
                    CellRunSpec(
                        cell.id, current_round(conn, cell.id), spec.today_iso, spec.max_iterations
                    ),
                    cell.level,
                )
                for cell in graph.get_eligible_for_validation(statuses)
                if cell.id not in active_app().conflicts
                and (not spec.cell_ids or cell.id in spec.cell_ids)
            ]
        )


@DBOS.step(name="petri.blocked_cells")
def blocked_cells(spec: ColonyRunSpec) -> tuple[str, ...]:
    return tuple(
        row["id"]
        for row in database.list_cells(active_app().db_path, colony=spec.colony_id)
        if row["status"] == "NEW" and (not spec.cell_ids or row["id"] in spec.cell_ids)
    )


async def enqueue_validation(spec: CellRunSpec, level: int):
    return await enqueue_with_priority(active_app(),spec,level)


@DBOS.workflow(name="petri.grow_colony")
async def grow_colony(spec: ColonyRunSpec) -> ColonyRunResult:
    from dbos._error import DBOSAwaitedWorkflowCancelledError
    from petri.execution.control import cancel_at_boundary

    app = active_app()
    cancel_at_boundary(app.petri_dir / "dbos.sqlite", spec.workflow_id)
    if any(parse_key(cell).get("colony_prefix") == spec.colony_id for cell in app.conflicts):
        return ColonyRunResult(
            spec.colony_id,
            (),
            tuple(
                row["id"]
                for row in database.list_cells(app.db_path, colony=spec.colony_id)
                if row["status"] == "NEW" and (not spec.cell_ids or row["id"] in spec.cell_ids)
            ),
        )
    completed: list[str] = []
    while candidates := await asyncio.to_thread(eligible_cells, spec):
        cancel_at_boundary(app.petri_dir / "dbos.sqlite", spec.workflow_id)
        handles = [await enqueue_validation(cell, level) for cell, level in candidates]
        try:
            await asyncio.gather(*(handle.get_result() for handle in handles))
        except DBOSAwaitedWorkflowCancelledError:
            cancel_at_boundary(app.petri_dir / "dbos.sqlite", spec.workflow_id)
            raise
        cancel_at_boundary(app.petri_dir / "dbos.sqlite", spec.workflow_id)
        completed.extend(cell.cell_id for cell, _ in candidates)
    blocked = await asyncio.to_thread(blocked_cells, spec)
    return _bounded(ColonyRunResult(spec.colony_id, tuple(completed), blocked))


@DBOS.step(name="petri.resolve_next_round")
def resolve_next_round(request: ReopenRequest) -> tuple[CellRunSpec, int]:
    path = active_app().db_path
    with database._write(path) as conn:
        row = conn.execute(
            "SELECT status,level FROM cells WHERE id = ?", (request.cell_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"Cell {request.cell_id!r} is not in the domain database")
        previous = conn.execute(
            "SELECT * FROM execution_round_requests WHERE cell_id = ? AND request_id = ?",
            (request.cell_id, request.request_id),
        ).fetchone()
        if previous is None:
            prior_status = row[0]
            if prior_status in {"NEW", "RESEARCH", "RED_TEAM", "EVALUATE"}:
                raise ValueError("Wait for the current cell round before requesting another")
            round_number = current_round(conn, request.cell_id) + 1
            conn.execute(
                "INSERT INTO execution_round_requests VALUES (?,?,?,?,?)",
                (
                    request.cell_id,
                    request.request_id,
                    round_number,
                    prior_status,
                    request.trigger,
                ),
            )
            conn.execute(
                "UPDATE cells SET status = ? WHERE id = ?", (CellStatus.NEW.value, request.cell_id)
            )
        else:
            round_number = previous["round_number"]
            prior_status = previous["prior_status"]
    append_event(
        database.SQLiteEventLog(path, request.cell_id),
        request.cell_id,
        "cell_reopened",
        "execution_backend",
        0,
        {"trigger": request.trigger, "prior_status": prior_status},
        round_number=round_number,
        node="reopen",
    )
    return (
        CellRunSpec(request.cell_id, round_number, request.today_iso, request.max_iterations),
        row["level"],
    )


@DBOS.workflow(name="petri.revalidate_cell")
async def revalidate_cell(request: ReopenRequest):
    if request.cell_id in active_app().conflicts:
        return SyncConflict(reason=active_app().conflicts[request.cell_id]["reason"])
    spec, level = await asyncio.to_thread(resolve_next_round, request)
    handle = await enqueue_validation(spec, level)
    return await handle.get_result()
