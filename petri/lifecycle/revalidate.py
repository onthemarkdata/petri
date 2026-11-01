"""Checkpoint evidence ingestion and fork settled cell rounds at research."""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from dbos import DBOS, DBOSClient, SetWorkflowID

from petri.execution.dbos_app import QUEUE_NAME, active_app
from petri.execution.dbos_backend import CellRunSpec, StepResult, _bounded
from petri.execution.models import FeedRequest
from petri.execution.pipeline import SyncConflict
from petri.execution.scheduler import current_round, enqueue_validation
from petri.lifecycle.propagation import propagate_upward
from petri.reasoning.ingest import ingest
from petri.storage import database
from petri.storage.colonies import load_colony
from petri.storage.event_log import derive_event_id
from petri.storage.evidence_files import write_once
from petri.models import Event, parse_key


@dataclass(frozen=True)
class EvidenceRef:
    path: str
    digest: str
    source_kind: str


@dataclass(frozen=True)
class ForkPlan:
    spec: CellRunSpec
    level: int
    original_workflow_id: str | None
    start_step: int | None


def research_boundary(steps) -> int | None:
    """Use the stored function ID, including retries, rather than a list index."""
    candidates = [
        step["function_id"]
        for step in steps
        if step["function_name"] == "petri.cell_phase"
        and isinstance(step["output"], StepResult)
        and getattr(step["output"], "phase", None) == "research"
    ]
    return min(candidates, default=None)


@DBOS.step(name="petri.ingest_evidence")
def ingest_step(request: FeedRequest) -> EvidenceRef:
    app = active_app()
    with app.step_activity():
        result = ingest(request.source)
        digest = hashlib.sha256(result.content.encode()).hexdigest()
        path = app.db_path.parent / "evidence" / f"{digest}.json"
        write_once(path, result.model_dump_json())
    return EvidenceRef(str(path.relative_to(app.db_path.parent)), digest, result.source_type)


@DBOS.step(name="petri.prepare_feed_fork")
def prepare_fork(request: FeedRequest, evidence: EvidenceRef) -> ForkPlan:
    with active_app().step_activity():
        return _prepare_fork(request, evidence)


def _prepare_fork(request: FeedRequest, evidence: EvidenceRef) -> ForkPlan:
    app = active_app()
    content = json.loads((app.db_path.parent / evidence.path).read_text(encoding="utf-8"))[
        "content"
    ]
    if hashlib.sha256(content.encode()).hexdigest() != evidence.digest:
        raise ValueError("Fed evidence differs from its ingested content hash")
    identity = f"feed:{evidence.digest}"
    client = DBOSClient(system_database_url=f"sqlite:///{app.petri_dir / 'dbos.sqlite'}")
    try:
        runs = client.list_workflows(name="petri.validate_cell")
        with database._write(app.db_path) as conn:
            row = conn.execute(
                "SELECT status,level FROM cells WHERE id=?", (request.cell_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"Cell {request.cell_id!r} is not in the domain database")
            previous = conn.execute(
                "SELECT * FROM execution_round_requests WHERE cell_id=? AND request_id=?",
                (request.cell_id, identity),
            ).fetchone()
            if previous is None:
                if row["status"] in {"NEW", "RESEARCH", "RED_TEAM", "EVALUATE"}:
                    raise ValueError(
                        "Wait for the current cell round before feeding another source"
                    )
                generation = current_round(conn, request.cell_id) + 1
                prior_status = row["status"]
                conn.execute(
                    "INSERT INTO execution_round_requests VALUES (?,?,?,?,?)",
                    (
                        request.cell_id,
                        identity,
                        generation,
                        prior_status,
                        "fed_evidence",
                    ),
                )
            else:
                generation, prior_status = previous["round_number"], previous["prior_status"]
            prefix = f"{request.cell_id}:r"
            originals = [
                (int(run.workflow_id[len(prefix) :]), run.workflow_id)
                for run in runs
                if run.workflow_id.startswith(prefix)
                and run.workflow_id[len(prefix) :].isdigit()
                and run.status == "SUCCESS"
                and int(run.workflow_id[len(prefix) :]) < generation
            ]
            original = max(originals, default=(0, None))[1]
            start = research_boundary(client.list_workflow_steps(original)) if original else None
            spec = CellRunSpec(
                request.cell_id, generation, request.today_iso, request.max_iterations
            )
            metadata = {
                "original_workflow_id": original,
                "forked_workflow_id": spec.workflow_id,
                "generation": generation,
            }

            def journal(kind, data):
                event, _ = database.append_event(
                    app.db_path,
                    Event(
                        id=derive_event_id(
                            request.cell_id,
                            round_number=generation,
                            node="feed",
                            agent="execution_backend",
                            iteration=0,
                            event_type=kind,
                        ),
                        cell_id=request.cell_id,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        type=kind,
                        agent="execution_backend",
                        iteration=0,
                        data=data,
                    ),
                    connection=conn,
                )
                return event

            journal(
                "evidence_fed",
                {
                    **metadata,
                    "content": content,
                    "evidence_path": evidence.path,
                    "content_sha256": evidence.digest,
                    "source_kind": evidence.source_kind,
                },
            )
            context = journal(
                "cell_reopened",
                {
                    **metadata,
                    "trigger": "fed_evidence",
                    "prior_status": prior_status,
                    "today_iso": request.today_iso,
                    "max_iterations": request.max_iterations,
                },
            ).data
            if previous is None:
                conn.execute("UPDATE cells SET status='NEW' WHERE id=?", (request.cell_id,))
            spec = CellRunSpec(
                request.cell_id, generation, context["today_iso"], context["max_iterations"]
            )
    finally:
        client.destroy()
    colony_id = str(parse_key(request.cell_id, strict=True)["colony_prefix"])
    graph, _ = load_colony(app.db_path, colony_id)
    propagate_upward(app.petri_dir, request.cell_id, graph, app.db_path.parent.name, identity)
    return _bounded(ForkPlan(spec, row["level"], original, start))


_FORK_LOCK = threading.Lock()


@DBOS.step(name="petri.fork_fed_cell")
def fork_step(plan: ForkPlan) -> str:
    """Attach after a crash between the SDK's fork commit and this checkpoint."""
    with active_app().step_activity():
        return _fork_once(plan)


def _fork_once(plan: ForkPlan) -> str:
    app = active_app()
    client = DBOSClient(system_database_url=f"sqlite:///{app.petri_dir / 'dbos.sqlite'}")
    try:
        with _FORK_LOCK:
            existing = client.list_workflows(workflow_ids=[plan.spec.workflow_id])
            if not existing:
                assert plan.original_workflow_id is not None and plan.start_step is not None
                with SetWorkflowID(plan.spec.workflow_id):
                    client.fork_workflow(
                        plan.original_workflow_id,
                        plan.start_step,
                        queue_name=QUEUE_NAME,
                    )
    finally:
        client.destroy()
    return plan.spec.workflow_id


@DBOS.workflow(name="petri.feed_evidence")
async def feed_workflow(request: FeedRequest):
    app = active_app()
    if request.cell_id in app.conflicts:
        return SyncConflict(reason=app.conflicts[request.cell_id]["reason"])
    evidence = await asyncio.to_thread(ingest_step, request)
    plan = await asyncio.to_thread(prepare_fork, request, evidence)
    from petri.execution.otel import set_cell_workflow_attributes

    set_cell_workflow_attributes(plan.spec, app, DBOS.span)
    handle: Any
    if plan.original_workflow_id is not None and plan.start_step is not None:
        workflow_id = await asyncio.to_thread(fork_step, plan)
        handle = await DBOS.retrieve_workflow_async(workflow_id)
    else:
        # Older checkpoints without phase metadata cannot identify a safe fork point.
        handle = await enqueue_validation(plan.spec, plan.level)
    return await handle.get_result()
