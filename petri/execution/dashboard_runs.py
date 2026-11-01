"""Durable preparation and fan-out for dashboard submissions."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from datetime import datetime, timezone

from dbos import DBOS, SetWorkflowID

from petri.execution.dbos_app import active_app
from petri.execution.run_controls import SeedRequest
from petri.execution.scheduler import ColonyRunSpec, grow_colony
from petri.execution.seed_workflow import SeedRunSpec, seed_colony
from petri.graph.colony import ColonyGraph
from petri.models import Cell, Colony, Event, build_cell_key
from petri.query.cells import list_colonies
from petri.storage import database
from petri.storage.backup import snapshot_before_destructive
from petri.storage.colonies import clear_unapproved_colony, reserve_colony, save_colony


@dataclass(frozen=True)
class DashboardSeedSpec:
    request: SeedRequest
    seed: SeedRunSpec

    @property
    def workflow_id(self) -> str:
        return self.seed.workflow_id


@dataclass(frozen=True)
class DashboardGrowSpec:
    id: str
    colonies: tuple[ColonyRunSpec, ...]


@DBOS.step(name="petri.prepare_dashboard_seed")
def prepare_dashboard_seed(spec: DashboardSeedSpec) -> SeedRunSpec:
    """Reserve and materialize a center only after the submission is durable."""
    app = active_app()
    with app.step_activity():
        return _prepare_dashboard_seed(app, spec)


def _prepare_dashboard_seed(app, spec: DashboardSeedSpec) -> SeedRunSpec:
    path = app.db_path
    seed = spec.seed
    identity = database.stable_event_id(seed.colony_id, "dashboard-seed", spec.workflow_id)
    with database.write_connection(path) as connection:
        if connection.execute("SELECT 1 FROM events WHERE id=?", (identity,)).fetchone():
            return seed
    existing = any(row["id"] == seed.colony_id for row in list_colonies(path))
    if existing and spec.request.regenerate:
        snapshot_before_destructive(app.petri_dir, seed.dish_id)
    now = datetime.now(timezone.utc).isoformat()
    center = Cell(
        id=build_cell_key(seed.dish_id, seed.colony_name, 0, 0),
        colony_id=seed.colony_id,
        claim_text=spec.request.claim,
        level=0,
        created_at=now,
    )
    graph = ColonyGraph(seed.colony_id)
    graph.add_cell(center)
    colony = Colony(
        id=seed.colony_id,
        dish=seed.dish_id,
        center_cell_id=center.id,
        center_claim=spec.request.claim,
        clarifications=spec.request.clarifications,
        created_at=now,
    )
    with database.write_connection(path) as connection:
        if connection.execute("SELECT 1 FROM events WHERE id=?", (identity,)).fetchone():
            return seed
        if existing and spec.request.regenerate:
            clear_unapproved_colony(
                path, seed.colony_id, connection=connection, release_reservation=False
            )
        else:
            reserve_colony(path, colony, connection=connection)
        save_colony(path, graph, colony, connection=connection)
        database.append_event(
            path,
            Event(
                id=identity,
                cell_id=center.id,
                timestamp=now,
                type="seed_started",
                agent="decomposition_lead",
                iteration=0,
                data={
                    "claim": spec.request.claim,
                    "workflow_id": spec.workflow_id,
                    "attempt": seed.attempt,
                },
            ),
            connection=connection,
        )
        database.append_event(
            path,
            Event(
                id=database.stable_event_id(center.id, "dashboard-decomposition", spec.workflow_id),
                cell_id=center.id,
                timestamp=now,
                type="decomposition_started",
                agent="decomposition_lead",
                iteration=0,
                data={"attempt": seed.attempt, "guidance": spec.request.guidance},
            ),
            connection=connection,
        )
    return seed


@DBOS.workflow(name="petri.dashboard_seed")
async def dashboard_seed(spec: DashboardSeedSpec):
    seed = await asyncio.to_thread(prepare_dashboard_seed, spec)
    return await seed_colony(seed)


@DBOS.step(name="petri.plan_dashboard_grow")
def plan_dashboard_grow(spec: DashboardGrowSpec) -> tuple[ColonyRunSpec, ...]:
    """Checkpoint each pass after external premises have reached their latest status."""
    from petri.execution.dbos_backend import DBOSExecutionBackend, _bounded

    app = active_app()
    backend = DBOSExecutionBackend()
    with app.step_activity():
        prior = DBOS.list_workflows(name="petri.grow_colony", load_output=False, load_input=False)
        planned = []
        for template in spec.colonies:
            eligible = backend.eligible_cells(
                app.petri_dir,
                colony_filter=template.colony_id,
                cell_ids=list(template.cell_ids) or None,
            )
            if not any(cell.id not in app.conflicts for cell in eligible):
                continue
            prefix = template.colony_id + ":grow:r"
            generations = []
            for run in prior:
                if run.workflow_id.startswith(prefix):
                    number = run.workflow_id[len(prefix) :].split(":", 1)[0]
                    if (
                        number.isdigit()
                        and replace(template, round_number=int(number)).workflow_id
                        == run.workflow_id
                    ):
                        generations.append(int(number))
            generation = max(template.round_number, max(generations, default=0) + 1)
            planned.append(replace(template, round_number=generation))
        return _bounded(tuple(planned))


@DBOS.workflow(name="petri.dashboard_grow")
async def dashboard_grow(spec: DashboardGrowSpec):
    results = []
    while colonies := await asyncio.to_thread(plan_dashboard_grow, spec):
        handles = []
        for colony in colonies:
            with SetWorkflowID(colony.workflow_id):
                handles.append(await DBOS.start_workflow_async(grow_colony, colony))
        results.extend(await asyncio.gather(*(handle.get_result() for handle in handles)))
    return results
