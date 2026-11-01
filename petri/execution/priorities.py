"""Opt-in graph priorities assigned at enqueue time without changing workflows."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from dbos import DBOS, SetEnqueueOptions, SetWorkflowID

from petri.analysis.edge_intelligence import EdgeIntelligenceReport, analyze_edges
from petri.config import load_dish_config
from petri.execution.config import SchedulingConfig
from petri.models import Event, parse_key
from petri.storage import database
from petri.storage.edge_registry import DishGraph
from petri.telemetry import get_tracer

if TYPE_CHECKING:
    from petri.execution.dbos_app import DBOSApp
    from petri.execution.dbos_backend import CellRunSpec


def priority_plan(report: EdgeIntelligenceReport) -> dict[str, int]:
    targets = set(report.convergence_points)
    ordered = report.convergence_points + sorted((cell_id for cell_id in report.cells if cell_id not in targets),
        key=lambda cell_id:(-report.cells[cell_id].importance,report.cells[cell_id].level,cell_id))
    return {cell_id:index+1 for index,cell_id in enumerate(ordered)}


def prepare_priority_pass(app: DBOSApp, colony_id: str) -> None:
    """Compute one report for this grow pass; eligibility is decided separately."""
    settings = SchedulingConfig.model_validate(load_dish_config(app.petri_dir).get('scheduling',{}))
    if settings.prioritize_targets:
        app.target_priority_plans[colony_id] = priority_plan(analyze_edges(DishGraph.read(app.db_path)))
    else:
        app.target_priority_plans[colony_id] = {}


def assigned_priority(app: DBOSApp, cell_id: str, level: int) -> int:
    if level < 0:
        raise ValueError('Cell level must be nonnegative')
    colony = str(parse_key(cell_id,strict=True)['colony_prefix'])
    if colony not in app.target_priority_plans:
        prepare_priority_pass(app,colony)
    return app.target_priority_plans[colony].get(cell_id,level+1)


async def enqueue_with_priority(app: DBOSApp, spec: CellRunSpec, level: int):
    from petri.execution.dbos_app import QUEUE_NAME
    from petri.execution.dbos_backend import validate_cell

    status = await DBOS.get_workflow_status_async(spec.workflow_id)
    if status is not None and status.status == "CANCELLED":
        return await DBOS.resume_workflow_async(spec.workflow_id, queue_name=QUEUE_NAME)
    priority = assigned_priority(app,spec.cell_id,level)
    event,_ = database.append_event(app.db_path,Event(
        id=database.stable_event_id(spec.cell_id,'queued',spec.round_number),cell_id=spec.cell_id,
        timestamp=datetime.now(timezone.utc).isoformat(),type='cell_queued',agent='execution_backend',iteration=0,
        data={'priority':priority,'workflow_id':spec.workflow_id,'round_number':spec.round_number}))
    priority = int(event.data['priority'])
    with get_tracer().start_as_current_span('petri.enqueue_cell',attributes={
        'petri.cell_id':spec.cell_id,'petri.colony_id':str(parse_key(spec.cell_id,strict=True)['colony_prefix']),
        'petri.dish_id':app.db_path.parent.name,'petri.queue.priority':priority,
    }):
        with SetWorkflowID(spec.workflow_id), SetEnqueueOptions(priority=priority):
            return await DBOS.enqueue_workflow_async(QUEUE_NAME,validate_cell,spec)
