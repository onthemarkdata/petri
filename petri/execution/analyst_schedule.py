"""Optional durable schedule for the model-free research Analyst."""

from datetime import datetime
from typing import Any, cast

from dbos import DBOS

from petri.analysis.analyst import AnalystConfig
from petri.analysis.health import run_analyst
from petri.config import load_dish_config
from petri.execution.dbos_app import active_app
from petri.storage.database import stable_event_id

SCHEDULE_NAME = 'petri-research-analyst'


def analyst_settings() -> AnalystConfig:
    return load_dish_config(active_app().petri_dir).analyst


@DBOS.step(name='petri.analyst_report')
def analyst_report(scheduled_at: datetime) -> str:
    app = active_app()
    identity = stable_event_id(app.db_path.parent.name,'analyst',scheduled_at.isoformat())
    return run_analyst(app.db_path,dish_id=app.db_path.parent.name,config=analyst_settings(),
                       now=scheduled_at,report_id=identity).id


@DBOS.workflow(name='petri.analyst_tick')
def analyst_tick(scheduled_at: datetime, context: dict[str, Any]) -> None:
    if context.get('cron') == analyst_settings().schedule:
        analyst_report(scheduled_at)


def register_analyst_schedule() -> None:
    """Keep one configured schedule; an absent key pauses any previous schedule."""
    settings = analyst_settings()
    existing = DBOS.get_schedule(SCHEDULE_NAME)
    if settings.schedule is None:
        if existing is not None:
            DBOS.pause_schedule(SCHEDULE_NAME)
        return
    if existing is not None and existing['schedule'] == settings.schedule:
        DBOS.resume_schedule(SCHEDULE_NAME)
        return
    if existing is not None:
        DBOS.delete_schedule(SCHEDULE_NAME)
    DBOS.create_schedule(schedule_name=SCHEDULE_NAME,workflow_fn=cast(Any,analyst_tick),
                         schedule=settings.schedule,context={'cron':settings.schedule})
