"""Run the same read-only domain scan manually or on a persisted DBOS schedule."""

from datetime import datetime
from typing import Any, cast

from dbos import DBOS

from petri.analysis.scanner import ScanIssue
from petri.analysis.scan_run import run_domain_scan
from petri.execution.dbos_app import active_app
from petri.telemetry.setup import get_tracer

SCHEDULE_NAME = "petri-domain-scan"


@DBOS.step()
def scan_step(trigger: str, run_id: str) -> list[dict[str, Any]]:
    app = active_app()
    return run_domain_scan(app.db_path,dish_id=app.db_path.parent.name,trigger=trigger,
                           run_id=run_id,tracer=get_tracer('petri.scan'))


@DBOS.workflow()
def scan_workflow(scheduled_at: datetime, context: dict[str, Any]) -> list[dict[str, Any]]:
    trigger = context.get("trigger", "scheduled")
    if trigger == "scheduled" and context.get("cron") is not None and context["cron"] != active_app().settings.scan_cron:
        return []
    return scan_step(trigger, str(DBOS.workflow_id))


def register_scan_schedule(cron: str | None) -> None:
    """Create or update only Petri's scan schedule; absence disables it."""
    existing = DBOS.get_schedule(SCHEDULE_NAME)
    if cron is None:
        if existing is not None:
            DBOS.pause_schedule(SCHEDULE_NAME)
        return
    if existing is not None and existing["schedule"] == cron:
        DBOS.resume_schedule(SCHEDULE_NAME)
        return
    if existing is not None:
        DBOS.delete_schedule(SCHEDULE_NAME)
    DBOS.create_schedule(
        schedule_name=SCHEDULE_NAME, workflow_fn=cast(Any, scan_workflow),
        schedule=cron, context={"trigger": "scheduled", "cron": cron},
    )


def run_scan_now() -> list[ScanIssue]:
    from datetime import UTC

    return [ScanIssue.model_validate(item) for item in scan_workflow(datetime.now(UTC), {"trigger": "manual"})]
