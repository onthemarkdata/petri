"""Run the same read-only domain scan manually or on a persisted DBOS schedule."""

from datetime import datetime
from typing import Any, cast

from dbos import DBOS
from opentelemetry.context import Context

from petri.analysis.scanner import ScanIssue, scan_database
from petri.execution.dbos_app import active_app
from petri.models import Event, EventType
from petri.storage import database
from petri.telemetry.setup import get_tracer

SCHEDULE_NAME = "petri-domain-scan"


@DBOS.step()
def scan_step(trigger: str, run_id: str) -> list[dict[str, Any]]:
    app = active_app()
    dish_id = app.db_path.parent.name
    with get_tracer("petri.scan").start_as_current_span(
        "petri.scan", context=Context(), attributes={
            "petri.command": "scan", "petri.dish_id": dish_id, "petri.scan.trigger": trigger,
        },
    ) as span:
        event_id = database.stable_event_id(dish_id, "scan", run_id)
        existing = database.read_events(app.db_path, event_type="scan_completed")
        previous = next((event for event in existing if event["id"] == event_id), None)
        if previous is not None:
            findings = previous["data"]["findings"]
        else:
            findings = [issue.model_dump() for issue in scan_database(app.db_path)]
            from datetime import UTC

            database.append_event(app.db_path, Event(
                id=event_id, cell_id=f"{dish_id}:scan", timestamp=datetime.now(UTC).isoformat(),
                type=EventType.scan_completed, agent="scanner", iteration=0,
                data={"findings": findings, "findings_count": len(findings), "trigger": trigger},
            ))
        span.set_attribute("petri.scan.findings_count", len(findings))
        return findings


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
