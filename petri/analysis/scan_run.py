"""Run and record one domain scan without starting an execution runtime."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from opentelemetry.context import Context
from opentelemetry.trace import Tracer

from petri.analysis.scanner import scan_database
from petri.models import Event, EventType
from petri.query.events import query_events
from petri.storage import database
from petri.telemetry.setup import get_tracer


def run_domain_scan(
    db_path: Path, *, dish_id: str, trigger: str, run_id: str, tracer: Tracer | None = None
) -> list[dict[str, Any]]:
    """Share findings, event identity, and tracing across CLI and scheduled runs."""
    with (tracer or get_tracer("petri.scan")).start_as_current_span(
        "petri.scan",
        context=Context(),
        attributes={
            "petri.command": "scan",
            "petri.dish_id": dish_id,
            "petri.scan.trigger": trigger,
        },
    ) as span:
        event_id = database.stable_event_id(dish_id, "scan", run_id)
        previous = next(
            (
                event
                for event in query_events(db_path, event_type="scan_completed", limit=None)
                if event["id"] == event_id
            ),
            None,
        )
        if previous is not None:
            findings = previous["data"]["findings"]
        else:
            findings = [issue.model_dump() for issue in scan_database(db_path)]
            event, _ = database.append_event(
                db_path,
                Event(
                    id=event_id,
                    cell_id=f"{dish_id}:scan",
                    timestamp=datetime.now(UTC).isoformat(),
                    type=EventType.scan_completed,
                    agent="scanner",
                    iteration=0,
                    data={
                        "findings": findings,
                        "findings_count": len(findings),
                        "trigger": trigger,
                    },
                ),
            )
            findings = event.data["findings"]
        span.set_attribute("petri.scan.findings_count", len(findings))
        return findings
