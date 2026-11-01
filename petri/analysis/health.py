"""Run the pure Analyst over read-only queries and append its diagnosis."""

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from petri.analysis.analyst import AnalystConfig, ResearchHealthReport, analyze_health
from petri.query.health import health_snapshot, latest_health
from petri.storage.analyst import append_report


def run_analyst(db_path: Path, *, dish_id: str, config: AnalystConfig | None = None,
                now: datetime | None = None, report_id: str | None = None) -> ResearchHealthReport:
    identity = report_id or str(uuid4())
    previous = latest_health(db_path,report_id=identity)
    if previous is not None:
        return previous
    report = analyze_health(health_snapshot(db_path),report_id=identity,dish_id=dish_id,
                            now=now or datetime.now(timezone.utc),config=config)
    append_report(db_path,report)
    return latest_health(db_path,report_id=identity) or report
