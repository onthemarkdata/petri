"""Append diagnostic reports without changing research cells, edges, or events."""

from pathlib import Path

from petri.analysis.analyst import ResearchHealthReport
from petri.storage.database import write_connection


def append_report(db_path: Path, report: ResearchHealthReport) -> bool:
    with write_connection(db_path) as connection:
        return connection.execute('INSERT INTO analyst_reports(id,dish_id,created_at,report_json) VALUES (?,?,?,?) '
            'ON CONFLICT(id) DO NOTHING',
            (report.id,report.dish_id,report.created_at,report.model_dump_json())).rowcount == 1
