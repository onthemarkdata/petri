"""Read committed Analyst inputs and append-only reports without domain writes."""

import json
from pathlib import Path

from petri.analysis.analyst import HealthSnapshot, ResearchHealthReport
from petri.analysis.edge_intelligence import analyze_edges
from petri.query.connection import has_table, read_connection
from petri.query.events import _event
from petri.storage.edge_registry import DishGraph


def health_snapshot(db_path: Path) -> HealthSnapshot:
    with read_connection(db_path) as connection:
        if connection is None:
            graph = DishGraph([],[])
            events, patterns = [], []
        else:
            graph = DishGraph.from_connection(connection)
            events = [dict(_event(row)) for row in connection.execute('SELECT * FROM events ORDER BY created_at,rowid')]
            patterns = [dict(row) for row in connection.execute('SELECT * FROM blocking_verdict_patterns')]
            for pattern in patterns:
                pattern['cell_ids'] = sorted(json.loads(pattern['cell_ids']))
        return HealthSnapshot(cells=list(graph.cells.values()),events=events,blocking_patterns=patterns,
            intelligence=analyze_edges(graph),dependencies={cell_id:graph.dependencies(cell_id) for cell_id in graph.cells})


def latest_health(db_path: Path, *, report_id: str | None = None) -> ResearchHealthReport | None:
    with read_connection(db_path) as connection:
        if connection is None or not has_table(connection,'analyst_reports'):
            return None
        row = connection.execute('SELECT report_json FROM analyst_reports' +
            (' WHERE id=?' if report_id is not None else '') + ' ORDER BY created_at DESC,rowid DESC LIMIT 1',
            (report_id,) if report_id is not None else ()).fetchone()
        return ResearchHealthReport.model_validate_json(row[0]) if row is not None else None
