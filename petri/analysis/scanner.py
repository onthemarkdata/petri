"""Find domain-data inconsistencies without changing research records."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError

from petri.models import CellStatus, Event, validate_event_data
from petri.query.connection import enumerate_dishes
from petri.query.integrity import integrity_rows


class ScanIssue(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    category: str
    description: str
    file_path: str | None = None


def scan_database(db_path: Path) -> list[ScanIssue]:
    """Check active graph endpoints, cell statuses, and every event envelope."""
    cells, edges, events = integrity_rows(db_path)
    issues: list[ScanIssue] = []

    def report(category: str, description: str) -> None:
        issues.append(ScanIssue(category=category, description=description, file_path=str(db_path)))

    ids = {cell["id"] for cell in cells}
    for cell in cells:
        try:
            CellStatus(cell["status"])
        except ValueError:
            report("cell_status", f"Cell {cell['id']} has unknown status {cell['status']!r}.")
    for edge in edges:
        for endpoint in ("from_cell", "to_cell"):
            if edge[endpoint] not in ids:
                report("phantom_edge", f"Edge {edge['from_cell']} → {edge['to_cell']} references missing cell {edge[endpoint]}.")
    for row in events:
        try:
            event = Event.model_validate({
                "id": row["id"], "cell_id": row["cell_id"], "timestamp": row["created_at"],
                "type": row["event_type"], "agent": row["agent"], "iteration": row["iteration"],
                "data": json.loads(row["payload"]),
            })
            validate_event_data(event.type.value, event.data)
        except (ValueError, TypeError, ValidationError) as error:
            # Report identity and schema failure without copying private event contents.
            report("event_payload", f"Event {row['id']} ({row['event_type']}) fails validation: {type(error).__name__}.")
    return issues


def scan(petri_dir: Path, generated_dir: Path | None = None) -> list[ScanIssue]:
    """Inspect each dish database; generated adapter files are outside this scan."""
    return [issue for dish in enumerate_dishes(petri_dir) for issue in scan_database(dish.db_path)]
