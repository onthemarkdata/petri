"""Run a domain scan and return a nonzero status when it finds inconsistencies."""

from __future__ import annotations

from uuid import uuid4

import typer

from petri.cli._bootstrap import bootstrap_telemetry, find_petri_dir, get_dish_id


def register(app: typer.Typer) -> None:
    @app.command()
    def scan() -> None:
        """Check stored cells, edges, and events without changing research data."""
        from petri.analysis.scanner import ScanIssue
        from petri.analysis.scan_run import run_domain_scan
        from petri.storage.paths import dish_db_path

        petri_dir = find_petri_dir()
        bootstrap_telemetry(petri_dir)
        dish_id = get_dish_id(petri_dir)
        db_path = dish_db_path(petri_dir, dish_id)
        issues = [
            ScanIssue.model_validate(item)
            for item in run_domain_scan(
                db_path, dish_id=dish_id, trigger="manual", run_id=str(uuid4())
            )
        ]
        if not issues:
            typer.echo("No inconsistencies found.")
        else:
            typer.echo(f"Found {len(issues)} inconsistencies:")
            for index, issue in enumerate(issues, 1):
                typer.echo(f"  {index}. [{issue.category}] {issue.description}")
        raise typer.Exit(code=0 if not issues else 1)
