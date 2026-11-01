"""Run a domain scan and return a nonzero status when it finds inconsistencies."""

from __future__ import annotations

from contextlib import closing

import typer

from petri.cli._bootstrap import bootstrap_telemetry, find_petri_dir, get_dish_id


def register(app: typer.Typer) -> None:
    @app.command()
    def scan() -> None:
        """Check stored cells, edges, and events without changing research data."""
        from petri.config import load_dish_config
        from petri.execution.config import ExecutionConfig
        from petri.execution.dbos_app import DBOSApp
        from petri.execution.scan_workflow import run_scan_now
        from petri.storage.database import connect
        from petri.storage.paths import dish_db_path

        petri_dir = find_petri_dir()
        bootstrap_telemetry(petri_dir)
        db_path = dish_db_path(petri_dir, get_dish_id(petri_dir))
        with closing(connect(db_path)) as connection:
            connection.commit()
        settings = ExecutionConfig.model_validate(load_dish_config(petri_dir).get("execution", {}))

        def no_inference(spec):
            raise RuntimeError("Scanning does not run research agents")

        runtime = DBOSApp(petri_dir, db_path, settings, no_inference).start()
        try:
            issues = run_scan_now()
        finally:
            runtime.close()
        if not issues:
            typer.echo("No inconsistencies found.")
        else:
            typer.echo(f"Found {len(issues)} inconsistencies:")
            for index, issue in enumerate(issues, 1):
                typer.echo(f"  {index}. [{issue.category}] {issue.description}")
        raise typer.Exit(code=0 if not issues else 1)
