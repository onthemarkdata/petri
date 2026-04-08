"""petri launch command."""

from __future__ import annotations

from pathlib import Path

import typer


def register(app: typer.Typer) -> None:
    @app.command()
    def launch(
        port: int = typer.Option(8090, "--port", help="Dashboard port"),
    ) -> None:
        """Launch the Petri Lab dashboard in your browser."""
        import threading
        import webbrowser

        import uvicorn

        from petri.dashboard.api import create_app

        petri_dir = Path.cwd() / ".petri"
        db_path = petri_dir / "petri.sqlite"

        from petri.dashboard.migrate import init_db, rebuild_sqlite

        if petri_dir.exists():
            typer.echo("Building event index...")
            count = rebuild_sqlite(petri_dir, db_path)
            typer.echo(f"Indexed {count} events.")
        else:
            # No dish yet -- frontend will show onboarding.
            # Ensure the db directory + schema exist so SSE works.
            petri_dir.mkdir(parents=True, exist_ok=True)
            init_db(db_path)

        dashboard_app = create_app(petri_dir, db_path)

        url = f"http://localhost:{port}"
        typer.echo(f"Launching Petri Lab at {url}")
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()

        uvicorn.run(dashboard_app, host="0.0.0.0", port=port, log_level="warning")
