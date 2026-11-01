"""Request durable cancellation of active validation workflows."""

from __future__ import annotations

import typer

from petri.cli._bootstrap import find_petri_dir
from petri.cli_ui import print_error_and_exit
from petri.execution import resolve_backend


def register(app: typer.Typer) -> None:
    @app.command()
    def stop(
        force: bool = typer.Option(False, "--force", help="Interrupt an active inference call"),
    ) -> None:
        """Cancel active runs, letting the current step finish unless --force is set."""
        petri_dir = find_petri_dir()
        try:
            backend = resolve_backend(petri_dir)
            count = backend.request_stop(petri_dir, force=force)
        except ValueError as error:
            print_error_and_exit(str(error))
        if count:
            mode = "immediately" if force else "after the current step"
            typer.echo(f"Cancellation requested for {count} runs {mode}.")
        else:
            typer.echo("No active runs to stop.")
