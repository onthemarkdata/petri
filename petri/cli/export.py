"""Export a dish as deterministic JSONL and Markdown."""

import typer

from petri.cli._bootstrap import find_petri_dir, get_dish_id
from petri.storage.export import export_dish


def register(app: typer.Typer) -> None:
    @app.command("export")
    def export_command(
        colony: str | None = typer.Option(None, "--colony", help="Export one colony"),
        cell: str | None = typer.Option(None, "--cell", help="Export one cell"),
        since: int | None = typer.Option(None, "--since", help="Rewrite cells with events after this rowid"),
    ) -> None:
        """Write derived text under the current dish's exports directory."""
        petri_dir = find_petri_dir()
        try:
            result = export_dish(petri_dir, get_dish_id(petri_dir), colony=colony, cell_id=cell, since=since)
        except (OSError, ValueError) as error:
            typer.echo(str(error), err=True)
            raise typer.Exit(1) from error
        typer.echo(f"Exported to {result}")
