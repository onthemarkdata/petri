"""Snapshot a dish's domain database and list saved copies."""

from typing import Optional

import typer

from petri.cli._bootstrap import find_petri_dir, get_dish_id
from petri.storage.backup import list_snapshots, snapshot_before_destructive


def register(app: typer.Typer) -> None:
    @app.command()
    def backup(
        dish: Optional[str] = typer.Option(None, "--dish", help="Dish ID; defaults to the current dish"),
        list_only: bool = typer.Option(False, "--list", help="List saved snapshots newest first"),
    ) -> None:
        """Back up committed cells, edges, events, and defaults."""
        petri_dir = find_petri_dir()
        try:
            if list_only:
                for snapshot in list_snapshots(petri_dir, dish):
                    typer.echo(f"{snapshot.name}\t{snapshot.dish_id}\t{snapshot.created_at}\t"
                               f"{snapshot.cell_count} cells\t{snapshot.size_bytes} bytes")
                return
            path = snapshot_before_destructive(petri_dir, dish or get_dish_id(petri_dir))
            typer.echo(f"Snapshot saved: {path}")
        except (OSError, RuntimeError, ValueError) as error:
            typer.echo(str(error), err=True)
            raise typer.Exit(1) from error
