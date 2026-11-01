"""Snapshot a dish's domain database and list saved copies."""

from typing import Optional

import typer

from petri.cli._bootstrap import find_petri_dir, get_dish_id
from petri.storage.backup import list_snapshots, restore_snapshot, snapshot_before_destructive


def register(app: typer.Typer) -> None:
    @app.command()
    def backup(
        dish: Optional[str] = typer.Option(None, "--dish", help="Dish ID; defaults to the current dish"),
        restore: Optional[str] = typer.Option(None, "--restore", help="Restore a saved snapshot by name"),
        force: bool = typer.Option(False, "--force", help="Allow restore to replace an existing dish database"),
        list_only: bool = typer.Option(False, "--list", help="List saved snapshots newest first"),
    ) -> None:
        """Back up committed cells, edges, events, and defaults."""
        petri_dir = find_petri_dir()
        try:
            if restore is not None:
                if list_only or dish is not None:
                    raise ValueError("Restore selects the snapshot's dish; omit --dish and --list")
                if any(petri_dir.rglob("dbos.sqlite")):
                    typer.echo("WARNING: DBOS execution state is present. Restoring domain data can "
                               "desynchronize it from workflow checkpoints. Run petri check before grow.", err=True)
                target = restore_snapshot(petri_dir, restore, force=force)
                typer.echo(f"Restored snapshot to {target}")
                return
            if force:
                raise ValueError("--force requires --restore")
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
