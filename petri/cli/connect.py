"""petri connect command."""

from __future__ import annotations

from typing import Optional

import typer

from petri.cli._bootstrap import (
    detect_interactive_mode,
    find_petri_dir,
    get_dish_id,
)
from petri.cli_ui import print_error_and_exit


def register(app: typer.Typer) -> None:
    @app.command()
    def connect(
        from_cell: Optional[str] = typer.Argument(
            None, help="Source cell ID"
        ),
        to_cell: Optional[str] = typer.Argument(
            None, help="Target cell ID"
        ),
    ) -> None:
        """Create a cross-colony edge between two cells."""
        from datetime import datetime, timezone
        from petri.storage.edge_registry import DishGraph, EdgeRecord, append
        from petri.storage.paths import dish_db_path

        petri_dir = find_petri_dir()
        dish_id = get_dish_id(petri_dir)

        # Interactive fallback when positional args omitted.
        if from_cell is None or to_cell is None:
            interactive, _ = detect_interactive_mode()
            if interactive:
                try:
                    import questionary

                    from_cell = questionary.text("First cell ID:").ask()
                    to_cell = questionary.text("Second cell ID:").ask()
                    if not from_cell or not to_cell:
                        typer.echo("Cancelled.")
                        raise typer.Exit(code=0)
                except ImportError:
                    print_error_and_exit(
                        "Usage: petri connect CELL1 CELL2",
                    )
            else:
                print_error_and_exit(
                    "Usage: petri connect CELL1 CELL2",
                )

        assert from_cell is not None and to_cell is not None
        path = dish_db_path(petri_dir,dish_id)
        graph = DishGraph.read(path)
        for cell_id in (from_cell,to_cell):
            if cell_id not in graph.cells:
                print_error_and_exit(f"Cell '{cell_id}' not found.")
        edge = EdgeRecord(from_cell=from_cell,to_cell=to_cell,edge_type='cross_colony',
                          created_at=datetime.now(timezone.utc).isoformat())
        try:
            append(path,edge)
        except ValueError as error:
            print_error_and_exit(f'Cannot create edge: {error}')
        first_cell_id,second_cell_id = from_cell,to_cell

        typer.echo(
            f"Cross-colony edge created: {first_cell_id} -> {second_cell_id}"
        )
        raise typer.Exit(code=0)
