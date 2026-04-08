"""petri connect command."""

from __future__ import annotations

from typing import Optional

import typer

from petri.cli._bootstrap import (
    detect_interactive_mode,
    find_petri_dir,
    get_dish_id,
    load_colonies,
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
        from petri.graph.colony import serialize_colony
        from petri.models import Edge

        petri_dir = find_petri_dir()
        dish_id = get_dish_id(petri_dir)

        # Interactive fallback when positional args omitted.
        if from_cell is None or to_cell is None:
            if detect_interactive_mode():
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

        first_cell_id, second_cell_id = from_cell, to_cell
        colonies = load_colonies(petri_dir, dish_id)

        # Find the cells across colonies
        source_graph = None
        source_colony = None
        target_found = False

        for colony_graph, colony in colonies:
            try:
                colony_graph.get_cell(first_cell_id)
                source_graph = colony_graph
                source_colony = colony
            except KeyError:
                pass
            try:
                colony_graph.get_cell(second_cell_id)
                target_found = True
            except KeyError:
                pass

        if source_graph is None:
            print_error_and_exit(f"Cell '{first_cell_id}' not found.")
        if not target_found:
            print_error_and_exit(f"Cell '{second_cell_id}' not found.")

        edge = Edge(
            from_cell=first_cell_id,
            to_cell=second_cell_id,
            edge_type="cross_colony",
        )

        try:
            source_graph.add_edge(edge)
        except ValueError as exc:
            print_error_and_exit(f"Cannot create edge: {exc}")

        # Re-serialize the colony
        colony_slug = source_colony.id.replace(f"{dish_id}-", "", 1)
        colony_path = petri_dir / "petri-dishes" / colony_slug
        serialize_colony(source_graph, source_colony, colony_path)

        typer.echo(
            f"Cross-colony edge created: {first_cell_id} -> {second_cell_id}"
        )
        raise typer.Exit(code=0)
