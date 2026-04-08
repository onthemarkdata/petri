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
        from_node: Optional[str] = typer.Argument(
            None, help="Source node ID"
        ),
        to_node: Optional[str] = typer.Argument(
            None, help="Target node ID"
        ),
    ) -> None:
        """Create a cross-colony edge between two nodes."""
        from petri.graph.colony import serialize_colony
        from petri.models import Edge

        petri_dir = find_petri_dir()
        dish_id = get_dish_id(petri_dir)

        # Interactive fallback when positional args omitted.
        if from_node is None or to_node is None:
            if detect_interactive_mode():
                try:
                    import questionary

                    from_node = questionary.text("First node ID:").ask()
                    to_node = questionary.text("Second node ID:").ask()
                    if not from_node or not to_node:
                        typer.echo("Cancelled.")
                        raise typer.Exit(code=0)
                except ImportError:
                    print_error_and_exit(
                        "Usage: petri connect NODE1 NODE2",
                    )
            else:
                print_error_and_exit(
                    "Usage: petri connect NODE1 NODE2",
                )

        node1_id, node2_id = from_node, to_node
        colonies = load_colonies(petri_dir, dish_id)

        # Find the nodes across colonies
        source_graph = None
        source_colony = None
        target_found = False

        for colony_graph, colony in colonies:
            try:
                colony_graph.get_node(node1_id)
                source_graph = colony_graph
                source_colony = colony
            except KeyError:
                pass
            try:
                colony_graph.get_node(node2_id)
                target_found = True
            except KeyError:
                pass

        if source_graph is None:
            print_error_and_exit(f"Node '{node1_id}' not found.")
        if not target_found:
            print_error_and_exit(f"Node '{node2_id}' not found.")

        edge = Edge(
            from_node=node1_id,
            to_node=node2_id,
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
            f"Cross-colony edge created: {node1_id} -> {node2_id}"
        )
        raise typer.Exit(code=0)
