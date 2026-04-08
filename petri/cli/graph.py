"""petri graph command."""

from __future__ import annotations

from typing import Optional

import typer

from petri.cli._bootstrap import find_petri_dir, get_dish_id, load_colonies
from petri.cli_ui import render_dot, render_text_tree


def register(app: typer.Typer) -> None:
    @app.command()
    def graph(
        colony_name: Optional[str] = typer.Option(
            None, "--colony", help="Filter to colony"
        ),
        format: str = typer.Option(
            "text", "--format", help="Output format: text or dot"
        ),
    ) -> None:
        """Render the colony DAG as text or Graphviz dot."""
        petri_dir = find_petri_dir()
        dish_id = get_dish_id(petri_dir)

        colonies = load_colonies(petri_dir, dish_id)

        if colony_name:
            colonies = [
                (colony_graph, colony)
                for colony_graph, colony in colonies
                if colony.id.endswith(f"-{colony_name}")
            ]

        if not colonies:
            typer.echo("No colonies found.")
            raise typer.Exit(code=0)

        for colony_graph, colony in colonies:
            if format == "dot":
                render_dot(colony_graph, colony)
            else:
                render_text_tree(colony_graph, colony)

        raise typer.Exit(code=0)
