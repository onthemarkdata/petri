"""petri feed command."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from petri.cli._bootstrap import find_petri_dir, get_dish_id, load_colonies
from petri.cli_ui import print_error_and_exit


def register(app: typer.Typer) -> None:
    @app.command()
    def feed(
        source: str = typer.Argument(..., help="URL, file path, or - for stdin"),
        colony_name: Optional[str] = typer.Option(
            None, "--colony", help="Limit matching to colony"
        ),
        auto_reopen: bool = typer.Option(
            False, "--auto-reopen", help="Skip re-open confirmation"
        ),
    ) -> None:
        """Provide new evidence to the colony."""
        petri_dir = find_petri_dir()
        dish_id = get_dish_id(petri_dir)

        # Load colonies
        colonies = load_colonies(petri_dir, dish_id)
        if not colonies:
            print_error_and_exit("No colonies found.")

        # Read source content
        if source == "-":
            import sys

            content = sys.stdin.read()
        elif Path(source).exists():
            content = Path(source).read_text()
        else:
            # Assume URL or text -- store as-is
            content = source

        # Without a InferenceProvider, we can't do intelligent matching.
        # Instead, show all nodes and let the user choose which to re-open.
        all_nodes = []
        for colony_graph, colony in colonies:
            if colony_name and colony.id.split("-", 1)[-1] != colony_name:
                continue
            for node in colony_graph.get_nodes():
                all_nodes.append((colony_graph, colony, node))

        # Display nodes that could be affected
        typer.echo(f"\nSource: {source}")
        typer.echo(f"Content length: {len(content)} characters\n")

        from petri.models import NodeStatus

        reopenable = [
            (colony_graph, colony, node)
            for colony_graph, colony, node in all_nodes
            if node.status
            in (
                NodeStatus.VALIDATED,
                NodeStatus.DISPROVEN,
                NodeStatus.DEFER_OPEN,
            )
            or node.status.value
            in ("VALIDATED", "DISPROVEN", "DEFER_OPEN")
        ]

        if not reopenable:
            typer.echo(
                "No nodes are in a re-openable state "
                "(VALIDATED, DISPROVEN, or DEFER_OPEN)."
            )
            raise typer.Exit(code=1)

        typer.echo(f"Re-openable nodes ({len(reopenable)}):")
        for index, (colony_graph, colony, node) in enumerate(reopenable, 1):
            status_val = (
                node.status.value
                if isinstance(node.status, NodeStatus)
                else str(node.status)
            )
            typer.echo(f"  {index}. [{status_val}] {node.id}: {node.claim_text}")

        # Re-open nodes
        from petri.engine.propagation import (
            get_impact_report,
            propagate_upward,
            reopen_node,
        )

        if auto_reopen:
            # Re-open all
            for colony_graph, colony, node in reopenable:
                reopen_node(petri_dir, node.id, trigger=f"New evidence from: {source}")
                flagged = propagate_upward(petri_dir, node.id, colony_graph, dish_id)
                typer.echo(
                    f"  Re-opened {node.id}, flagged {len(flagged)} dependents"
                )
        else:
            # In non-interactive mode, just show the impact report
            import sys

            if not sys.stdin.isatty():
                typer.echo(
                    "\nRun with --auto-reopen to re-open all affected nodes."
                )
                raise typer.Exit(code=0)

            # Interactive: ask for each node
            try:
                import questionary

                for colony_graph, colony, node in reopenable:
                    report = get_impact_report(petri_dir, node.id, colony_graph, dish_id)
                    typer.echo(f"\n{node.id}: {node.claim_text}")
                    typer.echo(
                        f"  Would affect {report['total_affected']} dependent nodes"
                    )

                    if questionary.confirm(
                        f"Re-open {node.id}?", default=False
                    ).ask():
                        reopen_node(
                            petri_dir,
                            node.id,
                            trigger=f"New evidence from: {source}",
                        )
                        flagged = propagate_upward(
                            petri_dir, node.id, colony_graph, dish_id
                        )
                        typer.echo(
                            f"  Re-opened. Flagged {len(flagged)} dependents."
                        )
            except (ImportError, OSError):
                typer.echo(
                    "\nRun with --auto-reopen in non-interactive mode."
                )
                raise typer.Exit(code=0)

        typer.echo("\nEvidence feed complete.")
