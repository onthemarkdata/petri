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
        # Instead, show all cells and let the user choose which to re-open.
        all_cells = []
        for colony_graph, colony in colonies:
            if colony_name and colony.id.split("-", 1)[-1] != colony_name:
                continue
            for cell in colony_graph.get_all_cells():
                all_cells.append((colony_graph, colony, cell))

        # Display cells that could be affected
        typer.echo(f"\nSource: {source}")
        typer.echo(f"Content length: {len(content)} characters\n")

        from petri.models import CellStatus

        reopenable = [
            (colony_graph, colony, cell)
            for colony_graph, colony, cell in all_cells
            if cell.status
            in (
                CellStatus.VALIDATED,
                CellStatus.DISPROVEN,
                CellStatus.DEFER_OPEN,
            )
            or cell.status.value
            in ("VALIDATED", "DISPROVEN", "DEFER_OPEN")
        ]

        if not reopenable:
            typer.echo(
                "No cells are in a re-openable state "
                "(VALIDATED, DISPROVEN, or DEFER_OPEN)."
            )
            raise typer.Exit(code=1)

        typer.echo(f"Re-openable cells ({len(reopenable)}):")
        for index, (colony_graph, colony, cell) in enumerate(reopenable, 1):
            status_val = (
                cell.status.value
                if isinstance(cell.status, CellStatus)
                else str(cell.status)
            )
            typer.echo(f"  {index}. [{status_val}] {cell.id}: {cell.claim_text}")

        # Re-open cells
        from petri.engine.propagation import (
            get_impact_report,
            propagate_upward,
            reopen_cell,
        )

        if auto_reopen:
            # Re-open all
            for colony_graph, colony, cell in reopenable:
                reopen_cell(petri_dir, cell.id, trigger=f"New evidence from: {source}")
                flagged = propagate_upward(petri_dir, cell.id, colony_graph, dish_id)
                typer.echo(
                    f"  Re-opened {cell.id}, flagged {len(flagged)} dependents"
                )
        else:
            # In non-interactive mode, just show the impact report
            import sys

            if not sys.stdin.isatty():
                typer.echo(
                    "\nRun with --auto-reopen to re-open all affected cells."
                )
                raise typer.Exit(code=0)

            # Interactive: ask for each cell
            try:
                import questionary

                for colony_graph, colony, cell in reopenable:
                    report = get_impact_report(petri_dir, cell.id, colony_graph, dish_id)
                    typer.echo(f"\n{cell.id}: {cell.claim_text}")
                    typer.echo(
                        f"  Would affect {report['total_affected']} dependent cells"
                    )

                    if questionary.confirm(
                        f"Re-open {cell.id}?", default=False
                    ).ask():
                        reopen_cell(
                            petri_dir,
                            cell.id,
                            trigger=f"New evidence from: {source}",
                        )
                        flagged = propagate_upward(
                            petri_dir, cell.id, colony_graph, dish_id
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
