"""petri feed command."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from uuid import uuid4
from typing import Optional

import typer

from petri.telemetry.commands import traced_command

from petri.cli._bootstrap import find_petri_dir, load_colonies
from petri.execution import resolve_backend
from petri.execution.models import FeedRequest
from petri.storage.dish import resolve_dish_id
from petri.cli_ui import print_error_and_exit


def register(app: typer.Typer) -> None:
    @app.command()
    @traced_command("feed")
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
        dish_id = resolve_dish_id(petri_dir)
        backend = resolve_backend(petri_dir)

        # Load colonies
        colonies = load_colonies(petri_dir, dish_id)
        if not colonies:
            print_error_and_exit("No colonies found.")

        if source == "-":
            import sys
            source = sys.stdin.read()
        request_id = hashlib.sha256(uuid4().bytes).hexdigest()

        all_cells = []
        for colony_graph, colony in colonies:
            if colony_name and colony.id.removeprefix(f"{dish_id}-") != colony_name:
                continue
            for cell in colony_graph.get_all_cells():
                all_cells.append((colony_graph, colony, cell))

        # Display cells that could be affected
        typer.echo(f"\nSource: {source}")

        from petri.models import CellStatus
        from petri.storage import database
        from petri.storage.paths import dish_db_path

        path = dish_db_path(petri_dir, dish_id)
        fed_cells = {
            event["cell_id"] for event in database.read_events(path, event_type="evidence_fed")
        } if path.is_file() else set()
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
            or (cell.status == CellStatus.NEW and cell.id in fed_cells)
        ]

        if not reopenable:
            typer.echo(
                "No cells are in a re-openable state "
                "(VALIDATED, DISPROVEN, or DEFER_OPEN)."
            )
            raise typer.Exit(code=1)

        typer.echo(f"Cells available for feed ({len(reopenable)}):")
        for index, (colony_graph, colony, cell) in enumerate(reopenable, 1):
            status_val = (
                cell.status.value
                if isinstance(cell.status, CellStatus)
                else str(cell.status)
            )
            typer.echo(f"  {index}. [{status_val}] {cell.id}: {cell.claim_text}")

        # Re-open cells
        from petri.lifecycle.propagation import get_impact_report

        def reopen(colony_graph, colony, cell):
            report = get_impact_report(petri_dir, cell.id, colony_graph, dish_id)
            typer.echo(f"  Impact: {report['total_affected']} dependent cells")
            from petri.reasoning.ingest import IngestError
            try:
                result = backend.feed_evidence(FeedRequest(
                    cell.id, request_id, source, datetime.now(timezone.utc).date().isoformat(),
                ), petri_dir=petri_dir)
                if getattr(result, "status", None) == "sync_conflict":
                    print_error_and_exit(result.reason)
            except IngestError as error:
                print_error_and_exit(str(error))
            typer.echo(f"  Revalidation finished for {cell.id}; dependents remain flagged")

        if auto_reopen:
            # Re-open all
            for colony_graph, colony, cell in reopenable:
                reopen(colony_graph, colony, cell)
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
                        reopen(colony_graph, colony, cell)
            except (ImportError, OSError):
                typer.echo(
                    "\nRun with --auto-reopen in non-interactive mode."
                )
                raise typer.Exit(code=0)

        typer.echo("\nEvidence feed complete.")
