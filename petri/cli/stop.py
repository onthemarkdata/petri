"""petri stop command."""

from __future__ import annotations

import typer

from petri.cli._bootstrap import find_petri_dir
from petri.cli_ui import print_error_and_exit
from petri.execution import resolve_backend
from petri.storage.database import SQLiteEventLog, get_cell
from petri.storage.paths import dish_db_path


def register(app: typer.Typer) -> None:
    @app.command()
    def stop(
        force: bool = typer.Option(False, "--force", help="Immediate stop"),
    ) -> None:
        """Gracefully stop all running tasks."""
        petri_dir = find_petri_dir()
        try:
            backend = resolve_backend(petri_dir)
        except ValueError as error:
            print_error_and_exit(str(error))

        from petri.storage.event_log import append_event

        # Signal the processor to stop (in-process threads + cross-process sentinel)
        backend.request_stop(petri_dir)

        # Find active cells and stall them
        active_states = {
            "socratic_active",
            "research_active",
            "critique_active",
            "mediating",
            "red_team_active",
            "evaluating",
        }
        entries = backend.list_entries(petri_dir)
        stopped_cells: list[str] = []

        for entry in entries:
            state = entry.get("queue_state", "")
            cell_id = entry.get("cell_id", "")
            if state in active_states and cell_id:
                try:
                    if force:
                        # Force stop: stall immediately
                        backend.update_state(petri_dir, cell_id, "stalled")
                    else:
                        # Graceful: stall
                        backend.update_state(petri_dir, cell_id, "stalled")

                    db_path = dish_db_path(petri_dir)
                    if get_cell(db_path, cell_id) is not None:
                        append_event(
                            events_path=SQLiteEventLog(db_path, cell_id), cell_id=cell_id,
                            event_type="verdict_issued", agent="cell_lead",
                            iteration=entry.get("iteration", 0), node="stop",
                            data={"verdict": "PIPELINE_STALLED", "summary":
                                  "Stop requested by user" + (" (force)" if force else "")},
                        )

                    stopped_cells.append(cell_id)
                except ValueError:
                    pass  # Transition not valid

        if stopped_cells:
            typer.echo(f"Stopped {len(stopped_cells)} cells:")
            for stopped_cell_id in stopped_cells:
                typer.echo(f"  {stopped_cell_id}: stalled")
        else:
            typer.echo("No active cells to stop.")

        # Also check for queued cells
        queued_cells = [entry for entry in entries if entry.get("queue_state") == "queued"]
        if queued_cells:
            typer.echo(f"\n{len(queued_cells)} cells remain queued (not started).")

        raise typer.Exit(code=0)
