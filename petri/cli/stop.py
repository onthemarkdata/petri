"""petri stop command."""

from __future__ import annotations

import json

import typer

from petri.cli._bootstrap import find_petri_dir
from petri.storage.paths import (
    cell_dir as cell_dir_for,
    events_path as events_file_path,
    metadata_path as metadata_file_path,
    parse_cell_id,
)


def register(app: typer.Typer) -> None:
    @app.command()
    def stop(
        force: bool = typer.Option(False, "--force", help="Immediate stop"),
    ) -> None:
        """Gracefully stop all running tasks."""
        petri_dir = find_petri_dir()

        from petri.engine.processor import request_stop, request_stop_file
        from petri.storage.queue import list_queue, update_state
        from petri.storage.event_log import append_event

        queue_path = petri_dir / "queue.json"

        # Signal the processor to stop (in-process threads + cross-process sentinel)
        request_stop()
        request_stop_file(petri_dir)

        # Find active cells and stall them
        active_states = {
            "socratic_active",
            "research_active",
            "critique_active",
            "mediating",
            "red_team_active",
            "evaluating",
        }
        entries = list_queue(queue_path)
        stopped_cells: list[str] = []

        for entry in entries:
            state = entry.get("queue_state", "")
            cell_id = entry.get("cell_id", "")
            if state in active_states and cell_id:
                try:
                    if force:
                        # Force stop: stall immediately
                        update_state(queue_path, cell_id, "stalled")
                    else:
                        # Graceful: stall
                        update_state(queue_path, cell_id, "stalled")

                    # Log stop event — walk each per-colony directory looking
                    # for a matching cell dir (using the zero-padded fallback
                    # layout, which is what the cell-ID alone can reconstruct).
                    try:
                        _, _, level_int, seq_int = parse_cell_id(cell_id)
                    except ValueError:
                        level_int = seq_int = None

                    if level_int is not None and seq_int is not None:
                        dishes_dir = petri_dir / "petri-dishes"
                        if dishes_dir.is_dir():
                            for colony_path in dishes_dir.iterdir():
                                if not colony_path.is_dir():
                                    continue
                                candidate_cell_dir = cell_dir_for(
                                    colony_path, level_int, seq_int
                                )
                                candidate_metadata = metadata_file_path(candidate_cell_dir)
                                if not candidate_metadata.exists():
                                    continue
                                try:
                                    meta = json.loads(candidate_metadata.read_text())
                                    if meta.get("id") == cell_id:
                                        append_event(
                                            events_path=events_file_path(candidate_cell_dir),
                                            cell_id=cell_id,
                                            event_type="verdict_issued",
                                            agent="cell_lead",
                                            iteration=entry.get("iteration", 0),
                                            data={
                                                "verdict": "PIPELINE_STALLED",
                                                "summary": "Stop requested by user"
                                                + (" (force)" if force else ""),
                                            },
                                        )
                                        break
                                except Exception:
                                    pass

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
        queued_cells = [
            entry for entry in entries if entry.get("queue_state") == "queued"
        ]
        if queued_cells:
            typer.echo(f"\n{len(queued_cells)} cells remain queued (not started).")

        raise typer.Exit(code=0)
