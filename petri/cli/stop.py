"""petri stop command."""

from __future__ import annotations

import json

import typer

from petri.cli._bootstrap import find_petri_dir
from petri.storage.paths import (
    events_path as events_file_path,
    metadata_path as metadata_file_path,
    node_dir as node_dir_for,
    parse_node_id,
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

        # Find active nodes and stall them
        active_states = {
            "socratic_active",
            "research_active",
            "critique_active",
            "mediating",
            "red_team_active",
            "evaluating",
        }
        entries = list_queue(queue_path)
        stopped_nodes: list[str] = []

        for entry in entries:
            state = entry.get("queue_state", "")
            node_id = entry.get("node_id", "")
            if state in active_states and node_id:
                try:
                    if force:
                        # Force stop: stall immediately
                        update_state(queue_path, node_id, "stalled")
                    else:
                        # Graceful: stall
                        update_state(queue_path, node_id, "stalled")

                    # Log stop event — walk each per-colony directory looking
                    # for a matching node dir (using the zero-padded fallback
                    # layout, which is what the node-ID alone can reconstruct).
                    try:
                        _, _, level_int, seq_int = parse_node_id(node_id)
                    except ValueError:
                        level_int = seq_int = None

                    if level_int is not None and seq_int is not None:
                        dishes_dir = petri_dir / "petri-dishes"
                        if dishes_dir.is_dir():
                            for colony_path in dishes_dir.iterdir():
                                if not colony_path.is_dir():
                                    continue
                                candidate_node_dir = node_dir_for(
                                    colony_path, level_int, seq_int
                                )
                                candidate_metadata = metadata_file_path(candidate_node_dir)
                                if not candidate_metadata.exists():
                                    continue
                                try:
                                    meta = json.loads(candidate_metadata.read_text())
                                    if meta.get("id") == node_id:
                                        append_event(
                                            events_path=events_file_path(candidate_node_dir),
                                            node_id=node_id,
                                            event_type="verdict_issued",
                                            agent="node_lead",
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

                    stopped_nodes.append(node_id)
                except ValueError:
                    pass  # Transition not valid

        if stopped_nodes:
            typer.echo(f"Stopped {len(stopped_nodes)} nodes:")
            for nid in stopped_nodes:
                typer.echo(f"  {nid}: stalled")
        else:
            typer.echo("No active nodes to stop.")

        # Also check for queued nodes
        queued_nodes = [
            entry for entry in entries if entry.get("queue_state") == "queued"
        ]
        if queued_nodes:
            typer.echo(f"\n{len(queued_nodes)} nodes remain queued (not started).")

        raise typer.Exit(code=0)
