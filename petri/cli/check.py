"""petri check command."""

from __future__ import annotations

import json
from typing import Optional

import typer

from petri.cli._bootstrap import find_petri_dir, get_dish_id, load_colonies
from petri.cli_ui import print_error_and_exit
from petri.storage.paths import (
    colony_dir as colony_dir_for,
    events_path as events_file_path,
    node_dir as node_dir_for,
    parse_node_id,
)


def register(app: typer.Typer) -> None:
    @app.command()
    def check(
        colony_name: Optional[str] = typer.Option(
            None, "--colony", help="Filter to one colony"
        ),
        node: Optional[str] = typer.Option(
            None, "--node", help="Detailed view of a single node"
        ),
        json_output: bool = typer.Option(False, "--json", help="JSON output"),
    ) -> None:
        """Show current state of the petri dish."""
        petri_dir = find_petri_dir()
        dish_id = get_dish_id(petri_dir)
        colonies = load_colonies(petri_dir, dish_id)

        if colony_name:
            colonies = [
                (graph, colony) for graph, colony in colonies
                if colony.id.endswith(f"-{colony_name}")
            ]

        if not colonies:
            typer.echo("No colonies found.")
            raise typer.Exit(code=0)

        # Load queue for state info
        from petri.storage.queue import load_queue

        queue_path = petri_dir / "queue.json"
        queue = load_queue(queue_path)
        queue_entries = queue.get("entries", {})

        # Detailed node view
        if node:
            found = False
            for graph, col in colonies:
                try:
                    node_obj = graph.get_node(node)
                except KeyError:
                    continue
                found = True
                queue_entry = queue_entries.get(node, {})

                if json_output:
                    detail = {
                        "node_id": node_obj.id,
                        "colony_id": node_obj.colony_id,
                        "claim_text": node_obj.claim_text,
                        "level": node_obj.level,
                        "status": node_obj.status.value,
                        "dependencies": node_obj.dependencies,
                        "dependents": node_obj.dependents,
                        "queue_state": queue_entry.get("queue_state", ""),
                        "iteration": queue_entry.get("iteration", 0),
                    }
                    typer.echo(json.dumps(detail, indent=2))
                else:
                    typer.echo(f"Node: {node_obj.id}")
                    typer.echo(f"  Colony:       {node_obj.colony_id}")
                    typer.echo(f"  Claim:        {node_obj.claim_text}")
                    typer.echo(f"  Level:        {node_obj.level}")
                    typer.echo(f"  Status:       {node_obj.status.value}")
                    typer.echo(
                        f"  Dependencies: {', '.join(node_obj.dependencies) or '(none)'}"
                    )
                    typer.echo(
                        f"  Dependents:   {', '.join(node_obj.dependents) or '(none)'}"
                    )
                    if queue_entry:
                        typer.echo(
                            f"  Queue State:  {queue_entry.get('queue_state', '')}"
                        )
                        typer.echo(
                            f"  Iteration:    {queue_entry.get('iteration', 0)}"
                        )

                    # Show events
                    from petri.storage.event_log import load_events

                    colony_base = colony_dir_for(petri_dir, dish_id, node_obj.colony_id)
                    # Look up path from colony.json first, fall back to the
                    # zero-padded convention based on the parsed node ID.
                    node_rel = col.node_paths.get(node_obj.id)
                    if node_rel:
                        node_events_path = colony_base / node_rel / "events.jsonl"
                    else:
                        _, _, level_int, seq_int = parse_node_id(node_obj.id)
                        node_events_path = events_file_path(
                            node_dir_for(colony_base, level_int, seq_int)
                        )
                    events = load_events(node_events_path)
                    if events:
                        typer.echo(f"  Events:       {len(events)}")
                        for evt in events[-5:]:
                            typer.echo(
                                f"    [{evt.get('type')}] {evt.get('agent')} "
                                f"iter={evt.get('iteration')} "
                                f"{evt.get('timestamp', '')[:19]}"
                            )
                break

            if not found:
                print_error_and_exit(f"Node '{node}' not found.", code=0)
            return

        # Table output
        all_data: list[dict] = []
        for graph, col in colonies:
            for node_obj in graph.get_nodes():
                queue_entry = queue_entries.get(node_obj.id, {})
                all_data.append(
                    {
                        "colony": col.id,
                        "level": node_obj.level,
                        "node_id": node_obj.id,
                        "claim": node_obj.claim_text,
                        "status": node_obj.status.value,
                        "queue_state": queue_entry.get("queue_state", ""),
                    }
                )

        if json_output:
            typer.echo(json.dumps(all_data, indent=2))
            return

        # Group by level
        levels: dict[int, list[dict]] = {}
        for item in all_data:
            levels.setdefault(item["level"], []).append(item)

        for level in sorted(levels.keys()):
            typer.echo(f"\nLevel {level}:")
            typer.echo(f"  {'Node ID':<40} {'Status':<16} {'Queue State':<16}")
            typer.echo(f"  {'-' * 40} {'-' * 16} {'-' * 16}")
            for item in levels[level]:
                typer.echo(
                    f"  {item['node_id']:<40} {item['status']:<16} "
                    f"{item['queue_state'] or '-':<16}"
                )
        typer.echo("")
