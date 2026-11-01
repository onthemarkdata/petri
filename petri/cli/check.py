"""petri check command."""

from __future__ import annotations

import json

import typer

from petri.cli._bootstrap import find_petri_dir, get_dish_id, load_colonies
from petri.cli_ui import print_error_and_exit
from petri.execution import resolve_backend
from petri.storage.database import SQLiteEventLog
from petri.storage.paths import dish_db_path


def register(app: typer.Typer) -> None:
    @app.command()
    def check(
        colony_name: str | None = typer.Option(None, "--colony", help="Filter to one colony"),
        cell: str | None = typer.Option(None, "--cell", help="Detailed view of a single cell"),
        targets: int = typer.Option(5, "--targets", min=1, help="Maximum validation targets"),
        json_output: bool = typer.Option(False, "--json", help="JSON output"),
    ) -> None:
        """Show current state of the petri dish."""
        petri_dir = find_petri_dir()
        try:
            backend = resolve_backend(petri_dir)
        except ValueError as error:
            print_error_and_exit(str(error))
        dish_id = get_dish_id(petri_dir)
        colonies = load_colonies(petri_dir, dish_id)

        if colony_name:
            colonies = [
                (graph, colony)
                for graph, colony in colonies
                if colony.id.endswith(f"-{colony_name}")
            ]

        queue_entries = backend.entries(petri_dir)
        conflicts = {
            key: value
            for key, value in queue_entries.items()
            if value.get("queue_state") == "sync_conflict"
            and (not colony_name or value.get("colony_id", "").endswith(f"-{colony_name}"))
        }

        if not colonies and not conflicts:
            typer.echo("No colonies found.")
            raise typer.Exit(code=0)

        from petri.query.intelligence import get_intelligence

        intelligence = get_intelligence(petri_dir)

        # Detailed cell view
        if cell:
            found = False
            for graph, col in colonies:
                try:
                    cell_obj = graph.get_cell(cell)
                except KeyError:
                    continue
                found = True
                queue_entry = queue_entries.get(cell, {})

                if json_output:
                    detail = {
                        "cell_id": cell_obj.id,
                        "colony_id": cell_obj.colony_id,
                        "claim_text": cell_obj.claim_text,
                        "level": cell_obj.level,
                        "status": cell_obj.status.value,
                        "dependencies": cell_obj.dependencies,
                        "dependents": cell_obj.dependents,
                        "queue_state": queue_entry.get("queue_state", ""),
                        "iteration": queue_entry.get("iteration", 0),
                    }
                    if queue_entry.get("sync_conflict_reason"):
                        detail["sync_conflict_reason"] = queue_entry["sync_conflict_reason"]
                    typer.echo(json.dumps(detail, indent=2))
                else:
                    typer.echo(f"Cell: {cell_obj.id}")
                    typer.echo(f"  Colony:       {cell_obj.colony_id}")
                    typer.echo(f"  Claim:        {cell_obj.claim_text}")
                    typer.echo(f"  Level:        {cell_obj.level}")
                    typer.echo(f"  Status:       {cell_obj.status.value}")
                    typer.echo(f"  Dependencies: {', '.join(cell_obj.dependencies) or '(none)'}")
                    typer.echo(f"  Dependents:   {', '.join(cell_obj.dependents) or '(none)'}")
                    if queue_entry:
                        typer.echo(f"  Queue State:  {queue_entry.get('queue_state', '')}")
                        typer.echo(f"  Iteration:    {queue_entry.get('iteration', 0)}")
                        if queue_entry.get("sync_conflict_reason"):
                            typer.echo(f"  Sync conflict: {queue_entry['sync_conflict_reason']}")

                    # Show events
                    from petri.storage.event_log import load_events

                    cell_events_path = SQLiteEventLog(dish_db_path(petri_dir, dish_id), cell_obj.id)
                    events = load_events(cell_events_path)
                    if events:
                        typer.echo(f"  Events:       {len(events)}")
                        for evt in events[-5:]:
                            typer.echo(
                                f"    [{evt.get('type')}] {evt.get('agent')} "
                                f"iter={evt.get('iteration')} "
                                f"{evt.get('timestamp', '')[:19]}"
                            )
                break

            if not found and cell in conflicts:
                detail = {"cell_id": cell, "status": "missing", **conflicts[cell]}
                typer.echo(
                    json.dumps(detail, indent=2)
                    if json_output
                    else f"Cell: {cell}\n  Sync conflict: {conflicts[cell]['sync_conflict_reason']}"
                )
                return
            if not found:
                print_error_and_exit(f"Cell '{cell}' not found.", code=0)
            return

        # Table output
        all_data: list[dict] = []
        for graph, col in colonies:
            for cell_obj in graph.get_all_cells():
                queue_entry = queue_entries.get(cell_obj.id, {})
                all_data.append(
                    {
                        "colony": col.id,
                        "level": cell_obj.level,
                        "cell_id": cell_obj.id,
                        "claim": cell_obj.claim_text,
                        "status": cell_obj.status.value,
                        "queue_state": queue_entry.get("queue_state", ""),
                        **(
                            {"sync_conflict_reason": queue_entry["sync_conflict_reason"]}
                            if queue_entry.get("sync_conflict_reason")
                            else {}
                        ),
                    }
                )

        known = {item["cell_id"] for item in all_data}
        for cell_id, conflict in conflicts.items():
            if cell_id not in known:
                all_data.append(
                    {
                        "colony": conflict.get("colony_id", ""),
                        "level": conflict.get("level", 0),
                        "cell_id": cell_id,
                        "claim": "",
                        "status": "missing",
                        "queue_state": "sync_conflict",
                        "sync_conflict_reason": conflict["sync_conflict_reason"],
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
            typer.echo(f"  {'Cell ID':<40} {'Status':<16} {'Queue State':<16}")
            typer.echo(f"  {'-' * 40} {'-' * 16} {'-' * 16}")
            for item in levels[level]:
                typer.echo(
                    f"  {item['cell_id']:<40} {item['status']:<16} {item['queue_state'] or '-':<16}"
                )
                if item.get("sync_conflict_reason"):
                    typer.echo(f"    Sync conflict: {item['sync_conflict_reason']}")
        typer.echo("")
        visible = {row['cell_id'] for row in all_data}
        ranked = [cell_id for cell_id in intelligence.convergence_points if cell_id in visible][:targets]
        typer.echo('Validation targets:')
        if not ranked:
            typer.echo('  No convergence points found.')
        for cell_id in ranked:
            metric = intelligence.cells[cell_id]
            typer.echo(f'  {cell_id}  importance={metric.importance}  fragility={metric.fragility}')

        from petri.query.health import latest_health

        health = latest_health(dish_db_path(petri_dir))
        typer.echo('Active health flags:')
        if health is None or not health.flags:
            typer.echo('  No active flags in the latest report.')
        else:
            for flag in health.flags:
                typer.echo(f'  [{flag.severity}] {flag.message}')
                typer.echo(f"    Cells: {', '.join(flag.affected_cells) or '-'}")
                typer.echo(f"    Agents: {', '.join(flag.affected_agents) or '-'}")
