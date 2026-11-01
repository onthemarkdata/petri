"""petri grow command."""

from __future__ import annotations

import typer

from petri.cli._bootstrap import find_petri_dir, resolve_provider
from petri.cli_ui import (
    MultiSpinner,
    grow_status_loop,
    print_error_and_exit,
    short_cell_id,
)
from petri.config import MAX_CONCURRENT
from petri.execution import resolve_backend
from petri.execution.models import GrowOutcome, NoProviderError, ProgressEvent


def register(app: typer.Typer) -> None:
    @app.command()
    def grow(
        cell: list[str] | None = typer.Option(
            None,
            "--cell",
            "-c",
            help=(
                "Cell ID to grow (repeat for multiple: --cell a --cell b). "
                "If omitted, every eligible cell across all colonies is "
                "processed."
            ),
        ),
        colony_name: str | None = typer.Option(
            None, "--colony", help="Grow all cells in a specific colony"
        ),
        max_concurrent: int = typer.Option(
            MAX_CONCURRENT, "--max-concurrent", help="Max parallel cells"
        ),
        dry_run: bool = typer.Option(
            False, "--dry-run", help="Show what would process without running"
        ),
    ) -> None:
        """Run the validation pipeline.

        By default, every eligible cell across every colony is processed.
        Scope to a subset with ``--cell`` (repeatable) or ``--colony``.

        Internally loops ``process_queue`` until every queue entry is in a
        terminal state, the cross-process stop sentinel appears, or two
        consecutive passes make no progress.  A daemon status thread
        prints periodic progress lines above a persistent spinner.
        """
        import threading

        petri_dir = find_petri_dir()
        try:
            backend = resolve_backend(petri_dir)
        except ValueError as error:
            print_error_and_exit(str(error))

        # The processor's process_queue API still takes three orthogonal
        # knobs — cell_ids, colony_filter, all_cells — and we keep using
        # them unchanged. We just no longer expose --all on the CLI
        # surface because it's the default: all_cells is true exactly
        # when the user didn't scope to a specific cell or colony.
        selected_cells: list[str] | None = cell or None
        all_cells: bool = not selected_cells and not colony_name

        check = backend.preflight()
        if not check.passed:
            print_error_and_exit(
                f"Error: {check.message}\nRun 'petri inspect' to check all prerequisites.",
            )

        # Resolve the inference provider from petri.yaml config
        provider = resolve_provider(petri_dir)

        queue_path = petri_dir / "queue.json"

        # Drop any stale stop sentinel from a previous interrupted run.
        backend.clear_stop(petri_dir)

        # ── dry-run path: single pass, no loop, no spinner ──
        if dry_run:
            try:
                dry_result = backend.process_queue(
                    petri_dir=petri_dir,
                    provider=provider,
                    max_concurrent=max_concurrent,
                    cell_ids=selected_cells,
                    colony_filter=colony_name,
                    all_cells=all_cells,
                    dry_run=True,
                )
            except NoProviderError:
                print_error_and_exit(
                    "Error: No inference provider configured.\n"
                    "Set model and harness in .petri/petri.yaml to enable processing.",
                )

            would_process = dry_result.would_process
            if would_process:
                typer.echo(f"Would process {len(would_process)} cells:")
                for cell_id in would_process:
                    typer.echo(f"  {cell_id}")
            else:
                typer.echo("No eligible cells found.")
            raise typer.Exit(code=0)

        # ── live path: loop until terminal/stopped/no-progress ──
        def _get_states() -> dict[str, int]:
            try:
                return backend.state_summary(petri_dir)
            except Exception:
                return {}

        def _is_stopped() -> bool:
            return backend.is_stopped(petri_dir)

        outcome: GrowOutcome | None = None
        interrupted = False
        no_provider = False
        status_stop_event = threading.Event()
        status_thread: threading.Thread | None = None

        try:
            with MultiSpinner("growing", slot_count=max_concurrent) as multi:
                # Render every row up-front with an "idle" label so all
                # N slots are visible from t=0 — before any worker has
                # actually picked up a cell.
                for slot_index in range(max_concurrent):
                    multi.update_slot(slot_index, "idle")

                def _on_event(event: ProgressEvent) -> None:
                    """Translate a processor lifecycle event into a row update."""
                    if event.slot_idx < 0 or event.slot_idx >= max_concurrent:
                        return
                    cell_label = short_cell_id(event.cell_id)
                    if event.kind == "started":
                        row_text = f"{cell_label} starting"
                    elif event.kind == "phase":
                        row_text = f"{cell_label} {event.phase}"
                    elif event.kind == "agent":
                        row_text = f"{cell_label} {event.phase} · {event.agent}"
                    elif event.kind == "verdict":
                        verdict_short = (event.verdict or "")[:24]
                        row_text = f"{cell_label} {event.phase} · {event.agent}: {verdict_short}"
                    elif event.kind == "agent_text":
                        # Streaming model text — overwrite the slot row
                        # with the latest chunk so the user sees the
                        # model thinking in real time, matching petri
                        # seed's streaming UX. The next "verdict" event
                        # will replace this with the final verdict
                        # label when the agent finishes.
                        text_excerpt = (event.text or "")[:120]
                        row_text = f"{cell_label} {event.phase} · {event.agent}: {text_excerpt}"
                    elif event.kind == "finished":
                        if event.error:
                            error_short = (event.error or "")[:60]
                            row_text = f"{cell_label} ✗ {error_short}"
                        else:
                            row_text = f"{cell_label} ✓"
                        # Do NOT sleep here — blocking the worker thread
                        # delays slot release. Let the next "started"
                        # event for this slot overwrite the result.
                        multi.update_slot(event.slot_idx, row_text)
                        return
                    else:
                        return
                    multi.update_slot(event.slot_idx, row_text)

                def _run_one_pass():
                    return backend.process_queue(
                        petri_dir=petri_dir,
                        provider=provider,
                        max_concurrent=max_concurrent,
                        cell_ids=selected_cells,
                        colony_filter=colony_name,
                        all_cells=all_cells,
                        dry_run=False,
                        on_event=_on_event,
                    )

                status_thread = threading.Thread(
                    target=grow_status_loop,
                    kwargs={
                        "petri_dir": petri_dir,
                        "queue_path": queue_path,
                        "spinner": multi,
                        "stop_event": status_stop_event,
                        "interval_seconds": backend.status_interval,
                        "get_states": _get_states,
                        "format_states": backend.format_state_summary,
                    },
                    daemon=True,
                )
                status_thread.start()

                try:
                    outcome = backend.grow_loop(
                        run_one_pass=_run_one_pass,
                        get_states=_get_states,
                        is_stopped=_is_stopped,
                    )
                except NoProviderError:
                    no_provider = True
                except KeyboardInterrupt:
                    interrupted = True
                    typer.echo("\nInterrupted by Ctrl+C")
        finally:
            status_stop_event.set()
            if status_thread is not None:
                status_thread.join(timeout=2.0)
            backend.clear_stop(petri_dir)

        if no_provider:
            print_error_and_exit(
                "Error: No inference provider configured.\n"
                "Set model and harness in .petri/petri.yaml to enable processing.",
            )

        if interrupted:
            raise typer.Exit(code=1)

        assert outcome is not None  # set unless an exception bubbled up

        final_states = outcome.final_states
        last_result = outcome.last_result

        typer.echo(f"\nGrow loop finished: {outcome.reason} after {outcome.passes_run} pass(es)")
        typer.echo(f"  Final queue: {backend.format_state_summary(final_states)}")

        if last_result is not None:
            typer.echo(f"  Last pass processed: {last_result.processed} cells")
            typer.echo(f"    Succeeded: {last_result.succeeded}")
            if last_result.stalled:
                typer.echo(f"    Stalled:   {last_result.stalled}")
            if last_result.failed:
                typer.echo(f"    Failed:    {last_result.failed}")
            for cell_result in last_result.results:
                typer.echo(
                    f"    {cell_result.cell_id}: {cell_result.final_state} "
                    f"({cell_result.iterations} iterations, "
                    f"{cell_result.events_logged} events)"
                )

        if outcome.reason == "all_terminal":
            raise typer.Exit(code=0)
        raise typer.Exit(code=1)
