"""petri grow command."""

from __future__ import annotations

from typing import Optional

import typer

from petri.cli._bootstrap import find_petri_dir, resolve_provider
from petri.cli_ui import grow_status_loop, print_error_and_exit
from petri.config import MAX_CONCURRENT
from petri.engine.grow_loop import (
    GROW_STATUS_INTERVAL_SECONDS,
    GrowLoopOutcome,
    format_state_summary,
    grow_loop,
)


def register(app: typer.Typer) -> None:
    @app.command()
    def grow(
        nodes: Optional[list[str]] = typer.Argument(None, help="Node IDs to grow"),
        colony_name: Optional[str] = typer.Option(
            None, "--colony", help="Grow all in colony"
        ),
        all_nodes: bool = typer.Option(False, "--all", help="Grow all eligible"),
        max_concurrent: int = typer.Option(
            MAX_CONCURRENT, "--max-concurrent", help="Max parallel nodes"
        ),
        dry_run: bool = typer.Option(
            False, "--dry-run", help="Show what would process"
        ),
    ) -> None:
        """Enqueue nodes and process through validation pipeline.

        Loops calling ``process_queue`` until every queue entry is in a
        terminal state, the cross-process stop sentinel appears, or two
        consecutive passes make no progress at all.  A daemon status thread
        prints periodic progress lines above a persistent spinner.
        """
        import threading

        petri_dir = find_petri_dir()

        # Preflight: check claude CLI before attempting processing
        from petri.engine.preflight import check_claude_cli

        claude_check = check_claude_cli()
        if not claude_check.passed:
            print_error_and_exit(
                f"Error: {claude_check.message}\n"
                "Run 'petri inspect' to check all prerequisites.",
            )

        from petri.cli_ui import Spinner
        from petri.engine.processor import (
            NoProviderError,
            clear_stop_file,
            is_stop_file_present,
            process_queue,
        )
        from petri.storage.queue import get_state_summary

        # Resolve the inference provider from petri.yaml config
        provider = resolve_provider(petri_dir)

        queue_path = petri_dir / "queue.json"

        # Drop any stale stop sentinel from a previous interrupted run.
        clear_stop_file(petri_dir)

        # ── dry-run path: single pass, no loop, no spinner ──
        if dry_run:
            try:
                dry_result = process_queue(
                    petri_dir=petri_dir,
                    provider=provider,
                    max_concurrent=max_concurrent,
                    node_ids=nodes,
                    colony_filter=colony_name,
                    all_nodes=all_nodes,
                    dry_run=True,
                )
            except NoProviderError:
                print_error_and_exit(
                    "Error: No inference provider configured.\n"
                    "Set model and harness in .petri/petri.yaml to enable processing.",
                )

            would_process = dry_result.would_process
            if would_process:
                typer.echo(f"Would process {len(would_process)} nodes:")
                for node_id in would_process:
                    typer.echo(f"  {node_id}")
            else:
                typer.echo("No eligible nodes found.")
            raise typer.Exit(code=0)

        # ── live path: loop until terminal/stopped/no-progress ──
        def _run_one_pass():
            return process_queue(
                petri_dir=petri_dir,
                provider=provider,
                max_concurrent=max_concurrent,
                node_ids=nodes,
                colony_filter=colony_name,
                all_nodes=all_nodes,
                dry_run=False,
            )

        def _get_states() -> dict[str, int]:
            try:
                return get_state_summary(queue_path)
            except Exception:
                return {}

        def _is_stopped() -> bool:
            return is_stop_file_present(petri_dir)

        outcome: GrowLoopOutcome | None = None
        interrupted = False
        no_provider = False
        status_stop_event = threading.Event()
        status_thread: threading.Thread | None = None

        try:
            with Spinner("growing") as spinner:
                # Surface the starting queue state immediately so the user
                # sees something other than a static "growing" label while
                # the first pass spins up.
                initial_states = _get_states()
                if initial_states:
                    spinner.update(
                        f"growing — {format_state_summary(initial_states)}"
                    )

                status_thread = threading.Thread(
                    target=grow_status_loop,
                    kwargs={
                        "petri_dir": petri_dir,
                        "queue_path": queue_path,
                        "spinner": spinner,
                        "stop_event": status_stop_event,
                        "interval_seconds": GROW_STATUS_INTERVAL_SECONDS,
                    },
                    daemon=True,
                )
                status_thread.start()

                def _on_pass_complete(state_counts: dict[str, int], pass_result) -> None:
                    spinner.update(
                        f"pass complete — {format_state_summary(state_counts)}"
                    )

                try:
                    outcome = grow_loop(
                        run_one_pass=_run_one_pass,
                        get_states=_get_states,
                        is_stopped=_is_stopped,
                        on_pass_complete=_on_pass_complete,
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
            clear_stop_file(petri_dir)

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

        typer.echo(
            f"\nGrow loop finished: {outcome.reason} "
            f"after {outcome.passes_run} pass(es)"
        )
        typer.echo(f"  Final queue: {format_state_summary(final_states)}")

        if last_result is not None:
            typer.echo(f"  Last pass processed: {last_result.processed} nodes")
            typer.echo(f"    Succeeded: {last_result.succeeded}")
            if last_result.stalled:
                typer.echo(f"    Stalled:   {last_result.stalled}")
            if last_result.failed:
                typer.echo(f"    Failed:    {last_result.failed}")
            for node_result in last_result.results:
                typer.echo(
                    f"    {node_result.node_id}: {node_result.final_state} "
                    f"({node_result.iterations} iterations, "
                    f"{node_result.events_logged} events)"
                )

        if outcome.reason == "all_terminal":
            raise typer.Exit(code=0)
        raise typer.Exit(code=1)
