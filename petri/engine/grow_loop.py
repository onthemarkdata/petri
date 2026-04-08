"""Grow-loop driver: pure logic for iterating petri's processing pipeline.

This module is intentionally Typer-free and Spinner-free. The CLI layer
wraps it with progress display, but the loop's termination logic, state
inspection, and outcome reporting all live here so they can be unit-tested
without spinning up a real engine or terminal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from petri.storage.queue import is_terminal_state

# Status thread tick interval. Module-level so tests can monkeypatch it
# down to a tiny value.  Configurable only via code edit for now.
GROW_STATUS_INTERVAL_SECONDS: float = 60.0


@dataclass
class GrowLoopOutcome:
    """Result of running ``grow_loop`` to termination."""

    reason: str  # "all_terminal" | "stop_signal" | "no_progress" | "interrupted"
    final_states: dict[str, int] = field(default_factory=dict)
    last_result: Any = None
    passes_run: int = 0


def all_states_terminal(state_counts: dict[str, int]) -> bool:
    """True iff every present state is terminal AND there is at least one entry.

    An empty dict is treated as not-yet-terminal — the queue may not be
    populated yet on the very first poll.
    """
    if not state_counts:
        return False
    return all(is_terminal_state(state) for state in state_counts.keys())


def grow_loop(
    *,
    run_one_pass: Callable[[], Any],
    get_states: Callable[[], dict[str, int]],
    is_stopped: Callable[[], bool],
    on_pass_complete: Callable[[dict[str, int], Any], None] | None = None,
    max_no_progress_passes: int = 2,
) -> GrowLoopOutcome:
    """Drive ``run_one_pass`` until a terminal condition is reached.

    Termination rules:
      * ``stop_signal``  — ``is_stopped()`` returns True (checked BEFORE each pass).
      * ``all_terminal`` — every present queue state is in ``TERMINAL_STATES``.
      * ``no_progress``  — ``max_no_progress_passes`` consecutive passes
        with ``processed == 0`` AND an unchanged state signature.
      * Otherwise loop continues.

    ``KeyboardInterrupt`` is propagated to the caller.
    """
    last_state_signature: tuple[tuple[str, int], ...] | None = None
    consecutive_zero_progress = 0
    last_result: Any = None
    passes_run = 0
    current_states: dict[str, int] = {}

    while True:
        # Stop check happens BEFORE every pass — including the first.  A
        # caller that pre-sets the stop sentinel gets zero passes run.
        if is_stopped():
            return GrowLoopOutcome(
                reason="stop_signal",
                final_states=current_states,
                last_result=last_result,
                passes_run=passes_run,
            )

        last_result = run_one_pass()
        passes_run += 1

        current_states = get_states()

        if on_pass_complete is not None:
            try:
                on_pass_complete(current_states, last_result)
            except Exception:
                # Callbacks must never break the loop
                pass

        if all_states_terminal(current_states):
            return GrowLoopOutcome(
                reason="all_terminal",
                final_states=current_states,
                last_result=last_result,
                passes_run=passes_run,
            )

        processed_count = getattr(last_result, "processed", 0) or 0
        state_signature = tuple(sorted(current_states.items()))

        if processed_count == 0 and state_signature == last_state_signature:
            consecutive_zero_progress += 1
        else:
            consecutive_zero_progress = 0

        last_state_signature = state_signature

        if consecutive_zero_progress >= max_no_progress_passes:
            return GrowLoopOutcome(
                reason="no_progress",
                final_states=current_states,
                last_result=last_result,
                passes_run=passes_run,
            )


def format_state_summary(state_counts: dict[str, int]) -> str:
    """Compact one-line summary like ``research_active=2 done=1``."""
    if not state_counts:
        return "queue empty"
    parts = [f"{state}={count}" for state, count in sorted(state_counts.items())]
    return " ".join(parts)
