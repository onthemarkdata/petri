"""Unit tests for the ``grow_loop`` driver in ``petri.engine.grow_loop``.

These tests exercise the pure ``grow_loop`` driver in isolation, with
fake polling primitives, so they never spin up the real engine, the real
spinner, or the real status thread.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from petri.engine.grow_loop import (
    GrowLoopOutcome,
    all_states_terminal,
    format_state_summary,
    grow_loop,
)


def _fake_pass_result(processed: int) -> SimpleNamespace:
    """Build a stand-in for ``QueueProcessingResult``.

    The loop only inspects ``.processed``; everything else is irrelevant.
    """
    return SimpleNamespace(
        processed=processed,
        succeeded=processed,
        failed=0,
        stalled=0,
        results=[],
    )


def _make_get_states(state_sequence: list[dict[str, int]]):
    """Return a callable that yields the next dict in ``state_sequence``.

    Once the sequence is exhausted the last value is returned forever.
    """
    iterator = iter(state_sequence)
    last = {"_unused": 0}

    def _get_states() -> dict[str, int]:
        nonlocal last
        try:
            last = next(iterator)
        except StopIteration:
            pass
        return last

    return _get_states


def _make_run_one_pass(processed_sequence: list[int]):
    """Return a callable that yields fake results from ``processed_sequence``."""
    iterator = iter(processed_sequence)
    last_processed = 0

    def _run_one_pass():
        nonlocal last_processed
        try:
            last_processed = next(iterator)
        except StopIteration:
            pass
        return _fake_pass_result(last_processed)

    return _run_one_pass


# ── 1. all-terminal exit ────────────────────────────────────────────────


def test_loop_exits_when_all_nodes_terminal() -> None:
    get_states = _make_get_states([{"done": 3}])
    run_one_pass = _make_run_one_pass([3])

    pass_call_count = {"count": 0}

    def _counting_pass():
        pass_call_count["count"] += 1
        return run_one_pass()

    outcome = grow_loop(
        run_one_pass=_counting_pass,
        get_states=get_states,
        is_stopped=lambda: False,
    )

    assert isinstance(outcome, GrowLoopOutcome)
    assert outcome.reason == "all_terminal"
    assert pass_call_count["count"] == 1
    assert outcome.passes_run == 1
    assert outcome.final_states == {"done": 3}


# ── 2. stop-signal exit ─────────────────────────────────────────────────


def test_loop_exits_on_stop_signal_after_first_pass() -> None:
    """``is_stopped`` flips to True after one pass — loop exits next check."""
    stop_calls = {"count": 0}

    def _is_stopped() -> bool:
        # First check (before pass 1) is False; second check (before pass 2)
        # is True.  Net result: exactly one pass runs.
        stop_calls["count"] += 1
        return stop_calls["count"] > 1

    get_states = _make_get_states([{"research_active": 2}])
    run_one_pass = _make_run_one_pass([1])

    outcome = grow_loop(
        run_one_pass=run_one_pass,
        get_states=get_states,
        is_stopped=_is_stopped,
    )

    assert outcome.reason == "stop_signal"
    assert outcome.passes_run == 1
    assert outcome.last_result is not None
    assert outcome.last_result.processed == 1


def test_loop_exits_on_stop_signal_before_first_pass() -> None:
    """If the sentinel is already set, zero passes run."""
    pass_call_count = {"count": 0}

    def _run_one_pass():
        pass_call_count["count"] += 1
        return _fake_pass_result(0)

    outcome = grow_loop(
        run_one_pass=_run_one_pass,
        get_states=lambda: {"queued": 5},
        is_stopped=lambda: True,
    )

    assert outcome.reason == "stop_signal"
    assert pass_call_count["count"] == 0
    assert outcome.passes_run == 0
    assert outcome.last_result is None


# ── 3. no-progress exit ─────────────────────────────────────────────────


def test_loop_exits_on_no_progress() -> None:
    pass_call_count = {"count": 0}

    def _run_one_pass():
        pass_call_count["count"] += 1
        return _fake_pass_result(0)

    def _get_states() -> dict[str, int]:
        return {"research_active": 2}

    outcome = grow_loop(
        run_one_pass=_run_one_pass,
        get_states=_get_states,
        is_stopped=lambda: False,
        max_no_progress_passes=2,
    )

    assert outcome.reason == "no_progress"
    # First pass: signature recorded, counter still 0 (signature was None
    # before).  Second pass: matching signature → counter 1.  Third pass:
    # matching signature → counter 2 → exit.
    assert pass_call_count["count"] == 3
    assert outcome.passes_run == 3
    assert outcome.last_result is not None
    assert outcome.last_result.processed == 0
    assert outcome.final_states == {"research_active": 2}


# ── 4. continue while progressing ───────────────────────────────────────


def test_loop_continues_while_progressing() -> None:
    state_sequence = [
        {"research_active": 3},
        {"research_active": 2},
        {"research_active": 1, "done": 2},
        {"done": 3},
    ]
    processed_sequence = [1, 1, 1, 0]

    get_states = _make_get_states(state_sequence)
    run_one_pass = _make_run_one_pass(processed_sequence)

    outcome = grow_loop(
        run_one_pass=run_one_pass,
        get_states=get_states,
        is_stopped=lambda: False,
    )

    assert outcome.reason == "all_terminal"
    assert outcome.passes_run == 4
    assert outcome.final_states == {"done": 3}


# ── 5. zero-progress counter resets on subsequent progress ──────────────


def test_loop_does_not_exit_on_single_zero_progress_pass() -> None:
    """One pass with processed=0 followed by a productive pass: no exit."""
    state_sequence = [
        {"research_active": 2},          # pass 1: stuck (zero progress)
        {"research_active": 2},          # pass 2: still stuck — counter 1
        {"research_active": 1, "done": 1},  # pass 3: progress! reset counter
        {"done": 2},                      # pass 4: terminal
    ]
    processed_sequence = [0, 0, 1, 0]

    get_states = _make_get_states(state_sequence)
    run_one_pass = _make_run_one_pass(processed_sequence)

    outcome = grow_loop(
        run_one_pass=run_one_pass,
        get_states=get_states,
        is_stopped=lambda: False,
        max_no_progress_passes=3,
    )

    assert outcome.reason == "all_terminal"
    assert outcome.passes_run == 4
    assert outcome.final_states == {"done": 2}


# ── on_pass_complete callback ───────────────────────────────────────────


def test_on_pass_complete_callback_is_invoked_each_pass() -> None:
    callback_calls: list[tuple[dict[str, int], int]] = []

    def _on_pass_complete(state_counts: dict[str, int], pass_result) -> None:
        callback_calls.append((dict(state_counts), pass_result.processed))

    state_sequence = [
        {"research_active": 1},
        {"done": 1},
    ]
    processed_sequence = [1, 0]

    outcome = grow_loop(
        run_one_pass=_make_run_one_pass(processed_sequence),
        get_states=_make_get_states(state_sequence),
        is_stopped=lambda: False,
        on_pass_complete=_on_pass_complete,
    )

    assert outcome.reason == "all_terminal"
    assert callback_calls == [
        ({"research_active": 1}, 1),
        ({"done": 1}, 0),
    ]


def test_callback_exception_does_not_break_loop() -> None:
    def _bad_callback(state_counts, pass_result) -> None:
        raise RuntimeError("callback boom")

    outcome = grow_loop(
        run_one_pass=_make_run_one_pass([1]),
        get_states=_make_get_states([{"done": 1}]),
        is_stopped=lambda: False,
        on_pass_complete=_bad_callback,
    )

    assert outcome.reason == "all_terminal"
    assert outcome.passes_run == 1


# ── KeyboardInterrupt propagation ───────────────────────────────────────


def test_keyboard_interrupt_propagates() -> None:
    def _raising_pass():
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        grow_loop(
            run_one_pass=_raising_pass,
            get_states=lambda: {"research_active": 1},
            is_stopped=lambda: False,
        )


# ── all_states_terminal helper ──────────────────────────────────────────


def test_all_states_terminal_empty_dict_returns_false() -> None:
    """Empty state snapshot means the queue hasn't been populated yet."""
    assert all_states_terminal({}) is False


def test_all_states_terminal_with_terminal_states_returns_true() -> None:
    """All 'done' entries are terminal → the loop should exit."""
    assert all_states_terminal({"done": 3}) is True


# ── format_state_summary helper ─────────────────────────────────────────


def test_format_state_summary_outputs_compact_format() -> None:
    """Summary is sorted ``key=value`` pairs joined by spaces."""
    summary = format_state_summary({"research_active": 2, "done": 1})
    assert summary == "done=1 research_active=2"

    assert format_state_summary({}) == "queue empty"
