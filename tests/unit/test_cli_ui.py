"""Unit tests for petri.cli_ui.

The Spinner is a pure UI primitive — it holds zero domain strings and
takes all content from its caller via ``update()`` / ``print_line()`` /
the required ``label``. These tests verify the rendering contract
without depending on real timing or terminal behaviour.
"""

from __future__ import annotations

import io
import re
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import typer

from petri.cli_ui import (
    Spinner,
    grow_status_loop,
    print_error_and_exit,
    render_dot,
    render_text_tree,
)


class _FakeTtyBuffer(io.StringIO):
    """StringIO that reports as a TTY."""

    def isatty(self) -> bool:  # type: ignore[override]
        return True


# ── plain (non-TTY) mode ─────────────────────────────────────────────────


class TestPlainMode:
    def test_label_banner_and_check_on_exit(self):
        buf = io.StringIO()
        with Spinner("substance check", stream=buf, force_plain=True) as sp:
            sp.update("hello")
        out = buf.getvalue()
        assert "─ substance check" in out
        assert "→ hello" in out
        assert re.search(r"✓ substance check \(\d+\.\d+s\)", out)

    def test_strips_newlines(self):
        buf = io.StringIO()
        with Spinner("stage", stream=buf, force_plain=True) as sp:
            sp.update("a\nb")
        out = buf.getvalue()
        assert "a b" in out

    def test_empty_update_is_noop(self):
        buf = io.StringIO()
        with Spinner("stage", stream=buf, force_plain=True) as sp:
            sp.update("")
            sp.update("   ")
            sp.update("\n")
        out = buf.getvalue()
        assert "→" not in out
        assert "✓" in out

    def test_exception_marks_failure_with_label(self):
        buf = io.StringIO()
        with pytest.raises(RuntimeError, match="boom"):
            with Spinner("decomposing", stream=buf, force_plain=True) as sp:
                sp.update("about to fail")
                raise RuntimeError("boom")
        out = buf.getvalue()
        assert re.search(r"✗ decomposing \(\d+\.\d+s\)", out)
        assert "✓" not in out

    def test_multiple_updates_each_become_a_line(self):
        buf = io.StringIO()
        with Spinner("stage", stream=buf, force_plain=True) as sp:
            sp.update("first")
            sp.update("second")
            sp.update("third")
        out = buf.getvalue()
        assert "→ first" in out
        assert "→ second" in out
        assert "→ third" in out


# ── print_line: permanent claim lines above the spinner ─────────────────


class TestPrintLine:
    def test_plain_mode(self):
        buf = io.StringIO()
        with Spinner("decomposing", stream=buf, force_plain=True) as sp:
            sp.print_line("first claim")
            sp.print_line("second claim")
        out = buf.getvalue()
        assert "  • first claim" in out
        assert "  • second claim" in out

    def test_strips_newlines(self):
        buf = io.StringIO()
        with Spinner("decomposing", stream=buf, force_plain=True) as sp:
            sp.print_line("a\nb\nc")
        out = buf.getvalue()
        assert "  • a b c" in out

    def test_empty_is_noop(self):
        buf = io.StringIO()
        with Spinner("decomposing", stream=buf, force_plain=True) as sp:
            sp.print_line("")
            sp.print_line("   ")
            sp.print_line("\n")
        out = buf.getvalue()
        assert "•" not in out

    def test_tty_mode_smoke(self):
        buf = _FakeTtyBuffer()
        with Spinner("decomposing", stream=buf) as sp:
            sp.print_line("permanent claim")
            time.sleep(0.05)
        out = buf.getvalue()
        assert "  • permanent claim" in out


# ── TTY mode (smoke) ─────────────────────────────────────────────────────


class TestTtyMode:
    def test_thread_starts_and_stops_cleanly(self):
        buf = _FakeTtyBuffer()
        with Spinner("stage", stream=buf) as sp:
            sp.update("first frame")
            time.sleep(0.05)  # let the spinner render at least one frame
            sp.update("second frame")
            time.sleep(0.05)
        out = buf.getvalue()
        # Spinner thread wrote *something*
        assert out
        # Final ✓ line includes the label
        assert "✓ stage" in out
        # At least one of the streamed payloads survives in the buffer
        assert ("first frame" in out) or ("second frame" in out)

    def test_update_thread_safe_smoke(self):
        buf = _FakeTtyBuffer()

        def writer(sp: Spinner) -> None:
            for i in range(20):
                sp.update(f"update-{i}")
                time.sleep(0.001)

        with Spinner("stage", stream=buf) as sp:
            t = threading.Thread(target=writer, args=(sp,))
            t.start()
            t.join(timeout=2.0)
        # Smoke: no exception escaped, the buffer received some output
        assert buf.getvalue()

    def test_exception_path_in_tty_mode(self):
        buf = _FakeTtyBuffer()
        with pytest.raises(ValueError):
            with Spinner("stage", stream=buf) as sp:
                sp.update("doomed")
                raise ValueError("nope")
        out = buf.getvalue()
        assert "✗ stage" in out


# ── print_error_and_exit ─────────────────────────────────────────────────


class TestPrintErrorAndExit:
    def test_prints_to_stderr_and_raises(self, capsys):
        with pytest.raises(typer.Exit) as exc_info:
            print_error_and_exit("Error: something bad happened")
        captured = capsys.readouterr()
        assert "Error: something bad happened" in captured.err
        # Default code is 1
        assert exc_info.value.exit_code == 1

    def test_default_code_is_one(self):
        with pytest.raises(typer.Exit) as exc_info:
            print_error_and_exit("boom")
        assert exc_info.value.exit_code == 1

    def test_custom_code_is_honored(self, capsys):
        with pytest.raises(typer.Exit) as exc_info:
            print_error_and_exit("nope", code=2)
        captured = capsys.readouterr()
        assert "nope" in captured.err
        assert exc_info.value.exit_code == 2


# ── render_text_tree / render_dot fixtures ──────────────────────────────


class _FakeNode:
    def __init__(self, node_id: str, level: int, claim_text: str) -> None:
        self.id = node_id
        self.level = level
        self.claim_text = claim_text


class _FakeEdge:
    def __init__(self, from_node: str, to_node: str, edge_type: str = "intra_colony") -> None:
        self.from_node = from_node
        self.to_node = to_node
        self.edge_type = edge_type


class _FakeGraph:
    def __init__(self, nodes: list[_FakeNode], edges: list[_FakeEdge], deps: dict[str, list[str]] | None = None) -> None:
        self._nodes = nodes
        self._edges = edges
        self._deps = deps or {}

    def get_nodes(self) -> list[_FakeNode]:
        return list(self._nodes)

    def get_edges(self) -> list[_FakeEdge]:
        return list(self._edges)

    def get_dependencies(self, node_id: str) -> list[str]:
        return list(self._deps.get(node_id, []))


class _FakeColony:
    def __init__(self, colony_id: str, center_claim: str) -> None:
        self.id = colony_id
        self.center_claim = center_claim


# ── render_text_tree ─────────────────────────────────────────────────────


class TestRenderTextTree:
    def test_outputs_levels_in_order(self, capsys):
        root_node = _FakeNode("dish-col-0-0", level=0, claim_text="root claim")
        child_node = _FakeNode("dish-col-1-0", level=1, claim_text="child claim")
        graph = _FakeGraph(
            nodes=[root_node, child_node],
            edges=[],
            deps={"dish-col-1-0": ["dish-col-0-0"]},
        )
        colony = _FakeColony("dish-col", "root claim")

        render_text_tree(graph, colony)
        captured = capsys.readouterr()
        output = captured.out

        # Level-0 line must appear before the level-1 line in stdout.
        level_zero_index = output.find("[L0] dish-col-0-0")
        level_one_index = output.find("[L1] dish-col-1-0")
        assert level_zero_index >= 0
        assert level_one_index >= 0
        assert level_zero_index < level_one_index

        # Colony header present.
        assert "Colony: dish-col" in output
        assert "Center: root claim" in output

        # Dependency arrow on the child line.
        assert "-> [dish-col-0-0]" in output

    def test_empty_graph_emits_empty_marker(self, capsys):
        graph = _FakeGraph(nodes=[], edges=[])
        colony = _FakeColony("dish-empty", "nothing here")
        render_text_tree(graph, colony)
        captured = capsys.readouterr()
        assert "(empty)" in captured.out


# ── render_dot ───────────────────────────────────────────────────────────


class TestRenderDot:
    def test_outputs_valid_dot_syntax(self, capsys):
        first_node = _FakeNode("dish-col-0-0", level=0, claim_text="first")
        second_node = _FakeNode("dish-col-1-0", level=1, claim_text="second")
        edge = _FakeEdge("dish-col-0-0", "dish-col-1-0")
        graph = _FakeGraph(nodes=[first_node, second_node], edges=[edge])
        colony = _FakeColony("dish-col", "root claim")

        render_dot(graph, colony)
        captured = capsys.readouterr()
        output = captured.out

        assert 'digraph "dish-col"' in output
        assert "{" in output
        assert "}" in output
        assert '"dish-col-0-0" -> "dish-col-1-0"' in output

    def test_cross_colony_edge_is_dashed_blue(self, capsys):
        first_node = _FakeNode("dish-a-0-0", level=0, claim_text="a")
        second_node = _FakeNode("dish-b-0-0", level=0, claim_text="b")
        edge = _FakeEdge("dish-a-0-0", "dish-b-0-0", edge_type="cross_colony")
        graph = _FakeGraph(nodes=[first_node, second_node], edges=[edge])
        colony = _FakeColony("dish-a", "claim")

        render_dot(graph, colony)
        captured = capsys.readouterr()
        assert "style=dashed" in captured.out
        assert "color=blue" in captured.out

    def test_long_label_is_truncated(self, capsys):
        long_text = "x" * 80
        node = _FakeNode("dish-col-0-0", level=0, claim_text=long_text)
        graph = _FakeGraph(nodes=[node], edges=[])
        colony = _FakeColony("dish-col", "claim")
        render_dot(graph, colony)
        captured = capsys.readouterr()
        assert "..." in captured.out


# ── grow_status_loop ─────────────────────────────────────────────────────


class _RecordingSpinner:
    """Tiny stand-in for Spinner that records print_line and update calls."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.updates: list[str] = []

    def print_line(self, text: str) -> None:
        self.lines.append(text)

    def update(self, text: str) -> None:
        self.updates.append(text)


class TestGrowStatusLoop:
    def test_exits_when_stop_event_set(self, tmp_path, monkeypatch):
        # Stop event is already set before the loop runs — it should exit
        # on the very first wait() and return within ~2× the interval.
        petri_dir = tmp_path / ".petri"
        petri_dir.mkdir()
        queue_path = petri_dir / "queue.json"

        # Patch out storage imports so no real filesystem access is needed.
        import petri.storage.event_log as event_log_mod
        import petri.storage.queue as queue_mod

        monkeypatch.setattr(
            queue_mod, "get_state_summary", lambda _path: {}, raising=True
        )
        monkeypatch.setattr(
            event_log_mod, "query_events", lambda *args, **kwargs: [], raising=True
        )

        spinner = _RecordingSpinner()
        stop_event = threading.Event()
        stop_event.set()  # Pre-set: loop should not iterate at all.

        interval_seconds = 0.05
        start_time = time.monotonic()
        grow_status_loop(
            petri_dir=petri_dir,
            queue_path=queue_path,
            spinner=spinner,
            stop_event=stop_event,
            interval_seconds=interval_seconds,
        )
        elapsed = time.monotonic() - start_time

        # Should have returned well under 2× the interval since stop was pre-set.
        assert elapsed < interval_seconds * 2 + 0.2
        # And should never have called print_line, because the loop exited
        # on the very first while-check without advancing.
        assert spinner.lines == []

    def test_calls_print_line_with_status_header(self, tmp_path, monkeypatch):
        petri_dir = tmp_path / ".petri"
        petri_dir.mkdir()
        queue_path = petri_dir / "queue.json"

        # Stub storage so we get one deterministic state snapshot.
        import petri.storage.event_log as event_log_mod
        import petri.storage.queue as queue_mod

        monkeypatch.setattr(
            queue_mod,
            "get_state_summary",
            lambda _path: {"research_active": 2, "done": 1},
            raising=True,
        )
        monkeypatch.setattr(
            event_log_mod, "query_events", lambda *args, **kwargs: [], raising=True
        )

        spinner = _RecordingSpinner()
        stop_event = threading.Event()

        # Fire a background thread that signals stop after one tick has
        # definitely elapsed. Interval is tiny.
        interval_seconds = 0.05

        def stop_after_one_tick() -> None:
            time.sleep(interval_seconds * 2)
            stop_event.set()

        stopper = threading.Thread(target=stop_after_one_tick)
        stopper.start()

        grow_status_loop(
            petri_dir=petri_dir,
            queue_path=queue_path,
            spinner=spinner,
            stop_event=stop_event,
            interval_seconds=interval_seconds,
        )

        stopper.join(timeout=1.0)

        # At least one status line should have been printed, and it should
        # start with the "status:" header.
        assert spinner.lines, "expected at least one print_line call"
        status_lines = [line for line in spinner.lines if "status" in line]
        assert status_lines, f"no status header in recorded lines: {spinner.lines}"
        # And the state summary text should include the counts we stubbed.
        joined = " ".join(status_lines)
        assert "research_active=2" in joined
        assert "done=1" in joined

    def test_empty_state_summary_prints_queue_empty(self, tmp_path, monkeypatch):
        petri_dir = tmp_path / ".petri"
        petri_dir.mkdir()
        queue_path = petri_dir / "queue.json"

        import petri.storage.event_log as event_log_mod
        import petri.storage.queue as queue_mod

        monkeypatch.setattr(
            queue_mod, "get_state_summary", lambda _path: {}, raising=True
        )
        monkeypatch.setattr(
            event_log_mod, "query_events", lambda *args, **kwargs: [], raising=True
        )

        spinner = _RecordingSpinner()
        stop_event = threading.Event()
        interval_seconds = 0.05

        def stop_after_one_tick() -> None:
            time.sleep(interval_seconds * 2)
            stop_event.set()

        stopper = threading.Thread(target=stop_after_one_tick)
        stopper.start()
        grow_status_loop(
            petri_dir=petri_dir,
            queue_path=queue_path,
            spinner=spinner,
            stop_event=stop_event,
            interval_seconds=interval_seconds,
        )
        stopper.join(timeout=1.0)

        assert any("queue empty" in line for line in spinner.lines)

    def test_first_status_tick_is_immediate(self, tmp_path, monkeypatch):
        """The first status line should land BEFORE any wait — earlier
        behavior delayed the first tick by one full interval, so a
        60s default left users staring at a static spinner."""
        petri_dir = tmp_path / ".petri"
        petri_dir.mkdir()
        queue_path = petri_dir / "queue.json"

        import petri.storage.event_log as event_log_mod
        import petri.storage.queue as queue_mod

        monkeypatch.setattr(
            queue_mod, "get_state_summary",
            lambda _path: {"socratic_active": 4}, raising=True
        )
        monkeypatch.setattr(
            event_log_mod, "query_events",
            lambda *args, **kwargs: [], raising=True
        )

        spinner = _RecordingSpinner()
        stop_event = threading.Event()
        # Long interval — if the first tick still runs, we know it's
        # because the loop printed BEFORE waiting.
        interval_seconds = 5.0

        def stop_after_short_delay() -> None:
            time.sleep(0.1)
            stop_event.set()

        stopper = threading.Thread(target=stop_after_short_delay)
        stopper.start()
        start_time = time.monotonic()
        grow_status_loop(
            petri_dir=petri_dir,
            queue_path=queue_path,
            spinner=spinner,
            stop_event=stop_event,
            interval_seconds=interval_seconds,
        )
        elapsed = time.monotonic() - start_time
        stopper.join(timeout=1.0)

        # Returned in well under one full interval — proving the loop
        # printed first and then exited via the stop signal during wait.
        assert elapsed < interval_seconds
        # And actually printed something meaningful.
        assert any("socratic_active=4" in line for line in spinner.lines)
        # Spinner bottom line was also updated with the state summary.
        assert any("socratic_active=4" in update for update in spinner.updates)

    def test_format_status_event_truncates_long_summary(self):
        """Verdict summaries can be hundreds of words; the CLI must keep
        each event on one line. Full text lives in evidence.md."""
        from petri.cli_ui import _format_status_event

        long_summary = "The claim rests on at least seven hidden assumptions, " * 30
        event = {
            "type": "verdict_issued",
            "node_id": "petri-ai-considered-commodity-12-001-002",
            "agent": "socratic_challenge_assumptions",
            "data": {
                "verdict": "ASSUMPTIONS_CHALLENGED",
                "summary": long_summary,
            },
        }
        line = _format_status_event(event)
        # Compact node id (last two segments only).
        assert "001-002" in line
        assert "petri-ai-considered" not in line
        # Verdict surfaced.
        assert "ASSUMPTIONS_CHALLENGED" in line
        # Summary truncated with ellipsis.
        assert "…" in line
        # Whole line is reasonable for one terminal row (~250 char ceiling
        # to give some headroom for prefix + verdict + node id).
        assert len(line) < 250

    def test_format_status_event_short_node_id_handles_simple_ids(self):
        from petri.cli_ui import _short_node_id

        assert _short_node_id("dish-colony-001-002") == "001-002"
        assert _short_node_id("petri-ai-considered-12-003-004") == "003-004"
        # Short input falls through unchanged.
        assert _short_node_id("foo") == "foo"
