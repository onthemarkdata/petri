"""Terminal UI helpers — content-free render primitives.

This module deliberately contains no domain strings. Anything the user sees
in a spinner originates from a caller (and ultimately from an LLM stream).
"""

from __future__ import annotations

import itertools
import shutil
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import NoReturn, Optional, TextIO

import typer


_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
_FRAME_INTERVAL = 0.08
_CLEAR_LINE = "\r\033[2K"


def _terminal_width(default: int = 80) -> int:
    try:
        return shutil.get_terminal_size((default, 20)).columns
    except OSError:
        return default


class Spinner:
    """Animated status line for one pipeline stage.

    ``label`` identifies the stage (e.g. ``decomposing claim``); it's
    printed as a banner on enter and echoed on the ``✓ / ✗`` exit line.
    The live spinner content itself comes from ``update()`` and is meant
    to be the model's streaming output.

    Permanent lines (e.g. each new node's claim text) can be written via
    ``print_line(text)`` — they appear as ``  • <text>`` above the live
    spinner without disrupting the animation.

    TTY mode: braille spinner that re-renders the most recent ``update``
    on every frame, on a single line, truncated to terminal width.

    Non-TTY mode: each ``update`` writes a new ``  → <text>`` line; on
    exit, writes a single ``✓`` (or ``✗`` on exception) line with elapsed
    seconds. No threads.
    """

    def __init__(
        self,
        label: str,
        *,
        stream: Optional[TextIO] = None,
        force_plain: bool = False,
    ) -> None:
        self._label = label.strip()
        self._stream = stream or sys.stdout
        self._is_tty = (not force_plain) and self._stream_isatty()
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._start_time: float = 0.0
        self._text: str = ""

    def update(self, text: str) -> None:
        """Set the live text. Safe from any thread."""
        if not text:
            return
        flat = text.replace("\n", " ").strip()
        if not flat:
            return
        with self._lock:
            self._text = flat
            if not self._is_tty:
                self._stream.write(f"  → {flat}\n")
                self._stream.flush()

    def print_line(self, text: str) -> None:
        """Write a permanent line above the live spinner.

        In TTY mode the current spinner line is cleared, the permanent
        line is written, and the next animation frame redraws the spinner
        below it. In plain mode the line is written verbatim. Safe to
        call from any thread.
        """
        if not text:
            return
        flat = text.replace("\n", " ").strip()
        if not flat:
            return
        with self._lock:
            if self._is_tty:
                self._stream.write(_CLEAR_LINE)
                self._stream.write(f"  • {flat}\n")
            else:
                self._stream.write(f"  • {flat}\n")
            self._stream.flush()

    def __enter__(self) -> "Spinner":
        self._start_time = time.monotonic()
        self._stream.write(f"\n─ {self._label}\n")
        self._stream.flush()
        if self._is_tty:
            self._thread = threading.Thread(target=self._spin, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        elapsed = time.monotonic() - self._start_time
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        with self._lock:
            mark = "✗" if exc_type is not None else "✓"
            line = f"{mark} {self._label} ({elapsed:.1f}s)\n"
            if self._is_tty:
                self._stream.write(_CLEAR_LINE + line)
            else:
                self._stream.write(line)
            self._stream.flush()
        return None

    # ── internal ─────────────────────────────────────────────────────

    def _stream_isatty(self) -> bool:
        try:
            return bool(self._stream.isatty())
        except (AttributeError, ValueError):
            return False

    def _spin(self) -> None:
        width = _terminal_width()
        for frame in itertools.cycle(_FRAMES):
            if self._stop_event.is_set():
                return
            with self._lock:
                visible = self._text or ""
                budget = max(10, width - 4)
                if len(visible) > budget:
                    visible = visible[: budget - 1] + "…"
                line = f"{_CLEAR_LINE}{frame} {visible}"
            try:
                self._stream.write(line)
                self._stream.flush()
            except (ValueError, OSError):
                return
            if self._stop_event.wait(_FRAME_INTERVAL):
                return


# ── error helper ─────────────────────────────────────────────────────────


def print_error_and_exit(message: str, *, code: int = 1) -> NoReturn:
    """Print an error message to stderr and exit with the given code.

    Replaces the 40+ inlined copies of::

        typer.echo(f"Error: ...", err=True)
        raise typer.Exit(code=1)

    in ``petri/cli.py``.
    """
    typer.echo(message, err=True)
    raise typer.Exit(code=code)


# ── colony visualization renderers ───────────────────────────────────────


def render_text_tree(graph, colony) -> None:
    """Render a colony as an indented text tree."""
    typer.echo(f"\nColony: {colony.id}")
    typer.echo(f"Center: {colony.center_claim}")
    typer.echo("")

    nodes = graph.get_nodes()
    if not nodes:
        typer.echo("  (empty)")
        return

    for node in nodes:
        indent = "  " * (node.level + 1)
        deps = graph.get_dependencies(node.id)
        dep_arrow = ""
        if deps:
            dep_arrow = f" -> [{', '.join(deps)}]"
        typer.echo(f"{indent}[L{node.level}] {node.id}: {node.claim_text}{dep_arrow}")

    typer.echo("")


def render_dot(graph, colony) -> None:
    """Render a colony as Graphviz DOT format."""
    typer.echo(f'digraph "{colony.id}" {{')
    typer.echo("  rankdir=TB;")
    typer.echo(f'  label="{colony.center_claim}";')
    typer.echo("")

    for node in graph.get_nodes():
        label = node.claim_text.replace('"', '\\"')
        if len(label) > 50:
            label = label[:47] + "..."
        typer.echo(f'  "{node.id}" [label="{label}\\nL{node.level}"];')

    typer.echo("")

    for edge in graph.get_edges():
        style = ""
        if edge.edge_type == "cross_colony":
            style = ' [style=dashed, color=blue]'
        typer.echo(f'  "{edge.from_node}" -> "{edge.to_node}"{style};')

    typer.echo("}")


# ── grow status loop ─────────────────────────────────────────────────────


def grow_status_loop(
    *,
    petri_dir: Path,
    queue_path: Path,
    spinner,
    stop_event,
    interval_seconds: float,
) -> None:
    """Daemon-thread body that prints periodic progress lines.

    Each tick (``interval_seconds``):

    * snapshot queue state via ``get_state_summary``
    * walk every ``events.jsonl`` under ``petri-dishes/`` and pull recent
      ``verdict_issued`` and ``convergence_checked`` events written since
      the previous tick (newest-first, top 5)
    * print all of that as permanent lines above the live spinner

    Shuts down promptly when ``stop_event`` is set.
    """
    from petri.engine.grow_loop import format_state_summary
    from petri.storage.event_log import query_events
    from petri.storage.paths import iter_events_files
    from petri.storage.queue import get_state_summary

    cutoff_iso = datetime.now(timezone.utc).isoformat()

    while not stop_event.is_set():
        # Wait first so the very first status line lands one interval in,
        # not at t=0 (which would race with the loop's own startup output).
        if stop_event.wait(timeout=interval_seconds):
            return

        try:
            state_counts = get_state_summary(queue_path)
        except Exception:
            state_counts = {}

        spinner.print_line(f"status: {format_state_summary(state_counts)}")

        recent_events: list[dict] = []
        for events_file in iter_events_files(petri_dir):
            try:
                for event_type in ("verdict_issued", "convergence_checked"):
                    recent_events.extend(
                        query_events(
                            events_file,
                            event_type=event_type,
                            since=cutoff_iso,
                        )
                    )
            except Exception:
                continue

        recent_events.sort(
            key=lambda evt: evt.get("timestamp", ""), reverse=True
        )

        for event in recent_events[:5]:
            event_type = event.get("type", "")
            node_id = event.get("node_id", "")
            agent = event.get("agent", "")
            data = event.get("data", {}) or {}
            verdict = data.get("verdict") or data.get("status") or ""
            summary_text = data.get("summary", "")
            line = f"{event_type} {node_id} {agent} {verdict}".strip()
            if summary_text:
                line = f"{line} — {summary_text}"
            spinner.print_line(line)

        # Advance the cutoff so the next tick only fetches truly new events
        cutoff_iso = datetime.now(timezone.utc).isoformat()
