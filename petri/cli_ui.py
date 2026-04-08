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

    Permanent lines (e.g. each new cell's claim text) can be written via
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


# ── multi-row spinner ────────────────────────────────────────────────────


# Prefix length reserved for ``"  ⠋ cell lead a  "`` before the slot text
# starts. Used to compute how much of each row text is visible on narrow
# terminals. ``"  ⠋ cell lead a  "`` is ~20 chars; 22 gives a little headroom.
_MULTI_CELL_LEAD_PREFIX_WIDTH = 22


def _cell_lead_label(slot_index: int) -> str:
    """Return the user-visible cell-lead identifier for a slot.

    Slots 0..25 map to letters ``a``..``z``. Anything beyond 25 falls
    back to the numeric index — concurrency settings above 26 are
    exceptional and deserve an ugly-but-unambiguous label.
    """
    if 0 <= slot_index < 26:
        return chr(ord("a") + slot_index)
    return str(slot_index)


class MultiSpinner:
    """N stacked status rows, one per concurrent worker slot.

    Each row is identified by a stable integer index in
    ``0..slot_count-1``. Rows update independently via
    :meth:`update_slot`. Permanent lines (e.g. a periodic aggregate
    state line) can be written above all rows via :meth:`print_line`
    without disturbing the row block.

    TTY mode: a daemon animation thread redraws all N rows on every
    frame using ``\\033[{slot_count+1}F`` (cursor-up) + ``\\033[2K``
    (clear-line) + the header row + the per-slot payload. The cursor
    is kept just below the block between frames. The block contains
    ``slot_count + 1`` lines: one optional header + N rows.

    A persistent header line lives at the top of the block and is
    updated in place via :meth:`set_header` — use it for aggregate
    state ("status: queued=9 ...") that should overwrite itself
    rather than scroll. :meth:`print_line` writes permanent lines
    that ARE meant to scroll above the block.

    Plain mode: each :meth:`update_slot` call writes a single
    ``[cell lead {letter}] <text>`` line; :meth:`set_header` writes
    a single ``status: <text>`` line each time it's called; no
    animation thread is spawned.

    ``MultiSpinner`` and the single-line :class:`Spinner` are
    intentionally independent classes — the seed wizard relies on
    the exact sequential behaviour of ``Spinner``, so this is a
    parallel primitive built for the concurrent ``petri grow`` path.
    """

    def __init__(
        self,
        label: str,
        slot_count: int,
        *,
        stream: Optional[TextIO] = None,
        force_plain: bool = False,
    ) -> None:
        self._label = label.strip()
        self._slot_count = max(0, int(slot_count))
        self._stream = stream or sys.stdout
        self._is_tty = (not force_plain) and self._stream_isatty()
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._animation_thread: Optional[threading.Thread] = None
        self._start_time: float = 0.0
        self._header_text: str = ""
        self._slot_texts: list[str] = [""] * self._slot_count
        self._frame_iterator = itertools.cycle(_FRAMES)
        self._current_frame: str = _FRAMES[0]
        self._terminal_width_cached: int = _terminal_width()

    @property
    def _block_line_count(self) -> int:
        """Total vertical lines managed by the animation thread.

        The block layout is ``[header row] + [slot rows...]``, so
        every ``\\033[NF`` cursor-up must move past all
        ``slot_count + 1`` lines to reach the header position.
        """
        return self._slot_count + 1

    def update_slot(self, slot_idx: int, text: str) -> None:
        """Set the live text for a single worker slot.

        Invalid indices are silently dropped — the engine's slot pool
        is the source of truth and a stale callback must never crash
        the UI. Safe from any thread.
        """
        if slot_idx < 0 or slot_idx >= self._slot_count:
            return
        if text is None:
            return
        flat = text.replace("\n", " ").strip()
        if self._is_tty:
            with self._lock:
                self._slot_texts[slot_idx] = flat
            return
        self._stream.write(f"[cell lead {_cell_lead_label(slot_idx)}] {flat}\n")
        try:
            self._stream.flush()
        except (ValueError, OSError):
            pass

    def set_header(self, text: str) -> None:
        """Set the persistent header text shown just above the slot rows.

        The header updates in place on each animation frame — it does
        NOT scroll. Use this for aggregate state (e.g.
        ``"status: queued=9 socratic_active=4"``) that should overwrite
        itself as the state evolves. Use :meth:`print_line` instead
        for permanent lines that should scroll above the block.

        Safe from any thread. Passing ``None`` or empty string clears
        the header (the row is still drawn, just blank).
        """
        flat = "" if text is None else text.replace("\n", " ").strip()
        if self._is_tty:
            with self._lock:
                self._header_text = flat
            return
        # Plain mode: each call becomes a single permanent line. This
        # matches update_slot's plain-mode behavior (one line per call)
        # and keeps non-TTY runs readable as a log.
        if flat:
            self._stream.write(flat + "\n")
            try:
                self._stream.flush()
            except (ValueError, OSError):
                pass

    def print_line(self, text: str) -> None:
        """Write a permanent line above the row block.

        TTY mode: clears the row block, writes the permanent line,
        then redraws all N rows below it. Plain mode: writes the
        line verbatim. Safe from any thread.
        """
        if text is None:
            return
        flat = text.replace("\n", " ").strip()
        if not flat:
            return
        if not self._is_tty:
            self._stream.write(flat + "\n")
            try:
                self._stream.flush()
            except (ValueError, OSError):
                pass
            return
        with self._lock:
            try:
                # Move up to the top of the block (the header row),
                # overwrite it with the permanent line, then redraw
                # the header + every slot row underneath. The net
                # effect is that the permanent line lands at the
                # current block-top position and the header+rows
                # shift down by one line — exactly the "scroll-up"
                # illusion the old print_line provided before the
                # header row existed.
                self._stream.write(f"\033[{self._block_line_count}F")
                self._stream.write("\033[2K")
                self._stream.write(flat + "\n")
                frame_char = self._current_frame
                # Redraw header
                self._stream.write("\033[2K")
                self._stream.write(self._header_text + "\n")
                # Redraw every slot row
                for slot_idx in range(self._slot_count):
                    self._stream.write("\033[2K")
                    self._stream.write(self._format_row(slot_idx, frame_char))
                self._stream.flush()
            except (ValueError, OSError):
                return

    def __enter__(self) -> "MultiSpinner":
        self._start_time = time.monotonic()
        self._stream.write(f"\n─ {self._label}\n")
        if self._is_tty and self._slot_count > 0:
            # Reserve vertical real estate for the header row + every
            # slot row so the animation thread can immediately do
            # ``\033[(N+1)F`` and land at the top of the block
            # (the header) without stomping whatever was on the
            # current line.
            self._stream.write("\n" * self._block_line_count)
        try:
            self._stream.flush()
        except (ValueError, OSError):
            pass
        if self._is_tty and self._slot_count > 0:
            self._animation_thread = threading.Thread(
                target=self._animate, daemon=True
            )
            self._animation_thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        elapsed = time.monotonic() - self._start_time
        self._stop_event.set()
        if self._animation_thread is not None:
            self._animation_thread.join(timeout=1.0)
        mark = "✗" if exc_type is not None else "✓"
        summary_line = f"{mark} {self._label} ({elapsed:.1f}s)\n"
        with self._lock:
            if self._is_tty and self._slot_count > 0:
                try:
                    # Rewind to the top of the block (header row)
                    # and blank out every line — header + all slot
                    # rows — so the summary line replaces the block
                    # entirely.
                    self._stream.write(f"\033[{self._block_line_count}F")
                    for _line_index in range(self._block_line_count):
                        self._stream.write("\033[2K\n")
                    # Move back up to the cleared header position and
                    # write the summary in its place.
                    self._stream.write(f"\033[{self._block_line_count}F")
                    self._stream.write(summary_line)
                    self._stream.flush()
                except (ValueError, OSError):
                    pass
            else:
                self._stream.write(summary_line)
                try:
                    self._stream.flush()
                except (ValueError, OSError):
                    pass
        return None

    # ── internal ─────────────────────────────────────────────────────

    def _stream_isatty(self) -> bool:
        try:
            return bool(self._stream.isatty())
        except (AttributeError, ValueError):
            return False

    def _format_row(self, slot_idx: int, frame_char: str) -> str:
        """Render one slot row including the trailing newline.

        Truncates with an ellipsis to fit the cached terminal width,
        leaving room for the ``"  ⠋ cell lead a  "`` prefix.
        """
        raw_text = self._slot_texts[slot_idx] if slot_idx < len(self._slot_texts) else ""
        budget = max(10, self._terminal_width_cached - _MULTI_CELL_LEAD_PREFIX_WIDTH)
        visible_text = raw_text
        if len(visible_text) > budget:
            visible_text = visible_text[: budget - 1] + "…"
        cell_lead_letter = _cell_lead_label(slot_idx)
        return f"  {frame_char} cell lead {cell_lead_letter}  {visible_text}\n"

    def _animate(self) -> None:
        while True:
            if self._stop_event.is_set():
                return
            frame_char = next(self._frame_iterator)
            self._current_frame = frame_char
            with self._lock:
                try:
                    # Step to the top of the block (header row) and
                    # redraw the header + every slot row. The trailing
                    # newline on the last row leaves the cursor one
                    # line below the block, ready for the next tick.
                    self._stream.write(f"\033[{self._block_line_count}F")
                    # Header row — cleared + rewritten every frame so
                    # ``set_header`` takes effect on the next tick.
                    self._stream.write("\033[2K")
                    self._stream.write(self._header_text + "\n")
                    # Slot rows
                    for slot_idx in range(self._slot_count):
                        self._stream.write("\033[2K")
                        self._stream.write(self._format_row(slot_idx, frame_char))
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

    cells = graph.get_all_cells()
    if not cells:
        typer.echo("  (empty)")
        return

    for cell in cells:
        indent = "  " * (cell.level + 1)
        deps = graph.get_dependencies(cell.id)
        dep_arrow = ""
        if deps:
            dep_arrow = f" -> [{', '.join(deps)}]"
        typer.echo(f"{indent}[L{cell.level}] {cell.id}: {cell.claim_text}{dep_arrow}")

    typer.echo("")


def render_dot(graph, colony) -> None:
    """Render a colony as Graphviz DOT format."""
    typer.echo(f'digraph "{colony.id}" {{')
    typer.echo("  rankdir=TB;")
    typer.echo(f'  label="{colony.center_claim}";')
    typer.echo("")

    for cell in graph.get_all_cells():
        label = cell.claim_text.replace('"', '\\"')
        if len(label) > 50:
            label = label[:47] + "..."
        typer.echo(f'  "{cell.id}" [label="{label}\\nL{cell.level}"];')

    typer.echo("")

    for edge in graph.get_edges():
        style = ""
        if edge.edge_type == "cross_colony":
            style = ' [style=dashed, color=blue]'
        typer.echo(f'  "{edge.from_cell}" -> "{edge.to_cell}"{style};')

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
    """Daemon-thread body that refreshes the aggregate status header.

    Each tick (``interval_seconds``):

    * snapshot queue state via ``get_state_summary``
    * push a ``status: <state_summary>`` line into the spinner header
      via ``spinner.set_header`` (if available) — this updates in
      place every animation frame rather than scrolling. For
      :class:`Spinner` (single-line, no header support) this falls
      back to ``print_line``.

    Per-cell lifecycle activity is already rendered in real time on
    the :class:`MultiSpinner` per-slot rows, so this loop no longer
    walks event logs for recent verdicts. Shuts down promptly when
    ``stop_event`` is set.
    """
    from petri.engine.grow_loop import format_state_summary
    from petri.storage.queue import get_state_summary

    set_header = getattr(spinner, "set_header", None)

    while True:
        if stop_event.is_set():
            return

        try:
            state_counts = get_state_summary(queue_path)
        except Exception:
            state_counts = {}

        status_line = f"status: {format_state_summary(state_counts)}"
        if set_header is not None:
            set_header(status_line)
        else:
            # Single-line Spinner fallback — no header row, so
            # fall back to scrolling the status above the spinner.
            spinner.print_line(status_line)

        # Sleep until next tick OR until stop_event is set
        if stop_event.wait(timeout=interval_seconds):
            return


# Maximum characters to show from a verdict's summary in the CLI status feed.
# Full summaries are persisted to evidence.md; the CLI is just a high-level
# activity feed, not a debugging surface.
_STATUS_SUMMARY_MAX_CHARS = 100


def short_cell_id(cell_id: str) -> str:
    """Return the trailing ``level-seq`` portion of a composite cell id.

    ``"petri-ai-considered-commodity-12-001-002"`` -> ``"001-002"``.
    Falls back to the original string if it has fewer than 2 hyphens.
    """
    parts = cell_id.rsplit("-", 2)
    if len(parts) < 3:
        return cell_id
    return f"{parts[-2]}-{parts[-1]}"


def _truncate_summary(summary_text: str, max_chars: int = _STATUS_SUMMARY_MAX_CHARS) -> str:
    """Collapse whitespace and truncate to ``max_chars`` with an ellipsis."""
    collapsed = " ".join(summary_text.split())
    if len(collapsed) <= max_chars:
        return collapsed
    return collapsed[: max_chars - 1].rstrip() + "…"


def _format_status_event(event: dict) -> str:
    """Render a verdict_issued or convergence_checked event as a one-line
    status entry: ``<short_cell> <agent>: <verdict> — <truncated summary>``.

    Drops the redundant event_type prefix and dish/colony portion of the
    cell id since both are obvious from context. Truncates the summary so
    one event fits on one terminal line — the full text lives in
    evidence.md.
    """
    cell_id = short_cell_id(event.get("cell_id", ""))
    agent = event.get("agent", "")
    data = event.get("data", {}) or {}
    verdict = data.get("verdict") or data.get("status") or ""
    summary_text = data.get("summary", "")

    head = f"{cell_id} {agent}: {verdict}".strip()
    if summary_text:
        return f"{head} — {_truncate_summary(summary_text)}"
    return head
