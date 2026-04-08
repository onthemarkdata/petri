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
from typing import Optional, TextIO


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
