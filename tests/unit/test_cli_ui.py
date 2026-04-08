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

import pytest

from petri.cli_ui import Spinner


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
