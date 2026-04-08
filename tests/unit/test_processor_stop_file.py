"""Tests for the cross-process stop sentinel helpers in ``processor``.

These helpers live alongside the in-process ``threading.Event`` stop
signal and let the CLI ``petri stop`` command halt a detached ``petri
grow`` worker running in a separate process.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def petri_dir(tmp_path):
    """Return a fresh empty .petri/ directory as a Path."""
    directory = tmp_path / ".petri"
    directory.mkdir()
    return directory


def test_request_stop_file_creates_file(petri_dir):
    from petri.engine.processor import request_stop_file

    stop_path = request_stop_file(petri_dir)
    assert stop_path.exists()
    assert stop_path.name == ".stop"
    assert stop_path.parent == petri_dir
    assert "stop" in stop_path.read_text()


def test_is_stop_file_present_detects_file(petri_dir):
    from petri.engine.processor import (
        clear_stop_file,
        is_stop_file_present,
        request_stop_file,
    )

    assert is_stop_file_present(petri_dir) is False
    request_stop_file(petri_dir)
    assert is_stop_file_present(petri_dir) is True
    clear_stop_file(petri_dir)
    assert is_stop_file_present(petri_dir) is False


def test_clear_stop_file_is_idempotent(petri_dir):
    """Calling clear_stop_file when no sentinel exists must not raise."""
    from petri.engine.processor import clear_stop_file, is_stop_file_present

    assert is_stop_file_present(petri_dir) is False
    # First call on an empty dir — must be a no-op, not a FileNotFoundError.
    clear_stop_file(petri_dir)
    # Second call — still a no-op.
    clear_stop_file(petri_dir)
    assert is_stop_file_present(petri_dir) is False
