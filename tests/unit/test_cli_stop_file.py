"""Verify ``petri stop`` writes the cross-process stop sentinel file.

The actual queue / event-log side-effects are covered by other tests; the
narrow contract here is just: after ``petri stop`` returns, the
``.petri/.stop`` sentinel must exist so detached worker processes can
observe the stop request.
"""

from __future__ import annotations

from typer.testing import CliRunner

from petri.cli import app


runner = CliRunner()


def test_stop_command_writes_stop_file(seeded_petri_dir) -> None:
    """``petri stop`` must create ``.petri/.stop`` even with no active cells."""
    result = runner.invoke(app, ["stop"])

    # The stop command always exits 0 — it never errors when nothing is
    # running, it just reports "no active cells to stop".
    assert result.exit_code == 0, result.output

    petri_dir = seeded_petri_dir["petri_dir"]
    stop_file = petri_dir / ".stop"
    assert stop_file.exists(), (
        f"expected stop sentinel at {stop_file}; cli output:\n{result.output}"
    )


def test_stop_command_writes_stop_file_with_force(seeded_petri_dir) -> None:
    """``petri stop --force`` also writes the sentinel."""
    result = runner.invoke(app, ["stop", "--force"])

    assert result.exit_code == 0, result.output

    petri_dir = seeded_petri_dir["petri_dir"]
    stop_file = petri_dir / ".stop"
    assert stop_file.exists()
