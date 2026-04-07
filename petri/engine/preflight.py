"""Prerequisite checks for Petri.

Validates that the runtime environment has everything Petri needs:
Python version and Claude Code CLI.
All checks use stdlib only (shutil, subprocess, sys).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass


@dataclass
class CheckResult:
    """Outcome of a single prerequisite check."""

    name: str
    passed: bool
    message: str


def check_python_version(minimum: tuple[int, int] = (3, 11)) -> CheckResult:
    """Check that the running Python meets the minimum version."""
    current = sys.version_info[:2]
    if current >= minimum:
        return CheckResult(
            name="Python",
            passed=True,
            message=f"{current[0]}.{current[1]}",
        )
    return CheckResult(
        name="Python",
        passed=False,
        message=(
            f"{current[0]}.{current[1]} (requires {minimum[0]}.{minimum[1]}+). "
            "Fix: uv venv --python 3.11 .venv && source .venv/bin/activate"
        ),
    )


def check_claude_cli() -> CheckResult:
    """Check that the ``claude`` CLI is on PATH and responds."""
    path = shutil.which("claude")
    if path is None:
        return CheckResult(
            name="Claude Code CLI",
            passed=False,
            message=(
                "Not found on PATH. "
                "Install: https://docs.anthropic.com/en/docs/claude-code"
            ),
        )
    try:
        result = subprocess.run(
            ["claude", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        version = result.stdout.strip().split("\n")[0] if result.stdout else "unknown"
        return CheckResult(name="Claude Code CLI", passed=True, message=version)
    except (subprocess.TimeoutExpired, OSError):
        return CheckResult(
            name="Claude Code CLI",
            passed=False,
            message=f"Found at {path} but not responding",
        )


def run_preflight() -> list[CheckResult]:
    """Run all prerequisite checks and return results."""
    results: list[CheckResult] = []
    results.append(check_python_version())
    results.append(check_claude_cli())
    return results
