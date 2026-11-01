"""Check Python and the selected inference backend without making model calls."""

from __future__ import annotations

import shutil
import subprocess
import sys
import os
import re
from collections.abc import Mapping
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
            f"Fix: uv venv --python {minimum[0]}.{minimum[1]} .venv && source .venv/bin/activate"
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


def check_pi(minimum_node: tuple[int, int, int] = (22, 19, 0)) -> CheckResult:
    """Report pi and Node versions, using the Node floor selected by the RPC spike."""
    install = "See README.md#install-pi-and-nodejs; or select harness.backend: claude-code."
    binary = os.environ.get("PETRI_PI_BIN") or "pi"
    path = shutil.which(binary)
    if path is None:
        return CheckResult("pi CLI", False, f"{binary} not found. {install}")
    node = shutil.which("node")
    if node is None:
        return CheckResult("pi CLI", False, f"pi found at {path}, but Node.js is missing. {install}")
    versions = []
    for name, executable in (("pi", path), ("Node.js", node)):
        try:
            result = subprocess.run([executable, "--version"], capture_output=True, text=True, timeout=10)
        except (subprocess.TimeoutExpired, OSError):
            return CheckResult("pi CLI", False, f"{name} did not respond. {install}")
        output = result.stdout.strip() or result.stderr.strip()
        if result.returncode != 0 or not output:
            return CheckResult("pi CLI", False, f"{name} --version failed (exit {result.returncode}): {result.stderr.strip()[:300]}. {install}")
        versions.append(output.split("\n", 1)[0])
    summary = f"pi {versions[0]}; Node.js {versions[1]}"
    match = re.fullmatch(r"v?(\d+)\.(\d+)\.(\d+)", versions[1])
    if match is None or tuple(map(int, match.groups())) < minimum_node:
        floor = ".".join(map(str, minimum_node))
        return CheckResult("pi CLI", False, f"{summary}; requires Node.js {floor}+. {install}")
    return CheckResult("pi CLI", True, summary)


def run_preflight(dish_config: Mapping[str, object] | None = None) -> list[CheckResult]:
    """Check the selected backend; absent harness blocks retain legacy checks."""
    results = [check_python_version()]
    harness = dish_config.get("harness") if dish_config is not None else None
    if isinstance(harness, Mapping):
        backend = harness.get("backend")
    else:
        backend = harness or "claude-code"
    if backend == "pi":
        results.append(check_pi())
    elif backend == "claude-code":
        results.append(check_claude_cli())
    elif backend != "provider":
        results.append(CheckResult("Harness", False, "harness.backend must be pi, claude-code, or provider."))
    return results
