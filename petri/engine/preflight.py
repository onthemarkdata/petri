"""Prerequisite checks for Petri.

Validates that the runtime environment has everything Petri needs:
Python version, Claude Code CLI, Ollama, and the configured model.
All checks use stdlib only (shutil, subprocess, sys).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass

# Model name prefixes that indicate cloud providers (skip Ollama checks).
_CLOUD_PREFIXES = ("claude-", "gpt-", "o1-", "o3-", "o4-")


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


def check_ollama(model_name: str | None = None) -> list[CheckResult]:
    """Check Ollama installation and optionally model availability.

    Returns 1-2 CheckResults: one for Ollama itself, one for the model
    (if *model_name* is provided and isn't a cloud model).
    """
    results: list[CheckResult] = []

    path = shutil.which("ollama")
    if path is None:
        results.append(
            CheckResult(
                name="Ollama",
                passed=False,
                message=(
                    "Not found on PATH. "
                    "Install: https://ollama.com/download"
                ),
            )
        )
        if model_name and not _is_cloud_model(model_name):
            results.append(
                CheckResult(
                    name=f"Model {model_name}",
                    passed=False,
                    message="Ollama not installed (required for local models)",
                )
            )
        return results

    # Ollama binary exists — check if it's running by listing models.
    try:
        list_result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if list_result.returncode != 0:
            results.append(
                CheckResult(
                    name="Ollama",
                    passed=False,
                    message=(
                        "Installed but not running. "
                        "Start it: ollama serve (or open /Applications/Ollama.app on macOS)"
                    ),
                )
            )
            if model_name and not _is_cloud_model(model_name):
                results.append(
                    CheckResult(
                        name=f"Model {model_name}",
                        passed=False,
                        message="Ollama not running",
                    )
                )
            return results

        results.append(CheckResult(name="Ollama", passed=True, message="running"))

        # Check model availability.
        if model_name and not _is_cloud_model(model_name):
            available = _parse_ollama_models(list_result.stdout)
            # Ollama model names: "gemma4:e4b" matches "gemma4:e4b" in list.
            # The list output uses NAME:TAG format.
            base_name = model_name.split(":")[0]
            if model_name in available or base_name in available:
                results.append(
                    CheckResult(
                        name=f"Model {model_name}",
                        passed=True,
                        message="available",
                    )
                )
            else:
                results.append(
                    CheckResult(
                        name=f"Model {model_name}",
                        passed=False,
                        message=f"Not found. Pull it: ollama pull {model_name}",
                    )
                )

    except (subprocess.TimeoutExpired, OSError):
        results.append(
            CheckResult(
                name="Ollama",
                passed=False,
                message="Installed but not responding",
            )
        )

    return results


def run_preflight(model_name: str | None = None) -> list[CheckResult]:
    """Run all prerequisite checks and return results."""
    results: list[CheckResult] = []
    results.append(check_python_version())
    results.append(check_claude_cli())
    results.extend(check_ollama(model_name))
    return results


def _is_cloud_model(model_name: str) -> bool:
    """Return True if the model name looks like a cloud provider model."""
    return any(model_name.startswith(prefix) for prefix in _CLOUD_PREFIXES)


def _parse_ollama_models(list_output: str) -> set[str]:
    """Parse ``ollama list`` output into a set of model name:tag strings."""
    models: set[str] = set()
    for line in list_output.strip().splitlines()[1:]:  # Skip header
        parts = line.split()
        if parts:
            models.add(parts[0])
    return models
