"""Shared test fixtures for Petri test suite."""

import json
import shutil
from pathlib import Path

import pytest


@pytest.fixture
def tmp_petri_dir(tmp_path):
    """Create a temporary .petri/ directory with default structure."""
    petri_dir = tmp_path / ".petri"
    petri_dir.mkdir()
    (petri_dir / "petri-dishes").mkdir()
    (petri_dir / "defaults").mkdir()

    # Default config
    config = {
        "name": "test-dish",
        "model": {"name": "gemma-3-4b-it", "provider": "local"},
        "harness": "claude-code",
        "max_iterations": 3,
    }
    (petri_dir / "petri.yaml").write_text(
        "# Petri configuration\n"
        f"name: {config['name']}\n"
        f"model:\n  name: {config['model']['name']}\n  provider: {config['model']['provider']}\n"
        f"harness: {config['harness']}\n"
        f"max_iterations: {config['max_iterations']}\n"
    )

    # Empty queue
    queue = {"version": 1, "last_updated": None, "entries": {}}
    (petri_dir / "queue.json").write_text(json.dumps(queue, indent=2) + "\n")

    return tmp_path


@pytest.fixture
def petri_defaults_dir():
    """Return the path to the package defaults directory."""
    return Path(__file__).parent.parent / "petri" / "defaults"
