"""Integration tests for the Petri CLI."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from petri.cli import app

runner = CliRunner()


@pytest.fixture
def clean_dir(tmp_path, monkeypatch):
    """Provide a clean temporary directory as cwd."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


# ── init ─────────────────────────────────────────────────────────────────


class TestInit:
    def test_init_creates_petri_dir(self, clean_dir):
        result = runner.invoke(app, ["init"])
        assert result.exit_code == 0, result.output

        petri_dir = clean_dir / ".petri"
        assert petri_dir.is_dir()

        # defaults/ directory with consolidated petri.yaml + constitution
        defaults = petri_dir / "defaults"
        assert defaults.is_dir()
        assert (defaults / "petri.yaml").is_file()
        assert (defaults / "constitution.md").is_file()

        # petri-dishes/ directory
        assert (petri_dir / "petri-dishes").is_dir()

        # queue.json with valid JSON
        queue_path = petri_dir / "queue.json"
        assert queue_path.is_file()
        queue = json.loads(queue_path.read_text())
        assert "version" in queue
        assert "entries" in queue

    def test_init_with_name(self, clean_dir):
        result = runner.invoke(app, ["init", "--name", "my-research"])
        assert result.exit_code == 0, result.output

        config_text = (clean_dir / ".petri" / "defaults" / "petri.yaml").read_text()
        assert "my-research" in config_text

    def test_init_already_exists(self, clean_dir):
        first = runner.invoke(app, ["init"])
        assert first.exit_code == 0, first.output

        second = runner.invoke(app, ["init"])
        assert second.exit_code == 1, second.output
        assert "already" in second.output.lower()


# ── seed coverage lives in tests/integration/test_seed_wizard.py ─────────


# ── check ────────────────────────────────────────────────────────────────
#
# These tests use the canonical ``seeded_petri_dir`` fixture from
# tests/conftest.py — a real .petri/ directory laid out on disk with the
# canonical 5-node diamond colony pre-serialized. They never call
# ``petri seed`` (which now requires the real Claude Code CLI) and never
# monkeypatch the inference provider.


class TestCheck:
    def test_check_shows_nodes(self, seeded_petri_dir):
        result = runner.invoke(app, ["check"])
        assert result.exit_code == 0, result.output

        out = result.output
        assert "Level 0" in out
        assert "Level 1" in out
        assert "Level 2" in out
        assert "NEW" in out

    def test_check_json_output(self, seeded_petri_dir):
        result = runner.invoke(app, ["check", "--json"])
        assert result.exit_code == 0, result.output

        data = json.loads(result.output)
        assert isinstance(data, list)
        # Canonical diamond has 5 nodes
        assert len(data) == 5


# ── graph ────────────────────────────────────────────────────────────────


class TestGraph:
    def test_graph(self, seeded_petri_dir):
        result = runner.invoke(app, ["graph"])
        assert result.exit_code == 0, result.output

        out = result.output
        # Composite IDs contain hyphens
        assert "-" in out
        # The canonical colony's center claim
        assert "Central thesis" in out
        # Level markers
        assert "L0" in out or "L1" in out or "L2" in out

    def test_graph_dot_format(self, seeded_petri_dir):
        result = runner.invoke(
            app, ["graph", "--format", "dot"]
        )
        assert result.exit_code == 0, result.output
        assert "digraph" in result.output


# ── full flow ────────────────────────────────────────────────────────────


class TestFullFlow:
    def test_full_flow(self, seeded_petri_dir):
        # check
        r = runner.invoke(app, ["check", "--json"])
        assert r.exit_code == 0, r.output
        check_data = json.loads(r.output)
        check_count = len(check_data)
        assert check_count == 5  # canonical diamond

        # graph
        r = runner.invoke(app, ["graph"])
        assert r.exit_code == 0, r.output

        # Node count consistency: check should report the same number
        # of nodes as live in the colony directory.
        dishes_dir = seeded_petri_dir["petri_dir"] / "petri-dishes"
        colony_dirs = [d for d in dishes_dir.iterdir() if d.is_dir()]
        assert len(colony_dirs) == 1

        metadata_files = list(colony_dirs[0].rglob("metadata.json"))
        assert len(metadata_files) == check_count


# ── on-disk integrity (SC-002) ──────────────────────────────────────────


class TestColonyOnDiskIntegrity:
    """Verify the canonical seeded layout is structurally sound.

    This is what survived from the old TestConcurrency: the parts that
    don't require running ``petri grow`` (which now needs the real Claude
    Code CLI in preflight). The state-machine + file-locking concurrency
    behaviour is exercised by tests under ``tests/unit/test_queue.py`` and
    ``tests/integration/test_processor*.py`` against the engine directly,
    not via the CLI.
    """

    def test_seeded_dir_structure_is_valid(self, seeded_petri_dir):
        petri_dir = seeded_petri_dir["petri_dir"]
        colony_dir = seeded_petri_dir["colony_path"]

        metadata_files = sorted(colony_dir.rglob("metadata.json"))
        node_dirs = [mf.parent for mf in metadata_files]
        # Canonical diamond has 5 nodes
        assert len(node_dirs) == 5

        for node_dir in node_dirs:
            meta_path = node_dir / "metadata.json"
            assert meta_path.is_file(), f"Missing metadata: {node_dir.name}"

            meta = json.loads(meta_path.read_text())
            assert "id" in meta, f"metadata missing 'id' in {node_dir.name}"

        # queue.json must be present, valid JSON, and have an entry per node
        queue_path = petri_dir / "queue.json"
        queue = json.loads(queue_path.read_text())
        assert "entries" in queue
        assert "version" in queue
        assert len(queue["entries"]) == 5
