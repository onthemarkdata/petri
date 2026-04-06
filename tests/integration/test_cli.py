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


# ── seed ─────────────────────────────────────────────────────────────────


class TestSeed:
    def test_seed_without_init(self, clean_dir):
        result = runner.invoke(app, ["seed", "thesis", "--no-questions"])
        assert result.exit_code == 3, result.output
        assert "no petri dish found" in result.output.lower()

    def test_seed_creates_colony(self, clean_dir):
        init_result = runner.invoke(app, ["init"])
        assert init_result.exit_code == 0, init_result.output

        result = runner.invoke(
            app, ["seed", "AI is viable", "--no-questions"]
        )
        assert result.exit_code == 0, result.output

        # Colony directory exists under petri-dishes/
        dishes_dir = clean_dir / ".petri" / "petri-dishes"
        colony_dirs = [d for d in dishes_dir.iterdir() if d.is_dir()]
        assert len(colony_dirs) >= 1, "No colony directory created"

        colony_dir = colony_dirs[0]

        # Node directories exist with required files (nested under level dirs)
        metadata_files = list(colony_dir.rglob("metadata.json"))
        assert len(metadata_files) > 0, "No node directories created"

        for mf in metadata_files:
            node_dir = mf.parent
            assert (
                node_dir / "events.jsonl"
            ).is_file(), f"Missing events.jsonl in {node_dir.name}"
            assert (
                node_dir / "evidence.md"
            ).is_file(), f"Missing evidence.md in {node_dir.name}"

        # Composite IDs visible in output
        # The default decomposition creates IDs like {dish}-{colony}-000-000
        assert "-" in result.output  # composite IDs contain hyphens


# ── check ────────────────────────────────────────────────────────────────


class TestCheck:
    def _init_and_seed(self, clean_dir):
        """Helper: run init + seed and return seed output."""
        r = runner.invoke(app, ["init"])
        assert r.exit_code == 0, r.output
        r = runner.invoke(app, ["seed", "AI is viable", "--no-questions"])
        assert r.exit_code == 0, r.output
        return r.output

    def test_check_shows_nodes(self, clean_dir):
        self._init_and_seed(clean_dir)

        result = runner.invoke(app, ["check"])
        assert result.exit_code == 0, result.output

        out = result.output
        assert "Level 0" in out
        assert "Level 1" in out
        assert "Level 2" in out
        assert "NEW" in out

    def test_check_json_output(self, clean_dir):
        self._init_and_seed(clean_dir)

        result = runner.invoke(app, ["check", "--json"])
        assert result.exit_code == 0, result.output

        data = json.loads(result.output)
        assert isinstance(data, list)
        assert len(data) > 0


# ── analyze ──────────────────────────────────────────────────────────────


class TestAnalyze:
    def _init_and_seed(self, clean_dir):
        """Helper: run init + seed and return seed output."""
        r = runner.invoke(app, ["init"])
        assert r.exit_code == 0, r.output
        r = runner.invoke(app, ["seed", "AI is viable", "--no-questions"])
        assert r.exit_code == 0, r.output
        return r.output

    def test_analyze_graph(self, clean_dir):
        self._init_and_seed(clean_dir)

        result = runner.invoke(app, ["analyze", "--graph"])
        assert result.exit_code == 0, result.output

        out = result.output
        # Should show node IDs (composite keys with dashes)
        assert "-" in out
        # Should show claim text
        assert "AI is viable" in out
        # Should show levels
        assert "L0" in out or "L1" in out or "L2" in out

    def test_analyze_graph_dot_format(self, clean_dir):
        self._init_and_seed(clean_dir)

        result = runner.invoke(
            app, ["analyze", "--graph", "--format", "dot"]
        )
        assert result.exit_code == 0, result.output

        out = result.output
        assert "digraph" in out


# ── full flow ────────────────────────────────────────────────────────────


class TestFullFlow:
    def test_full_flow(self, clean_dir):
        # 1. init
        r = runner.invoke(app, ["init"])
        assert r.exit_code == 0, r.output

        # 2. seed
        r = runner.invoke(app, ["seed", "AI is viable", "--no-questions"])
        assert r.exit_code == 0, r.output
        seed_output = r.output

        # 3. check
        r = runner.invoke(app, ["check", "--json"])
        assert r.exit_code == 0, r.output
        check_data = json.loads(r.output)
        check_count = len(check_data)
        assert check_count > 0

        # 4. analyze --graph
        r = runner.invoke(app, ["analyze", "--graph"])
        assert r.exit_code == 0, r.output

        # Node count consistency: the default decomposition creates 6 nodes
        # (1 center + 3 level-1 + 2 level-2). Verify check reported
        # the same number we see in the colony directory.
        dishes_dir = clean_dir / ".petri" / "petri-dishes"
        colony_dirs = [d for d in dishes_dir.iterdir() if d.is_dir()]
        assert len(colony_dirs) == 1

        metadata_files = list(colony_dirs[0].rglob("metadata.json"))
        assert len(metadata_files) == check_count


# ── concurrency (SC-002) ────────────────────────────────────────────────


class TestConcurrency:
    """Verify concurrent processing doesn't cause data loss or corruption."""

    def test_concurrent_grow_preserves_events(self, clean_dir):
        """Seed a colony, grow --all with concurrency, verify event integrity.

        Covers SC-002: concurrent processing must not lose or corrupt data.
        """
        # 1. init
        r = runner.invoke(app, ["init"])
        assert r.exit_code == 0, r.output

        # 2. seed — default decomposition creates ~6 nodes across 3 levels
        r = runner.invoke(
            app, ["seed", "Multi-node thesis for concurrency", "--no-questions"]
        )
        assert r.exit_code == 0, r.output

        # Count nodes created
        dishes_dir = clean_dir / ".petri" / "petri-dishes"
        colony_dirs = [d for d in dishes_dir.iterdir() if d.is_dir()]
        assert len(colony_dirs) == 1
        colony_dir = colony_dirs[0]

        metadata_files = sorted(colony_dir.rglob("metadata.json"))
        node_dirs = [mf.parent for mf in metadata_files]
        total_nodes = len(node_dirs)
        assert total_nodes >= 3, f"Expected >=3 nodes, got {total_nodes}"

        # 3. grow --all (no LLM provider → uses no-op fallback, exercises
        #    the state machine and file locking under concurrency)
        r = runner.invoke(app, ["grow", "--all"])
        # May exit 0 (success) or 1 (some stalled) — both are valid
        assert r.exit_code in (0, 1), f"Unexpected exit {r.exit_code}: {r.output}"

        # 4. Verify data integrity: every node directory still has
        #    valid metadata.json and events.jsonl
        for node_dir in node_dirs:
            meta_path = node_dir / "metadata.json"
            events_path = node_dir / "events.jsonl"

            assert meta_path.is_file(), f"Missing metadata: {node_dir.name}"
            assert events_path.is_file(), f"Missing events: {node_dir.name}"

            # metadata.json must be valid JSON
            meta = json.loads(meta_path.read_text())
            assert "id" in meta, f"metadata missing 'id' in {node_dir.name}"

            # events.jsonl: every non-empty line must be valid JSON
            events_text = events_path.read_text()
            event_count = 0
            for i, line in enumerate(events_text.splitlines()):
                line = line.strip()
                if not line:
                    continue
                try:
                    evt = json.loads(line)
                    assert "node_id" in evt, f"Event missing node_id: {node_dir.name} line {i}"
                    event_count += 1
                except json.JSONDecodeError:
                    pytest.fail(
                        f"Corrupt JSONL in {node_dir.name} line {i}: {line!r}"
                    )

            # Each node should have at least the initial decomposition event
            assert event_count >= 1, (
                f"Node {node_dir.name} has {event_count} events (expected >=1)"
            )

        # 5. Verify queue.json is valid
        queue_path = clean_dir / ".petri" / "queue.json"
        queue = json.loads(queue_path.read_text())
        assert "entries" in queue, "queue.json missing 'entries'"
        assert "version" in queue, "queue.json missing 'version'"
