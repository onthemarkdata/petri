"""Comprehensive unit tests for petri/event_log.py."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

from petri.event_log import (
    append_event,
    get_searches,
    get_sources,
    get_verdicts,
    load_events,
    query_events,
    rollup_to_combined,
)


# ── Helpers ──────────────────────────────────────────────────────────────


def _write_event_line(path: Path, event: dict) -> None:
    """Write a single compact JSON event line to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(event, separators=(",", ":")) + "\n")


def _make_event(
    *,
    node_id: str = "dish-col-001-001",
    event_id: str = "dish-col-001-001-aabbccdd",
    event_type: str = "verdict_issued",
    agent: str = "analyst",
    iteration: int = 1,
    timestamp: str = "2026-01-15T10:00:00+00:00",
    data: dict | None = None,
) -> dict:
    """Build a raw event dict (as it would appear serialised in JSONL)."""
    if data is None:
        data = {"verdict": "VALIDATED", "summary": "Looks good"}
    return {
        "id": event_id,
        "node_id": node_id,
        "timestamp": timestamp,
        "type": event_type,
        "agent": agent,
        "iteration": iteration,
        "data": data,
    }


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def events_path(tmp_path: Path) -> Path:
    """Return a path to a fresh events.jsonl file (does not yet exist)."""
    return tmp_path / "events.jsonl"


@pytest.fixture
def populated_events_path(events_path: Path) -> Path:
    """Create an events.jsonl with several varied events already written."""
    events = [
        _make_event(
            event_id="dish-col-001-001-00000001",
            node_id="node-a",
            event_type="search_executed",
            agent="searcher",
            iteration=1,
            timestamp="2026-01-10T08:00:00+00:00",
            data={"query": "python best practices", "sources_found": 5},
        ),
        _make_event(
            event_id="dish-col-001-001-00000002",
            node_id="node-a",
            event_type="source_reviewed",
            agent="reviewer",
            iteration=1,
            timestamp="2026-01-10T09:00:00+00:00",
            data={"url": "https://example.com/a", "title": "Source A"},
        ),
        _make_event(
            event_id="dish-col-001-001-00000003",
            node_id="node-b",
            event_type="verdict_issued",
            agent="analyst",
            iteration=1,
            timestamp="2026-01-11T10:00:00+00:00",
            data={"verdict": "VALIDATED", "summary": "All clear"},
        ),
        _make_event(
            event_id="dish-col-001-001-00000004",
            node_id="node-a",
            event_type="verdict_issued",
            agent="analyst",
            iteration=2,
            timestamp="2026-01-12T11:00:00+00:00",
            data={"verdict": "NEEDS_EXPERIMENT", "summary": "Needs more data"},
        ),
        _make_event(
            event_id="dish-col-001-001-00000005",
            node_id="node-a",
            event_type="source_reviewed",
            agent="reviewer",
            iteration=2,
            timestamp="2026-01-12T12:00:00+00:00",
            data={"url": "https://example.com/a", "title": "Source A again"},
        ),
        _make_event(
            event_id="dish-col-001-001-00000006",
            node_id="node-a",
            event_type="source_reviewed",
            agent="reviewer",
            iteration=2,
            timestamp="2026-01-12T13:00:00+00:00",
            data={"url": "https://example.com/b", "title": "Source B"},
        ),
        _make_event(
            event_id="dish-col-001-001-00000007",
            node_id="node-b",
            event_type="search_executed",
            agent="searcher",
            iteration=2,
            timestamp="2026-01-13T14:00:00+00:00",
            data={"query": "advanced testing", "sources_found": 3},
        ),
    ]
    for evt in events:
        _write_event_line(events_path, evt)
    return events_path


# ── append_event Tests ───────────────────────────────────────────────────


class TestAppendEvent:
    """Tests for append_event."""

    def test_appends_valid_event_returns_event_model(self, events_path: Path):
        result = append_event(
            events_path,
            node_id="dish-col-001-001",
            event_type="verdict_issued",
            agent="analyst",
            iteration=1,
            data={"verdict": "VALIDATED", "summary": "Confirmed"},
        )
        assert result.node_id == "dish-col-001-001"
        assert result.type.value == "verdict_issued"
        assert result.agent == "analyst"
        assert result.iteration == 1
        assert result.data["verdict"] == "VALIDATED"
        assert result.data["summary"] == "Confirmed"

    def test_event_id_format(self, events_path: Path):
        result = append_event(
            events_path,
            node_id="dish-col-001-001",
            event_type="search_executed",
            agent="searcher",
            iteration=0,
            data={"query": "test", "sources_found": 1},
        )
        # ID should be {node_id}-{8hex}
        pattern = r"^dish-col-001-001-[0-9a-f]{8}$"
        assert re.match(pattern, result.id), f"ID {result.id!r} doesn't match expected format"

    def test_timestamp_is_utc_iso8601(self, events_path: Path):
        result = append_event(
            events_path,
            node_id="dish-col-001-001",
            event_type="search_executed",
            agent="searcher",
            iteration=0,
            data={"query": "test", "sources_found": 1},
        )
        ts = datetime.fromisoformat(result.timestamp)
        assert ts.tzinfo is not None
        assert ts.tzinfo == timezone.utc or ts.utcoffset().total_seconds() == 0

    def test_valid_event_data_accepted(self, events_path: Path):
        """Pydantic validates data -- valid payload is accepted without error."""
        result = append_event(
            events_path,
            node_id="n1",
            event_type="source_reviewed",
            agent="rev",
            iteration=1,
            data={
                "url": "https://example.com",
                "title": "Example",
                "finding": "Relevant info",
            },
        )
        assert result.data["url"] == "https://example.com"
        assert result.data["title"] == "Example"

    def test_invalid_event_data_raises_valueerror(self, events_path: Path):
        """Pydantic rejects invalid data -- sources_found must be >= 0."""
        with pytest.raises((ValueError, Exception)):
            append_event(
                events_path,
                node_id="n1",
                event_type="search_executed",
                agent="searcher",
                iteration=0,
                data={"query": "test", "sources_found": -1},
            )

    def test_invalid_event_type_raises_valueerror(self, events_path: Path):
        with pytest.raises(ValueError, match="Unknown event type"):
            append_event(
                events_path,
                node_id="n1",
                event_type="not_a_real_type",
                agent="agent",
                iteration=0,
                data={},
            )

    def test_file_created_if_not_exists(self, events_path: Path):
        assert not events_path.exists()
        append_event(
            events_path,
            node_id="n1",
            event_type="search_executed",
            agent="searcher",
            iteration=0,
            data={"query": "hello", "sources_found": 0},
        )
        assert events_path.exists()

    def test_multiple_appends_produce_multiple_lines(self, events_path: Path):
        for i in range(5):
            append_event(
                events_path,
                node_id="n1",
                event_type="search_executed",
                agent="searcher",
                iteration=i,
                data={"query": f"q{i}", "sources_found": i},
            )
        lines = [ln for ln in events_path.read_text().splitlines() if ln.strip()]
        assert len(lines) == 5

    def test_compact_json_format(self, events_path: Path):
        append_event(
            events_path,
            node_id="n1",
            event_type="search_executed",
            agent="searcher",
            iteration=0,
            data={"query": "test", "sources_found": 1},
        )
        raw_line = events_path.read_text().splitlines()[0]
        # Compact JSON: no spaces after colons or commas
        assert ": " not in raw_line
        assert ", " not in raw_line
        # Should still parse as valid JSON
        parsed = json.loads(raw_line)
        assert parsed["node_id"] == "n1"


# ── load_events Tests ────────────────────────────────────────────────────


class TestLoadEvents:
    """Tests for load_events."""

    def test_loads_valid_jsonl(self, populated_events_path: Path):
        events = load_events(populated_events_path)
        assert len(events) == 7
        assert events[0]["id"] == "dish-col-001-001-00000001"
        assert events[-1]["id"] == "dish-col-001-001-00000007"

    def test_empty_file_returns_empty_list(self, events_path: Path):
        events_path.parent.mkdir(parents=True, exist_ok=True)
        events_path.write_text("")
        result = load_events(events_path)
        assert result == []

    def test_nonexistent_file_returns_empty_list(self, tmp_path: Path):
        nonexistent = tmp_path / "does_not_exist.jsonl"
        result = load_events(nonexistent)
        assert result == []

    def test_malformed_lines_skipped_with_warning(self, events_path: Path):
        events_path.parent.mkdir(parents=True, exist_ok=True)
        events_path.write_text("this is not json\n")
        with pytest.warns(UserWarning, match="Skipping malformed line 1"):
            result = load_events(events_path)
        assert result == []

    def test_mix_of_valid_and_malformed(self, events_path: Path):
        valid_event = _make_event()
        events_path.parent.mkdir(parents=True, exist_ok=True)
        content = (
            json.dumps(valid_event, separators=(",", ":")) + "\n"
            + "INVALID LINE\n"
            + json.dumps(valid_event, separators=(",", ":")) + "\n"
        )
        events_path.write_text(content)
        with pytest.warns(UserWarning, match="Skipping malformed line 2"):
            result = load_events(events_path)
        assert len(result) == 2


# ── query_events Tests ───────────────────────────────────────────────────


class TestQueryEvents:
    """Tests for query_events."""

    def test_filter_by_node_id(self, populated_events_path: Path):
        result = query_events(populated_events_path, node_id="node-a")
        assert len(result) == 5
        assert all(e["node_id"] == "node-a" for e in result)

    def test_filter_by_iteration(self, populated_events_path: Path):
        result = query_events(populated_events_path, iteration=2)
        assert len(result) == 4
        assert all(e["iteration"] == 2 for e in result)

    def test_filter_by_event_type(self, populated_events_path: Path):
        result = query_events(populated_events_path, event_type="verdict_issued")
        assert len(result) == 2
        assert all(e["type"] == "verdict_issued" for e in result)

    def test_filter_by_agent(self, populated_events_path: Path):
        result = query_events(populated_events_path, agent="searcher")
        assert len(result) == 2
        assert all(e["agent"] == "searcher" for e in result)

    def test_filter_by_since(self, populated_events_path: Path):
        result = query_events(
            populated_events_path, since="2026-01-12T00:00:00+00:00"
        )
        assert len(result) == 4
        for e in result:
            assert e["timestamp"] >= "2026-01-12T00:00:00+00:00"

    def test_multiple_filters_combined_and_logic(self, populated_events_path: Path):
        result = query_events(
            populated_events_path,
            node_id="node-a",
            iteration=2,
            event_type="source_reviewed",
        )
        assert len(result) == 2
        for e in result:
            assert e["node_id"] == "node-a"
            assert e["iteration"] == 2
            assert e["type"] == "source_reviewed"

    def test_no_filters_returns_all(self, populated_events_path: Path):
        result = query_events(populated_events_path)
        assert len(result) == 7

    def test_no_matches_returns_empty_list(self, populated_events_path: Path):
        result = query_events(populated_events_path, node_id="nonexistent-node")
        assert result == []


# ── get_verdicts Tests ───────────────────────────────────────────────────


class TestGetVerdicts:
    """Tests for get_verdicts."""

    def test_returns_verdict_events_with_extracted_fields(
        self, populated_events_path: Path
    ):
        verdicts = get_verdicts(populated_events_path)
        assert len(verdicts) == 2
        v1 = verdicts[0]
        assert v1["node_id"] == "node-b"
        assert v1["verdict"] == "VALIDATED"
        assert v1["summary"] == "All clear"
        assert v1["agent"] == "analyst"
        assert v1["iteration"] == 1

    def test_filter_by_iteration(self, populated_events_path: Path):
        verdicts = get_verdicts(populated_events_path, iteration=2)
        assert len(verdicts) == 1
        assert verdicts[0]["verdict"] == "NEEDS_EXPERIMENT"

    def test_filter_by_agent(self, populated_events_path: Path):
        verdicts = get_verdicts(populated_events_path, agent="analyst")
        assert len(verdicts) == 2

    def test_empty_when_no_verdicts(self, events_path: Path):
        events_path.parent.mkdir(parents=True, exist_ok=True)
        _write_event_line(
            events_path,
            _make_event(
                event_type="search_executed",
                data={"query": "test", "sources_found": 0},
            ),
        )
        verdicts = get_verdicts(events_path)
        assert verdicts == []


# ── get_sources Tests ────────────────────────────────────────────────────


class TestGetSources:
    """Tests for get_sources."""

    def test_returns_source_reviewed_events(self, populated_events_path: Path):
        sources = get_sources(populated_events_path)
        assert len(sources) >= 1
        for s in sources:
            assert s["type"] == "source_reviewed"

    def test_deduplication_by_url(self, populated_events_path: Path):
        """There are two source_reviewed events with URL 'https://example.com/a'.
        Only the first should appear in the deduplicated result."""
        sources = get_sources(populated_events_path)
        urls = [s["data"]["url"] for s in sources]
        assert urls.count("https://example.com/a") == 1
        assert urls.count("https://example.com/b") == 1
        assert len(sources) == 2


# ── get_searches Tests ───────────────────────────────────────────────────


class TestGetSearches:
    """Tests for get_searches."""

    def test_returns_search_executed_events(self, populated_events_path: Path):
        searches = get_searches(populated_events_path)
        assert len(searches) == 2
        for s in searches:
            assert s["type"] == "search_executed"
        queries = [s["data"]["query"] for s in searches]
        assert "python best practices" in queries
        assert "advanced testing" in queries


# ── rollup_to_combined Tests ─────────────────────────────────────────────


class TestRollupToCombined:
    """Tests for rollup_to_combined."""

    def test_combines_multiple_colonies(self, tmp_path: Path):
        dishes = tmp_path / "petri-dishes"

        # Colony alpha, node 1
        alpha_node1 = dishes / "alpha" / "level-001"
        alpha_node1.mkdir(parents=True)
        _write_event_line(
            alpha_node1 / "events.jsonl",
            _make_event(event_id="a1", node_id="alpha-001"),
        )
        _write_event_line(
            alpha_node1 / "events.jsonl",
            _make_event(event_id="a2", node_id="alpha-001"),
        )

        # Colony alpha, node 2
        alpha_node2 = dishes / "alpha" / "level-002"
        alpha_node2.mkdir(parents=True)
        _write_event_line(
            alpha_node2 / "events.jsonl",
            _make_event(event_id="a3", node_id="alpha-002"),
        )

        # Colony beta, node 1
        beta_node1 = dishes / "beta" / "level-001"
        beta_node1.mkdir(parents=True)
        _write_event_line(
            beta_node1 / "events.jsonl",
            _make_event(event_id="b1", node_id="beta-001"),
        )

        combined_path = rollup_to_combined(tmp_path)
        assert combined_path == tmp_path / "combined.jsonl"
        assert combined_path.exists()

        lines = [ln for ln in combined_path.read_text().splitlines() if ln.strip()]
        assert len(lines) == 4

        ids = [json.loads(ln)["id"] for ln in lines]
        assert ids == ["a1", "a2", "a3", "b1"]

    def test_combined_contains_all_events(self, tmp_path: Path):
        """Verify the combined file faithfully reproduces all source events."""
        dishes = tmp_path / "petri-dishes"
        colony_dir = dishes / "only" / "node-001"
        colony_dir.mkdir(parents=True)

        source_events = [
            _make_event(event_id=f"evt-{i}", node_id="only-001", iteration=i)
            for i in range(10)
        ]
        for evt in source_events:
            _write_event_line(colony_dir / "events.jsonl", evt)

        rollup_to_combined(tmp_path)
        combined = tmp_path / "combined.jsonl"
        lines = [ln for ln in combined.read_text().splitlines() if ln.strip()]
        assert len(lines) == 10

    def test_empty_nodes_skipped(self, tmp_path: Path):
        dishes = tmp_path / "petri-dishes"

        # Node with events
        node_with = dishes / "col" / "node-with"
        node_with.mkdir(parents=True)
        _write_event_line(
            node_with / "events.jsonl",
            _make_event(event_id="x1"),
        )

        # Node with empty JSONL
        node_empty = dishes / "col" / "node-empty"
        node_empty.mkdir(parents=True)
        (node_empty / "events.jsonl").write_text("")

        combined_path = rollup_to_combined(tmp_path)
        lines = [ln for ln in combined_path.read_text().splitlines() if ln.strip()]
        assert len(lines) == 1
        assert json.loads(lines[0])["id"] == "x1"

    def test_returns_combined_path(self, tmp_path: Path):
        # Even with no petri-dishes dir, should return the path
        combined_path = rollup_to_combined(tmp_path)
        assert combined_path == tmp_path / "combined.jsonl"
        assert combined_path.exists()
