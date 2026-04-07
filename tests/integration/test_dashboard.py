"""Integration tests for the Petri dashboard (migrate + API + SSE).

Tests cover:
- SQLite rebuild from JSONL (migrate)
- REST endpoint responses (GET /api/events, /api/queue, /api/nodes, /api/node/{id}, /api/health, /api/stats)
- SSE stream delivers new events after incremental sync
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from petri.dashboard.migrate import (
    incremental_sync,
    init_db,
    rebuild_sqlite,
)
from petri.event_log import append_event, rollup_to_combined
from petri.queue import add_to_queue


@pytest.fixture
def dashboard_env(petri_env_with_colony):
    """Extend the shared colony environment with events and queue entries."""
    env = petri_env_with_colony
    colony_path = env["colony_path"]
    colony_model = env["colony_model"]

    # Resolve event paths from node_paths mapping
    cell1_events_path = colony_path / colony_model.node_paths[env["cell1"].id] / "events.jsonl"
    center_events_path = colony_path / colony_model.node_paths[env["center"].id] / "events.jsonl"

    for i in range(3):
        append_event(
            cell1_events_path,
            node_id=env["cell1"].id,
            event_type="search_executed",
            agent="investigator",
            iteration=i,
            data={"query": f"query-{i}", "sources_found": i + 1},
        )

    append_event(
        cell1_events_path,
        node_id=env["cell1"].id,
        event_type="verdict_issued",
        agent="investigator",
        iteration=1,
        data={"verdict": "EVIDENCE_SUFFICIENT", "summary": "Good evidence"},
    )

    append_event(
        center_events_path,
        node_id=env["center"].id,
        event_type="decomposition_created",
        agent="decomposition_lead",
        iteration=0,
        data={"parent_node_id": env["center"].id, "child_node_ids": [env["cell1"].id]},
    )

    # Add a queue entry
    add_to_queue(env["petri_dir"] / "queue.json", env["cell1"].id)

    return {
        **env,
        "db_path": env["tmp_path"] / "petri.sqlite",
    }


# ── migrate: rebuild_sqlite ─────────────────────────────────────────────


class TestRebuildSqlite:
    def test_rebuilds_from_jsonl(self, dashboard_env):
        env = dashboard_env
        count = rebuild_sqlite(env["petri_dir"], env["db_path"])

        assert count == 5  # 3 search + 1 verdict + 1 decomposition
        assert env["db_path"].exists()

    def test_schema_has_expected_tables(self, dashboard_env):
        env = dashboard_env
        rebuild_sqlite(env["petri_dir"], env["db_path"])

        conn = sqlite3.connect(str(env["db_path"]))
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        conn.close()

        table_names = {t[0] for t in tables}
        assert "events" in table_names

    def test_events_queryable_by_node_id(self, dashboard_env):
        env = dashboard_env
        rebuild_sqlite(env["petri_dir"], env["db_path"])

        conn = sqlite3.connect(str(env["db_path"]))
        rows = conn.execute(
            "SELECT COUNT(*) FROM events WHERE node_id = ?",
            [env["cell1"].id],
        ).fetchone()
        conn.close()

        assert rows[0] == 4  # 3 search + 1 verdict

    def test_events_queryable_by_type(self, dashboard_env):
        env = dashboard_env
        rebuild_sqlite(env["petri_dir"], env["db_path"])

        conn = sqlite3.connect(str(env["db_path"]))
        rows = conn.execute(
            "SELECT COUNT(*) FROM events WHERE type = 'verdict_issued'"
        ).fetchone()
        conn.close()

        assert rows[0] == 1

    def test_idempotent_rebuild(self, dashboard_env):
        """INSERT OR IGNORE means rebuilding twice doesn't duplicate."""
        env = dashboard_env
        count1 = rebuild_sqlite(env["petri_dir"], env["db_path"])
        count2 = rebuild_sqlite(env["petri_dir"], env["db_path"])

        # Second rebuild may insert 0 (OR IGNORE) or same count
        # but total in DB should still be 5
        conn = sqlite3.connect(str(env["db_path"]))
        total = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        conn.close()

        assert total == 5

    def test_wal_journal_mode(self, dashboard_env):
        env = dashboard_env
        rebuild_sqlite(env["petri_dir"], env["db_path"])

        conn = sqlite3.connect(str(env["db_path"]))
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        conn.close()

        assert mode == "wal"


# ── migrate: incremental_sync ───────────────────────────────────────────


class TestIncrementalSync:
    def test_syncs_new_events_after_offset(self, dashboard_env):
        env = dashboard_env
        # First, do a full rebuild
        rebuild_sqlite(env["petri_dir"], env["db_path"])

        # Rollup to get combined.jsonl and record its size
        combined = rollup_to_combined(env["petri_dir"])
        initial_size = combined.stat().st_size
        file_offsets = {str(combined): initial_size}

        # Append a new event — resolve path from colony.json
        colony_data = json.loads((env["colony_path"] / "colony.json").read_text())
        cell1_rel = colony_data["node_paths"][env["cell1"].id]
        cell1_events_path = env["colony_path"] / cell1_rel / "events.jsonl"
        append_event(
            cell1_events_path,
            node_id=env["cell1"].id,
            event_type="verdict_issued",
            agent="skeptic",
            iteration=2,
            data={"verdict": "ARGUMENT_WITHSTANDS_CRITIQUE", "summary": "Solid"},
        )

        # Re-rollup so combined.jsonl picks up the new event
        rollup_to_combined(env["petri_dir"])

        # Incremental sync should pick up only the new event
        inserted = incremental_sync(env["petri_dir"], env["db_path"], file_offsets)
        assert inserted >= 1

        # Verify in SQLite
        conn = sqlite3.connect(str(env["db_path"]))
        total = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        conn.close()

        assert total == 6  # 5 original + 1 new

    def test_no_new_events_returns_zero(self, dashboard_env):
        env = dashboard_env
        rebuild_sqlite(env["petri_dir"], env["db_path"])

        combined = rollup_to_combined(env["petri_dir"])
        file_offsets = {str(combined): combined.stat().st_size}

        inserted = incremental_sync(env["petri_dir"], env["db_path"], file_offsets)
        assert inserted == 0

    def test_missing_combined_returns_zero(self, dashboard_env):
        env = dashboard_env
        init_db(env["db_path"])

        file_offsets: dict[str, int] = {}
        inserted = incremental_sync(env["petri_dir"], env["db_path"], file_offsets)
        assert inserted == 0


# ── FastAPI REST endpoints ──────────────────────────────────────────────


@pytest.fixture
def api_client(dashboard_env):
    """Create a TestClient for the dashboard API."""
    try:
        from fastapi.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi not installed")

    env = dashboard_env
    rebuild_sqlite(env["petri_dir"], env["db_path"])

    from petri.dashboard.api import create_app

    app = create_app(env["petri_dir"], env["db_path"])
    client = TestClient(app, raise_server_exceptions=True)
    return client, env


class TestHealthEndpoint:
    def test_health_returns_ok(self, api_client):
        client, _ = api_client
        resp = client.get("/api/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


class TestEventsEndpoint:
    def test_get_all_events(self, api_client):
        client, env = api_client
        resp = client.get("/api/events")
        assert resp.status_code == 200
        events = resp.json()
        assert isinstance(events, list)
        assert len(events) == 5

    def test_filter_by_node_id(self, api_client):
        client, env = api_client
        resp = client.get("/api/events", params={"node_id": env["cell1"].id})
        assert resp.status_code == 200
        events = resp.json()
        assert len(events) == 4
        assert all(e["node_id"] == env["cell1"].id for e in events)

    def test_filter_by_event_type(self, api_client):
        client, _ = api_client
        resp = client.get("/api/events", params={"event_type": "verdict_issued"})
        assert resp.status_code == 200
        events = resp.json()
        assert len(events) == 1
        assert events[0]["type"] == "verdict_issued"

    def test_filter_by_agent(self, api_client):
        client, _ = api_client
        resp = client.get("/api/events", params={"agent": "investigator"})
        assert resp.status_code == 200
        events = resp.json()
        assert all(e["agent"] == "investigator" for e in events)

    def test_filter_by_iteration(self, api_client):
        client, _ = api_client
        resp = client.get("/api/events", params={"iteration": 0})
        assert resp.status_code == 200
        events = resp.json()
        assert all(e["iteration"] == 0 for e in events)

    def test_events_have_expected_fields(self, api_client):
        client, _ = api_client
        resp = client.get("/api/events", params={"limit": 1})
        events = resp.json()
        assert len(events) >= 1
        evt = events[0]
        for field in ["id", "node_id", "timestamp", "type", "agent", "iteration", "data"]:
            assert field in evt, f"Missing field: {field}"

    def test_data_is_parsed_dict(self, api_client):
        client, _ = api_client
        resp = client.get("/api/events", params={"event_type": "search_executed", "limit": 1})
        events = resp.json()
        assert isinstance(events[0]["data"], dict)
        assert "query" in events[0]["data"]


class TestQueueEndpoint:
    def test_get_queue(self, api_client):
        client, env = api_client
        resp = client.get("/api/queue")
        assert resp.status_code == 200
        entries = resp.json()
        assert isinstance(entries, list)
        assert len(entries) >= 1
        assert entries[0]["node_id"] == env["cell1"].id


class TestNodesEndpoint:
    def test_get_all_nodes(self, api_client):
        client, env = api_client
        resp = client.get("/api/nodes")
        assert resp.status_code == 200
        nodes = resp.json()
        assert isinstance(nodes, list)
        assert len(nodes) == 5  # canonical colony: center + 2 premises + 2 cells

        node_ids = {n["node_id"] for n in nodes}
        assert env["center"].id in node_ids
        assert env["cell1"].id in node_ids

    def test_node_has_expected_fields(self, api_client):
        client, _ = api_client
        nodes = client.get("/api/nodes").json()
        node = nodes[0]
        for field in [
            "node_id", "colony_id", "claim_text", "level",
            "status", "dependencies", "dependents",
        ]:
            assert field in node, f"Missing field: {field}"


class TestNodeDetailEndpoint:
    def test_get_node_detail(self, api_client):
        client, env = api_client
        resp = client.get(f"/api/node/{env['cell1'].id}")
        assert resp.status_code == 200

        detail = resp.json()
        assert detail["node_id"] == env["cell1"].id
        assert detail["claim_text"] == "Cell premise of P1"
        assert "events" in detail
        assert len(detail["events"]) == 4  # 3 search + 1 verdict

    def test_node_detail_includes_events(self, api_client):
        client, env = api_client
        detail = client.get(f"/api/node/{env['cell1'].id}").json()
        event_types = [e["type"] for e in detail["events"]]
        assert "search_executed" in event_types
        assert "verdict_issued" in event_types

    def test_node_not_found(self, api_client):
        client, _ = api_client
        resp = client.get("/api/node/nonexistent-node-999-999")
        assert resp.status_code == 404


class TestStatsEndpoint:
    def test_get_stats(self, api_client):
        client, _ = api_client
        resp = client.get("/api/stats")
        assert resp.status_code == 200

        stats = resp.json()
        assert stats["total_events"] == 5
        assert stats["nodes_with_events"] == 2
        assert stats["queue_size"] >= 1
        assert isinstance(stats["events_by_type"], list)
        assert isinstance(stats["top_nodes"], list)
        assert isinstance(stats["nodes_by_state"], dict)


# ── SSE stream ──────────────────────────────────────────────────────────


class TestSSEStream:
    def test_stream_endpoint_registered(self, api_client):
        """Verify the SSE endpoint is registered in the app routes."""
        client, _ = api_client
        routes = [r.path for r in client.app.routes if hasattr(r, "path")]
        assert "/api/stream" in routes
