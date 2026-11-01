"""Dashboard REST and SSE integration over per-dish SQLite."""

from __future__ import annotations


import pytest

from petri.storage.event_log import append_event


@pytest.fixture
def dashboard_env(petri_env_with_colony):
    """Extend the shared colony environment with events and queue entries."""
    env = petri_env_with_colony
    colony_path = env["colony_path"]
    colony_model = env["colony_model"]

    # Resolve event paths from cell_paths mapping
    cell1_events_path = colony_path / colony_model.cell_paths[env["cell1"].id] / "events.jsonl"
    center_events_path = colony_path / colony_model.cell_paths[env["center"].id] / "events.jsonl"

    for i in range(3):
        append_event(
            cell1_events_path,
            cell_id=env["cell1"].id,
            event_type="search_executed",
            agent="investigator",
            iteration=i,
            data={"query": f"query-{i}", "sources_found": i + 1},
        )

    append_event(
        cell1_events_path,
        cell_id=env["cell1"].id,
        event_type="verdict_issued",
        agent="investigator",
        iteration=1,
        data={"verdict": "EVIDENCE_SUFFICIENT", "summary": "Good evidence"},
    )

    append_event(
        center_events_path,
        cell_id=env["center"].id,
        event_type="decomposition_created",
        agent="decomposition_lead",
        iteration=0,
        data={"parent_cell_id": env["center"].id, "child_cell_ids": [env["cell1"].id]},
    )

    from petri.models import Event
    from petri.storage import database
    from petri.storage.event_log import load_events
    from petri.storage.paths import dish_db_path

    for path in (cell1_events_path, center_events_path):
        for event in load_events(path):
            database.append_event(dish_db_path(env["petri_dir"]), Event.model_validate(event))

    # Add a queue entry

    return {
        **env,
        "db_path": env["tmp_path"] / "petri.sqlite",
    }


@pytest.fixture
def api_client(dashboard_env):
    """Create a TestClient for the dashboard API."""
    try:
        from fastapi.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi not installed")

    env = dashboard_env

    from petri.dashboard.api import create_app

    app = create_app(env["petri_dir"])
    client = TestClient(app, raise_server_exceptions=True)
    return client, env


class TestHealthEndpoint:
    def test_health_returns_ok(self, api_client):
        client, _ = api_client
        resp = client.get("/api/ping")
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

    def test_filter_by_cell_id(self, api_client):
        client, env = api_client
        resp = client.get("/api/events", params={"cell_id": env["cell1"].id})
        assert resp.status_code == 200
        events = resp.json()
        assert len(events) == 4
        assert all(e["cell_id"] == env["cell1"].id for e in events)

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
        for field in ["id", "cell_id", "timestamp", "type", "agent", "iteration", "data"]:
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
        assert any(row["cell_id"] == env["cell1"].id for row in entries)


class TestCellsEndpoint:
    def test_get_all_cells(self, api_client):
        client, env = api_client
        resp = client.get("/api/cells")
        assert resp.status_code == 200
        cells = resp.json()
        assert isinstance(cells, list)
        assert len(cells) == 5  # canonical colony: center + 2 premises + 2 cells

        cell_ids = {entry["cell_id"] for entry in cells}
        assert env["center"].id in cell_ids
        assert env["cell1"].id in cell_ids

    def test_cell_has_expected_fields(self, api_client):
        client, _ = api_client
        cells = client.get("/api/cells").json()
        cell = cells[0]
        for field in [
            "cell_id", "colony_id", "claim_text", "level",
            "status", "dependencies", "dependents",
        ]:
            assert field in cell, f"Missing field: {field}"


class TestCellDetailEndpoint:
    def test_get_cell_detail(self, api_client):
        client, env = api_client
        resp = client.get(f"/api/cell/{env['cell1'].id}")
        assert resp.status_code == 200

        detail = resp.json()
        assert detail["cell_id"] == env["cell1"].id
        assert detail["claim_text"] == "Leaf premise of P1"
        assert "events" in detail
        assert len(detail["events"]) == 4  # 3 search + 1 verdict

    def test_cell_detail_includes_events(self, api_client):
        client, env = api_client
        detail = client.get(f"/api/cell/{env['cell1'].id}").json()
        event_types = [e["type"] for e in detail["events"]]
        assert "search_executed" in event_types
        assert "verdict_issued" in event_types

    def test_cell_not_found(self, api_client):
        client, _ = api_client
        resp = client.get("/api/cell/nonexistent-cell-999-999")
        assert resp.status_code == 404


class TestStatsEndpoint:
    def test_get_stats(self, api_client):
        client, _ = api_client
        resp = client.get("/api/stats")
        assert resp.status_code == 200

        stats = resp.json()
        assert stats["total_events"] == 5
        assert stats["cells_with_events"] == 2
        assert stats["queue_size"] >= 1
        assert isinstance(stats["events_by_type"], list)
        assert isinstance(stats["top_cells"], list)
        assert isinstance(stats["cells_by_state"], dict)


# ── SSE stream ──────────────────────────────────────────────────────────


class TestRootEndpoint:
    def test_root_returns_html(self, api_client):
        """GET / serves the frontend HTML generated from template."""
        client, _ = api_client
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "PETRI LAB" in resp.text

    def test_root_html_has_substituted_config(self, api_client):
        """Verify template variables were substituted."""
        client, _ = api_client
        resp = client.get("/")
        # Should contain JSON-serialized status colors, not raw template vars
        assert "STATUS_COLORS" in resp.text
        assert "$status_colors_json" not in resp.text
        assert "$event_colors_json" not in resp.text


class TestDishesEndpoint:
    def test_dishes_returns_list(self, api_client):
        """GET /api/dishes returns colony directories."""
        client, env = api_client
        resp = client.get("/api/dishes")
        assert resp.status_code == 200
        dishes = resp.json()
        assert isinstance(dishes, list)
        # The test fixture creates a colony, so there should be one dish
        assert len(dishes) >= 1

    def test_dishes_empty_when_no_colonies(self, dashboard_env):
        """GET /api/dishes returns [] when petri-dishes is empty."""
        import shutil

        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi not installed")

        env = dashboard_env

        # Remove all colony directories
        dishes_dir = env["petri_dir"] / "petri-dishes"
        if dishes_dir.exists():
            shutil.rmtree(dishes_dir)
            dishes_dir.mkdir()

        from petri.dashboard.api import create_app

        app = create_app(env["petri_dir"])
        client = TestClient(app, raise_server_exceptions=True)

        resp = client.get("/api/dishes")
        assert resp.status_code == 200
        assert resp.json() == []


class TestFrontendBuilder:
    def test_build_frontend_html_returns_string(self):
        """build_frontend_html returns a non-empty HTML string."""
        from petri.dashboard.frontend import build_frontend_html

        html = build_frontend_html()
        assert isinstance(html, str)
        assert len(html) > 1000
        assert html.startswith("<!DOCTYPE html>")

    def test_build_frontend_html_substitutes_all_variables(self):
        """No unresolved $variable placeholders remain."""
        from petri.dashboard.frontend import build_frontend_html

        html = build_frontend_html(version="1.2.3")
        assert "$status_colors_json" not in html
        assert "$event_colors_json" not in html
        assert "$queue_active_states_json" not in html
        assert "$pass_verdicts_json" not in html
        assert "$petri_version" not in html
        assert "$starter_claim" not in html
        assert "v1.2.3" in html

    def test_build_frontend_html_contains_config_data(self):
        """Verify config-derived data appears in the output."""
        from petri.dashboard.frontend import build_frontend_html

        html = build_frontend_html()
        # Status colors should be JSON-embedded
        assert '"NEW"' in html
        assert '"VALIDATED"' in html
        assert "#22c55e" in html
        # Event types
        assert "search_executed" in html
        assert "verdict_issued" in html


class TestSSEStream:
    def test_stream_endpoint_registered(self, api_client):
        """Verify the SSE endpoint is registered in the app routes."""
        client, _ = api_client
        routes = [r.path for r in client.app.routes if hasattr(r, "path")]
        assert "/api/stream" in routes
