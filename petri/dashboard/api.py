"""Dashboard REST and SSE reads over per-dish SQLite query functions."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from petri.storage.dish import resolve_dish_id as _get_dish_id
from petri.dashboard.frontend import build_frontend_html
from petri.execution import resolve_backend
from petri.execution.backend import ExecutionBackend
from petri import query

ASSETS_DIR = Path(__file__).parent.parent / "assets"


# ── Factory ───────────────────────────────────────────────────────────────


def create_app(
    petri_dir: Path, *, execution_backend: ExecutionBackend | None = None,
    allowed_origins: list[str] | None = None
) -> FastAPI:
    """Build and return the FastAPI application.

    Parameters
    ----------
    petri_dir:
        Path to the ``.petri`` directory.
    """
    backend = execution_backend or resolve_backend(petri_dir)
    from petri.config import load_dish_config
    from petri.dashboard.config import DashboardConfig
    from petri.dashboard.runs_api import DashboardOwner, create_runs_router

    owner = DashboardOwner(petri_dir,backend)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if (petri_dir/'defaults/petri.yaml').is_file() or (petri_dir/'petri.yaml').is_file():
            await owner.start()
        try:
            yield
        finally:
            await owner.close()

    app = FastAPI(title="Petri Dashboard API",lifespan=lifespan)
    app.state.execution_owner = owner
    app.include_router(create_runs_router(petri_dir,backend,owner))
    from petri.dashboard.traces_api import create_traces_router

    app.include_router(create_traces_router(petri_dir))
    app.add_middleware(
        CORSMiddleware,
        allow_origins=(DashboardConfig(allowed_origins=allowed_origins).allowed_origins
                       if allowed_origins is not None else load_dish_config(petri_dir).dashboard.allowed_origins),
        allow_methods=["GET","POST"],
        allow_headers=["Content-Type","Last-Event-ID"],
    )

    # ── Root + Static ────────────────────────────────────────────────

    @app.get("/")
    def root():
        return HTMLResponse(build_frontend_html())

    app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")

    # ── Health ────────────────────────────────────────────────────────

    @app.get("/api/ping")
    def ping():
        return {"status": "ok"}

    # ── Dishes ────────────────────────────────────────────────────────

    @app.get("/api/dishes")
    def get_dishes():
        """List available dishes (empty list triggers onboarding)."""
        return [{'id': dish.dish_id, 'path': str(dish.db_path.parent)} for dish in query.enumerate_dishes(petri_dir)]

    # ── Init (web-based onboarding) ──────────────────────────────────

    @app.post("/api/init")
    async def post_init(body: dict):
        """Run petri init logic from the web UI.

        Delegates to the shared ``create_petri_dish`` helper so the
        skeleton-creation logic lives in exactly one place (used by
        ``petri init``, ``petri launch`` auto-heal, and this endpoint).
        Preserves the ``already_exists`` short-circuit so the web
        wizard running twice in one session doesn't silently wipe
        user customizations.
        """
        from petri.cli.init import create_petri_dish
        from petri.config import load_config

        defaults = load_config()

        dish_name = body.get("name", petri_dir.parent.name)
        model_name = body.get("model", defaults.model)
        max_concurrent = int(body.get("max_concurrent", defaults.execution.worker_concurrency))
        max_iterations = int(body.get("max_iterations", defaults.max_iterations))

        config_exists = (petri_dir / "defaults" / "petri.yaml").exists()
        if config_exists:
            await owner.start()
            return {"status": "already_exists", "dish_id": dish_name}

        try:
            create_petri_dish(
                petri_dir,
                dish_name=dish_name,
                model_name=model_name,
                max_concurrent=max_concurrent,
                max_iterations=max_iterations,
            )
            await owner.start()
            return {"status": "ok", "dish_id": dish_name}
        except OSError as exc:
            raise HTTPException(500, f"Init failed: {exc}")

    # ── Events ────────────────────────────────────────────────────────

    @app.get("/api/events")
    def get_events(
        cell_id: Optional[str] = None,
        iteration: Optional[int] = None,
        event_type: Optional[str] = None,
        agent: Optional[str] = None,
        limit: int = Query(default=500, ge=1, le=5000),
    ):
        events = [
            event for dish in query.enumerate_dishes(petri_dir)
            for event in query.query_events(dish.db_path, cell_id=cell_id or None,
                iteration=iteration, event_type=event_type or None, agent=agent or None, limit=limit)
        ]
        return sorted(events, key=lambda event: event['timestamp'])[:limit]

    # ── Queue ─────────────────────────────────────────────────────────

    @app.get("/api/queue")
    def get_queue():
        return backend.list_entries(petri_dir)

    # ── Cells ─────────────────────────────────────────────────────────

    @app.get("/api/cells")
    def get_cells():
        """Read cell metadata and relationships from each v2 dish database."""
        return [
            {'dish_id': dish.dish_id, **{key: cell[key] for key in ('cell_id', 'colony_id', 'claim_text', 'level', 'status', 'dependencies', 'dependents')}}
            for dish in query.enumerate_dishes(petri_dir) for cell in query.list_cells(dish.db_path)
        ]

    from petri.storage.edge_registry import EdgeRecord

    @app.get('/api/edges', response_model=list[EdgeRecord])
    def get_edges(cell_id: str | None = None, colony: str | None = None):
        from petri.telemetry import get_tracer
        attributes = {'petri.dish_id':_get_dish_id(petri_dir),'http.route':'/api/edges'}
        if cell_id is not None:
            attributes['petri.cell_id'] = cell_id
        if colony is not None:
            attributes['petri.colony_id'] = colony
        with get_tracer().start_as_current_span('petri.dashboard.edges',attributes=attributes) as span:
            result = [edge for dish in query.enumerate_dishes(petri_dir)
                      for edge in query.get_edges(dish.db_path,cell_id=cell_id,colony=colony)]
            span.set_attribute('petri.outcome','ok')
            return result

    from petri.analysis.analyst import ResearchHealthReport

    @app.get('/api/health', response_model=ResearchHealthReport | None)
    def get_health():
        from petri.query.health import latest_health
        reports = [report for dish in query.enumerate_dishes(petri_dir)
                   if (report := latest_health(dish.db_path)) is not None]
        latest = max(reports,key=lambda report:report.created_at,default=None)
        return latest.model_dump(mode='json') if latest is not None else None

    @app.get('/api/intelligence')
    def get_intelligence():
        from petri.query.intelligence import get_intelligence as read_intelligence
        from petri.telemetry import get_tracer
        with get_tracer().start_as_current_span('petri.dashboard.intelligence',attributes={
            'petri.dish_id':_get_dish_id(petri_dir),'http.route':'/api/intelligence',
        }) as span:
            report = read_intelligence(petri_dir)
            span.set_attribute('petri.outcome','ok')
            return report.model_dump(mode='json')

    @app.get('/api/cell/{cell_id}')
    def get_cell_detail(cell_id: str):
        """Read one cell's content and events without assembling a colony."""
        for dish in query.enumerate_dishes(petri_dir):
            cell = query.get_cell(dish.db_path, cell_id)
            if cell is not None:
                result: dict[str, Any] = {key: cell[key] for key in ('cell_id', 'colony_id', 'claim_text', 'level', 'status', 'dependencies', 'dependents', 'evidence_md', 'summary_md')}
                result['dish_id'] = dish.dish_id
                result['events'] = query.query_events(dish.db_path, cell_id=cell_id, limit=None)
                return result
        raise HTTPException(404, 'Cell %s not found' % cell_id)

    # ── Stats ─────────────────────────────────────────────────────────

    @app.get("/api/stats")
    def get_stats():
        summaries = [query.event_stats(dish.db_path) for dish in query.enumerate_dishes(petri_dir)]
        types: dict[str, int] = {}
        cells: dict[str, int] = {}
        for summary in summaries:
            for row in summary['events_by_type']:
                types[row['type']] = types.get(row['type'], 0) + row['count']
            for row in summary['top_cells']:
                cells[row['cell_id']] = cells.get(row['cell_id'], 0) + row['event_count']

        entries = backend.list_entries(petri_dir)
        cells_by_state = backend.state_summary(petri_dir)

        return {
            "total_events": sum(summary["total_events"] for summary in summaries),
            "cells_with_events": sum(summary["cells_with_events"] for summary in summaries),
            "queue_size": len(entries),
            "cells_by_state": cells_by_state,
            'events_by_type': [{'type': key, 'count': value} for key, value in sorted(types.items(), key=lambda item: (-item[1], item[0]))],
            'top_cells': [{'cell_id': key, 'event_count': value} for key, value in sorted(cells.items(), key=lambda item: (-item[1], item[0]))[:10]],
        }

    from petri.dashboard.stream_api import create_stream_router

    app.include_router(create_stream_router(petri_dir))

    return app
