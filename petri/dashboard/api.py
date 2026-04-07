"""FastAPI REST + SSE application for the Petri dashboard.

SQLite is a disposable read index -- agents never write to it.  The
background tail task watches ``combined.jsonl`` and syncs new events
into SQLite automatically via ``incremental_sync``.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from petri.dashboard.migrate import incremental_sync, rebuild_sqlite
from petri.storage.event_log import rollup_to_combined
from petri.storage.queue import list_queue, load_queue


# ── SQLite connection ─────────────────────────────────────────────────────


def get_db(db_path: Path) -> sqlite3.Connection:
    """Open a WAL-mode SQLite connection with row factory."""
    conn = sqlite3.connect(str(db_path), timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.row_factory = sqlite3.Row
    return conn


# ── Factory ───────────────────────────────────────────────────────────────


def create_app(petri_dir: Path, db_path: Path) -> FastAPI:
    """Build and return the FastAPI application.

    Parameters
    ----------
    petri_dir:
        Path to the ``.petri`` directory.
    db_path:
        Path to the SQLite read index file.
    """
    file_offsets: dict[str, int] = {}

    # Record the initial combined.jsonl size so incremental_sync starts
    # from the right place (rebuild_sqlite has already ingested everything).
    combined_path = petri_dir / "combined.jsonl"
    if combined_path.exists():
        try:
            file_offsets[str(combined_path)] = combined_path.stat().st_size
        except OSError:
            pass

    # ── Lifespan ──────────────────────────────────────────────────────

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Start the background tail loop; cancel on shutdown."""
        task = asyncio.create_task(_tail_loop(petri_dir, db_path, file_offsets))
        yield
        task.cancel()

    app = FastAPI(title="Petri Dashboard API", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Health ────────────────────────────────────────────────────────

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    # ── Events ────────────────────────────────────────────────────────

    @app.get("/api/events")
    def get_events(
        node_id: Optional[str] = None,
        iteration: Optional[int] = None,
        event_type: Optional[str] = None,
        agent: Optional[str] = None,
        limit: int = Query(default=500, le=5000),
    ):
        conditions: list[str] = []
        params: list[object] = []

        if node_id:
            conditions.append("node_id = ?")
            params.append(node_id)
        if iteration is not None:
            conditions.append("iteration = ?")
            params.append(iteration)
        if event_type:
            conditions.append("type = ?")
            params.append(event_type)
        if agent:
            conditions.append("agent = ?")
            params.append(agent)

        where = (" WHERE " + " AND ".join(conditions)) if conditions else ""
        sql = (
            "SELECT id, node_id, timestamp, type, agent, iteration, data "
            "FROM events" + where + " ORDER BY timestamp ASC LIMIT ?"
        )
        params.append(limit)

        conn = get_db(db_path)
        rows = conn.execute(sql, params).fetchall()
        conn.close()

        return [
            {
                "id": r["id"],
                "node_id": r["node_id"],
                "timestamp": r["timestamp"],
                "type": r["type"],
                "agent": r["agent"],
                "iteration": r["iteration"],
                "data": json.loads(r["data"]),
            }
            for r in rows
        ]

    # ── Queue ─────────────────────────────────────────────────────────

    @app.get("/api/queue")
    def get_queue():
        queue_path = petri_dir / "queue.json"
        return list_queue(queue_path)

    # ── Nodes ─────────────────────────────────────────────────────────

    @app.get("/api/nodes")
    def get_nodes():
        """Return all nodes with colony graph data."""
        from petri.graph.colony import deserialize_colony

        dishes_dir = petri_dir / "petri-dishes"
        if not dishes_dir.is_dir():
            return []

        # Derive dish_id from petri.yaml or parent directory name
        dish_id = _get_dish_id(petri_dir)

        all_nodes: list[dict] = []
        for colony_dir in sorted(dishes_dir.iterdir()):
            if not colony_dir.is_dir():
                continue
            try:
                graph, colony = deserialize_colony(colony_dir, dish_id)
            except Exception:
                continue

            for node in graph.get_nodes():
                all_nodes.append(
                    {
                        "node_id": node.id,
                        "colony_id": node.colony_id,
                        "claim_text": node.claim_text,
                        "level": node.level,
                        "status": node.status.value,
                        "dependencies": node.dependencies,
                        "dependents": node.dependents,
                    }
                )

        return all_nodes

    # ── Single node detail ────────────────────────────────────────────

    @app.get("/api/node/{node_id}")
    def get_node_detail(node_id: str):
        """Full detail for one node: metadata + events."""
        from petri.graph.colony import deserialize_colony

        dishes_dir = petri_dir / "petri-dishes"
        dish_id = _get_dish_id(petri_dir)

        # Find the node across all colonies
        if dishes_dir.is_dir():
            for colony_dir in sorted(dishes_dir.iterdir()):
                if not colony_dir.is_dir():
                    continue
                try:
                    graph, colony = deserialize_colony(colony_dir, dish_id)
                    node = graph.get_node(node_id)
                except (Exception, KeyError):
                    continue

                # Fetch events from SQLite
                conn = get_db(db_path)
                rows = conn.execute(
                    "SELECT id, node_id, timestamp, type, agent, iteration, data "
                    "FROM events WHERE node_id = ? ORDER BY timestamp ASC",
                    [node_id],
                ).fetchall()
                conn.close()

                events = [
                    {
                        "id": r["id"],
                        "node_id": r["node_id"],
                        "timestamp": r["timestamp"],
                        "type": r["type"],
                        "agent": r["agent"],
                        "iteration": r["iteration"],
                        "data": json.loads(r["data"]),
                    }
                    for r in rows
                ]

                return {
                    "node_id": node.id,
                    "colony_id": node.colony_id,
                    "claim_text": node.claim_text,
                    "level": node.level,
                    "status": node.status.value,
                    "dependencies": node.dependencies,
                    "dependents": node.dependents,
                    "events": events,
                }

        raise HTTPException(404, "Node %s not found" % node_id)

    # ── Stats ─────────────────────────────────────────────────────────

    @app.get("/api/stats")
    def get_stats():
        conn = get_db(db_path)
        total_events = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        nodes_with_events = conn.execute(
            "SELECT COUNT(DISTINCT node_id) FROM events"
        ).fetchone()[0]

        # Per-type counts
        type_rows = conn.execute(
            "SELECT type, COUNT(*) as cnt FROM events GROUP BY type ORDER BY cnt DESC"
        ).fetchall()

        # Top nodes by event count
        top_nodes = conn.execute(
            "SELECT node_id, COUNT(*) as cnt FROM events "
            "GROUP BY node_id ORDER BY cnt DESC LIMIT 10"
        ).fetchall()

        conn.close()

        # Queue stats
        queue_path = petri_dir / "queue.json"
        queue = load_queue(queue_path)
        entries = queue.get("entries", {})

        nodes_by_state: dict[str, int] = {}
        for entry in entries.values():
            state = entry.get("queue_state", "unknown")
            nodes_by_state[state] = nodes_by_state.get(state, 0) + 1

        return {
            "total_events": total_events,
            "nodes_with_events": nodes_with_events,
            "queue_size": len(entries),
            "nodes_by_state": nodes_by_state,
            "events_by_type": [
                {"type": r["type"], "count": r["cnt"]} for r in type_rows
            ],
            "top_nodes": [
                {"node_id": r["node_id"], "event_count": r["cnt"]}
                for r in top_nodes
            ],
        }

    # ── SSE stream ────────────────────────────────────────────────────

    @app.get("/api/stream")
    async def event_stream():
        async def generate():
            conn = get_db(db_path)
            row = conn.execute("SELECT MAX(rowid) FROM events").fetchone()
            last_rowid = row[0] or 0
            conn.close()

            yield {
                "event": "connected",
                "data": json.dumps(
                    {"status": "ok", "last_rowid": last_rowid}
                ),
            }

            while True:
                await asyncio.sleep(2)
                try:
                    conn = get_db(db_path)
                    new_rows = conn.execute(
                        "SELECT rowid, id, node_id, type, agent, iteration "
                        "FROM events WHERE rowid > ? "
                        "ORDER BY rowid ASC LIMIT 50",
                        [last_rowid],
                    ).fetchall()
                    conn.close()

                    for evt in new_rows:
                        last_rowid = evt["rowid"]
                        yield {
                            "event": "event_inserted",
                            "data": json.dumps(
                                {
                                    "id": evt["id"],
                                    "node_id": evt["node_id"],
                                    "type": evt["type"],
                                    "agent": evt["agent"],
                                    "iteration": evt["iteration"],
                                }
                            ),
                        }
                except Exception:
                    pass  # non-fatal, retries next cycle

        return EventSourceResponse(generate())

    return app


# ── Internal helpers ──────────────────────────────────────────────────────


def _get_dish_id(petri_dir: Path) -> str:
    """Get the dish ID from petri.yaml or derive from parent directory."""
    config_path = petri_dir / "petri.yaml"
    if config_path.exists():
        # Simple line parser (no yaml dependency required)
        for line in config_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("name:") and not line.startswith(" "):
                return line.split(":", 1)[1].strip()
    return petri_dir.parent.name


async def _tail_loop(
    petri_dir: Path,
    db_path: Path,
    file_offsets: dict[str, int],
) -> None:
    """Background task: re-roll-up and sync new events every 5 seconds."""
    while True:
        await asyncio.sleep(5)
        try:
            # Re-rollup so combined.jsonl picks up any new per-node events
            rollup_to_combined(petri_dir)
            incremental_sync(petri_dir, db_path, file_offsets)
        except Exception:
            pass  # non-fatal, retries next cycle
