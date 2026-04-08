"""FastAPI REST + SSE application for the Petri dashboard.

SQLite is a disposable read index -- agents never write to it.  The
background tail task watches ``combined.jsonl`` and syncs new events
into SQLite automatically via ``incremental_sync``.
"""

from __future__ import annotations

import asyncio
import base64
import errno
import json
import os
import pty
import signal
import sqlite3
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

from petri.dashboard.frontend import build_frontend_html
from petri.dashboard.migrate import incremental_sync, rebuild_sqlite
from petri.storage.event_log import rollup_to_combined
from petri.storage.queue import list_queue, load_queue

ASSETS_DIR = Path(__file__).parent.parent / "assets"


# ── Petri subprocess execution (for the Computer tab terminal) ───────────

# The set of petri subcommands the Computer tab is allowed to spawn. Frozen
# at import time; any unknown command is rejected at the endpoint boundary.
_PETRI_SUBCOMMANDS = frozenset(
    {
        "init",
        "seed",
        "grow",
        "check",
        "feed",
        "stop",
        "scan",
        "graph",
        "connect",
        "inspect",
        "launch",
    }
)

# Safety caps: see /api/proc/start for the specific defenses each one enforces.
_MAX_LIVE_SESSIONS = 4
_MAX_ARGS = 64
_MAX_ARG_LEN = 4096

# Local alias for asyncio's argv-list subprocess spawner. This is the
# execFile-equivalent pattern (argv list, no shell interpretation) — the
# rename is a local convention, not a behavioral change.
_spawn_argv_subprocess = asyncio.create_subprocess_exec


@dataclass
class _ProcSession:
    """One running (or just-finished) petri subprocess invocation.

    Output is streamed through a PTY so rich/typer see a real terminal and
    emit their live multi-spinner + ANSI colors. Raw bytes (including
    escape sequences) flow through the session's queue as base64 strings
    and get rendered by xterm.js on the frontend.
    """

    stream_id: str
    process: asyncio.subprocess.Process
    queue: asyncio.Queue  # items are (kind: str, text: str) tuples
    master_fd: int = -1  # PTY master side; -1 when not using a PTY
    detach: bool = False
    finished: bool = False


# Keyed by stream_id. Lives at module scope so the /start endpoint can hand
# off a stream_id that the /stream endpoint picks up on a separate request.
_proc_sessions: Dict[str, _ProcSession] = {}


async def _drain_pty(master_fd: int, queue: asyncio.Queue) -> None:
    """Copy raw bytes from the PTY master into the session's queue.

    Chunks are base64-encoded so SSE JSON transport is binary-safe —
    ANSI escape sequences and UTF-8 multi-byte characters survive
    intact. The frontend decodes each chunk and feeds it to xterm.js
    for rendering.

    On Linux, reading from a PTY master whose slave was closed raises
    OSError(EIO) instead of returning b''. Both are treated as EOF.
    """
    loop = asyncio.get_running_loop()
    chunk_size = 4096
    while True:
        try:
            chunk = await loop.run_in_executor(
                None, os.read, master_fd, chunk_size
            )
        except OSError as exc:
            if exc.errno == errno.EIO:
                return  # slave closed
            raise
        if not chunk:
            return
        await queue.put(("stdout", base64.b64encode(chunk).decode("ascii")))


async def _await_exit(session: _ProcSession) -> None:
    """Wait for the subprocess to finish, then emit the terminal 'done'
    marker and close the PTY master fd."""
    code = await session.process.wait()
    await session.queue.put(("done", str(code)))
    session.finished = True
    if session.master_fd >= 0:
        try:
            os.close(session.master_fd)
        except OSError:
            pass
        session.master_fd = -1


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

    # ── Root + Static ────────────────────────────────────────────────

    @app.get("/")
    def root():
        return HTMLResponse(build_frontend_html())

    app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")

    # ── Health ────────────────────────────────────────────────────────

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    # ── Petri subprocess spawning (Computer tab) ─────────────────────
    #
    # Nine defenses layered here:
    #   1. argv list (no shell) — no substring interpretation
    #   2. frozen subcommand whitelist
    #   3. arg type validation (list of strings)
    #   4. argv[0..2] hardcoded to sys.executable / -m / petri
    #   5. cwd pinned to the project root
    #   6. localhost bind default (enforced in cli/launch.py)
    #   7. concurrent session cap
    #   8. arg count + length caps
    #   9. cleanup on disconnect via finally + killpg

    @app.post("/api/proc/start")
    async def proc_start(body: dict):
        command = body.get("command")
        args = body.get("args") or []
        detach = bool(body.get("detach"))

        # Defense 2: subcommand whitelist.
        if command not in _PETRI_SUBCOMMANDS:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown petri subcommand: {command!r}",
            )
        # Defense 3: arg type validation.
        if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
            raise HTTPException(
                status_code=400,
                detail="args must be a list of strings",
            )
        # Defense 8: arg count + length caps.
        if len(args) > _MAX_ARGS:
            raise HTTPException(
                status_code=400,
                detail=f"too many args (max {_MAX_ARGS})",
            )
        if any(len(a) > _MAX_ARG_LEN for a in args):
            raise HTTPException(
                status_code=400,
                detail=f"arg too long (max {_MAX_ARG_LEN} chars)",
            )
        # Defense 7: concurrent session cap.
        live_count = sum(1 for s in _proc_sessions.values() if not s.finished)
        if live_count >= _MAX_LIVE_SESSIONS:
            raise HTTPException(
                status_code=429,
                detail=f"too many live sessions (max {_MAX_LIVE_SESSIONS})",
            )

        # Defense 4: argv[0..2] hardcoded; argv[3] is whitelisted; argv[4:]
        # are user-supplied strings (already validated by defense 3).
        argv = [sys.executable, "-m", "petri", command, *args]

        # Allocate a PTY so the child process (rich/typer) thinks it's
        # talking to a real terminal and emits its live multi-spinner +
        # ANSI colors. Without this, rich detects non-TTY and silently
        # downgrades to boring single-line output.
        master_fd, slave_fd = pty.openpty()

        # Give the child a sensible terminal environment. Inheriting the
        # dashboard's env keeps PATH / HOME / CLAUDE_* intact; the keys we
        # set force color + unbuffered output even through whatever
        # buffering layers might still exist.
        child_env = os.environ.copy()
        child_env.update(
            {
                "TERM": "xterm-256color",
                "PYTHONUNBUFFERED": "1",
                "FORCE_COLOR": "1",
                "COLORTERM": "truecolor",
                "COLUMNS": "120",
                "LINES": "40",
            }
        )

        # Defense 1: argv-list spawn (execFile-equivalent) — no shell in
        # between. Defense 5: cwd pinned to the parent of .petri.
        # stdin = DEVNULL means interactive prompts (petri init without
        # --no-questions) will hang and need to be stopped with STOP.
        # That's fine for v1; the Computer tab is not a full TTY.
        try:
            process = await _spawn_argv_subprocess(
                *argv,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=slave_fd,
                stderr=slave_fd,
                cwd=str(petri_dir.parent),
                env=child_env,
                start_new_session=True,  # new process group for clean SIGTERM
                close_fds=True,
            )
        except Exception:
            os.close(master_fd)
            os.close(slave_fd)
            raise

        # The parent no longer needs the slave end; closing it lets the
        # master read return EOF cleanly when the child exits.
        os.close(slave_fd)

        stream_id = uuid.uuid4().hex
        queue: asyncio.Queue = asyncio.Queue()
        session = _ProcSession(
            stream_id=stream_id,
            process=process,
            queue=queue,
            master_fd=master_fd,
            detach=detach,
        )
        _proc_sessions[stream_id] = session

        asyncio.create_task(_drain_pty(master_fd, queue))
        asyncio.create_task(_await_exit(session))

        return {
            "stream_id": stream_id,
            "pid": process.pid,
            "argv": argv,
        }

    @app.get("/api/proc/stream/{stream_id}")
    async def proc_stream(stream_id: str):
        session = _proc_sessions.get(stream_id)
        if session is None:
            raise HTTPException(status_code=404, detail="unknown stream_id")
        if session.detach:
            raise HTTPException(status_code=400, detail="stream detached")

        async def generate():
            try:
                while True:
                    kind, text = await session.queue.get()
                    if kind == "done":
                        yield {
                            "event": "done",
                            "data": json.dumps({"code": int(text)}),
                        }
                        return
                    yield {
                        "event": kind,
                        "data": json.dumps({"line": text}),
                    }
            finally:
                # Defense 9: drop the session once the consumer disconnects
                # AND the subprocess has exited.
                if session.finished:
                    _proc_sessions.pop(stream_id, None)

        return EventSourceResponse(generate())

    @app.post("/api/proc/stop/{stream_id}")
    async def proc_stop(stream_id: str):
        session = _proc_sessions.get(stream_id)
        if session is None:
            raise HTTPException(status_code=404, detail="unknown stream_id")
        if session.process.returncode is not None:
            return {"status": "already_done", "code": session.process.returncode}

        # SIGTERM the whole process group so any petri-spawned children
        # (e.g. the claude CLI) get cleaned up too.
        try:
            os.killpg(os.getpgid(session.process.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        return {"status": "stopping", "pid": session.process.pid}

    # ── Dishes ────────────────────────────────────────────────────────

    @app.get("/api/dishes")
    def get_dishes():
        """List available dishes (empty list triggers onboarding)."""
        dishes_dir = petri_dir / "petri-dishes"
        if not dishes_dir.is_dir():
            return []
        result = []
        for entry in sorted(dishes_dir.iterdir()):
            if entry.is_dir():
                result.append({"id": entry.name, "path": str(entry)})
        return result

    # ── Init (web-based onboarding) ──────────────────────────────────

    @app.post("/api/init")
    def post_init(body: dict):
        """Run petri init logic from the web UI."""
        import shutil

        from petri.config import LLM_INFERENCE_MODEL, MAX_CONCURRENT, MAX_ITERATIONS

        dish_name = body.get("name", petri_dir.parent.name)
        model_name = body.get("model", LLM_INFERENCE_MODEL)
        max_concurrent = int(body.get("max_concurrent", MAX_CONCURRENT))
        max_iterations = int(body.get("max_iterations", MAX_ITERATIONS))

        defaults_dir = Path(__file__).parent.parent / "defaults"

        # Check for proper initialization (not just directory existence)
        config_exists = (petri_dir / "defaults" / "petri.yaml").exists()
        if config_exists:
            return {"status": "already_exists", "dish_id": dish_name}

        try:
            petri_dir.mkdir(parents=True, exist_ok=True)
            defaults_dest = petri_dir / "defaults"
            defaults_dest.mkdir(exist_ok=True)

            src_config = defaults_dir / "petri.yaml"
            if src_config.exists():
                try:
                    import yaml

                    with open(src_config) as f_cfg:
                        config = yaml.safe_load(f_cfg)
                    config["name"] = dish_name
                    config.setdefault("model", {})["name"] = model_name
                    config["max_concurrent"] = max_concurrent
                    config["max_iterations"] = max_iterations
                    with open(defaults_dest / "petri.yaml", "w") as f_out:
                        f_out.write("# Petri Dish Configuration\n")
                        f_out.write(f"# Initialized for dish: {dish_name}\n\n")
                        yaml.dump(config, f_out, default_flow_style=False, sort_keys=False)
                except ImportError:
                    (defaults_dest / "petri.yaml").write_text(
                        f"name: {dish_name}\nmodel:\n  name: {model_name}\n"
                        f"max_concurrent: {max_concurrent}\nmax_iterations: {max_iterations}\n",
                    )
            else:
                (defaults_dest / "petri.yaml").write_text(
                    f"name: {dish_name}\nmodel:\n  name: {model_name}\n"
                    f"max_concurrent: {max_concurrent}\nmax_iterations: {max_iterations}\n",
                )

            src_constitution = defaults_dir / "constitution.md"
            if src_constitution.exists():
                shutil.copy2(src_constitution, defaults_dest / "constitution.md")

            (petri_dir / "petri-dishes").mkdir(exist_ok=True)

            queue_data = {"version": 1, "last_updated": None, "entries": {}}
            (petri_dir / "queue.json").write_text(
                json.dumps(queue_data, indent=2) + "\n",
            )

            return {"status": "ok", "dish_id": dish_name}
        except OSError as exc:
            raise HTTPException(500, f"Init failed: {exc}")

    # ── Seed (web-based colony creation) ─────────────────────────────

    @app.post("/api/seed")
    def post_seed(body: dict):
        """Seed a new colony from the web UI."""
        from datetime import datetime, timezone

        claim = body.get("claim", "")
        colony_name_input = body.get("colony")

        if not claim:
            raise HTTPException(400, "claim is required")

        dish_id = _get_dish_id(petri_dir)

        from petri.reasoning.decomposer import (
            decompose_claim,
            generate_colony_name,
        )

        colony_name = colony_name_input or generate_colony_name(claim)

        try:
            result = decompose_claim(
                claim=claim,
                clarifications=[],
                dish_id=dish_id,
                colony_name=colony_name,
            )
        except Exception as exc:
            raise HTTPException(500, f"Decomposition failed: {exc}")

        from petri.graph.colony import ColonyGraph, serialize_colony
        from petri.models import Colony
        from petri.storage.event_log import append_event

        now = datetime.now(timezone.utc).isoformat()
        colony_id = f"{dish_id}-{colony_name}"

        graph = ColonyGraph(colony_id=colony_id)
        cells = result.cells
        edges = result.edges

        for cell in cells:
            graph.add_cell(cell)
        for edge in edges:
            graph.add_edge(edge)

        center_cell_id = cells[0].id if cells else ""
        colony_model = Colony(
            id=colony_id,
            dish=dish_id,
            center_claim=claim,
            center_cell_id=center_cell_id,
            clarifications=[],
            created_at=now,
        )

        colony_path = petri_dir / "petri-dishes" / colony_name
        serialize_colony(graph, colony_model, colony_path)

        for cell in cells:
            child_ids = [e.from_cell for e in edges if e.to_cell == cell.id]
            cell_rel_path = colony_model.cell_paths.get(
                cell.id, f"{cell.id.split('-')[-2]}-{cell.id.split('-')[-1]}"
            )
            events_path = colony_path / cell_rel_path / "events.jsonl"
            append_event(
                events_path=events_path,
                cell_id=cell.id,
                event_type="decomposition_created",
                agent="decomposition_lead",
                iteration=0,
                data={"parent_cell_id": cell.id, "child_cell_ids": child_ids},
            )

        return {
            "status": "ok",
            "colony_id": colony_id,
            "colony_name": colony_name,
            "cell_count": len(cells),
            "cells": [
                {"id": c.id, "claim_text": c.claim_text, "level": c.level}
                for c in cells
            ],
        }

    # ── Events ────────────────────────────────────────────────────────

    @app.get("/api/events")
    def get_events(
        cell_id: Optional[str] = None,
        iteration: Optional[int] = None,
        event_type: Optional[str] = None,
        agent: Optional[str] = None,
        limit: int = Query(default=500, le=5000),
    ):
        conditions: list[str] = []
        params: list[object] = []

        if cell_id:
            conditions.append("cell_id = ?")
            params.append(cell_id)
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
            "SELECT id, cell_id, timestamp, type, agent, iteration, data "
            "FROM events" + where + " ORDER BY timestamp ASC LIMIT ?"
        )
        params.append(limit)

        conn = get_db(db_path)
        try:
            rows = conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError:
            rows = []
        conn.close()

        return [
            {
                "id": r["id"],
                "cell_id": r["cell_id"],
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

    # ── Cells ─────────────────────────────────────────────────────────

    @app.get("/api/cells")
    def get_cells():
        """Return all cells with colony graph data."""
        from petri.graph.colony import deserialize_colony

        dishes_dir = petri_dir / "petri-dishes"
        if not dishes_dir.is_dir():
            return []

        # Derive dish_id from petri.yaml or parent directory name
        dish_id = _get_dish_id(petri_dir)

        all_cells: list[dict] = []
        for colony_dir in sorted(dishes_dir.iterdir()):
            if not colony_dir.is_dir():
                continue
            try:
                graph, colony = deserialize_colony(colony_dir, dish_id)
            except Exception:
                continue

            for cell in graph.get_all_cells():
                all_cells.append(
                    {
                        "cell_id": cell.id,
                        "colony_id": cell.colony_id,
                        "claim_text": cell.claim_text,
                        "level": cell.level,
                        "status": cell.status.value,
                        "dependencies": cell.dependencies,
                        "dependents": cell.dependents,
                    }
                )

        return all_cells

    # ── Single cell detail ────────────────────────────────────────────

    @app.get("/api/cell/{cell_id}")
    def get_cell_detail(cell_id: str):
        """Full detail for one cell: metadata + events + evidence markdown."""
        from petri.graph.colony import deserialize_colony

        dishes_dir = petri_dir / "petri-dishes"
        dish_id = _get_dish_id(petri_dir)

        # Find the cell across all colonies
        if dishes_dir.is_dir():
            for colony_dir in sorted(dishes_dir.iterdir()):
                if not colony_dir.is_dir():
                    continue
                try:
                    graph, colony = deserialize_colony(colony_dir, dish_id)
                    cell = graph.get_cell(cell_id)
                except (Exception, KeyError):
                    continue

                # Fetch events from SQLite
                conn = get_db(db_path)
                try:
                    rows = conn.execute(
                        "SELECT id, cell_id, timestamp, type, agent, iteration, data "
                        "FROM events WHERE cell_id = ? ORDER BY timestamp ASC",
                        [cell_id],
                    ).fetchall()
                except sqlite3.OperationalError:
                    rows = []
                conn.close()

                events = [
                    {
                        "id": r["id"],
                        "cell_id": r["cell_id"],
                        "timestamp": r["timestamp"],
                        "type": r["type"],
                        "agent": r["agent"],
                        "iteration": r["iteration"],
                        "data": json.loads(r["data"]),
                    }
                    for r in rows
                ]

                # Read evidence.md from the cell directory. The colony stores
                # the per-cell relative path in cell_paths; fall back to the
                # default {level}-{seq} slug used by serialize_colony.
                evidence_md = ""
                cell_rel_path = colony.cell_paths.get(
                    cell.id,
                    f"{cell.id.split('-')[-2]}-{cell.id.split('-')[-1]}",
                )
                evidence_file = colony_dir / cell_rel_path / "evidence.md"
                if evidence_file.is_file():
                    try:
                        evidence_md = evidence_file.read_text()
                    except OSError:
                        evidence_md = ""

                return {
                    "cell_id": cell.id,
                    "colony_id": cell.colony_id,
                    "claim_text": cell.claim_text,
                    "level": cell.level,
                    "status": cell.status.value,
                    "dependencies": cell.dependencies,
                    "dependents": cell.dependents,
                    "evidence_md": evidence_md,
                    "events": events,
                }

        raise HTTPException(404, "Cell %s not found" % cell_id)

    # ── Stats ─────────────────────────────────────────────────────────

    @app.get("/api/stats")
    def get_stats():
        conn = get_db(db_path)
        try:
            total_events = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
            cells_with_events = conn.execute(
                "SELECT COUNT(DISTINCT cell_id) FROM events"
            ).fetchone()[0]
            type_rows = conn.execute(
                "SELECT type, COUNT(*) as cnt FROM events GROUP BY type ORDER BY cnt DESC"
            ).fetchall()
            top_cells = conn.execute(
                "SELECT cell_id, COUNT(*) as cnt FROM events "
                "GROUP BY cell_id ORDER BY cnt DESC LIMIT 10"
            ).fetchall()
        except sqlite3.OperationalError:
            total_events = 0
            cells_with_events = 0
            type_rows = []
            top_cells = []
        conn.close()

        # Queue stats
        queue_path = petri_dir / "queue.json"
        try:
            queue = load_queue(queue_path)
        except Exception:
            queue = {"entries": {}}
        entries = queue.get("entries", {})

        cells_by_state: dict[str, int] = {}
        for entry in entries.values():
            state = entry.get("queue_state", "unknown")
            cells_by_state[state] = cells_by_state.get(state, 0) + 1

        return {
            "total_events": total_events,
            "cells_with_events": cells_with_events,
            "queue_size": len(entries),
            "cells_by_state": cells_by_state,
            "events_by_type": [
                {"type": r["type"], "count": r["cnt"]} for r in type_rows
            ],
            "top_cells": [
                {"cell_id": r["cell_id"], "event_count": r["cnt"]}
                for r in top_cells
            ],
        }

    # ── SSE stream ────────────────────────────────────────────────────

    @app.get("/api/stream")
    async def event_stream():
        async def generate():
            conn = get_db(db_path)
            try:
                row = conn.execute("SELECT MAX(rowid) FROM events").fetchone()
                last_rowid = row[0] or 0
            except sqlite3.OperationalError:
                last_rowid = 0
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
                        "SELECT rowid, id, cell_id, type, agent, iteration "
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
                                    "cell_id": evt["cell_id"],
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
