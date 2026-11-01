"""Enqueue durable work and expose backend-owned status and cancellation."""

from __future__ import annotations

import asyncio
from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, HTTPException, Response

from petri.dashboard.broker import AgentProgress, publish
from petri.engine.streaming import progress_handler
from petri.execution.backend import ExecutionBackend
from petri.execution.run_controls import RunReceipt, RunRequest, RunStatus, SeedRequest


class DashboardOwner:
    def __init__(self, petri_dir: Path, backend: ExecutionBackend):
        self.petri_dir = petri_dir
        self.backend = backend
        self.started = False
        self.lock = asyncio.Lock()

    async def start(self) -> None:
        async with self.lock:
            if self.started:
                return

            def phase_progress(event):
                if event.agent is not None:
                    publish(
                        AgentProgress(
                            cell_id=event.cell_id,
                            agent=event.agent,
                            kind=event.kind,
                            text_delta=event.verdict or event.phase,
                        )
                    )

            def stream_factory(cell_id, agent):
                return progress_handler(cell_id=cell_id, agent=agent)

            await self.backend.start_owner(
                self.petri_dir,
                on_phase_progress=phase_progress,
                event_stream_handler_factory=stream_factory,
            )
            self.started = True

    async def close(self) -> None:
        if self.started:
            await self.backend.shutdown_owner()
            self.started = False


def create_runs_router(
    petri_dir: Path, backend: ExecutionBackend, owner: DashboardOwner
) -> APIRouter:
    router = APIRouter()

    def receipt(run: RunStatus, response: Response, colony_id: str | None = None) -> RunReceipt:
        url = "/api/runs/" + quote(run.id, safe="")
        response.headers["Location"] = url
        return RunReceipt(
            id=run.id, workflow_id=run.id, status=run.status, status_url=url, colony_id=colony_id
        )

    @router.post("/api/seed", status_code=202, response_model=RunReceipt)
    async def post_seed(body: SeedRequest, response: Response):
        try:
            run = await backend.submit_seed(body)
        except ValueError as error:
            raise HTTPException(409, str(error)) from error
        except RuntimeError as error:
            raise HTTPException(503, str(error)) from error
        colony_id = run.id.split(":seed-attempt", 1)[0]
        return receipt(run, response, colony_id)

    @router.post("/api/runs", status_code=202, response_model=RunReceipt)
    async def post_run(body: RunRequest, response: Response):
        try:
            run = await backend.submit_run(body)
        except KeyError as error:
            raise HTTPException(404, str(error)) from error
        except ValueError as error:
            raise HTTPException(409, str(error)) from error
        except RuntimeError as error:
            raise HTTPException(503, str(error)) from error
        return receipt(run, response)

    @router.get("/api/runs/{run_id:path}", response_model=RunStatus)
    async def get_run(run_id: str):
        run = await backend.get_run(petri_dir, run_id)
        if run is None:
            raise HTTPException(404, "Run not found.")
        return run

    @router.post("/api/runs/{run_id:path}/cancel", response_model=RunStatus)
    async def cancel_run(run_id: str):
        run = await backend.cancel_run(petri_dir, run_id)
        if run is None:
            raise HTTPException(404, "Run not found.")
        return run

    return router
