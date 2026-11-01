"""Execution protocol; this module does not import an engine or queue store."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol, runtime_checkable

from petri.execution.models import GrowOutcome, PreflightCheck, ProgressEvent
from petri.execution.run_controls import RunRequest, RunStatus, SeedRequest
from petri.models import InferenceProvider, QueueProcessingResult


@runtime_checkable
class ExecutionBackend(Protocol):
    async def start_owner(self, petri_dir: Path, *, on_phase_progress=None,
                          event_stream_handler_factory=None) -> None: ...

    async def shutdown_owner(self) -> None: ...

    async def submit_seed(self, request: SeedRequest) -> RunStatus: ...

    async def submit_run(self, request: RunRequest) -> RunStatus: ...

    async def get_run(self, petri_dir: Path, run_id: str) -> RunStatus | None: ...

    async def cancel_run(self, petri_dir: Path, run_id: str) -> RunStatus | None: ...

    @property
    def status_interval(self) -> float: ...

    def preflight(self) -> PreflightCheck: ...

    def seed_colony(
        self,
        *,
        petri_dir: Path,
        claim,
        clarifications,
        dish_id,
        colony_name,
        provider,
        settings,
        guidance="",
        db_path=None,
        on_progress=None,
        on_cell_created=None,
        center=None,
        agent_runner=None,
        resume=False,
    ): ...

    def reopen_cell(self, request, *, petri_dir: Path): ...

    def process_queue(
        self,
        *,
        petri_dir: Path,
        provider: InferenceProvider | None,
        max_concurrent: int,
        cell_ids: list[str] | None,
        colony_filter: str | None,
        all_cells: bool,
        dry_run: bool,
        on_event: Callable[[ProgressEvent], None] | None = None,
    ) -> QueueProcessingResult: ...

    def grow_loop(
        self,
        *,
        run_one_pass: Callable[[], QueueProcessingResult],
        get_states: Callable[[], dict[str, int]],
        is_stopped: Callable[[], bool],
    ) -> GrowOutcome: ...

    def state_summary(self, petri_dir: Path) -> dict[str, int]: ...

    def format_state_summary(self, states: dict[str, int]) -> str: ...

    def clear_stop(self, petri_dir: Path) -> None: ...

    def is_stopped(self, petri_dir: Path) -> bool: ...

    def request_stop(self, petri_dir: Path, *, force: bool = False) -> int: ...

    def entries(self, petri_dir: Path) -> dict[str, dict]: ...

    def list_entries(self, petri_dir: Path) -> list[dict]: ...

    def update_state(self, petri_dir: Path, cell_id: str, state: str) -> None: ...
