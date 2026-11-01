"""Delegate the execution protocol to the unchanged v1 engine and queue."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from petri.engine import grow_loop as loop
from petri.engine import preflight, processor
from petri.execution.models import GrowOutcome, NoProviderError, PreflightCheck
from petri.storage import queue
from petri.models import QueueProcessingResult


class V1ExecutionBackend:
    def __init__(self, config: dict | None = None):
        self.config = config

    @property
    def status_interval(self) -> float:
        return loop.GROW_STATUS_INTERVAL_SECONDS

    def preflight(self) -> PreflightCheck:
        if self.config is None or not isinstance(self.config.get("harness"), dict):
            return PreflightCheck(**asdict(preflight.check_claude_cli()))
        checks = preflight.run_preflight(self.config)
        return PreflightCheck(
            name="Inference backend", passed=all(check.passed for check in checks),
            message="; ".join(f"{check.name}: {check.message}" for check in checks),
        )

    def process_queue(self, **kwargs: Any) -> QueueProcessingResult:
        try:
            return processor.process_queue(**kwargs)
        except processor.NoProviderError as error:
            raise NoProviderError(str(error)) from error

    def grow_loop(self, **kwargs: Any) -> GrowOutcome:
        result = loop.grow_loop(**kwargs)
        return GrowOutcome(
            reason=result.reason,
            final_states=result.final_states,
            last_result=result.last_result,
            passes_run=result.passes_run,
        )

    def state_summary(self, petri_dir: Path) -> dict[str, int]:
        return queue.get_state_summary(petri_dir / "queue.json")

    def format_state_summary(self, states: dict[str, int]) -> str:
        return loop.format_state_summary(states)

    def clear_stop(self, petri_dir: Path) -> None:
        processor.clear_stop_file(petri_dir)

    def is_stopped(self, petri_dir: Path) -> bool:
        return processor.is_stop_file_present(petri_dir)

    def request_stop(self, petri_dir: Path) -> None:
        processor.request_stop()
        processor.request_stop_file(petri_dir)

    def entries(self, petri_dir: Path) -> dict[str, dict]:
        return queue.load_queue(petri_dir / "queue.json").get("entries", {})

    def list_entries(self, petri_dir: Path) -> list[dict]:
        return queue.list_queue(petri_dir / "queue.json")

    def update_state(self, petri_dir: Path, cell_id: str, state: str) -> None:
        queue.update_state(petri_dir / "queue.json", cell_id, state)
