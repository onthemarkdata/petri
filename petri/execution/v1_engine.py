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
    backend_name = "v1"

    def seed_colony(self, *, petri_dir, **kwargs):
        from petri.reasoning.decomposer import decompose_claim

        return decompose_claim(**kwargs)

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
        from petri.storage.paths import dish_db_path
        from petri.storage.colonies import list_colonies

        db_path = dish_db_path(kwargs["petri_dir"]) if kwargs.get("dry_run") else None
        if db_path is not None and db_path.is_file():
            selected = kwargs.get("cell_ids")
            eligible = []
            if selected:
                eligible = list(selected)
            else:
                for graph, colony in list_colonies(db_path):
                    name = colony.id.removeprefix(f"{colony.dish}-")
                    if kwargs.get("colony_filter") and name != kwargs["colony_filter"]:
                        continue
                    statuses = {cell.id: cell.status for cell in graph.get_all_cells()}
                    eligible.extend(cell.id for cell in graph.get_eligible_for_validation(statuses))
                    if not kwargs.get("all_cells") and not kwargs.get("colony_filter"):
                        break
            return QueueProcessingResult(processed=0, succeeded=0, failed=0, stalled=0,
                                         dry_run=True, would_process=eligible)
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
