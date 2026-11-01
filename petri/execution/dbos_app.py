"""Own one dish's DBOS process and enforce its admission budget."""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable
from contextlib import closing, contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from filelock import FileLock, Timeout

from petri.execution.config import ExecutionConfig
from petri.storage import database

if TYPE_CHECKING:
    from petri.execution.dbos_backend import CellRunSpec
    from petri.execution.pipeline import PipelineDeps

QUEUE_NAME = "petri-cells"
APPLICATION_VERSION = "petri-cell-pipeline-1"
CHECKPOINT_LIMIT = 2 * 1024 * 1024
_ACTIVE: DBOSApp | None = None


class ExecutionOwnerError(RuntimeError):
    """Another process already owns execution for this dish."""


def active_app() -> DBOSApp:
    if _ACTIVE is None or _ACTIVE.closing:
        raise RuntimeError("The dish execution runtime has not been started")
    return _ACTIVE


class DBOSApp:
    def __init__(
        self,
        petri_dir: Path,
        db_path: Path,
        settings: ExecutionConfig,
        dependencies: Callable[[CellRunSpec], PipelineDeps],
        *,
        seed_runner_factory=None,
        seed_on_progress=None,
        seed_on_cell_created=None,
        owns_seed_runners=False,
    ):
        self.petri_dir = Path(petri_dir).resolve()
        self.db_path = Path(db_path).resolve()
        self.settings = settings
        self.owns_seed_runners = owns_seed_runners
        self.closing = False
        self._activity = threading.Condition()
        self._active_steps = 0
        self.seed_runner_factory = seed_runner_factory
        self.seed_on_progress = seed_on_progress
        self.seed_on_cell_created = seed_on_cell_created
        self._seed_runners: dict[str, object] = {}
        self._factory = dependencies
        self._dependencies: dict[str, PipelineDeps] = {}
        self._deps_lock = threading.Lock()
        self._owner = FileLock(self.petri_dir / ".execution-owner.lock", thread_local=False)
        self._slots = threading.BoundedSemaphore(settings.worker_concurrency)
        self.started = False
        self.conflicts: dict[str, dict] = {}

    def dependencies(self, spec: CellRunSpec) -> PipelineDeps:
        with self._deps_lock:
            if spec.workflow_id not in self._dependencies:
                self._dependencies[spec.workflow_id] = self._factory(spec)
            return self._dependencies[spec.workflow_id]

    @contextmanager
    def step_activity(self):
        with self._activity:
            if self.closing:
                raise RuntimeError("The execution runtime is stopping")
            self._active_steps += 1
        try:
            yield
        finally:
            with self._activity:
                self._active_steps -= 1
                self._activity.notify_all()

    def detach(self):
        """Retain pending workflows after all active steps finish."""
        global _ACTIVE
        from dbos import DBOS

        with self._activity:
            self.closing = True
            if not self._activity.wait_for(lambda: self._active_steps == 0, timeout=10):
                raise RuntimeError(
                    "An inference call is still running; execution ownership is retained"
                )
        DBOS.destroy(workflow_completion_timeout_sec=0)
        self.started = False
        _ACTIVE = None
        self._owner.release()

    def seed_runner(self, spec):
        if self.seed_runner_factory is None:
            raise RuntimeError("Configure a typed seed runner before starting decomposition")
        if spec.workflow_id not in self._seed_runners:
            self._seed_runners[spec.workflow_id] = self.seed_runner_factory(spec)
        return self._seed_runners[spec.workflow_id]

    def start(self) -> DBOSApp:
        global _ACTIVE
        from dbos import DBOS

        # Importing registers the fixed workflow and step definitions before launch.
        from petri.execution import analyst_schedule, dbos_backend, scan_workflow, scheduler, seed_workflow  # noqa: F401
        from petri.lifecycle import redecompose  # noqa: F401

        self.petri_dir.mkdir(parents=True, exist_ok=True)
        try:
            self._owner.acquire(timeout=0)
        except Timeout as error:
            raise ExecutionOwnerError(
                "Another process owns this dish's execution. Attach to it or stop it first."
            ) from error
        if _ACTIVE is not None:
            self._owner.release()
            raise ExecutionOwnerError("This process already owns a dish execution runtime")
        _ACTIVE = self
        try:
            self._prepare_admission()
            scheduler.prepare_rounds(self.db_path)
            from petri.execution.reconciliation import reconcile

            self.conflicts = reconcile(self.db_path, self.petri_dir / "dbos.sqlite")
            DBOS(
                config={
                    "name": "petri",
                    "application_version": APPLICATION_VERSION,
                    "system_database_url": f"sqlite:///{self.petri_dir / 'dbos.sqlite'}",
                    "executor_id": "petri-dish-owner",
                    "use_listen_notify": False,
                    "enable_otlp": False,
                    "log_level": "WARNING",
                }
            )
            DBOS.launch()
            scan_workflow.register_scan_schedule(self.settings.scan_cron)
            analyst_schedule.register_analyst_schedule()
            DBOS.register_queue(
                QUEUE_NAME,
                worker_concurrency=self.settings.worker_concurrency,
                global_concurrency=self.settings.worker_concurrency,
                limiter=(
                    {"limit": self.settings.limiter.limit, "period": self.settings.limiter.period}
                    if self.settings.limiter
                    else None
                ),
                polling_interval_sec=self.settings.polling_interval_seconds,
            )
            self.started = True
            return self
        except BaseException:
            DBOS.destroy()
            _ACTIVE = None
            self._owner.release()
            raise

    def close(self) -> None:
        global _ACTIVE
        if self.started:
            from dbos import DBOS

            # Callers must await known handles before releasing ownership.
            pending = DBOS.list_workflows(status=["PENDING", "ENQUEUED"], limit=1)
            if pending:
                raise RuntimeError("Wait for or cancel active workflows before closing execution")
            DBOS.destroy(workflow_completion_timeout_sec=10)
            for runner in self._seed_runners.values() if self.owns_seed_runners else ():
                if hasattr(runner, "close"):
                    runner.close()
            self.started = False
            _ACTIVE = None
            self._owner.release()

    async def acquire_slot(self) -> None:
        while not self._slots.acquire(blocking=False):
            if self.closing:
                raise RuntimeError("The execution runtime is stopping")
            await asyncio.sleep(0.01)

    def release_slot(self) -> None:
        self._slots.release()

    def _prepare_admission(self) -> None:
        with closing(database.connect(self.db_path)) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS execution_admissions "
                "(workflow_id TEXT PRIMARY KEY, started_at REAL NOT NULL)"
            )
            conn.commit()

    def admission_delay(self, workflow_id: str) -> float:
        """Atomically admit one start, preserving the rate window across restarts."""
        limiter = self.settings.limiter
        with closing(database.connect(self.db_path)) as conn:
            conn.execute("BEGIN IMMEDIATE")
            if conn.execute(
                "SELECT 1 FROM execution_admissions WHERE workflow_id = ?", (workflow_id,)
            ).fetchone():
                conn.commit()
                return 0
            now = time.time()
            if limiter:
                starts = conn.execute(
                    "SELECT started_at FROM execution_admissions WHERE started_at > ? "
                    "ORDER BY started_at",
                    (now - limiter.period,),
                ).fetchall()
                if len(starts) >= limiter.limit:
                    conn.commit()
                    return max(0.001, starts[-limiter.limit][0] + limiter.period - now)
            conn.execute("INSERT INTO execution_admissions VALUES (?, ?)", (workflow_id, now))
            conn.commit()
            return 0

    def resource_report(self) -> str:
        rate = (
            f"{self.settings.limiter.limit} starts/{self.settings.limiter.period:g}s"
            if self.settings.limiter
            else "unlimited starts"
        )
        return (
            f"DBOS: {self.settings.worker_concurrency} concurrent cells; {rate}; "
            "one execution owner per dish. Concurrency is bounded by execution.worker_concurrency."
        )
