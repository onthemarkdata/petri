"""Run the real grow command with local, checkpointed fake phases."""

import asyncio
import json
import os
from dataclasses import replace
from pathlib import Path
import sqlite3
import sys

from typer.testing import CliRunner
from dbos import DBOSClient

from petri.cli import app
from petri.execution.dbos_backend import DBOSExecutionBackend
from petri.execution.pipeline import NeedsHuman
from petri.harness.errors import PermanentHarnessError
from petri.models import Cell
from petri.storage import database
from petri.storage.colonies import save_colony
from petri.storage.paths import dish_db_path
from petri.storage.event_log import append_event
from tests.conftest import _build_canonical_colony
from tests.pipeline_stubs import StubPhases


def main():
    directory = Path(sys.argv[1])
    park_sibling = "--park-sibling" in sys.argv[2:]
    parallel_stop = "--parallel-stop" in sys.argv[2:]
    directory.mkdir(parents=True, exist_ok=True)
    os.chdir(directory)
    (directory / "petri.yaml").write_text("name: test-dish\nexecution:\n  backend: dbos\n")
    path = dish_db_path(directory)
    colony = _build_canonical_colony()
    if parallel_stop:
        colony["graph"].add_cell(Cell(
            id=f"{colony['colony_id']}-002-099", colony_id=colony["colony_id"],
            claim_text="Queued offline sibling", level=2,
        ))
    if not path.exists():
        save_colony(path, colony["graph"], colony["colony_model"])
    with sqlite3.connect(directory / "audit.sqlite") as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS calls(cell_id TEXT, phase TEXT)")
    phases = StubPhases()
    original = phases.dependencies()
    researching = set()
    held_children = {
        colony["cell1"].id, colony["cell2"].id, f"{colony['colony_id']}-002-099"
    }

    def wrapped(name):
        async def call(value):
            if parallel_stop and name == "research" and value.cell_id in held_children:
                researching.add(value.cell_id)
                if len(researching) == 2:
                    (directory / "in-research").write_text("two active children")
                while not (directory / "release").exists():
                    await asyncio.sleep(0.02)
            if not parallel_stop and name == "research" and value.cell_id == colony["premise1"].id:
                client = DBOSClient(system_database_url=f"sqlite:///{directory / 'dbos.sqlite'}")
                try:
                    while True:
                        statuses = client.list_workflows(workflow_ids=[f"{colony['premise2'].id}:r1"])
                        if statuses and statuses[0].status == "SUCCESS":
                            break
                        await asyncio.sleep(0.02)
                finally:
                    client.destroy()
                (directory / "in-research").write_text("ready")
                while not (directory / "release").exists():
                    await asyncio.sleep(0.02)
            with sqlite3.connect(directory / "audit.sqlite") as conn:
                conn.execute("INSERT INTO calls VALUES (?,?)", (value.cell_id, name))
            append_event(
                database.SQLiteEventLog(path), value.cell_id, "evidence_appended", "stub", 0,
                {"summary": f"Completed offline {name}"}, round_number=1, node=name,
            )
            if park_sibling and name == "research" and value.cell_id == colony["premise2"].id:
                raise PermanentHarnessError("Offline sibling needs review")
            return await getattr(original, name)(value)
        return call

    async def finish(value, result):
        if isinstance(result, NeedsHuman):
            append_event(
                database.SQLiteEventLog(path), value.cell_id, "agent_execution_failed", "stub", 0,
                {"phase": "research", "error_type": "PermanentHarnessError",
                 "message": result.reason, "failure_detail": {}},
                round_number=1, node="finish",
            )
        with database._write(path) as conn:
            status = "STALLED" if isinstance(result, NeedsHuman) else result.status.upper()
            conn.execute("UPDATE cells SET status=? WHERE id=?", (status, value.cell_id))

    deps = replace(original, **{
        name: wrapped(name) for name in (
            "socratic", "research", "critique", "convergence", "red_team", "evaluate", "sync_check"
        )
    }, finish=finish)

    class LocalBackend(DBOSExecutionBackend):
        def start(self, petri_dir, **options):
            options["dependencies"] = lambda _: deps
            return super().start(petri_dir, **options)

    backend = LocalBackend(config={"execution": {"backend": "dbos", "worker_concurrency": 2}})
    import importlib
    grow = importlib.import_module("petri.cli.grow")
    grow.find_petri_dir = lambda: directory
    grow.resolve_backend = lambda _: backend
    grow.resolve_provider = lambda _: None
    result = CliRunner().invoke(app, ["grow"])
    (directory / "cli-result.json").write_text(json.dumps({
        "exit_code": result.exit_code, "output": result.output, "exception": repr(result.exception)
    }))
    expected_needs_human = (
        park_sibling and result.exit_code == 1
        and "Grow loop finished: needs_human" in result.output
    )
    if result.exit_code and not expected_needs_human:
        raise RuntimeError(result.output) from result.exception


if __name__ == "__main__":
    main()
