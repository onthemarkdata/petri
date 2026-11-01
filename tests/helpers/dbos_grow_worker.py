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
from petri.storage import database
from petri.storage.colonies import save_colony
from petri.storage.paths import dish_db_path
from tests.conftest import _build_canonical_colony
from tests.pipeline_stubs import StubPhases


def main():
    directory = Path(sys.argv[1])
    directory.mkdir(parents=True, exist_ok=True)
    os.chdir(directory)
    (directory / "petri.yaml").write_text("name: test-dish\nexecution:\n  backend: dbos\n")
    path = dish_db_path(directory)
    colony = _build_canonical_colony()
    if not path.exists():
        save_colony(path, colony["graph"], colony["colony_model"])
    with sqlite3.connect(directory / "audit.sqlite") as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS calls(cell_id TEXT, phase TEXT)")
    phases = StubPhases()
    original = phases.dependencies()

    def wrapped(name):
        async def call(value):
            if name == "research" and value.cell_id == colony["premise1"].id:
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
            return await getattr(original, name)(value)
        return call

    async def finish(value, result):
        with database._write(path) as conn:
            conn.execute("UPDATE cells SET status=? WHERE id=?", (result.status.upper(), value.cell_id))

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
    if result.exit_code:
        raise RuntimeError(result.output) from result.exception


if __name__ == "__main__":
    main()
