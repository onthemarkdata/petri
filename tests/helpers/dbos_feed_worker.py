"""Interrupt real SQLite feed persistence without calling a model service."""

import asyncio
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sqlite3
import sys
import time

from dbos import DBOSClient

from petri.execution.dbos_backend import CellRunSpec, DBOSExecutionBackend
from petri.storage import database
from petri.storage.colonies import save_colony
from petri.storage.paths import dish_db_path
from tests.conftest import _build_canonical_colony
from tests.pipeline_stubs import StubPhases


async def main():
    directory, point = Path(sys.argv[1]), sys.argv[2]
    directory.mkdir(parents=True, exist_ok=True)
    os.chdir(directory)
    (directory / "petri.yaml").write_text("name: test-dish\ntelemetry:\n  enabled: false\n")
    path = dish_db_path(directory)
    colony = _build_canonical_colony()
    if not path.exists():
        save_colony(path, colony["graph"], colony["colony_model"])
    with sqlite3.connect(directory / "audit.sqlite") as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS calls(phase TEXT)")
    phases = StubPhases()
    original = phases.dependencies()

    def wrapped(name):
        async def call(value):
            with sqlite3.connect(directory / "audit.sqlite") as conn:
                conn.execute("INSERT INTO calls VALUES (?)", (name,))
            return await getattr(original, name)(value)

        return call

    async def finish(value, result):
        with database._write(path) as conn:
            conn.execute(
                "UPDATE cells SET status=? WHERE id=?", (result.status.upper(), value.cell_id)
            )

    deps = replace(
        original,
        **{
            name: wrapped(name)
            for name in (
                "socratic",
                "research",
                "critique",
                "convergence",
                "red_team",
                "evaluate",
                "sync_check",
            )
        },
        finish=finish,
    )
    import petri.lifecycle.revalidate as module

    publish = module.write_once
    fork = DBOSClient.fork_workflow

    def pause():
        (directory / "interrupted").write_text(point)
        while not (directory / "release").exists():
            time.sleep(0.02)

    def publish_and_pause(output, content):
        publish(output, content)
        if point == "ingest" and output.parent.name == "evidence":
            pause()

    def fork_and_pause(self, *args, **kwargs):
        result = fork(self, *args, **kwargs)
        if point == "fork":
            # Keep the parent fork step uncheckpointed while its child settles.
            # Killing an uncheckpointed child inference could legitimately repeat it.
            result.get_result()
            pause()
        return result

    module.write_once = publish_and_pause
    DBOSClient.fork_workflow = fork_and_pause

    class LocalBackend(DBOSExecutionBackend):
        def start(self, petri_dir, **options):
            options["dependencies"] = lambda _: deps
            return super().start(petri_dir, **options)

    backend = LocalBackend()
    has_feed = False
    if (directory / "dbos.sqlite").is_file():
        client = DBOSClient(system_database_url=f"sqlite:///{directory / 'dbos.sqlite'}")
        try:
            has_feed = bool(client.list_workflows(name="petri.feed_evidence"))
        finally:
            client.destroy()
    cell_id = colony["cell1"].id
    if not has_feed:
        await backend.start_async(directory, db_path=path)
        try:
            handle = await backend.enqueue_cell(CellRunSpec(cell_id, 1, "2026-11-01"))
            await handle.get_result()
        finally:
            await backend.close_async()
    source = directory / "source.txt"
    source.write_text("A new measurement supports this claim.")
    import importlib
    from typer.testing import CliRunner
    from petri.cli import app

    module = importlib.import_module("petri.cli.feed")
    module.find_petri_dir = lambda: directory
    module.resolve_backend = lambda _: backend
    with ThreadPoolExecutor(max_workers=1) as executor:
        result = await asyncio.get_running_loop().run_in_executor(
            executor,
            lambda: CliRunner().invoke(app, ["feed", str(source), "--auto-reopen"]),
        )
    if result.exit_code:
        raise RuntimeError(result.output) from result.exception
    (directory / "result.json").write_text(
        json.dumps(
            {
                "status": database.get_cell(path, cell_id)["status"].lower(),
                "cell_id": cell_id,
            }
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
