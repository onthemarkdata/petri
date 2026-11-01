"""Leave one local research workflow pending for the manual-scan regression."""

import asyncio
import sys
from dataclasses import replace
from pathlib import Path

from petri.execution.dbos_backend import CellRunSpec, DBOSExecutionBackend
from petri.storage import database
from petri.storage.paths import dish_db_path
from tests.pipeline_stubs import StubPhases


def main():
    directory = Path(sys.argv[1])
    directory.mkdir()
    (directory / "petri.yaml").write_text("name: dish\n")
    path = dish_db_path(directory)
    cell_id = "dish-colony-001-001"
    database.put_cell(
        path,
        {
            "id": cell_id,
            "colony": "dish-colony",
            "level": 1,
            "seq": 1,
            "claim_text": "Local scan recovery fixture",
            "created_at": "2026-11-01",
        },
    )
    phases = StubPhases()

    async def research(value):
        (directory / "research-started").write_text("Pending")
        while True:
            await asyncio.sleep(0.05)

    deps = replace(phases.dependencies(), research=research)
    backend = DBOSExecutionBackend(config={"execution": {"backend": "dbos"}})
    backend.start(directory, db_path=path, dependencies=lambda spec: deps)

    async def run():
        handle = await backend.enqueue_cell(CellRunSpec(cell_id, 1, "2026-11-01"))
        await handle.get_result()

    asyncio.run(run())


if __name__ == "__main__":
    main()
