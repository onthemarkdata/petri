"""Local subprocess fixture for the DBOS cell-round recovery tests."""

import asyncio
import json
import sqlite3
import sys
from dataclasses import asdict, replace
from pathlib import Path

from dbos import DBOS

from petri.execution.dbos_backend import CellRunSpec, DBOSExecutionBackend
from petri.execution.scheduler import ColonyRunSpec
from petri.storage.colonies import save_colony
from tests.conftest import _build_canonical_colony
from petri.models import Event
from petri.storage import database
from tests.pipeline_stubs import StubPhases


def main():
    directory = Path(sys.argv[1])
    mode = sys.argv[2]
    domain = directory / "petri.sqlite"
    phases = StubPhases()
    colony_mode = mode.startswith("colony")
    colony = _build_canonical_colony() if colony_mode else None
    if colony_mode and not domain.exists():
        save_colony(domain, colony["graph"], colony["colony_model"])
    elif not colony_mode and mode == "start":
        database.put_cell(
            domain,
            {
                "id": "dish-colony-002-001",
                "colony": "dish-colony",
                "level": 2,
                "seq": 1,
                "claim_text": "A local recovery fixture",
                "status": "NEW",
                "created_at": "2026-11-01T00:00:00Z",
            },
        )
    directory.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(directory / "audit.sqlite") as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS calls(phase TEXT, cell_id TEXT)")

    def record(phase, value):
        with sqlite3.connect(directory / "audit.sqlite") as conn:
            conn.execute("INSERT INTO calls VALUES (?,?)", (phase, value.cell_id))

    def event(phase, value):
        database.append_event(
            domain,
            Event(
                id=f"{value.cell_id}:{phase}",
                cell_id=value.cell_id,
                timestamp="2026-11-01T00:00:00Z",
                type="verdict_issued",
                agent="investigator",
                iteration=0,
                data={"verdict": "EVIDENCE_SUFFICIENT", "summary": phase},
            ),
        )

    async def socratic(value):
        record("socratic", value)
        event("socratic", value)
        return await phases.socratic(value)

    async def research(value):
        record("research", value)
        event("research", value)
        if not colony_mode or value.cell_id == colony["premise1"].id:
            (directory / "in-research").write_text("ready")
            while not (directory / "release").exists():
                await asyncio.sleep(0.02)
        return await phases.research(value)

    async def finish(value, result):
        with database._write(domain) as conn:
            conn.execute(
                "UPDATE cells SET status = ? WHERE id = ?",
                (result.status.upper(), value.cell_id),
            )

    deps = replace(phases.dependencies(), socratic=socratic, research=research, finish=finish)
    backend = DBOSExecutionBackend(config={"execution": {"backend": "dbos"}})
    backend.start(directory, db_path=domain, dependencies=lambda spec: deps)
    if mode == "probe":
        raise AssertionError("A second process acquired the dish execution lock")

    async def run():
        spec = (
            ColonyRunSpec(colony["colony_id"], 1, "2026-11-01")
            if colony_mode
            else CellRunSpec("dish-colony-002-001", 1, "2026-11-01")
        )
        if mode.endswith("resume"):
            handle = await DBOS.retrieve_workflow_async(spec.workflow_id)
        elif colony_mode:
            handle = await backend.enqueue_colony(spec)
        else:
            handle = await backend.enqueue_cell(spec, level=2)
        result = await handle.get_result()
        (directory / "result.json").write_text(
            json.dumps(asdict(result) if colony_mode else result.model_dump())
        )

    asyncio.run(run())
    backend.close()


if __name__ == "__main__":
    main()
