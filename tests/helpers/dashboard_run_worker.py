"""Run a dashboard seed with local model fixtures across a process interruption."""

import asyncio
import json
import sqlite3
import sys
import threading
from pathlib import Path

from httpx import ASGITransport, AsyncClient

from petri.cli.init import create_petri_dish
from petri.dashboard.api import create_app
from petri.execution.dbos_backend import DBOSExecutionBackend
from petri.reasoning.decomposition_schemas import InitialDecomposition, WhyExpansion
from petri.storage.paths import dish_db_path
from tests.pipeline_stubs import StubPhases


async def main():
    root = Path(sys.argv[1])
    create_petri_dish(root, dish_name="dish")
    (root / "defaults/petri.yaml").write_text("name: dish\ntelemetry:\n  enabled: false\n")
    audit = root / "calls.sqlite"
    with sqlite3.connect(audit) as connection:
        connection.execute("CREATE TABLE IF NOT EXISTS calls(name TEXT)")

    class Runner:
        def initial_decomposition(self, request, on_progress=None):
            with sqlite3.connect(audit) as connection:
                connection.execute("INSERT INTO calls VALUES (?)", ("initial",))
            return InitialDecomposition.model_validate(
                {
                    "nodes": [
                        {"claim_text": "The meter is a defined unit"},
                        {"claim_text": "The second is a defined unit", "relation": "limits"},
                    ]
                }
            )

        def expand_why(self, request, on_progress=None):
            if not (root / "release").exists():
                (root / "blocked").write_text("ready")
                threading.Event().wait()
            with sqlite3.connect(audit) as connection:
                connection.execute("INSERT INTO calls VALUES (?)", (request.claim,))
            return WhyExpansion.model_validate(
                {"is_atomic": True, "bedrock_reason": "A definition", "sub_premises": []}
            )

    backend = DBOSExecutionBackend()
    await backend.start_async(
        root,
        db_path=dish_db_path(root, "dish"),
        dependencies=lambda spec: StubPhases().dependencies(),
        seed_runner_factory=lambda spec: Runner(),
    )
    try:
        async with AsyncClient(
            transport=ASGITransport(app=create_app(root, execution_backend=backend)),
            base_url="http://localhost:8090",
        ) as client:
            if sys.argv[2] == "submit":
                response = await client.post(
                    "/api/seed", json={"claim": "Physical units", "colony": "units"}
                )
                assert response.status_code == 202, response.text
                (root / "receipt.json").write_text(response.text)
            for _ in range(1000):
                response = await client.get("/api/runs/dish-units")
                if response.status_code == 200 and response.json()["status"] not in {
                    "PENDING",
                    "ENQUEUED",
                }:
                    (root / "result.json").write_text(json.dumps(response.json()))
                    assert response.json()["status"] == "SUCCESS", response.text
                    return
                await asyncio.sleep(0.05)
            raise TimeoutError("Dashboard seed did not finish.")
    finally:
        await backend.close_async()


if __name__ == "__main__":
    asyncio.run(main())
