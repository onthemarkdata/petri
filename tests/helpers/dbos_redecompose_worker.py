"""Pause after a recorded model response so recovery can prove it skips inference."""

import asyncio
import json
import sys
import time
from pathlib import Path

from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel

from petri.execution.dbos_backend import DBOSExecutionBackend
from petri.graph.colony import ColonyGraph
from petri.harness.legacy_bridge import HarnessInferenceProvider
from petri.lifecycle.redecompose import RedecomposeSpec, propose
from petri.lifecycle.triggers import RedecompositionTrigger
from petri.models import Cell, Colony
from petri.storage.colonies import save_colony
from tests.pipeline_stubs import StubPhases


async def main(root: Path, mode: str) -> None:
    path = root / "petri-dishes/dish/petri.sqlite"
    parent = "dish-case-000-000"
    if not path.exists():
        graph = ColonyGraph("dish-case")
        graph.add_cell(Cell(id=parent, colony_id="dish-case", claim_text="The stated mechanism explains the observed result", level=0))
        save_colony(path, graph, Colony(id="dish-case", dish="dish", center_claim="The stated mechanism explains the observed result", center_cell_id=parent))

    def respond(messages, info):
        with (root / "calls.jsonl").open("a") as file:
            file.write(json.dumps({"call": "decomposer"}) + "\n")
        return ModelResponse(parts=[TextPart(json.dumps({"is_atomic": False, "sub_premises": [
            {"claim_text": "An independent measurement reproduces the result", "relation": "supports"},
            {"claim_text": "A calibration bias could explain the observation", "relation": "limits"},
        ]}))])

    from petri.lifecycle import redecompose as module
    original = module.journal_output

    def journal(*args):
        original(*args)
        (root / "response-recorded").touch()
        if mode == "start":
            while True:
                time.sleep(0.05)

    module.journal_output = journal
    runner = HarnessInferenceProvider(FunctionModel(respond))
    backend = DBOSExecutionBackend(config={"execution": {"backend": "dbos"}})
    await backend.start_async(root, db_path=path, dependencies=lambda _: StubPhases().dependencies(), seed_runner_factory=lambda _: runner)
    try:
        spec = RedecomposeSpec(trigger=RedecompositionTrigger(cell_id=parent, trigger_verdict="EVIDENCE_INCONCLUSIVE", agent="evidence_evaluator", fire_point="settled_round"))
        result = await propose(spec)
        (root / "result.json").write_text(result.model_dump_json())
    finally:
        await backend.close_async()
        runner.close()


if __name__ == "__main__":
    asyncio.run(main(Path(sys.argv[1]), sys.argv[2]))
