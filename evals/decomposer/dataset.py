"""Pydantic Evals cases with offline scripts and an explicit live task."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml
from pydantic_ai import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.messages import ModelRequest, ToolReturnPart
from pydantic_ai.models.function import FunctionModel
from pydantic_evals import Case, Dataset

from evals.decomposer.evaluators import QUALITY_EVALUATORS
from evals.decomposer.types import EvalBatch, EvalInput
from petri.harness.legacy_bridge import HarnessInferenceProvider
from petri.models import Cell, DecompositionResult, Edge
from petri.reasoning.decomposer import decompose_claim
from petri.reasoning.decomposition_settings import DecompositionSettings

ROOT = Path(__file__).parent


def load_inputs() -> list[EvalInput]:
    return [EvalInput.model_validate(item) for item in yaml.safe_load((ROOT / "claims.yaml").read_text())]


def build_dataset(*, include_negatives: bool = True) -> Dataset:
    return Dataset(name="decomposition-quality", cases=[Case(name=item.name, inputs=item)
                   for item in load_inputs() if include_negatives or item.kind != "negative"],
                   evaluators=[kind() for kind in QUALITY_EVALUATORS])


def negative_fixture(inputs: EvalInput) -> EvalBatch:
    if inputs.fixture is None or Path(inputs.fixture).name != inputs.fixture:
        raise ValueError("Negative case requires a fixture filename")
    raw = json.loads((ROOT / "fixtures" / inputs.fixture).read_text())
    root = Cell(id="eval-negative-000-000", colony_id="eval-negative", level=0, claim_text=raw["claim"])
    cells = [root]
    edges = []
    for index, text in enumerate(raw["children"], 1):
        cell = Cell(id=f"eval-negative-001-{index:03d}", colony_id="eval-negative", level=1, claim_text=text)
        cells.append(cell)
        edges.append(Edge(from_cell=root.id, to_cell=cell.id))
    for index, text in enumerate(raw.get("grandchildren", []), 1):
        cell = Cell(id=f"eval-negative-002-{index:03d}", colony_id="eval-negative", level=2, claim_text=text)
        cells.append(cell)
        edges.append(Edge(from_cell=cells[2].id, to_cell=cell.id))
    return EvalBatch(results=[DecompositionResult(cells=cells, edges=edges,
                            center_claim=root.claim_text, colony_name="negative")])


def scripted_plan(claim: str, sizes: list[int], *, prefix: str, axiom: str | None = None) -> dict[str, dict]:
    """Create known graph shapes; the synthetic text carries no research findings."""
    plan: dict[str, dict] = {}
    levels = []
    for level, count in enumerate(sizes, 1):
        levels.append([axiom if axiom and level == 1 else
                       f"Constraint {prefix}L{level}N{index} predicts an observable outcome for condition {level}-{index}."
                       for index in range(count)])
    plan[claim] = {"nodes": [{"claim_text": text, "relation": "rebuts" if index == len(levels[0]) - 1
                              and len(levels[0]) > 1 else "supports"}
                             for index, text in enumerate(levels[0])]}
    for position, parents in enumerate(levels):
        remaining = list(levels[position + 1]) if position + 1 < len(levels) else []
        for parent in parents:
            children, remaining = remaining[:4], remaining[4:]
            if children:
                plan[parent] = {"is_atomic": False, "sub_premises": [
                    {"claim_text": text, "relation": "rebuts" if index == len(children) - 1
                     and len(children) > 1 else "supports"} for index, text in enumerate(children)]}
            else:
                plan[parent] = {"is_atomic": True, "bedrock_reason": "A stipulated measurement in this synthetic fixture.",
                                "sub_premises": []}
    return plan


def scripted_model(plan: dict[str, dict]) -> FunctionModel:
    def response(messages, info):
        instructions = info.instructions or ""
        parent = next((text for text in plan if f'Claim: "{text}"' in instructions
                       or f'Premise: "{text}"' in instructions), None)
        if parent is None:
            raise AssertionError("Fixture plan does not contain this requested claim")
        payload = plan[parent]
        latest = messages[-1]
        returned = isinstance(latest, ModelRequest) and any(isinstance(part, ToolReturnPart) for part in latest.parts)
        if not returned:
            candidates = payload.get("nodes", payload.get("sub_premises", []))
            query = next((item["claim_text"] for item in candidates if "claim_text" in item), parent)
            return ModelResponse(parts=[ToolCallPart("search_cells", {"query": query})])
        return ModelResponse(parts=[TextPart(json.dumps(payload))])

    return FunctionModel(response)


def scripted_task(inputs: EvalInput) -> EvalBatch:
    if inputs.kind == "negative":
        return negative_fixture(inputs)
    results: list[DecompositionResult] = []
    expected = set()
    with TemporaryDirectory(prefix="petri-offline-eval-") as directory:
        path = Path(directory) / "dish.sqlite"
        for index, (claim, sizes) in enumerate(zip(inputs.claims, inputs.layer_sizes, strict=True)):
            plan = scripted_plan(claim, sizes, prefix=f"case{index}",
                                 axiom=inputs.axiom_texts[0] if inputs.axiom_texts else None)
            if inputs.kind == "reference" and index == 1:
                target = results[0].cells[1]
                expected.add(target.id)
                plan[claim] = {"nodes": [{"existing_cell_id": target.id, "reason": "Reuse the earlier measured premise"}]}
                # Search with the existing text so its real ID is available to validation.
                seen = False

                def reference_response(messages, info):
                    nonlocal seen
                    if not seen:
                        seen = True
                        return ModelResponse(parts=[ToolCallPart("search_cells", {"query": target.claim_text})])
                    return ModelResponse(parts=[TextPart(json.dumps(plan[claim]))])

                model = FunctionModel(reference_response)
            else:
                model = scripted_model(plan)
            with HarnessInferenceProvider(model) as runner:
                result = decompose_claim(claim, [], "eval", f"case{index}", None, agent_runner=runner,
                                         db_path=path, settings=DecompositionSettings(
                                             depth=len(sizes) + 1, per_layer=inputs.layer_ceiling,
                                             level_one_cap=inputs.layer_ceiling, model="offline-function-model"))
            results.append(result)
    return EvalBatch(results=results, expected_reference_ids=expected)


def live_task(inputs: EvalInput, *, model_name: str) -> EvalBatch:
    """Use an explicitly configured model; callers own cost consent and gating."""
    if inputs.kind == "negative":
        return negative_fixture(inputs)
    results: list[DecompositionResult] = []
    expected = set()
    with TemporaryDirectory(prefix="petri-live-eval-") as directory:
        path = Path(directory) / "dish.sqlite"
        for index, claim in enumerate(inputs.claims):
            if inputs.kind == "reference" and index == 1:
                expected = {cell.id for cell in results[0].cells if cell.level > 0}
            with HarnessInferenceProvider(model_name) as runner:
                result = decompose_claim(claim, [], "eval", f"case{index}", None, agent_runner=runner,
                                         db_path=path, settings=DecompositionSettings(
                                             depth=3, per_layer=inputs.layer_ceiling, model=model_name))
            results.append(result)
    return EvalBatch(results=results, expected_reference_ids=expected)


def assert_neutral_instructions() -> None:
    from petri.reasoning.decomposition_agents import DecompositionInput, initial_instructions, why_instructions

    for builder in (initial_instructions, why_instructions):
        text = builder(DecompositionInput("A scoped claim"))
        assert "Pick the TOP" not in text and "AT MOST" not in text, "Numeric quota phrasing returned"
        assert "let the structure determine the count" in text.lower()
