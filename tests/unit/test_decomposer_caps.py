"""Models choose decomposition counts; code enforces parent and safety limits."""

import json
import re
from collections import Counter

import pytest
from pydantic import ValidationError
from pydantic_ai import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel

from petri.harness.legacy_bridge import HarnessInferenceProvider
from petri.reasoning import prompts
from petri.reasoning.decomposer import decompose_claim
from petri.reasoning.decomposition_agents import DecompositionInput, initial_instructions, why_instructions
from petri.reasoning.decomposition_settings import DecompositionSettings
from tests.conftest import FakeProvider


def candidates(count, prefix):
    return [{"claim_text": f"{prefix} assumption {index}",
             "relation": "rebuts" if index == count - 1 and count > 1 else "supports"}
            for index in range(count)]


def run_script(tmp_path, root_count, child_counts, **limits):
    calls = []
    payloads = [{"nodes": candidates(root_count, "Root")}]
    payloads.extend({"is_atomic": False, "sub_premises": candidates(count, f"Branch {index}")}
                    for index, count in enumerate(child_counts))

    async def response(messages, info):
        calls.append(messages)
        return ModelResponse(parts=[TextPart(json.dumps(payloads[len(calls) - 1]))])

    settings = DecompositionSettings(model="test", **{"depth": 2, "per_layer": 50, **limits})
    with HarnessInferenceProvider(FunctionModel(response)) as runner:
        result = decompose_claim("A scoped claim", [], "dish", "colony", None,
                                 agent_runner=runner, db_path=tmp_path / "dish.sqlite", settings=settings)
    return result, calls


def test_ten_children_are_capped_per_parent_and_counterargument_survives(tmp_path):
    result, calls = run_script(tmp_path, 1, [10])
    children = [cell for cell in result.cells if cell.level == 2]
    assert len(children) == 4
    assert children[-1].relation == "rebuts"
    assert len(calls) == 2


def test_cumulative_layer_ceiling_still_applies(tmp_path):
    result, calls = run_script(tmp_path, 3, [10, 10], per_layer=5)
    assert Counter(cell.level for cell in result.cells) == {0: 1, 1: 3, 2: 5}
    assert len(calls) == 3
    assert [len(cell.dependencies) for cell in result.cells if cell.level == 1] == [4, 1, 0]


def test_root_truncates_to_its_own_cap(tmp_path):
    result, calls = run_script(tmp_path, 15, [], depth=1)
    assert Counter(cell.level for cell in result.cells) == {0: 1, 1: 10}
    assert len(calls) == 1


def test_counts_follow_model_outputs_until_capped(tmp_path):
    result, calls = run_script(tmp_path, 2, [5, 1])
    assert Counter(cell.level for cell in result.cells) == {0: 1, 1: 2, 2: 5}
    assert [len(cell.dependencies) for cell in result.cells if cell.level == 1] == [4, 1]
    assert len(calls) == 3


def test_depth_ceiling_prevents_further_calls(tmp_path):
    result, calls = run_script(tmp_path, 2, [], depth=1)
    assert max(cell.level for cell in result.cells) == 1
    assert len(calls) == 1


@pytest.mark.parametrize("cap", [17, 53, 97])
def test_both_agents_and_legacy_suffixes_have_no_numeric_quota(cap):
    texts = [initial_instructions(DecompositionInput("Scoped claim")),
             why_instructions(DecompositionInput("Scoped premise", parent_level=83, parent_seq=79)),
             prompts.build_decompose_claim_prompt("Scoped claim", [], max_premises=cap),
             prompts.build_decompose_why_prompt("Scoped premise", 83, 79, max_premises=cap),
             prompts.decompose_claim_output_suffix(cap), prompts.decompose_why_output_suffix(cap)]
    for text in texts:
        assert not re.search(rf"\b(?:{cap}|83|79)\b", text)
        assert "Pick the TOP" not in text and "AT MOST" not in text
    assert all("let the structure determine the count; do not pad to a quota; fewer is fine" in text.lower()
               for text in texts[:4])


def test_bridge_schema_contains_no_count_limit():
    instructions = []

    async def response(messages, info):
        instructions.append(info.instructions)
        return ModelResponse(parts=[TextPart(json.dumps({"nodes": [{"claim_text": f"Premise {index}"} for index in range(12)]}))])

    with HarnessInferenceProvider(FunctionModel(response)) as provider:
        result = provider.decompose_claim("claim", [], max_premises=17)
    assert len(result.cells) == 12
    assert "maxItems" not in "".join(instructions)
    assert not re.search(r"\b17\b", "".join(instructions))


@pytest.mark.parametrize("key", ["per_parent_cap", "level_one_cap"])
def test_caps_are_positive_and_fit_the_layer_ceiling(key):
    for value in (0, -1):
        with pytest.raises(ValidationError):
            DecompositionSettings(depth=2, per_layer=5, model="test", **{key: value})
    settings = DecompositionSettings(depth=2, per_layer=2, model="test")
    assert settings.per_parent_cap == settings.level_one_cap == 2


def test_legacy_decomposer_enforces_parent_and_root_caps():
    provider = FakeProvider()
    provider.decompose_response = {"nodes": candidates(15, "Root")}
    provider.why_response = candidates(10, "Branch")
    result = decompose_claim("Claim", [], "dish", "colony", provider,
                             settings=DecompositionSettings(depth=2, per_layer=50, model="test"))
    assert sum(cell.level == 1 for cell in result.cells) == 10
    assert all(len(cell.dependencies) == 4 for cell in result.cells if cell.level == 1)


def test_shipped_config_accepts_caps_and_documents_safety_ceilings():
    from pathlib import Path
    from petri.agents.config import load_agent_config

    path = Path(__file__).parents[2] / "petri/defaults/petri.yaml"
    config = load_agent_config(path)
    assert config.per_parent_cap == 4 and config.level_one_cap == 10
    assert "Safety ceilings only" in path.read_text()
    assert "never model targets" in path.read_text()


def test_dish_caps_are_loaded_afresh(tmp_path):
    path = tmp_path / "defaults" / "petri.yaml"
    path.parent.mkdir()
    path.write_text("max_nodes_per_layer: 20\nper_parent_cap: 2\nlevel_one_cap: 7\n")
    first = DecompositionSettings.from_dish(tmp_path)
    assert (first.per_parent_cap, first.level_one_cap) == (2, 7)
    path.write_text("max_nodes_per_layer: 20\nper_parent_cap: 3\nlevel_one_cap: 6\n")
    second = DecompositionSettings.from_dish(tmp_path)
    assert (second.per_parent_cap, second.level_one_cap) == (3, 6)
