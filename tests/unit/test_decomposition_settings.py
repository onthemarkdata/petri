"""Dish limits remain independent within one running process."""

from pathlib import Path

import pytest

from petri.config import load_config
from petri.reasoning.decomposer import _provider_decompose
from petri.reasoning.decomposition_settings import DecompositionSettings
from tests.conftest import FakeProvider


def test_two_dishes_and_edits_do_not_share_cached_settings(tmp_path):
    dishes = []
    for name, depth, cap in [("first", 1, 2), ("second", 4, 7)]:
        dish = tmp_path / name / ".petri"
        (dish / "defaults").mkdir(parents=True)
        (dish / "defaults" / "petri.yaml").write_text(
            f"max_decomposition_depth: {depth}\nmax_nodes_per_layer: {cap}\nmodel: test:{name}\n"
        )
        dishes.append(dish)
    first, second = map(DecompositionSettings.from_dish, dishes)
    assert (first.depth, first.per_layer, first.model) == (1, 2, "test:first")
    assert (second.depth, second.per_layer, second.model) == (4, 7, "test:second")
    (dishes[0] / "defaults" / "petri.yaml").write_text("max_decomposition_depth: 8\n")
    assert DecompositionSettings.from_dish(dishes[0]).depth == 8
    assert DecompositionSettings.from_dish(dishes[1]) == second


def test_explicit_missing_config_never_falls_back(tmp_path):
    with pytest.raises(FileNotFoundError, match="Petri config not found"):
        load_config(tmp_path / "missing" / "petri.yaml")


@pytest.mark.parametrize("depth,cap", [(1, 1), (2, 2), (4, 3)])
def test_provider_uses_injected_depth_and_layer_cap(depth, cap):
    provider = FakeProvider()
    provider.decompose_response = {
        "nodes": [{"claim_text": f"Premise {n}"} for n in range(10)], "edges": [],
    }
    provider.why_response = [{"claim_text": "Subpremise", "is_atomic": False}]
    result = _provider_decompose(
        "Claim", [], "dish", "colony", provider,
        settings=DecompositionSettings(depth=depth, per_layer=cap, model="test:model"),
    )
    assert max(cell.level for cell in result.cells) == depth
    assert sum(cell.level == 1 for cell in result.cells) == cap
    assert all(sum(cell.level == level for cell in result.cells) <= cap
               for level in range(1, depth + 1))
    assert provider.decompose_calls[0]["max_premises"] == 5
    assert bool(provider.why_calls) == (depth > 1)


def test_decomposer_has_no_global_limit_getters():
    source = (Path(__file__).parents[2] / "petri/reasoning/decomposer.py").read_text()
    assert "get_max_decomposition_depth" not in source
    assert "get_max_nodes_per_layer" not in source
