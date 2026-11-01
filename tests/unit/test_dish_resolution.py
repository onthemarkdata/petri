"""All entry points use one dish identity and one composite-key parser."""

import ast
from pathlib import Path

import pytest

from petri.cli._bootstrap import get_dish_id
from petri.dashboard.api import _get_dish_id as dashboard_dish
from petri.models import parse_key
from petri.storage.dish import resolve_dish_id
from petri.storage.paths import parse_cell_id


def test_all_entry_points_share_resolver(tmp_path):
    petri_dir = tmp_path / ".petri"
    (petri_dir / "defaults").mkdir(parents=True)
    resolvers = [get_dish_id, dashboard_dish]
    assert all(resolver is resolve_dish_id for resolver in resolvers)
    assert resolve_dish_id(petri_dir) == tmp_path.name
    (petri_dir / "petri.yaml").write_text('name: "old-dish"\n')
    assert resolve_dish_id(petri_dir) == "old-dish"
    (petri_dir / "defaults" / "petri.yaml").write_text('name: "my-dish" # comment\n')
    assert {resolver(petri_dir) for resolver in resolvers} == {"my-dish"}
    (petri_dir / "defaults" / "petri.yaml").write_text("model: test\n")
    assert resolve_dish_id(petri_dir) == tmp_path.name


def test_hyphenated_dish_and_colony():
    cell_id = "my-research-dish-counter-claim-002-007"
    result = parse_key(cell_id, "my-research-dish")
    assert (result["dish"], result["colony"], result["level"], result["seq"]) == (
        "my-research-dish", "counter-claim", 2, 7,
    )
    with pytest.warns(DeprecationWarning, match="parse_key"):
        assert parse_cell_id(cell_id, "my-research-dish") == (
            "my-research-dish", "counter-claim", 2, 7,
        )


def test_no_duplicate_resolvers_or_suffix_parsers():
    root = Path(__file__).parents[2] / "petri"
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                assert node.name not in {"get_dish_id", "_get_dish_id"}, path
            if path.name != "models.py" and isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute) and node.func.attr == "rsplit":
                    assert not (node.args and isinstance(node.args[0], ast.Constant)
                                and node.args[0].value == "-"), path
