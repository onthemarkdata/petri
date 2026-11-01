"""Dish search keeps exact matches and deterministic lexical rankings."""

import pytest

from petri.reasoning.cell_search import CellSearchCandidate, search
from petri.storage import database


@pytest.fixture
def search_dish(tmp_path):
    path = tmp_path / "dish.sqlite"
    for colony, claims in [("foundation", ["Energy research", "Solar panels generate electricity",
                                          "Wind turbines generate electricity", "Batteries store electricity", "Inverters convert power"]),
                           ("delivery", ["Logistics research", "Solar panels reduce imports", "Delivery depends on roads",
                                         "Road traffic limits delivery", "Inverters convert power"])]:
        for level, claim in enumerate(claims):
            database.put_cell(path, {"id": f"dish-{colony}-{level:03d}-001", "colony": f"dish-{colony}",
                              "level": level, "seq": 1, "claim_text": claim, "created_at": "2026-11-01T00:00:00Z"})
    return path


def test_exact_phrase_retrieves_the_specific_cell(search_dish):
    result = search(search_dish, "Solar panels generate electricity", 3)
    assert isinstance(result[0], CellSearchCandidate)
    assert result[0].id == "dish-foundation-001-001"
    assert result[0].level == 1 and result[0].score == 1
    assert [cell.claim_text for cell in result[1:]] == [
        "Solar panels reduce imports", "Wind turbines generate electricity",
    ]
    assert result[1].score == result[2].score


def test_deterministic_ties_and_case_punctuation_normalization(search_dish):
    result = search(search_dish, "INVERTERS, convert POWER!", 10)
    assert [cell.id for cell in result if cell.score == 1] == [
        "dish-delivery-004-001", "dish-foundation-004-001",
    ]
    assert result == search(search_dish, "INVERTERS, convert POWER!", 10)


def test_current_colony_can_be_excluded_before_ranking(search_dish):
    assert search(search_dish, "Inverters convert power", 1,
                  exclude_colony="dish-delivery")[0].colony == "dish-foundation"


def test_empty_no_match_and_invalid_limits(search_dish, tmp_path):
    assert search(search_dish, "!!!") == []
    assert search(search_dish, "unrelatedword") == []
    assert search(tmp_path / "absent.sqlite", "query") == []
    for limit in (0, -1, True):
        with pytest.raises(ValueError):
            search(search_dish, "query", limit)
