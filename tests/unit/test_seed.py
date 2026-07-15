"""Regression tests for petri seed helpers."""

from petri.cli.seed import _clarifications_to_models
from petri.models import ClarifyingQuestion


def test_clarification_dicts_convert_to_models():
    """seed stores clarifications as dicts, but decompose_claim reads
    .question/.answer/.options on ClarifyingQuestion objects. Passing raw
    dicts crashed with AttributeError once a user answered a question; the
    converter must turn the stored dicts into models."""
    stored = [
        {"question": "What scale?", "answer": "national"},  # no options key
        {"question": "Which year?", "answer": "", "options": ["2023", "2024"]},
    ]
    assert _clarifications_to_models(stored) == [
        ClarifyingQuestion(question="What scale?", answer="national", options=[]),
        ClarifyingQuestion(
            question="Which year?", answer="", options=["2023", "2024"]
        ),
    ]


def test_clarifications_empty_list():
    assert _clarifications_to_models([]) == []
