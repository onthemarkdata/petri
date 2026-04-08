"""Unit tests for petri.reasoning.decomposer.

These tests use dependency injection — the FakeProvider from
``tests/conftest.py`` is passed directly to functions under test. No
monkeypatching, no real ``claude`` CLI calls. Anything that would require
simulating the real Claude Code CLI through monkeypatching is verified as a
manual check, not here.
"""

from __future__ import annotations

import pytest

from petri.reasoning.decomposer import (
    decompose_claim,
    generate_clarifying_questions,
    generate_colony_name,
)
from tests.conftest import FakeProvider


# ── generate_clarifying_questions ────────────────────────────────────────


class TestGenerateClarifyingQuestions:
    def test_requires_provider(self):
        """No hardcoded fallback — None provider must raise."""
        with pytest.raises(ValueError, match="requires an InferenceProvider"):
            generate_clarifying_questions("any claim", provider=None)

    def test_delegates_to_provider(self):
        provider = FakeProvider()
        provider.questions_response = [
            {"question": "Q1?", "options": ["A", "B"]},
            {"question": "Q2?", "options": []},
        ]
        result = generate_clarifying_questions("a claim", provider=provider)

        assert len(result) == 2
        assert result[0].question == "Q1?"
        assert result[0].options == ["A", "B"]
        assert result[1].question == "Q2?"
        assert result[1].options == []
        # Recorded the call so the test can assert the contract
        assert provider.questions_calls == [("a claim", 5)]

    def test_respects_max_questions(self):
        provider = FakeProvider()
        provider.questions_response = [
            {"question": f"Q{i}?", "options": []} for i in range(10)
        ]
        result = generate_clarifying_questions(
            "claim", provider=provider, max_questions=3
        )
        assert len(result) == 3


# ── decompose_claim ──────────────────────────────────────────────────────


class TestDecomposeClaim:
    def test_requires_provider(self):
        """No hardcoded fallback — None provider must raise."""
        with pytest.raises(ValueError, match="requires an InferenceProvider"):
            decompose_claim(
                claim="any claim",
                clarifications=[],
                dish_id="d",
                colony_name="c",
                provider=None,
            )

    def test_threads_guidance_to_provider(self):
        """Free-text regenerate guidance must reach the provider."""
        provider = FakeProvider()
        decompose_claim(
            claim="a claim",
            clarifications=[],
            dish_id="dish",
            colony_name="colony",
            provider=provider,
            guidance="focus on measurement methodology",
        )

        assert len(provider.decompose_calls) == 1
        recorded = provider.decompose_calls[0]
        assert recorded["claim"] == "a claim"
        assert recorded["guidance"] == "focus on measurement methodology"

    def test_empty_guidance_default(self):
        provider = FakeProvider()
        decompose_claim(
            claim="a claim",
            clarifications=[],
            dish_id="dish",
            colony_name="colony",
            provider=provider,
        )
        assert provider.decompose_calls[0]["guidance"] == ""

    def test_clarifications_serialized_to_provider(self):
        from petri.models import ClarifyingQuestion

        provider = FakeProvider()
        clarifications = [
            ClarifyingQuestion(
                question="Q1?", options=["A", "B"], answer="A"
            ),
            ClarifyingQuestion(question="Q2?", options=[], answer="free text"),
        ]
        decompose_claim(
            claim="claim",
            clarifications=clarifications,
            dish_id="dish",
            colony_name="colony",
            provider=provider,
        )

        recorded_clarifications = provider.decompose_calls[0]["clarifications"]
        assert recorded_clarifications == [
            {"question": "Q1?", "answer": "A", "options": ["A", "B"]},
            {"question": "Q2?", "answer": "free text", "options": []},
        ]

    def test_raises_when_provider_returns_no_nodes(self):
        """The deleted ``_default_decompose`` fallback must NOT be reinstated.

        When the LLM returns nothing usable, the call must fail loudly so the
        user can regenerate with guidance — not silently produce a generic
        boilerplate tree.
        """
        provider = FakeProvider()
        provider.decompose_response = {"nodes": [], "edges": []}

        with pytest.raises(RuntimeError, match="no level-1 premises"):
            decompose_claim(
                claim="a claim",
                clarifications=[],
                dish_id="dish",
                colony_name="colony",
                provider=provider,
            )

    def test_raises_when_provider_only_returns_center(self):
        """A response with only a level-0 node is not enough."""
        provider = FakeProvider()
        provider.decompose_response = {
            "nodes": [{"level": 0, "seq": 0, "claim_text": "claim"}],
            "edges": [],
        }
        with pytest.raises(RuntimeError, match="no level-1 premises"):
            decompose_claim(
                claim="a claim",
                clarifications=[],
                dish_id="dish",
                colony_name="colony",
                provider=provider,
            )

    def test_builds_center_and_level_one_nodes(self):
        provider = FakeProvider()
        provider.decompose_response = {
            "nodes": [
                {"level": 1, "seq": 1, "claim_text": "first premise"},
                {"level": 1, "seq": 2, "claim_text": "second premise"},
            ],
            "edges": [],
        }
        result = decompose_claim(
            claim="a claim",
            clarifications=[],
            dish_id="dish",
            colony_name="colony",
            provider=provider,
        )

        # 1 center + 2 level-1 nodes
        assert len(result.nodes) == 3
        center = result.nodes[0]
        assert center.level == 0
        assert center.claim_text == "a claim"

        level_one = [n for n in result.nodes if n.level == 1]
        assert len(level_one) == 2
        assert {n.claim_text for n in level_one} == {
            "first premise",
            "second premise",
        }

        # Center depends on both level-1 nodes
        assert sorted(center.dependencies) == sorted(n.id for n in level_one)
        # Each level-1 node lists center as a dependent
        for node in level_one:
            assert node.dependents == [center.id]

        # Edges go center → each level-1 node
        assert len(result.edges) == 2
        for edge in result.edges:
            assert edge.from_node == center.id
            assert edge.to_node in {n.id for n in level_one}


# ── generate_colony_name (unchanged behaviour, smoke check) ──────────────


class TestGenerateColonyName:
    def test_strips_stop_words(self):
        assert generate_colony_name("The quick brown fox") == "quick-brown-fox"

    def test_falls_back_to_all_words_when_only_stop_words(self):
        # When stop-word filtering removes everything, fall back to all words
        assert generate_colony_name("the a an") == "the-a-an"

    def test_truncates_to_30_chars(self):
        name = generate_colony_name(
            "alpha bravo charlie delta echo foxtrot golf hotel"
        )
        assert len(name) <= 30

    def test_returns_colony_for_empty_input(self):
        assert generate_colony_name("") == "colony"
