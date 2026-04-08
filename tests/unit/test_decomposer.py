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


# ── per-layer cap (max_nodes_per_layer) ─────────────────────────────────


class TestPerLayerCap:
    """The seed-time decomposer caps each level at ``max_nodes_per_layer``
    so a single seed run never produces 100+ nodes. The cap is enforced
    in two places: the prompt asks for the top N most important premises,
    and the decomposer hard-truncates the model's response as a safety
    net (in case the model ignores the prompt).
    """

    def _force_cap(self, monkeypatch, value: int) -> None:
        """Override the per-layer cap for the duration of one test."""
        import petri.config

        monkeypatch.setattr(
            petri.config,
            "get_max_nodes_per_layer",
            lambda config=None: value,
        )

    def test_level_one_truncated_to_cap(self, monkeypatch):
        """If the model returns 10 level-1 nodes, only the first 5 survive."""
        self._force_cap(monkeypatch, 5)
        provider = FakeProvider()
        provider.decompose_response = {
            "nodes": [
                {"level": 1, "seq": i, "claim_text": f"premise {i}"}
                for i in range(1, 11)
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

        level_one = [n for n in result.nodes if n.level == 1]
        assert len(level_one) == 5

    def test_max_premises_passed_to_decompose_claim(self, monkeypatch):
        """The configured cap must reach the provider as max_premises."""
        self._force_cap(monkeypatch, 7)
        provider = FakeProvider()
        provider.decompose_response = {
            "nodes": [{"level": 1, "seq": 1, "claim_text": "p"}],
            "edges": [],
        }

        decompose_claim(
            claim="a claim",
            clarifications=[],
            dish_id="dish",
            colony_name="colony",
            provider=provider,
        )

        assert provider.decompose_calls[0]["max_premises"] == 7

    def test_five_whys_respects_per_layer_budget(self, monkeypatch):
        """Across multiple parents, the level-2 total never exceeds the cap.

        Three level-1 parents, each ``decompose_why`` returns 4 children.
        With cap=5, the level-2 total must be exactly 5 — not 12.
        """
        self._force_cap(monkeypatch, 5)
        provider = FakeProvider()
        provider.decompose_response = {
            "nodes": [
                {"level": 1, "seq": 1, "claim_text": "premise A"},
                {"level": 1, "seq": 2, "claim_text": "premise B"},
                {"level": 1, "seq": 3, "claim_text": "premise C"},
            ],
            "edges": [],
        }
        provider.why_response = [
            {"claim_text": f"sub-{i}", "is_atomic": True}
            for i in range(1, 5)
        ]

        result = decompose_claim(
            claim="a claim",
            clarifications=[],
            dish_id="dish",
            colony_name="colony",
            provider=provider,
        )

        level_two = [n for n in result.nodes if n.level == 2]
        assert len(level_two) == 5

    def test_remaining_budget_passed_to_decompose_why(self, monkeypatch):
        """Each Five Whys call should receive the remaining level budget,
        not the global cap, so the model knows how many slots are left.
        """
        self._force_cap(monkeypatch, 5)
        provider = FakeProvider()
        provider.decompose_response = {
            "nodes": [
                {"level": 1, "seq": 1, "claim_text": "premise A"},
                {"level": 1, "seq": 2, "claim_text": "premise B"},
            ],
            "edges": [],
        }
        provider.why_response = [
            {"claim_text": "sub-1", "is_atomic": True},
            {"claim_text": "sub-2", "is_atomic": True},
            {"claim_text": "sub-3", "is_atomic": True},
        ]

        decompose_claim(
            claim="a claim",
            clarifications=[],
            dish_id="dish",
            colony_name="colony",
            provider=provider,
        )

        # First decompose_why call: full budget of 5
        assert provider.why_calls[0]["max_premises"] == 5
        # After 3 children consumed, second call gets remaining budget of 2
        assert provider.why_calls[1]["max_premises"] == 2

    def test_skips_decompose_why_when_layer_full(self, monkeypatch):
        """Once a layer is full, subsequent Five Whys calls for that
        layer must be skipped entirely (no wasted LLM round-trips).
        """
        self._force_cap(monkeypatch, 3)
        provider = FakeProvider()
        provider.decompose_response = {
            "nodes": [
                {"level": 1, "seq": 1, "claim_text": "premise A"},
                {"level": 1, "seq": 2, "claim_text": "premise B"},
                {"level": 1, "seq": 3, "claim_text": "premise C"},
            ],
            "edges": [],
        }
        provider.why_response = [
            {"claim_text": f"sub-{i}", "is_atomic": True}
            for i in range(1, 5)
        ]

        decompose_claim(
            claim="a claim",
            clarifications=[],
            dish_id="dish",
            colony_name="colony",
            provider=provider,
        )

        # First call alone fills level-2 to the cap of 3 → at most 1 why call
        assert len(provider.why_calls) == 1


# ── on_progress + on_node_created callbacks ──────────────────────────────


class TestOnProgress:
    def test_threads_progress_chunks_to_caller(self):
        """Chunks emitted by the provider must reach the caller's callback.

        Use a single level-1 premise so the call count is deterministic:
        1 decompose_claim call + 1 decompose_why call = 2 LLM calls total,
        each emitting the canned chunks.
        """
        provider = FakeProvider()
        provider.progress_chunks = ["thinking-1", "thinking-2"]
        provider.decompose_response = {
            "nodes": [{"level": 1, "seq": 1, "claim_text": "single premise"}],
            "edges": [],
        }
        recorded: list[str] = []

        decompose_claim(
            claim="a claim",
            clarifications=[],
            dish_id="dish",
            colony_name="colony",
            provider=provider,
            on_progress=recorded.append,
        )

        # 2 LLM calls × 2 canned chunks = 4 forwarded chunks, in order
        assert recorded == [
            "thinking-1",
            "thinking-2",
            "thinking-1",
            "thinking-2",
        ]

    def test_threads_progress_through_five_whys(self):
        """Each Five Whys iteration must forward chunks to the same callback."""
        provider = FakeProvider()
        provider.progress_chunks = ["chunk"]
        provider.decompose_response = {
            "nodes": [
                {"level": 1, "seq": 1, "claim_text": "premise A"},
                {"level": 1, "seq": 2, "claim_text": "premise B"},
            ],
            "edges": [],
        }
        # decompose_why returns one atomic sub-premise per call so the loop
        # runs but doesn't recurse forever.
        provider.why_response = [
            {"claim_text": "sub-premise", "is_atomic": True}
        ]
        recorded: list[str] = []

        decompose_claim(
            claim="a claim",
            clarifications=[],
            dish_id="dish",
            colony_name="colony",
            provider=provider,
            on_progress=recorded.append,
        )

        # 1 chunk from decompose_claim + 1 chunk per decompose_why call
        # (one per level-1 premise = 2 calls). All forwarded to the same
        # callback, no developer-authored prefixes interleaved.
        assert recorded == ["chunk", "chunk", "chunk"]

    def test_progress_optional(self):
        provider = FakeProvider()
        provider.progress_chunks = ["a", "b"]
        # Must not raise even though progress_chunks is set
        decompose_claim(
            claim="a claim",
            clarifications=[],
            dish_id="dish",
            colony_name="colony",
            provider=provider,
        )


class TestOnNodeCreated:
    def _three_layer_provider(self) -> FakeProvider:
        """Provider that yields 2 level-1 premises and one atomic
        sub-premise for each, so the Five Whys loop runs once per premise.
        """
        provider = FakeProvider()
        provider.decompose_response = {
            "nodes": [
                {"level": 1, "seq": 1, "claim_text": "premise A"},
                {"level": 1, "seq": 2, "claim_text": "premise B"},
            ],
            "edges": [],
        }
        provider.why_response = [
            {"claim_text": "deeper premise", "is_atomic": True}
        ]
        return provider

    def test_called_for_every_node_except_center(self):
        provider = self._three_layer_provider()
        recorded: list = []

        result = decompose_claim(
            claim="a claim",
            clarifications=[],
            dish_id="dish",
            colony_name="colony",
            provider=provider,
            on_node_created=lambda node, edges: recorded.append((node, edges)),
        )

        # Center is created by the CLI before decompose_claim runs, so the
        # callback fires for every node *except* the center.
        non_center = [n for n in result.nodes if n.level > 0]
        assert len(recorded) == len(non_center)
        assert {n.id for n, _ in recorded} == {n.id for n in non_center}

    def test_called_in_topological_order_with_correct_parent_edges(self):
        """A child must never appear in the callback before its parent."""
        provider = self._three_layer_provider()
        seen_ids: list[str] = []

        def _callback(node, edges):
            # Every parent referenced in edges must already have been seen
            # (or be the center, which the CLI creates upfront).
            for edge in edges:
                assert (
                    edge.from_node in seen_ids
                    or edge.from_node.endswith("-000-000")
                ), f"child {node.id} appeared before parent {edge.from_node}"
            seen_ids.append(node.id)

        decompose_claim(
            claim="a claim",
            clarifications=[],
            dish_id="dish",
            colony_name="colony",
            provider=provider,
            on_node_created=_callback,
        )

        assert len(seen_ids) > 0

    def test_callback_optional(self):
        provider = FakeProvider()
        # Must not raise without on_node_created
        result = decompose_claim(
            claim="a claim",
            clarifications=[],
            dish_id="dish",
            colony_name="colony",
            provider=provider,
        )
        assert len(result.nodes) >= 1


# ── caller-supplied center node (bottom-up inversion fix) ────────────────


class TestCallerSuppliedCenter:
    """The CLI's Phase C creates its own ``center_node`` object and then
    calls ``decompose_claim``. Before the fix, the decomposer constructed
    its *own* ``center = Node(...)`` internally and wired dependencies on
    that local object — leaving the CLI's ``center_node`` with an empty
    dependencies list, so the engine treated the center as a leaf. The
    new ``center`` parameter lets the CLI pass in its own Node and have
    the decomposer mutate it in place.
    """

    def test_decompose_claim_populates_passed_center_dependencies(self):
        """When a ``center`` Node is passed in, its dependencies must be
        populated with the IDs of every level-1 node built from the LLM
        response — mutating the caller's object in place.
        """
        from petri.models import Node, build_node_key

        provider = FakeProvider()
        provider.decompose_response = {
            "nodes": [
                {"level": 1, "seq": 1, "claim_text": "premise alpha"},
                {"level": 1, "seq": 2, "claim_text": "premise beta"},
                {"level": 1, "seq": 3, "claim_text": "premise gamma"},
            ],
            "edges": [],
        }

        dish_id = "dish"
        colony_name = "colony"
        center_id = build_node_key(dish_id, colony_name, 0, 0)
        caller_center = Node(
            id=center_id,
            colony_id=f"{dish_id}-{colony_name}",
            claim_text="a claim",
            level=0,
            dependencies=[],
        )

        result = decompose_claim(
            claim="a claim",
            clarifications=[],
            dish_id=dish_id,
            colony_name=colony_name,
            provider=provider,
            center=caller_center,
        )

        # The exact Python object passed in must have been mutated — not
        # a different Node with the same id.
        level_one_nodes = [
            node for node in result.nodes if node.level == 1
        ]
        assert len(level_one_nodes) == 3
        expected_ids = {node.id for node in level_one_nodes}
        assert set(caller_center.dependencies) == expected_ids

        # And the DecompositionResult's level-0 node must be the very
        # same object the caller handed in (identity, not equality).
        result_center = next(
            node for node in result.nodes if node.level == 0
        )
        assert result_center is caller_center

    def test_decompose_claim_creates_center_when_none_passed(self):
        """Backwards-compat: omitting ``center`` must still produce a
        level-0 node with correct dependencies.
        """
        provider = FakeProvider()
        provider.decompose_response = {
            "nodes": [
                {"level": 1, "seq": 1, "claim_text": "premise one"},
                {"level": 1, "seq": 2, "claim_text": "premise two"},
            ],
            "edges": [],
        }

        result = decompose_claim(
            claim="another claim",
            clarifications=[],
            dish_id="dish",
            colony_name="colony",
            provider=provider,
        )

        level_zero_nodes = [
            node for node in result.nodes if node.level == 0
        ]
        assert len(level_zero_nodes) == 1
        internal_center = level_zero_nodes[0]
        assert internal_center.claim_text == "another claim"

        level_one_ids = {
            node.id for node in result.nodes if node.level == 1
        }
        assert len(level_one_ids) == 2
        assert set(internal_center.dependencies) == level_one_ids


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
