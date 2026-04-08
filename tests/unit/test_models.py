"""Unit tests for petri/models.py — domain constraints, key utilities, and protocols."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from petri.models import (
    DecompositionResult,
    Edge,
    Event,
    EventType,
    Node,
    NodeStatus,
    QueueEntry,
    InferenceProvider,
    SearchExecutedData,
    build_event_key,
    build_node_key,
    parent_key,
    parse_key,
    validate_event_data,
    validate_slug,
)


# ── Domain Constraint Tests ──────────────────────────────────────────────
# These test constraints specific to Petri's domain logic (ge=0, enums, etc).


class TestSearchExecutedData:
    def test_negative_sources_rejected(self):
        with pytest.raises(ValidationError):
            SearchExecutedData(query="q", sources_found=-1)


class TestEvent:
    def test_negative_iteration_rejected(self):
        with pytest.raises(ValidationError):
            Event(
                id="evt-001",
                node_id="n1",
                timestamp="2025-06-01T00:00:00Z",
                type=EventType.search_executed,
                agent="researcher",
                iteration=-1,
                data={},
            )

    def test_invalid_event_type(self):
        with pytest.raises(ValidationError):
            Event(
                id="evt-001",
                node_id="n1",
                timestamp="2025-06-01T00:00:00Z",
                type="not_a_real_event_type",
                agent="researcher",
                iteration=0,
                data={},
            )


class TestNode:
    def test_negative_level_rejected(self):
        with pytest.raises(ValidationError):
            Node(
                id="n",
                colony_id="c",
                claim_text="x",
                level=-1,
            )


class TestQueueEntry:
    def test_negative_iteration_rejected(self):
        with pytest.raises(ValidationError):
            QueueEntry(node_id="n1", iteration=-1)

    def test_max_iterations_zero_rejected(self):
        with pytest.raises(ValidationError):
            QueueEntry(node_id="n1", max_iterations=0)

    def test_negative_cycle_start_rejected(self):
        with pytest.raises(ValidationError):
            QueueEntry(node_id="n1", cycle_start_iteration=-1)


class TestDecompositionResult:
    def test_composes_node_and_edge(self):
        node = Node(id="n1", colony_id="c1", claim_text="x", level=0)
        edge = Edge(from_node="n1", to_node="n2")
        decomposition_result = DecompositionResult(
            nodes=[node],
            edges=[edge],
            colony_name="market",
            center_claim="Market claim",
        )
        assert len(decomposition_result.nodes) == 1
        assert len(decomposition_result.edges) == 1
        assert decomposition_result.colony_name == "market"


# ── validate_event_data Tests ────────────────────────────────────────────
# These test the dispatch function that routes event types to Pydantic models.


class TestValidateEventData:
    def test_valid_search_executed(self):
        result = validate_event_data(
            "search_executed", {"query": "test", "sources_found": 5}
        )
        assert result == {"query": "test", "sources_found": 5}

    def test_valid_verdict_issued(self):
        result = validate_event_data(
            "verdict_issued", {"verdict": "confirmed", "summary": "OK"}
        )
        assert result["verdict"] == "confirmed"

    def test_valid_evidence_appended(self):
        result = validate_event_data(
            "evidence_appended", {"summary": "New data"}
        )
        assert result["summary"] == "New data"

    def test_valid_debate_mediated(self):
        result = validate_event_data(
            "debate_mediated",
            {"from_agent": "a", "to_agent": "b", "exchange_summary": "s"},
        )
        assert result["from_agent"] == "a"

    def test_valid_convergence_checked(self):
        result = validate_event_data(
            "convergence_checked", {"converged": True}
        )
        assert result["converged"] is True
        assert "blocking_verdicts" not in result
        assert "weakest_link" not in result

    def test_valid_node_reopened(self):
        result = validate_event_data(
            "node_reopened",
            {"trigger": "new_evidence", "prior_status": "VALIDATED"},
        )
        assert result["trigger"] == "new_evidence"

    def test_valid_propagation_triggered(self):
        result = validate_event_data(
            "propagation_triggered",
            {"reopened_node_id": "n1", "flagged_dependents": ["n2"]},
        )
        assert result["flagged_dependents"] == ["n2"]

    def test_valid_decomposition_created(self):
        result = validate_event_data(
            "decomposition_created",
            {"parent_node_id": "p1", "child_node_ids": ["c1"]},
        )
        assert result["parent_node_id"] == "p1"

    def test_unknown_event_type_raises(self):
        with pytest.raises(ValueError, match="Unknown event type"):
            validate_event_data("nonexistent_event", {})

    def test_invalid_data_for_known_type(self):
        with pytest.raises(ValidationError):
            validate_event_data("search_executed", {"query": "q"})

    def test_invalid_negative_value(self):
        with pytest.raises(ValidationError):
            validate_event_data(
                "search_executed", {"query": "q", "sources_found": -1}
            )


# ── Composite Key Utility Tests ──────────────────────────────────────────


class TestBuildNodeKey:
    def test_basic(self):
        assert (
            build_node_key("research1", "market", 2, 3)
            == "research1-market-002-003"
        )

    def test_zero_padding(self):
        assert (
            build_node_key("my-dish", "ai-eval", 0, 0)
            == "my-dish-ai-eval-000-000"
        )

    def test_large_numbers(self):
        assert (
            build_node_key("d", "c", 100, 999)
            == "d-c-100-999"
        )


class TestBuildEventKey:
    def test_basic(self):
        assert (
            build_event_key("research1-market-002-003", "a1b2c3d4")
            == "research1-market-002-003-a1b2c3d4"
        )


class TestParseKey:
    def test_node_key(self):
        result = parse_key("research1-market-002-003")
        assert result["raw"] == "research1-market-002-003"
        assert result["level"] == 2
        assert result["seq"] == 3
        assert result["colony_prefix"] == "research1-market"
        assert "event_hex" not in result

    def test_node_key_with_dish_id(self):
        result = parse_key("research1-market-002-003", dish_id="research1")
        assert result["dish"] == "research1"
        assert result["colony"] == "market"
        assert result["level"] == 2
        assert result["seq"] == 3

    def test_hyphenated_dish_id(self):
        result = parse_key(
            "my-research-ai-eval-001-002", dish_id="my-research"
        )
        assert result["dish"] == "my-research"
        assert result["colony"] == "ai-eval"
        assert result["level"] == 1
        assert result["seq"] == 2

    def test_event_key(self):
        result = parse_key("research1-market-002-003-a1b2c3d4")
        assert result["event_hex"] == "a1b2c3d4"
        assert result["level"] == 2
        assert result["seq"] == 3
        assert result["colony_prefix"] == "research1-market"

    def test_raw_always_present(self):
        result = parse_key("anything")
        assert result["raw"] == "anything"

    def test_short_key_no_crash(self):
        result = parse_key("a-b")
        assert result["raw"] == "a-b"
        assert "level" not in result


class TestParentKey:
    def test_event_to_node(self):
        assert (
            parent_key("research1-market-002-003-a1b2c3d4")
            == "research1-market-002-003"
        )

    def test_node_to_colony_level(self):
        assert parent_key("research1-market-002-003") == "research1-market-002"

    def test_single_segment(self):
        assert parent_key("nosegments") == "nosegments"


# ── Slug Validation Tests ────────────────────────────────────────────────


class TestValidateSlug:
    @pytest.mark.parametrize(
        "slug",
        ["research1", "my-research", "ai-eval", "abc", "a1b2", "hello-world-test"],
    )
    def test_valid_slugs(self, slug):
        assert validate_slug(slug) is True

    @pytest.mark.parametrize(
        "slug,reason",
        [
            ("123", "no letters"),
            ("", "empty string"),
            ("My-Research", "uppercase"),
            ("research_1", "underscores"),
            ("-leading", "leading hyphen"),
            ("trailing-", "trailing hyphen"),
            ("double--hyphen", "consecutive hyphens"),
            ("ALLCAPS", "all uppercase"),
            ("has space", "spaces"),
        ],
    )
    def test_invalid_slugs(self, slug, reason):
        assert validate_slug(slug) is False, f"Should reject: {reason}"


# ── InferenceProvider Protocol Tests ──────────────────────────────────────


class TestInferenceProviderProtocol:
    def test_conforming_class_satisfies_protocol(self):
        class GoodProvider:
            def assess_claim_substance(self, claim: str) -> dict:
                return {
                    "is_substantive": True,
                    "reason": "",
                    "suggested_rewrite": "",
                }

            def generate_clarifying_questions(
                self, claim: str, max_questions: int = 5
            ) -> list[dict]:
                return []

            def decompose_claim(
                self,
                claim: str,
                clarifications: list[dict],
                guidance: str = "",
            ) -> dict:
                return {"nodes": [], "edges": []}

            def assess_node(
                self,
                node_id: str,
                claim_text: str,
                context: dict,
                agent_role: str,
            ) -> dict:
                return {}

            def match_evidence(
                self, content: str, nodes: list[dict]
            ) -> list[dict]:
                return []

        assert isinstance(GoodProvider(), InferenceProvider)
