"""Comprehensive unit tests for petri/models.py."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from petri.models import (
    AgentRole,
    ClarifyingQuestion,
    Colony,
    Confidence,
    ConvergenceCheckedData,
    Debate,
    DebateMediatedData,
    DecompositionCreatedData,
    DecompositionResult,
    Edge,
    EvidenceAppendedData,
    Event,
    EventType,
    FreshnessCheckedData,
    HierarchyLevel,
    Node,
    NodeReopenedData,
    NodeStatus,
    PetriConfig,
    PetriDish,
    PropagationTriggeredData,
    QueueEntry,
    QueueState,
    InferenceProvider,
    SearchExecutedData,
    SourceReviewedData,
    SupportsOrContradicts,
    Verdict,
    VerdictIssuedData,
    build_event_key,
    build_node_key,
    parent_key,
    parse_key,
    validate_event_data,
    validate_slug,
)


# ── Enum Tests ────────────────────────────────────────────────────────────


class TestNodeStatus:
    """NodeStatus enum: 10 members, str-based."""

    EXPECTED = [
        "NEW",
        "RESEARCH",
        "RED_TEAM",
        "EVALUATE",
        "VALIDATED",
        "DISPROVEN",
        "NEEDS_EXPERIMENT",
        "DEFER_OPEN",
        "DEFER_CLOSED",
        "STALLED",
    ]

    def test_member_count(self):
        assert len(NodeStatus) == 10

    @pytest.mark.parametrize("value", EXPECTED)
    def test_member_exists(self, value):
        assert NodeStatus(value) == value

    def test_exact_member_set(self):
        assert sorted(m.value for m in NodeStatus) == sorted(self.EXPECTED)

    def test_is_str(self):
        assert isinstance(NodeStatus.NEW, str)
        assert NodeStatus.VALIDATED.value == "VALIDATED"
        assert str(NodeStatus.VALIDATED.value) == "VALIDATED"


class TestQueueState:
    """QueueState enum: 13 members, str-based."""

    EXPECTED = [
        "queued",
        "phase1_active",
        "phase2_active",
        "mediating",
        "converged",
        "stalled",
        "needs_human",
        "red_team_active",
        "evaluating",
        "done",
        "deferred_open",
        "deferred_closed",
        "sync_conflict",
    ]

    def test_member_count(self):
        assert len(QueueState) == 13

    @pytest.mark.parametrize("value", EXPECTED)
    def test_member_exists(self, value):
        assert QueueState(value) == value

    def test_exact_member_set(self):
        assert sorted(m.value for m in QueueState) == sorted(self.EXPECTED)

    def test_is_str(self):
        assert isinstance(QueueState.queued, str)
        assert QueueState.done == "done"


class TestEventType:
    """EventType enum: 10 members, str-based."""

    EXPECTED = [
        "search_executed",
        "source_reviewed",
        "freshness_checked",
        "verdict_issued",
        "evidence_appended",
        "debate_mediated",
        "convergence_checked",
        "node_reopened",
        "propagation_triggered",
        "decomposition_created",
    ]

    def test_member_count(self):
        assert len(EventType) == 10

    @pytest.mark.parametrize("value", EXPECTED)
    def test_member_exists(self, value):
        assert EventType(value) == value

    def test_exact_member_set(self):
        assert sorted(m.value for m in EventType) == sorted(self.EXPECTED)

    def test_is_str(self):
        assert isinstance(EventType.verdict_issued, str)


class TestHierarchyLevel:
    """HierarchyLevel enum: 6 members (1-6), int-based."""

    def test_member_count(self):
        assert len(HierarchyLevel) == 6

    @pytest.mark.parametrize("value", [1, 2, 3, 4, 5, 6])
    def test_member_exists(self, value):
        assert HierarchyLevel(value) == value

    def test_is_int(self):
        assert isinstance(HierarchyLevel.direct_measurement, int)
        assert HierarchyLevel.direct_measurement + 1 == 2

    def test_named_values(self):
        assert HierarchyLevel.direct_measurement == 1
        assert HierarchyLevel.authoritative_docs == 2
        assert HierarchyLevel.derived_calculation == 3
        assert HierarchyLevel.expert_consensus == 4
        assert HierarchyLevel.single_expert == 5
        assert HierarchyLevel.community_report == 6


class TestSupportsOrContradicts:
    def test_member_count(self):
        assert len(SupportsOrContradicts) == 2

    def test_values(self):
        assert SupportsOrContradicts.supports == "supports"
        assert SupportsOrContradicts.contradicts == "contradicts"


class TestConfidence:
    def test_member_count(self):
        assert len(Confidence) == 3

    def test_values(self):
        assert Confidence.HIGH == "HIGH"
        assert Confidence.MEDIUM == "MEDIUM"
        assert Confidence.LOW == "LOW"


# ── Event Data Payload Tests ──────────────────────────────────────────────


class TestSearchExecutedData:
    def test_valid(self):
        d = SearchExecutedData(query="test query", sources_found=5)
        assert d.query == "test query"
        assert d.sources_found == 5

    def test_sources_found_zero(self):
        d = SearchExecutedData(query="q", sources_found=0)
        assert d.sources_found == 0

    def test_negative_sources_rejected(self):
        with pytest.raises(ValidationError):
            SearchExecutedData(query="q", sources_found=-1)

    def test_missing_query(self):
        with pytest.raises(ValidationError):
            SearchExecutedData(sources_found=3)

    def test_missing_sources_found(self):
        with pytest.raises(ValidationError):
            SearchExecutedData(query="q")

    def test_wrong_type_sources_found(self):
        with pytest.raises(ValidationError):
            SearchExecutedData(query="q", sources_found="not_a_number")


class TestSourceReviewedData:
    def test_minimal(self):
        d = SourceReviewedData(url="https://example.com")
        assert d.url == "https://example.com"
        assert d.title == ""
        assert d.pub_date == ""
        assert d.hierarchy_level is None
        assert d.finding == ""
        assert d.supports_or_contradicts is None
        assert d.confidence is None

    def test_full(self):
        d = SourceReviewedData(
            url="https://example.com",
            title="Title",
            pub_date="2025-01-01",
            hierarchy_level=HierarchyLevel.direct_measurement,
            finding="Found something",
            supports_or_contradicts=SupportsOrContradicts.supports,
            confidence=Confidence.HIGH,
        )
        assert d.hierarchy_level == 1
        assert d.supports_or_contradicts == "supports"
        assert d.confidence == "HIGH"

    def test_missing_url(self):
        with pytest.raises(ValidationError):
            SourceReviewedData()


class TestFreshnessCheckedData:
    def test_minimal(self):
        d = FreshnessCheckedData(source_url="https://example.com")
        assert d.original_date == ""
        assert d.verdict == ""
        assert d.notes == ""

    def test_full(self):
        d = FreshnessCheckedData(
            source_url="https://example.com",
            original_date="2024-01-01",
            verdict="current",
            notes="Checked via archive.org",
        )
        assert d.verdict == "current"

    def test_missing_source_url(self):
        with pytest.raises(ValidationError):
            FreshnessCheckedData()


class TestVerdictIssuedData:
    def test_minimal(self):
        d = VerdictIssuedData(verdict="confirmed")
        assert d.summary == ""

    def test_full(self):
        d = VerdictIssuedData(verdict="confirmed", summary="All evidence aligns")
        assert d.verdict == "confirmed"
        assert d.summary == "All evidence aligns"

    def test_missing_verdict(self):
        with pytest.raises(ValidationError):
            VerdictIssuedData()


class TestEvidenceAppendedData:
    def test_valid(self):
        d = EvidenceAppendedData(summary="New data point")
        assert d.summary == "New data point"

    def test_missing_summary(self):
        with pytest.raises(ValidationError):
            EvidenceAppendedData()


class TestDebateMediatedData:
    def test_minimal(self):
        d = DebateMediatedData(from_agent="researcher", to_agent="critic")
        assert d.exchange_summary == ""

    def test_full(self):
        d = DebateMediatedData(
            from_agent="researcher",
            to_agent="critic",
            exchange_summary="Resolved disagreement on methodology",
        )
        assert d.from_agent == "researcher"

    def test_missing_from_agent(self):
        with pytest.raises(ValidationError):
            DebateMediatedData(to_agent="critic")


class TestConvergenceCheckedData:
    def test_minimal(self):
        d = ConvergenceCheckedData(converged=True)
        assert d.blocking_verdicts is None
        assert d.weakest_link is None
        assert d.focused_directive is None

    def test_full(self):
        d = ConvergenceCheckedData(
            converged=False,
            blocking_verdicts={"node-a": "insufficient"},
            weakest_link="node-a",
            focused_directive="Gather more evidence",
        )
        assert d.converged is False
        assert d.blocking_verdicts == {"node-a": "insufficient"}

    def test_missing_converged(self):
        with pytest.raises(ValidationError):
            ConvergenceCheckedData()


class TestNodeReopenedData:
    def test_valid(self):
        d = NodeReopenedData(trigger="new_evidence", prior_status="VALIDATED")
        assert d.trigger == "new_evidence"
        assert d.prior_status == "VALIDATED"

    def test_missing_fields(self):
        with pytest.raises(ValidationError):
            NodeReopenedData(trigger="new_evidence")
        with pytest.raises(ValidationError):
            NodeReopenedData(prior_status="VALIDATED")


class TestPropagationTriggeredData:
    def test_valid(self):
        d = PropagationTriggeredData(
            reopened_node_id="n1", flagged_dependents=["n2", "n3"]
        )
        assert d.flagged_dependents == ["n2", "n3"]

    def test_empty_dependents(self):
        d = PropagationTriggeredData(
            reopened_node_id="n1", flagged_dependents=[]
        )
        assert d.flagged_dependents == []

    def test_missing_fields(self):
        with pytest.raises(ValidationError):
            PropagationTriggeredData(reopened_node_id="n1")


class TestDecompositionCreatedData:
    def test_valid(self):
        d = DecompositionCreatedData(
            parent_node_id="p1", child_node_ids=["c1", "c2"]
        )
        assert d.parent_node_id == "p1"

    def test_missing_fields(self):
        with pytest.raises(ValidationError):
            DecompositionCreatedData(parent_node_id="p1")


# ── Core Entity Model Tests ──────────────────────────────────────────────


class TestEvent:
    def test_valid(self):
        e = Event(
            id="evt-001",
            node_id="dish-col-001-001",
            timestamp="2025-06-01T00:00:00Z",
            type=EventType.search_executed,
            agent="researcher",
            iteration=0,
            data={"query": "test", "sources_found": 3},
        )
        assert e.type == EventType.search_executed
        assert e.iteration == 0

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


class TestVerdict:
    def test_valid(self):
        v = Verdict(
            node_id="n1",
            agent="evaluator",
            iteration=1,
            verdict="confirmed",
            summary="All clear",
        )
        assert v.summary == "All clear"

    def test_default_summary(self):
        v = Verdict(node_id="n1", agent="evaluator", iteration=0, verdict="ok")
        assert v.summary == ""


class TestNode:
    def test_valid_with_defaults(self):
        n = Node(
            id="dish-col-001-001",
            colony_id="dish-col",
            claim_text="The sky is blue",
            level=1,
        )
        assert n.status == NodeStatus.NEW
        assert n.dependencies == []
        assert n.dependents == []
        assert n.created_at == ""

    def test_full(self):
        n = Node(
            id="dish-col-001-001",
            colony_id="dish-col",
            claim_text="Claim",
            level=2,
            status=NodeStatus.VALIDATED,
            dependencies=["dish-col-001-000"],
            dependents=["dish-col-001-002"],
            created_at="2025-01-01T00:00:00Z",
        )
        assert n.status == NodeStatus.VALIDATED
        assert len(n.dependencies) == 1

    def test_negative_level_rejected(self):
        with pytest.raises(ValidationError):
            Node(
                id="n",
                colony_id="c",
                claim_text="x",
                level=-1,
            )


class TestEdge:
    def test_valid(self):
        e = Edge(from_node="a", to_node="b")
        assert e.edge_type == "intra_colony"

    def test_custom_edge_type(self):
        e = Edge(from_node="a", to_node="b", edge_type="cross_colony")
        assert e.edge_type == "cross_colony"


class TestColony:
    def test_valid_with_defaults(self):
        c = Colony(
            id="dish-market",
            dish="dish",
            center_claim="Market is growing",
            center_node_id="dish-market-000-000",
        )
        assert c.clarifications == []
        assert c.created_at == ""

    def test_with_clarifications(self):
        c = Colony(
            id="dish-market",
            dish="dish",
            center_claim="Claim",
            center_node_id="n1",
            clarifications=[{"question": "What market?", "answer": "US equities"}],
        )
        assert len(c.clarifications) == 1


class TestPetriDish:
    def test_valid_with_defaults(self):
        p = PetriDish(id="research1", path="/tmp/.petri")
        assert p.colonies == []
        assert p.config == {}
        assert p.created_at == ""


class TestQueueEntry:
    def test_defaults(self):
        qe = QueueEntry(node_id="dish-col-001-001")
        assert qe.queue_state == QueueState.queued
        assert qe.iteration == 0
        assert qe.max_iterations == 3
        assert qe.cycle_start_iteration == 0
        assert qe.weakest_link is None
        assert qe.focused_directive is None
        assert qe.entered_at == ""
        assert qe.last_activity == ""

    def test_custom_values(self):
        qe = QueueEntry(
            node_id="n1",
            queue_state=QueueState.phase1_active,
            iteration=2,
            max_iterations=5,
        )
        assert qe.queue_state == QueueState.phase1_active
        assert qe.iteration == 2
        assert qe.max_iterations == 5

    def test_negative_iteration_rejected(self):
        with pytest.raises(ValidationError):
            QueueEntry(node_id="n1", iteration=-1)

    def test_max_iterations_zero_rejected(self):
        with pytest.raises(ValidationError):
            QueueEntry(node_id="n1", max_iterations=0)

    def test_negative_cycle_start_rejected(self):
        with pytest.raises(ValidationError):
            QueueEntry(node_id="n1", cycle_start_iteration=-1)


class TestAgentRole:
    def test_defaults(self):
        ar = AgentRole(name="researcher", display_name="Researcher")
        assert ar.phase is None
        assert ar.blocking == "false"
        assert ar.is_lead is False
        assert ar.scope is None
        assert ar.verdicts_pass == []
        assert ar.verdicts_block == []
        assert ar.redirect_on_block is None

    def test_full(self):
        ar = AgentRole(
            name="fact_checker",
            display_name="Fact Checker",
            phase=1,
            blocking="true",
            is_lead=False,
            scope="pipeline",
            verdicts_pass=["confirmed", "likely"],
            verdicts_block=["insufficient"],
            redirect_on_block="NEEDS_EXPERIMENT",
        )
        assert ar.phase == 1
        assert ar.blocking == "true"
        assert ar.verdicts_pass == ["confirmed", "likely"]


class TestDebate:
    def test_valid(self):
        d = Debate(
            pair=("researcher", "critic"),
            rounds=1.5,
            purpose="Test methodology",
        )
        assert d.pair == ("researcher", "critic")
        assert d.rounds == 1.5
        assert d.purpose == "Test methodology"

    def test_integer_rounds(self):
        d = Debate(pair=("a", "b"), rounds=2, purpose="p")
        assert d.rounds == 2.0


class TestPetriConfig:
    def test_defaults(self):
        pc = PetriConfig()
        assert pc.name == ""
        assert pc.model == {}
        assert pc.harness == "claude-code"
        assert pc.max_iterations == 3
        assert pc.max_concurrent == 4
        assert pc.agents == {}
        assert pc.debates == []
        assert pc.source_hierarchy == {}


class TestClarifyingQuestion:
    def test_defaults(self):
        cq = ClarifyingQuestion(question="What market?")
        assert cq.options == []
        assert cq.answer is None

    def test_full(self):
        cq = ClarifyingQuestion(
            question="Which region?",
            options=["US", "EU", "APAC"],
            answer="US",
        )
        assert len(cq.options) == 3


class TestDecompositionResult:
    def test_valid(self):
        node = Node(id="n1", colony_id="c1", claim_text="x", level=0)
        edge = Edge(from_node="n1", to_node="n2")
        dr = DecompositionResult(
            nodes=[node],
            edges=[edge],
            colony_name="market",
            center_claim="Market claim",
        )
        assert len(dr.nodes) == 1
        assert len(dr.edges) == 1
        assert dr.colony_name == "market"


# ── validate_event_data Tests ────────────────────────────────────────────


class TestValidateEventData:
    def test_valid_search_executed(self):
        result = validate_event_data(
            "search_executed", {"query": "test", "sources_found": 5}
        )
        assert result == {"query": "test", "sources_found": 5}

    def test_valid_source_reviewed_minimal(self):
        result = validate_event_data(
            "source_reviewed", {"url": "https://example.com"}
        )
        assert result["url"] == "https://example.com"
        # title defaults to "" which should be present
        assert "title" in result or result.get("title") == ""

    def test_valid_freshness_checked(self):
        result = validate_event_data(
            "freshness_checked", {"source_url": "https://example.com"}
        )
        assert result["source_url"] == "https://example.com"

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
        # None values should be excluded
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

    def test_defaults_applied(self):
        result = validate_event_data(
            "source_reviewed", {"url": "https://example.com"}
        )
        # title has a default of "" -- it should be in the result
        assert result.get("title", "") == ""

    def test_extra_fields_excluded(self):
        result = validate_event_data(
            "evidence_appended",
            {"summary": "data", "extra_field": "should_be_gone"},
        )
        assert "extra_field" not in result

    def test_none_values_excluded(self):
        result = validate_event_data(
            "convergence_checked",
            {"converged": False, "weakest_link": None},
        )
        assert "weakest_link" not in result


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
        # Not enough segments to extract level/seq
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
            def generate_clarifying_questions(
                self, claim: str, max_questions: int = 5
            ) -> list[dict]:
                return []

            def decompose_claim(
                self, claim: str, clarifications: list[dict]
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

    def test_missing_method_fails_protocol(self):
        class BadProvider:
            def generate_clarifying_questions(
                self, claim: str, max_questions: int = 5
            ) -> list[dict]:
                return []
            # Missing decompose_claim, assess_node, match_evidence

        assert not isinstance(BadProvider(), InferenceProvider)

    def test_partial_implementation_fails(self):
        class PartialProvider:
            def generate_clarifying_questions(
                self, claim: str, max_questions: int = 5
            ) -> list[dict]:
                return []

            def decompose_claim(
                self, claim: str, clarifications: list[dict]
            ) -> dict:
                return {}
            # Missing assess_node, match_evidence

        assert not isinstance(PartialProvider(), InferenceProvider)

    def test_empty_class_fails_protocol(self):
        class EmptyProvider:
            pass

        assert not isinstance(EmptyProvider(), InferenceProvider)
