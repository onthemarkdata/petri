"""Unit tests for petri/debate.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from petri.debate import (
    get_held_messages,
    load_debate_pairings,
    log_debate,
    mediate_debate,
)
from petri.event_log import load_events
from petri.models import Debate


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def debate_pairings():
    """Load default debate pairings."""
    return load_debate_pairings()


@pytest.fixture
def skeptic_champion_debate(debate_pairings):
    """Return the skeptic vs champion debate (1.5 rounds)."""
    for d in debate_pairings:
        if d.pair == ("skeptic", "champion"):
            return d
    pytest.fail("skeptic vs champion debate not found in defaults")


@pytest.fixture
def skeptic_pragmatist_debate(debate_pairings):
    """Return the skeptic vs pragmatist debate (1 round)."""
    for d in debate_pairings:
        if d.pair == ("skeptic", "pragmatist"):
            return d
    pytest.fail("skeptic vs pragmatist debate not found in defaults")


def _agent_output(agent: str, verdict: str, summary: str = "", arguments: str = "") -> dict:
    return {
        "agent": agent,
        "verdict": verdict,
        "summary": summary,
        "arguments": arguments,
    }


# ── load_debate_pairings ────────────────────────────────────────────────


class TestLoadDebatePairings:
    def test_loads_four_default_pairings(self, debate_pairings):
        assert len(debate_pairings) == 4

    def test_pairings_are_debate_models(self, debate_pairings):
        for d in debate_pairings:
            assert isinstance(d, Debate)

    def test_skeptic_champion_is_1_5_rounds(self, debate_pairings):
        sc = [d for d in debate_pairings if d.pair == ("skeptic", "champion")]
        assert len(sc) == 1
        assert sc[0].rounds == 1.5

    def test_other_debates_are_1_round(self, debate_pairings):
        others = [d for d in debate_pairings if d.pair != ("skeptic", "champion")]
        for d in others:
            assert d.rounds == 1.0

    def test_all_pairings_have_purpose(self, debate_pairings):
        for d in debate_pairings:
            assert d.purpose

    def test_expected_pairs(self, debate_pairings):
        pairs = {d.pair for d in debate_pairings}
        assert ("skeptic", "champion") in pairs
        assert ("skeptic", "pragmatist") in pairs
        assert ("simplifier", "impact_assessor") in pairs
        assert ("triage", "impact_assessor") in pairs

    def test_fallback_to_defaults_on_missing_path(self):
        pairings = load_debate_pairings(Path("/nonexistent/debates.yaml"))
        assert len(pairings) == 4


# ── mediate_debate ───────────────────────────────────────────────────────


class TestMediateDebate:
    def test_1_5_round_debate_has_3_exchanges(self, skeptic_champion_debate):
        """1.5 rounds = presentation + response + rebuttal (3 exchanges)."""
        a = _agent_output("skeptic", "CRITICAL_FLAW_FOUND", "There's a flaw", "Arg A")
        b = _agent_output("champion", "STRONG_CASE", "It's strong", "Arg B")
        result = mediate_debate(a, b, skeptic_champion_debate)

        assert len(result["exchanges"]) == 3
        assert result["exchanges"][0]["speaker"] == "skeptic"
        assert result["exchanges"][1]["speaker"] == "champion"
        assert result["exchanges"][2]["speaker"] == "skeptic"  # final word

    def test_1_round_debate_has_2_exchanges(self, skeptic_pragmatist_debate):
        """1 round = presentation + response (2 exchanges)."""
        a = _agent_output("skeptic", "CRITICAL_FLAW_FOUND", "Flaw found")
        b = _agent_output("pragmatist", "PRODUCTION_READY", "Ship it")
        result = mediate_debate(a, b, skeptic_pragmatist_debate)

        assert len(result["exchanges"]) == 2
        assert result["exchanges"][0]["speaker"] == "skeptic"
        assert result["exchanges"][1]["speaker"] == "pragmatist"

    def test_debate_result_structure(self, skeptic_champion_debate):
        a = _agent_output("skeptic", "CRITICAL_FLAW_FOUND")
        b = _agent_output("champion", "STRONG_CASE")
        result = mediate_debate(a, b, skeptic_champion_debate)

        assert "pair" in result
        assert "rounds" in result
        assert "purpose" in result
        assert "exchanges" in result
        assert "summary" in result

    def test_debate_preserves_pair_info(self, skeptic_champion_debate):
        a = _agent_output("skeptic", "CRITICAL_FLAW_FOUND")
        b = _agent_output("champion", "STRONG_CASE")
        result = mediate_debate(a, b, skeptic_champion_debate)

        assert result["pair"] == ("skeptic", "champion")
        assert result["rounds"] == 1.5
        assert result["purpose"] == skeptic_champion_debate.purpose

    def test_exchange_round_numbering(self, skeptic_champion_debate):
        a = _agent_output("skeptic", "CRITICAL_FLAW_FOUND")
        b = _agent_output("champion", "STRONG_CASE")
        result = mediate_debate(a, b, skeptic_champion_debate)

        assert result["exchanges"][0]["round"] == 1
        assert result["exchanges"][1]["round"] == 1
        assert result["exchanges"][2]["round"] == 1.5

    def test_content_includes_verdict(self, skeptic_champion_debate):
        a = _agent_output("skeptic", "CRITICAL_FLAW_FOUND", "Bad logic")
        b = _agent_output("champion", "STRONG_CASE", "Good logic")
        result = mediate_debate(a, b, skeptic_champion_debate)

        assert "CRITICAL_FLAW_FOUND" in result["exchanges"][0]["content"]
        assert "STRONG_CASE" in result["exchanges"][1]["content"]

    def test_content_includes_arguments(self, skeptic_pragmatist_debate):
        a = _agent_output("skeptic", "CRITICAL_FLAW_FOUND", "Flaw", "The data is wrong")
        b = _agent_output("pragmatist", "PRODUCTION_READY", "Fine", "Works in practice")
        result = mediate_debate(a, b, skeptic_pragmatist_debate)

        assert "The data is wrong" in result["exchanges"][0]["content"]
        assert "Works in practice" in result["exchanges"][1]["content"]

    def test_empty_outputs_produce_fallback_content(self, skeptic_pragmatist_debate):
        a = _agent_output("skeptic", "", "", "")
        b = _agent_output("pragmatist", "", "", "")
        result = mediate_debate(a, b, skeptic_pragmatist_debate)

        # Should not crash; exchanges should have fallback content
        assert len(result["exchanges"]) == 2
        for ex in result["exchanges"]:
            assert ex["content"]  # not empty

    def test_summary_contains_agent_names(self, skeptic_champion_debate):
        a = _agent_output("skeptic", "CRITICAL_FLAW_FOUND")
        b = _agent_output("champion", "STRONG_CASE")
        result = mediate_debate(a, b, skeptic_champion_debate)

        assert "skeptic" in result["summary"]
        assert "champion" in result["summary"]

    def test_summary_contains_verdicts(self, skeptic_champion_debate):
        a = _agent_output("skeptic", "CRITICAL_FLAW_FOUND")
        b = _agent_output("champion", "STRONG_CASE")
        result = mediate_debate(a, b, skeptic_champion_debate)

        assert "CRITICAL_FLAW_FOUND" in result["summary"]
        assert "STRONG_CASE" in result["summary"]


# ── log_debate ───────────────────────────────────────────────────────────


class TestLogDebate:
    def test_logs_debate_mediated_event(self, tmp_path, skeptic_champion_debate):
        events_path = tmp_path / "events.jsonl"
        node_id = "test-colony-001-001"

        a = _agent_output("skeptic", "CRITICAL_FLAW_FOUND", "Flaw found")
        b = _agent_output("champion", "STRONG_CASE", "Strong case")
        debate_result = mediate_debate(a, b, skeptic_champion_debate)

        log_debate(events_path, node_id, iteration=1, debate_result=debate_result)

        events = load_events(events_path)
        assert len(events) == 1
        evt = events[0]
        assert evt["type"] == "debate_mediated"
        assert evt["agent"] == "node_lead"
        assert evt["node_id"] == node_id
        assert evt["iteration"] == 1

    def test_logs_correct_agents_in_data(self, tmp_path, skeptic_champion_debate):
        events_path = tmp_path / "events.jsonl"
        node_id = "test-colony-001-001"

        a = _agent_output("skeptic", "CRITICAL_FLAW_FOUND")
        b = _agent_output("champion", "STRONG_CASE")
        debate_result = mediate_debate(a, b, skeptic_champion_debate)

        log_debate(events_path, node_id, iteration=0, debate_result=debate_result)

        events = load_events(events_path)
        data = events[0]["data"]
        assert data["from_agent"] == "skeptic"
        assert data["to_agent"] == "champion"

    def test_exchange_summary_in_event(self, tmp_path, skeptic_champion_debate):
        events_path = tmp_path / "events.jsonl"
        node_id = "test-colony-001-001"

        a = _agent_output("skeptic", "CRITICAL_FLAW_FOUND", "Problem here")
        b = _agent_output("champion", "STRONG_CASE", "No problem")
        debate_result = mediate_debate(a, b, skeptic_champion_debate)

        log_debate(events_path, node_id, iteration=0, debate_result=debate_result)

        events = load_events(events_path)
        summary = events[0]["data"]["exchange_summary"]
        assert "skeptic" in summary
        assert "champion" in summary

    def test_multiple_debates_logged(self, tmp_path, debate_pairings):
        """Log all 4 default debates for the same node."""
        events_path = tmp_path / "events.jsonl"
        node_id = "test-colony-001-001"

        for debate in debate_pairings:
            a = _agent_output(debate.pair[0], "PASS_VERDICT")
            b = _agent_output(debate.pair[1], "PASS_VERDICT")
            result = mediate_debate(a, b, debate)
            log_debate(events_path, node_id, iteration=0, debate_result=result)

        events = load_events(events_path)
        assert len(events) == 4
        assert all(e["type"] == "debate_mediated" for e in events)


# ── get_held_messages ────────────────────────────────────────────────────


class TestGetHeldMessages:
    def test_phase2_debates_produce_held_messages(self, skeptic_champion_debate):
        a = _agent_output("skeptic", "CRITICAL_FLAW_FOUND", "Flaw")
        b = _agent_output("champion", "STRONG_CASE", "Strong")
        debate_result = mediate_debate(a, b, skeptic_champion_debate)

        held = get_held_messages([debate_result], current_phase=2)
        assert len(held) > 0
        for msg in held:
            assert msg["hold_until_phase"] == 1
            assert msg["from_agent"] in ("skeptic", "champion")
            assert msg["to_agent"] in ("skeptic", "champion")

    def test_phase1_produces_no_held_messages(self, skeptic_champion_debate):
        a = _agent_output("skeptic", "CRITICAL_FLAW_FOUND")
        b = _agent_output("champion", "STRONG_CASE")
        debate_result = mediate_debate(a, b, skeptic_champion_debate)

        held = get_held_messages([debate_result], current_phase=1)
        assert len(held) == 0

    def test_held_messages_have_content(self, skeptic_champion_debate):
        a = _agent_output("skeptic", "CRITICAL_FLAW_FOUND", "A problem", "Args")
        b = _agent_output("champion", "STRONG_CASE", "No problem", "Counter")
        debate_result = mediate_debate(a, b, skeptic_champion_debate)

        held = get_held_messages([debate_result], current_phase=2)
        for msg in held:
            assert msg["content"]  # non-empty

    def test_multiple_debates_accumulate_held_messages(self, debate_pairings):
        results = []
        for debate in debate_pairings:
            a = _agent_output(debate.pair[0], "VERDICT_A")
            b = _agent_output(debate.pair[1], "VERDICT_B")
            results.append(mediate_debate(a, b, debate))

        held = get_held_messages(results, current_phase=2)
        # Each debate produces exchanges, all held when phase=2
        assert len(held) > 4  # at least one per debate

    def test_empty_debates_no_held_messages(self):
        held = get_held_messages([], current_phase=2)
        assert held == []
