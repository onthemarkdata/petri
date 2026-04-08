"""Unit tests for ``petri.reasoning.claude_code_provider``.

Focus: verdict parsing and ``assess_node`` error-handling semantics. The
critical invariants tested here are:

  1. ``_parse_verdict`` raises ``ValueError`` on no-match (it must NOT
     silently return the first pass verdict — that was the old bug).
  2. ``assess_node`` surfaces ``EXECUTION_ERROR`` when the model output
     is unparseable or returns an unknown verdict value.
  3. ``assess_node`` raises ``ValueError`` for undeclared agent roles
     rather than falling back to a default ``["PASS"]`` verdict list.

These tests stub out ``_ask`` so they never touch the real ``claude``
CLI subprocess. The stub mirrors the ``FakeProvider`` pattern used
elsewhere in the test suite (see ``tests/conftest.py``).
"""

from __future__ import annotations

import pytest

from petri.reasoning.claude_code_provider import (
    ClaudeCodeProvider,
    _parse_verdict,
)


class StubProvider(ClaudeCodeProvider):
    """ClaudeCodeProvider subclass that returns a canned ``_ask`` response.

    Avoids the ``claude`` CLI dependency check in ``__init__`` by skipping
    the parent's ``__init__`` entirely — we only need the instance methods.
    """

    def __init__(self, canned_response: str) -> None:
        # Intentionally skip ClaudeCodeProvider.__init__ so we don't need
        # the real claude CLI on PATH. Set the minimal attributes the
        # instance methods touch.
        self.model = "test-model"
        self._canned_response = canned_response
        self.last_prompt: str | None = None

    def _ask(self, prompt, on_progress=None):  # type: ignore[override]
        self.last_prompt = prompt
        return self._canned_response


# ── _parse_verdict ────────────────────────────────────────────────────────


def test_parse_verdict_returns_match_when_present():
    """Happy path: the first recognized verdict in the text wins."""
    valid_verdicts = ["EVIDENCE_SUFFICIENT", "NEEDS_MORE_EVIDENCE"]
    raw_text = "After analysis the verdict is EVIDENCE_SUFFICIENT for this claim."
    assert _parse_verdict(raw_text, valid_verdicts) == "EVIDENCE_SUFFICIENT"


def test_parse_verdict_raises_when_no_match():
    """Critical: must raise ValueError, NOT silently return valid_verdicts[0]."""
    valid_verdicts = ["EVIDENCE_SUFFICIENT", "NEEDS_MORE_EVIDENCE"]
    raw_text = "The model panicked and returned gibberish."
    with pytest.raises(ValueError) as exception_info:
        _parse_verdict(raw_text, valid_verdicts)
    error_message = str(exception_info.value)
    assert "did not contain any recognized verdict" in error_message
    assert "EVIDENCE_SUFFICIENT" in error_message


# ── assess_node error handling ────────────────────────────────────────────


def test_assess_node_returns_execution_error_on_unparseable_output():
    """When ``_ask`` returns unparseable text, verdict must be EXECUTION_ERROR.

    Regression guard against the old silent-PASS bug where failing calls
    became the strongest PASS verdict.
    """
    provider = StubProvider("Execution error")
    result = provider.assess_node(
        node_id="test-dish-colony-001-001",
        claim_text="A sample claim",
        context={},
        agent_role="investigator",
    )
    assert result.verdict == "EXECUTION_ERROR"
    assert result.agent == "investigator"
    # The summary should mention the raw output to aid debugging.
    assert "Execution error" in result.summary
    # And it must NOT be the first pass verdict (what the old code returned).
    assert result.verdict != "EVIDENCE_SUFFICIENT"


def test_assess_node_returns_execution_error_on_invalid_verdict_field():
    """When JSON has a verdict field with a bogus value, surface EXECUTION_ERROR."""
    provider = StubProvider('{"verdict": "MAYBE", "summary": "x"}')
    result = provider.assess_node(
        node_id="test-dish-colony-001-001",
        claim_text="A sample claim",
        context={},
        agent_role="investigator",
    )
    assert result.verdict == "EXECUTION_ERROR"
    assert "MAYBE" in result.summary


def test_assess_node_raises_on_unknown_agent_role():
    """Unknown agent roles must raise, not fall back to a PASS sentinel."""
    provider = StubProvider('{"verdict": "EVIDENCE_SUFFICIENT", "summary": "ok"}')
    with pytest.raises(ValueError) as exception_info:
        provider.assess_node(
            node_id="test-dish-colony-001-001",
            claim_text="A sample claim",
            context={},
            agent_role="not_a_real_agent",
        )
    error_message = str(exception_info.value)
    assert "not_a_real_agent" in error_message
    assert "agents.yaml" in error_message


def test_assess_node_returns_first_pass_verdict_when_model_says_so():
    """Positive control: the refactor didn't break the happy path."""
    provider = StubProvider(
        '{"verdict": "EVIDENCE_SUFFICIENT", '
        '"summary": "Three sources confirm.", '
        '"confidence": "HIGH", '
        '"sources_cited": []}'
    )
    result = provider.assess_node(
        node_id="test-dish-colony-001-001",
        claim_text="A sample claim",
        context={},
        agent_role="investigator",
    )
    assert result.verdict == "EVIDENCE_SUFFICIENT"
    assert result.summary == "Three sources confirm."
    assert result.confidence == "HIGH"


def test_assess_node_recovers_from_invalid_json_verdict_via_raw_text():
    """If JSON verdict is bogus but raw text still contains a valid verdict,
    the raw-text salvage path should succeed (no EXECUTION_ERROR).
    """
    provider = StubProvider(
        '{"verdict": "MAYBE", "summary": "x"} '
        "Final decision: EVIDENCE_SUFFICIENT based on three primary sources."
    )
    result = provider.assess_node(
        node_id="test-dish-colony-001-001",
        claim_text="A sample claim",
        context={},
        agent_role="investigator",
    )
    assert result.verdict == "EVIDENCE_SUFFICIENT"


def test_assess_node_accepts_socratic_questioner_role():
    """``socratic_questioner`` was added to defaults/petri.yaml; verify the
    config loader sees it so the Socratic phase doesn't trip the new
    "unknown agent" error.
    """
    provider = StubProvider(
        '{"verdict": "CLARIFIED", "summary": "terms defined"}'
    )
    result = provider.assess_node(
        node_id="test-dish-colony-001-001",
        claim_text="A sample claim",
        context={"phase": "socratic_clarify"},
        agent_role="socratic_questioner",
    )
    assert result.verdict == "CLARIFIED"
