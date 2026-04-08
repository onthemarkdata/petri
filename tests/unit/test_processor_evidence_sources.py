"""Regression tests for the evidence.md formatters' source rendering.

Every formatter in ``petri.engine.processor`` receives
``AssessmentResult`` Pydantic objects from its phase runner. Those
objects carry ``sources_cited: list[SourceCitation]`` — Pydantic
models, NOT plain dicts. Before this test file existed, every
formatter did ``if not isinstance(source, dict): continue`` and
silently dropped 100% of the cited sources on the floor. The
``verdict_issued`` events on disk had the full source payloads
(because ``_verdict_data`` correctly normalized via ``model_dump``),
but ``evidence.md`` ended up with zero URLs.

This file pins the fix: every formatter must accept ``SourceCitation``
objects and render their URLs in the output.
"""

from __future__ import annotations

from petri.engine.processor import (
    _format_evaluation_evidence,
    _format_phase1_evidence,
    _format_phase2_evidence,
    _format_red_team_evidence,
    _iter_verdict_sources,
    _render_source_line,
    _source_to_dict,
)
from petri.models import AssessmentResult, SourceCitation


# ── _source_to_dict normalization helper ────────────────────────────────


def test_source_to_dict_handles_pydantic_source():
    """SourceCitation Pydantic objects must normalize to a plain dict
    via model_dump. Before the fix, formatters treated Pydantic
    sources as non-dicts and silently skipped them."""
    source = SourceCitation(
        url="https://example.org/a",
        title="A paper",
        hierarchy_level=2,
        finding="supports the claim",
        supports_or_contradicts="supports",
    )
    result = _source_to_dict(source)
    assert isinstance(result, dict)
    assert result["url"] == "https://example.org/a"
    assert result["title"] == "A paper"
    assert result["hierarchy_level"] == 2


def test_source_to_dict_passes_plain_dict_through():
    raw = {"url": "https://example.org/b", "title": "Raw dict"}
    assert _source_to_dict(raw) is raw


def test_source_to_dict_returns_none_for_garbage():
    assert _source_to_dict(None) is None
    assert _source_to_dict("a string") is None
    assert _source_to_dict(42) is None


# ── Phase 1 (Research) formatter ────────────────────────────────────────


def test_format_phase1_evidence_renders_urls_from_pydantic_sources():
    """Regression for the silent-drop bug. Phase 1 research agents
    return ``AssessmentResult`` objects whose ``sources_cited`` is a
    list of ``SourceCitation`` Pydantic objects. The formatter MUST
    render their URLs in the evidence.md output."""
    research_result = AssessmentResult(
        agent="investigator",
        verdict="EVIDENCE_SUFFICIENT",
        summary="Three primary sources confirm the claim.",
        sources_cited=[
            SourceCitation(
                url="https://arxiv.org/abs/2410.12345",
                title="Language Model Scaling Laws, Kaplan et al (2020)",
                hierarchy_level=2,
                finding="Log-linear loss scaling holds through 7B parameters",
                supports_or_contradicts="supports",
            ),
            SourceCitation(
                url="https://lmsys.org/blog/2025-02-arena",
                title="LMSYS Chatbot Arena Leaderboard (Feb 2025)",
                hierarchy_level=1,
                finding="GPT-4o leads Claude by 42 Elo points",
                supports_or_contradicts="supports",
            ),
        ],
    )

    rendered = _format_phase1_evidence([research_result], iteration=1)

    # Both URLs must appear verbatim in the output.
    assert "https://arxiv.org/abs/2410.12345" in rendered
    assert "https://lmsys.org/blog/2025-02-arena" in rendered
    # Titles too.
    assert "Language Model Scaling Laws" in rendered
    assert "Chatbot Arena Leaderboard" in rendered
    # Findings appear in context.
    assert "Log-linear loss scaling" in rendered
    # Hierarchy level names are rendered.
    assert "Level 2" in rendered
    assert "Level 1" in rendered
    # And the agent summary still shows up.
    assert "Three primary sources confirm" in rendered


def test_format_phase1_evidence_handles_plain_dict_sources_too():
    """Dict-shaped sources (e.g. from a replay or test fixture) must
    still work — the formatter is format-agnostic."""
    verdict_entry = {
        "agent": "investigator",
        "verdict": "EVIDENCE_SUFFICIENT",
        "summary": "One source found.",
        "sources_cited": [
            {
                "url": "https://example.org/plain",
                "title": "Plain dict source",
                "hierarchy_level": 3,
                "finding": "plain dict finding",
                "supports_or_contradicts": "supports",
            }
        ],
    }
    rendered = _format_phase1_evidence([verdict_entry], iteration=1)
    assert "https://example.org/plain" in rendered
    assert "Plain dict source" in rendered


def test_format_phase1_evidence_placeholder_when_no_sources():
    """When zero agents cited anything, the formatter emits a clear
    placeholder so the reader knows the emptiness is real (not a
    rendering bug)."""
    result = AssessmentResult(
        agent="investigator",
        verdict="NEEDS_MORE_EVIDENCE",
        summary="Could not find sources.",
        sources_cited=[],
    )
    rendered = _format_phase1_evidence([result], iteration=1)
    assert "No sources cited by research agents" in rendered


# ── Phase 2 (Critique) formatter ────────────────────────────────────────


def test_format_phase2_evidence_renders_sources_from_critique_agents():
    """Phase 2 was previously a missing feature: critique agents
    (skeptic/champion/pragmatist) cite counter-arguments with URLs,
    but the formatter only built a 3-column verdict table and never
    surfaced the citations. This test locks in the fix."""
    skeptic = AssessmentResult(
        agent="skeptic",
        verdict="PASS",
        summary="Two counter-sources found.",
        sources_cited=[
            SourceCitation(
                url="https://nature.com/articles/skeptic-study",
                title="Counter-evidence on scaling limits (2024)",
                hierarchy_level=2,
                finding="Scaling diminishes above 100B",
                supports_or_contradicts="contradicts",
            ),
        ],
    )
    champion = AssessmentResult(
        agent="champion",
        verdict="PASS",
        summary="One supporting source.",
        sources_cited=[
            SourceCitation(
                url="https://epoch.ai/trends",
                title="Epoch AI compute trends dashboard",
                hierarchy_level=2,
                finding="Investment in scaling is accelerating",
                supports_or_contradicts="supports",
            ),
        ],
    )

    rendered = _format_phase2_evidence(
        [skeptic, champion], debates=[], iteration=1
    )

    assert "https://nature.com/articles/skeptic-study" in rendered
    assert "https://epoch.ai/trends" in rendered
    assert "Counter-evidence on scaling limits" in rendered
    assert "Epoch AI compute trends dashboard" in rendered
    # The existing verdict table is still there.
    assert "| skeptic |" in rendered
    assert "| champion |" in rendered


def test_format_phase2_evidence_placeholder_when_no_sources():
    verdict_entry = {
        "agent": "skeptic",
        "verdict": "PASS",
        "summary": "No sources.",
        "sources_cited": [],
    }
    rendered = _format_phase2_evidence([verdict_entry], debates=[], iteration=1)
    assert "No sources cited by critique agents" in rendered


# ── Red Team formatter ──────────────────────────────────────────────────


def test_format_red_team_evidence_renders_pydantic_sources():
    red_team_result = AssessmentResult(
        agent="red_team_lead",
        verdict="CANNOT_DISPROVE",
        summary="Tried hard; claim holds.",
        sources_cited=[
            SourceCitation(
                url="https://openreview.net/forum?id=refutation",
                title="Attempted refutation paper",
                hierarchy_level=2,
                finding="Theoretical limits imply the claim cannot hold",
                supports_or_contradicts="contradicts",
            ),
        ],
    )

    rendered = _format_red_team_evidence(red_team_result, iteration=1)

    assert "https://openreview.net/forum?id=refutation" in rendered
    assert "Attempted refutation paper" in rendered
    assert "Theoretical limits" in rendered
    assert "Contradicts claim" in rendered
    assert "CANNOT_DISPROVE" in rendered


def test_format_red_team_evidence_placeholder_when_no_sources():
    result = AssessmentResult(
        agent="red_team_lead",
        verdict="CANNOT_DISPROVE",
        summary="Found nothing.",
        sources_cited=[],
    )
    rendered = _format_red_team_evidence(result, iteration=1)
    assert "No sources cited by red team" in rendered


# ── Evaluation formatter ────────────────────────────────────────────────


def test_format_evaluation_evidence_renders_pydantic_sources():
    evaluation_result = AssessmentResult(
        agent="evidence_evaluator",
        verdict="EVIDENCE_CONFIRMS",
        summary="Strong terminal-level evidence.",
        confidence="HIGH",
        sources_cited=[
            SourceCitation(
                url="https://nist.gov/measurement",
                title="NIST measurement record",
                hierarchy_level=1,
                finding="Direct measurement matches claim",
                supports_or_contradicts="supports",
            ),
            SourceCitation(
                url="https://bls.gov/data/series",
                title="BLS time series",
                hierarchy_level=2,
                finding="Authoritative documentation",
                supports_or_contradicts="supports",
            ),
        ],
    )

    rendered = _format_evaluation_evidence(
        evaluation_result,
        source_validation={"meets_terminal_threshold": True, "max_hierarchy_level": 1},
        iteration=1,
    )

    # Both URLs appear in the numbered list AND the table.
    assert rendered.count("https://nist.gov/measurement") >= 2
    assert rendered.count("https://bls.gov/data/series") >= 2
    assert "NIST measurement record" in rendered
    assert "BLS time series" in rendered
    # The inventory table header is present.
    assert "| # | Source | URL |" in rendered
    # Verdict + confidence + terminal validation still render.
    assert "EVIDENCE_CONFIRMS" in rendered
    assert "HIGH" in rendered
    assert "Meets terminal: Yes" in rendered


def test_format_evaluation_evidence_placeholder_when_no_sources():
    result = AssessmentResult(
        agent="evidence_evaluator",
        verdict="INSUFFICIENT",
        summary="No sources.",
        sources_cited=[],
    )
    rendered = _format_evaluation_evidence(
        result,
        source_validation={"meets_terminal_threshold": False, "max_hierarchy_level": None},
        iteration=1,
    )
    assert "No sources cited by evidence evaluator" in rendered
    # The table is NOT emitted when there are no sources.
    assert "| # | Source | URL |" not in rendered


# ── _iter_verdict_sources helper ────────────────────────────────────────


def test_iter_verdict_sources_yields_mixed_shapes():
    """Mixed lists (Pydantic + dict + garbage) should yield only the
    valid ones, normalized to dicts."""
    entry = {
        "sources_cited": [
            SourceCitation(url="https://a.com", title="pydantic"),
            {"url": "https://b.com", "title": "plain dict"},
            "not a source",  # should be skipped
            None,  # should be skipped
            42,  # should be skipped
        ]
    }
    sources = list(_iter_verdict_sources(entry))
    assert len(sources) == 2
    assert sources[0]["url"] == "https://a.com"
    assert sources[1]["url"] == "https://b.com"


def test_iter_verdict_sources_empty_for_missing_field():
    assert list(_iter_verdict_sources({})) == []
    assert list(_iter_verdict_sources({"sources_cited": None})) == []
    assert list(_iter_verdict_sources({"sources_cited": "not a list"})) == []


# ── _render_source_line helper ──────────────────────────────────────────


def test_render_source_line_includes_all_fields():
    line = _render_source_line(
        3,
        {
            "url": "https://example.org/x",
            "title": "Example paper",
            "hierarchy_level": 2,
            "finding": "key finding",
            "supports_or_contradicts": "supports",
        },
    )
    assert "Source 3" in line
    assert "Level 2" in line
    assert "Authoritative Documentation" in line
    assert "Example paper" in line
    assert "https://example.org/x" in line
    assert "key finding" in line
    assert "Supports claim." in line


def test_render_source_line_handles_missing_url():
    line = _render_source_line(
        1,
        {
            "title": "No URL here",
            "hierarchy_level": 5,
            "finding": "dubious",
            "supports_or_contradicts": "contradicts",
        },
    )
    assert "No URL here" in line
    assert "Contradicts claim." in line
    # No "— https..." segment should appear.
    assert "https" not in line
