"""Render evidence sections from typed assessments and persisted domain events."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from petri.agents.contract import AGENT_SPECS, AgentName, Phase
from petri.agents.outputs import SourceCitation


class RecordedAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    agent: str
    verdict: str
    summary: str = ""
    arguments: str = ""
    evidence: str = ""
    confidence: str = ""
    sources_cited: list[SourceCitation] = Field(default_factory=list)


class Assessment(Protocol):
    agent: str
    verdict: str
    summary: str
    arguments: str
    sources_cited: list[SourceCitation]


_LEVEL_NAMES = {
    1: "Direct Measurement",
    2: "Authoritative Documentation",
    3: "Derived Calculation",
    4: "Expert Consensus",
    5: "Single Expert",
    6: "Community Report",
}


def render_source(index: int, source: SourceCitation) -> str:
    direction = (
        "Contradicts claim."
        if source.supports_or_contradicts == "contradicts"
        else "Supports claim."
    )
    return (
        f"**Source {index} (Level {source.hierarchy_level} — {_LEVEL_NAMES[source.hierarchy_level]}):** "
        f"{source.title or 'Unknown source'} — {source.url} — {source.finding} **{direction}**"
    )


def _sources(assessments: Sequence[Assessment]) -> list[str]:
    lines = []
    seen: set[str] = set()
    for assessment in assessments:
        for source in assessment.sources_cited:
            if source.url not in seen:
                seen.add(source.url)
                lines.extend([render_source(len(seen), source), ""])
    return lines


def format_research(assessments: Sequence[Assessment], iteration: int) -> str:
    lines = [f"### Iteration {iteration} — Phase 1 Research\n"]
    lines.extend(_sources(assessments) or ["_No sources cited by research agents._", ""])
    for assessment in assessments:
        if assessment.summary:
            display = assessment.agent.replace("_", " ").title()
            lines.extend([f"**{display}:** {assessment.summary}", ""])
    return "\n".join(lines)


def format_critique(
    assessments: Sequence[Assessment], debates: Sequence[dict], iteration: int
) -> str:
    lines = [f"### Iteration {iteration} — Phase 2 Critique\n"]
    lines.extend(_sources(assessments) or ["_No sources cited by critique agents._", ""])
    lines.extend(["| Agent | Verdict | Summary |", "|-------|---------|---------|"])
    for assessment in assessments:
        lines.append(f"| {assessment.agent} | {assessment.verdict} | {assessment.summary[:120]} |")
    lines.append("")
    if debates:
        lines.append("**Debate Outcomes:**")
        for debate in debates:
            if debate["summary"]:
                lines.append(f"- {debate['pair'][0]} vs {debate['pair'][1]}: {debate['summary']}")
        lines.append("")
    return "\n".join(lines)


def format_socratic(assessments: Sequence[Assessment]) -> str:
    lines = ["### Socratic Analysis\n"]
    for assessment in assessments:
        if assessment.agent == "socratic_verifier":
            if assessment.summary:
                lines.extend(
                    [f"**Socratic Verification:** {assessment.verdict}", f"{assessment.summary}\n"]
                )
            for source in assessment.sources_cited:
                lines.append(render_source(len(lines), source))
            continue
        step = assessment.agent.removeprefix("socratic_")
        lines.append(f"**{step.replace('_', ' ').title()}:**")
        if assessment.summary:
            lines.append(assessment.summary)
        if assessment.arguments:
            lines.append(f"\n{assessment.arguments}")
        seen: set[str] = set()
        for source in assessment.sources_cited:
            if source.url not in seen:
                seen.add(source.url)
                lines.append(render_source(len(lines), source))
        lines.append("")
    return "\n".join(lines)


def debate_summary(data: dict) -> dict:
    """Render actual debate turns through the legacy critique-summary layout."""
    detail = json.loads(data["exchange_summary"])
    summary = " ".join(
        f"{turn['speaker']}: {turn['assessment']['summary']}" for turn in detail["exchanges"]
    )
    return {"pair": (data["from_agent"], data["to_agent"]), "summary": summary}


def render_evidence(events: Iterable[dict]) -> str:
    """Rebuild the appended evidence body using events alone, in append order."""
    assessments: dict[tuple[int, str], list[RecordedAssessment]] = {}
    debates: dict[int, list[dict]] = {}
    sections = []
    for event in events:
        iteration = event["iteration"]
        if event["type"] == "verdict_issued":
            agent = event["agent"]
            if agent.startswith("socratic_"):
                phase = "socratic"
            elif agent in AgentName._value2member_map_:
                phase = AGENT_SPECS[AgentName(agent)].phase.value
                if phase not in {Phase.RESEARCH.value, Phase.CRITIQUE.value}:
                    continue
            else:
                continue
            assessment = RecordedAssessment.model_validate({"agent": agent, **event["data"]})
            assessments.setdefault((iteration, phase), []).append(assessment)
        elif event["type"] == "debate_mediated":
            debates.setdefault(iteration, []).append(debate_summary(event["data"]))
        elif event["type"] == "evidence_appended":
            summary = event["data"]["summary"]
            marker_phase = next(
                (
                    name
                    for name in ("socratic", "research", "critique")
                    if summary == f"Appended {name} findings (iteration {iteration})"
                ),
                None,
            )
            if marker_phase is None:
                continue
            values = assessments.pop((iteration, marker_phase), [])
            if marker_phase == "socratic":
                content = format_socratic(values)
            elif marker_phase == "research":
                content = format_research(values, iteration)
            else:
                content = format_critique(values, debates.pop(iteration, []), iteration)
            sections.append(f"\n\n---\n\n{content}")
    return "".join(sections)
