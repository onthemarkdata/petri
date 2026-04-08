"""Pipeline processor for the Petri Research Orchestration Framework.

Processes nodes through the full validation pipeline:
research -> critique -> convergence -> red team -> evaluation.

Pure library logic -- no CLI, no UI.  The CLI (``petri grow``) calls this.
"""

from __future__ import annotations

import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from petri.config import MAX_CONCURRENT, MAX_ITERATIONS
from petri.analysis.convergence import (
    check_convergence,
    compute_circuit_breaker,
    evaluate_short_circuits,
    identify_weakest_link,
    load_agent_roles,
)
from petri.reasoning.debate import load_debate_pairings, log_debate, mediate_debate
from petri.storage.event_log import append_event, get_verdicts, query_events
from petri.models import (
    ConvergenceOutcome,
    EvaluationResult,
    InferenceProvider,
    NodeStatus,
    ProcessNodeResult,
    QueueProcessingResult,
    QueueState,
)
from petri.storage.queue import (
    add_to_queue,
    get_next,
    list_queue,
    new_cycle,
    set_focused_directive,
    set_iteration,
    set_weakest_link,
    update_state,
)
from petri.analysis.validators import validate_terminal_sources

logger = logging.getLogger(__name__)

# ── Graceful Stop ────────────────────────────────────────────────────────

_stop_event = threading.Event()


def request_stop() -> None:
    """Signal the processor to stop after current phases complete."""
    _stop_event.set()


def is_stop_requested() -> bool:
    """Check if a stop has been requested."""
    return _stop_event.is_set()


def reset_stop() -> None:
    """Clear the stop signal."""
    _stop_event.clear()


# ── File-based Stop Sentinel ─────────────────────────────────────────────
# The in-process ``threading.Event`` above only signals threads inside the
# current process.  ``grow`` may run in a detached child process, so the CLI
# also writes a sentinel file that the worker polls.

_STOP_FILE_NAME = ".stop"


def request_stop_file(petri_dir: Path) -> Path:
    """Create the cross-process stop sentinel file."""
    stop_path = petri_dir / _STOP_FILE_NAME
    stop_path.write_text("stop\n", encoding="utf-8")
    return stop_path


def is_stop_file_present(petri_dir: Path) -> bool:
    """Return True if the cross-process stop sentinel is currently set."""
    return (petri_dir / _STOP_FILE_NAME).exists()


def clear_stop_file(petri_dir: Path) -> None:
    """Remove the cross-process stop sentinel if present (idempotent)."""
    stop_path = petri_dir / _STOP_FILE_NAME
    try:
        stop_path.unlink()
    except FileNotFoundError:
        pass


# ── Path Helpers ─────────────────────────────────────────────────────────


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _queue_path(petri_dir: Path) -> Path:
    return petri_dir / "queue.json"


def _get_dish_id(petri_dir: Path) -> str:
    """Get the dish ID from config or derive from directory name."""
    config_path = petri_dir / "defaults" / "petri.yaml"
    if config_path.exists():
        try:
            import yaml

            with open(config_path) as f:
                config = yaml.safe_load(f) or {}
            return config.get("name", petri_dir.parent.name)
        except Exception:
            pass
    return petri_dir.parent.name


def _colony_slug(node_id: str, dish_id: str) -> str:
    """Extract the colony slug from a node's composite key."""
    dish_prefix = dish_id + "-"
    if node_id.startswith(dish_prefix):
        colony_and_rest = node_id[len(dish_prefix):]
        return colony_and_rest.rsplit("-", 2)[0]
    parts = node_id.split("-")
    return "-".join(parts[:-2])


def _load_node_paths(petri_dir: Path, node_id: str, dish_id: str) -> dict[str, str]:
    """Load node_paths from colony.json for the node's colony."""
    slug = _colony_slug(node_id, dish_id)
    colony_json = petri_dir / "petri-dishes" / slug / "colony.json"
    if colony_json.exists():
        data = json.loads(colony_json.read_text())
        return data.get("node_paths", {})
    return {}


def _node_dir(petri_dir: Path, node_id: str, dish_id: str) -> Path:
    """Derive the filesystem path for a node.

    Looks up the human-readable path from colony.json's node_paths mapping.
    Falls back to flat {level}-{seq} layout for backwards compatibility.
    """
    slug = _colony_slug(node_id, dish_id)
    colony_base = petri_dir / "petri-dishes" / slug

    # Try node_paths from colony.json
    node_paths = _load_node_paths(petri_dir, node_id, dish_id)
    if node_id in node_paths:
        return colony_base / node_paths[node_id]

    # Fallback: flat layout
    parts = node_id.split("-")
    return colony_base / f"{parts[-2]}-{parts[-1]}"


def _events_path(petri_dir: Path, node_id: str, dish_id: str) -> Path:
    return _node_dir(petri_dir, node_id, dish_id) / "events.jsonl"


def _metadata_path(petri_dir: Path, node_id: str, dish_id: str) -> Path:
    return _node_dir(petri_dir, node_id, dish_id) / "metadata.json"


def _load_node_metadata(petri_dir: Path, node_id: str, dish_id: str) -> dict:
    """Load a node's metadata.json."""
    path = _metadata_path(petri_dir, node_id, dish_id)
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


# ── Node Status Update ──────────────────────────────────────────────────


def _update_node_status(petri_dir: Path, node_id: str, new_status: str) -> None:
    """Update a node's status in its metadata.json file."""
    dish_id = _get_dish_id(petri_dir)
    path = _metadata_path(petri_dir, node_id, dish_id)
    if not path.exists():
        logger.warning("metadata.json not found for node %s at %s", node_id, path)
        return
    with open(path) as f:
        metadata = json.load(f)
    metadata["status"] = new_status
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")


# ── Data Coercion ────────────────────────────────────────────────────────


def _to_str(value: object) -> str:
    """Coerce a value to string. Handles lists/dicts from LLM output."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                # e.g. {"point": "...", "detail": "..."}
                parts.append("; ".join(f"{k}: {v}" for k, v in item.items()))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    if isinstance(value, dict):
        return json.dumps(value)
    return str(value) if value else ""


def _get(result: object, key: str, default: str = "") -> object:
    """Get a field from an AssessmentResult model or a dict."""
    if hasattr(result, key):
        return getattr(result, key)
    if isinstance(result, dict):
        return result.get(key, default)
    return default


def _verdict_data(result: object) -> dict:
    """Build a verdict_issued data dict from an AssessmentResult or dict."""
    sources = _get(result, "sources_cited", [])
    serialized_sources = []
    for source_entry in sources:
        if hasattr(source_entry, "model_dump"):
            serialized_sources.append(source_entry.model_dump(exclude_none=True))
        elif isinstance(source_entry, dict):
            serialized_sources.append(source_entry)
    return {
        "verdict": _get(result, "verdict", "PASS"),
        "summary": _to_str(_get(result, "summary", "")),
        "arguments": "",
        "evidence": "",
        "confidence": _to_str(_get(result, "confidence", "")),
        "sources_cited": serialized_sources,
    }


# Hierarchy level display names.
_LEVEL_NAMES = {
    1: "Direct Measurement",
    2: "Authoritative Documentation",
    3: "Derived Calculation",
    4: "Expert Consensus",
    5: "Single Expert",
    6: "Community Report",
}


# ── Evidence File Helpers ────────────────────────────────────────────────


def _log_sources_from_result(
    events_file: Path, node_id: str, agent: str, iteration: int, result: object,
) -> None:
    """Extract sources_cited from agent output and log source_reviewed events."""
    sources = _get(result, "sources_cited", [])
    if not isinstance(sources, list):
        return
    for source_entry in sources:
        # Handle both SourceCitation models and raw dicts
        if hasattr(source_entry, "url"):
            data = {
                "url": getattr(source_entry, "url", "") or getattr(source_entry, "url_or_name", ""),
                "title": getattr(source_entry, "title", ""),
                "pub_date": getattr(source_entry, "pub_date", ""),
                "hierarchy_level": getattr(source_entry, "hierarchy_level", None),
                "finding": getattr(source_entry, "finding", ""),
                "supports_or_contradicts": getattr(source_entry, "supports_or_contradicts", None),
                "confidence": getattr(source_entry, "confidence", None),
            }
        elif isinstance(source_entry, dict):
            data = {
                "url": source_entry.get("url", source_entry.get("url_or_name", "")),
                "title": source_entry.get("title", ""),
                "pub_date": source_entry.get("pub_date", ""),
                "hierarchy_level": source_entry.get("hierarchy_level"),
                "finding": source_entry.get("finding", ""),
                "supports_or_contradicts": source_entry.get("supports_or_contradicts"),
                "confidence": source_entry.get("confidence"),
            }
        else:
            continue
        if not data["url"].startswith(("http://", "https://")):
            logger.warning(
                "Source without valid URL from %s: %s",
                agent, data.get("title", "unknown"),
            )
        append_event(
            events_path=events_file,
            node_id=node_id,
            event_type="source_reviewed",
            agent=agent,
            iteration=iteration,
            data=data,
        )


def _append_evidence(
    petri_dir: Path, node_id: str, dish_id: str,
    phase: str, iteration: int, content: str,
) -> None:
    """Append a section to a node's evidence.md and log evidence_appended."""
    node_path = _node_dir(petri_dir, node_id, dish_id)
    evidence_path = node_path / "evidence.md"
    if not evidence_path.exists():
        return

    with open(evidence_path, "a") as f:
        f.write(f"\n\n---\n\n{content}")

    events_file = _events_path(petri_dir, node_id, dish_id)
    append_event(
        events_path=events_file,
        node_id=node_id,
        event_type="evidence_appended",
        agent="node_lead",
        iteration=iteration,
        data={"summary": f"Appended {phase} findings (iteration {iteration})"},
    )


def _update_evidence_status(
    petri_dir: Path, node_id: str, dish_id: str, new_status: str,
) -> None:
    """Update the Status line in a node's evidence.md file."""
    import re as _re

    node_path = _node_dir(petri_dir, node_id, dish_id)
    evidence_path = node_path / "evidence.md"
    if not evidence_path.exists():
        return
    content = evidence_path.read_text()
    updated = _re.sub(
        r"\*\*Status:\*\*\s*\S+.*",
        f"**Status:** {new_status}",
        content,
        count=1,
    )
    evidence_path.write_text(updated)


def _format_phase1_evidence(verdicts: list[dict], iteration: int) -> str:
    """Format Phase 1 (Research) findings as markdown — citation-first."""
    lines = [f"### Iteration {iteration} — Phase 1 Research\n"]

    # Collect all sources across phase 1 agents into a numbered list.
    source_num = 0
    agent_summaries: dict[str, tuple[str, str]] = {}  # agent -> (verdict, summary)

    for verdict_entry in verdicts:
        agent_name = _get(verdict_entry, "agent", "unknown")
        verdict_value = _get(verdict_entry, "verdict", "")
        summary_text = _get(verdict_entry, "summary", "")
        agent_summaries[agent_name] = (verdict_value, summary_text)

        cited_sources = _get(verdict_entry, "sources_cited", [])
        if isinstance(cited_sources, list):
            for source in cited_sources:
                if not isinstance(source, dict):
                    continue
                source_num += 1
                level = source.get("hierarchy_level", 6)
                level_name = _LEVEL_NAMES.get(level, "Unknown")
                title = source.get("title", "Unknown source")
                url = source.get("url", source.get("url_or_name", ""))
                finding = source.get("finding", "")
                direction = source.get("supports_or_contradicts", "supports")
                direction_label = direction.capitalize() + "s" if not direction.endswith("s") else direction.capitalize()

                url_part = f" — {url}" if url else ""
                lines.append(
                    f"**Source {source_num} (Level {level} — {level_name}):** "
                    f"{title}{url_part} — {finding} "
                    f"**{direction_label} claim.**"
                )
                lines.append("")

    # Agent summaries
    for agent_name, (verdict_value, summary_text) in agent_summaries.items():
        display = agent_name.replace("_", " ").title()
        if summary_text:
            lines.append(f"**{display}:** {summary_text}")
            lines.append("")

    return "\n".join(lines)


def _format_phase2_evidence(
    verdicts: list, debates: list, iteration: int,
) -> str:
    """Format Phase 2 (Critique) assessment as markdown — citation-first."""
    lines = [f"### Iteration {iteration} — Phase 2 Critique\n"]

    # Verdict summary table
    lines.append("| Agent | Verdict | Summary |")
    lines.append("|-------|---------|---------|")
    for verdict_entry in verdicts:
        agent_name = _get(verdict_entry, "agent", "unknown")
        verdict_value = _get(verdict_entry, "verdict", "")
        summary_text = str(_get(verdict_entry, "summary", ""))[:120]
        lines.append(f"| {agent_name} | {verdict_value} | {summary_text} |")
    lines.append("")

    # Debate summaries
    if debates:
        lines.append("**Debate Outcomes:**")
        for debate_entry in debates:
            pair = _get(debate_entry, "pair", ("", ""))
            debate_summary = _get(debate_entry, "summary", "")
            if debate_summary:
                lines.append(f"- {pair[0]} vs {pair[1]}: {debate_summary}")
        lines.append("")

    return "\n".join(lines)


def _format_red_team_evidence(result: dict, iteration: int) -> str:
    """Format Red Team review as markdown — citation-first."""
    verdict = _get(result, "verdict", "")
    summary = _to_str(_get(result, "summary", ""))

    lines = [f"### Red Team Review (Iteration {iteration})\n"]

    # Numbered counter-arguments from sources
    sources = _get(result, "sources_cited", [])
    if isinstance(sources, list) and sources:
        for source_idx, source_entry in enumerate(sources, 1):
            if not isinstance(source_entry, dict):
                continue
            level = source_entry.get("hierarchy_level", 6)
            level_name = _LEVEL_NAMES.get(level, "Unknown")
            title = source_entry.get("title", "Unknown source")
            source_url = source_entry.get("url", source_entry.get("url_or_name", ""))
            finding = source_entry.get("finding", "")
            url_part = f" — {source_url}" if source_url else ""
            lines.append(
                f"{source_idx}. **(Level {level} — {level_name})** "
                f"{title}{url_part} — {finding} **Contradicts claim.**"
            )
        lines.append("")

    lines.append(f"**Red Team Verdict:** {verdict} — {summary}")
    lines.append("")

    return "\n".join(lines)


def _format_evaluation_evidence(
    result: dict, source_validation: dict, iteration: int,
) -> str:
    """Format Evidence Evaluation as markdown — citation-first."""
    verdict = _get(result, "verdict", "")
    summary = _to_str(_get(result, "summary", ""))
    confidence = _to_str(_get(result, "confidence", ""))

    lines = [f"### Evidence Evaluation (Iteration {iteration})\n"]

    # Source inventory table
    sources = _get(result, "sources_cited", [])
    if isinstance(sources, list) and sources:
        lines.append("| # | Source | Level | Direction | Key Finding |")
        lines.append("|---|--------|-------|-----------|-------------|")
        for source_idx, source_entry in enumerate(sources, 1):
            if not isinstance(source_entry, dict):
                continue
            title = source_entry.get("title", "Unknown")
            level = source_entry.get("hierarchy_level", "—")
            direction = source_entry.get("supports_or_contradicts", "—")
            finding = source_entry.get("finding", "—")
            lines.append(f"| {source_idx} | {title} | {level} | {direction} | {finding} |")
        lines.append("")

    lines.append(f"**Verdict:** {verdict} | **Confidence:** {confidence}")
    lines.append("")
    if summary:
        lines.append(f"**Assessment:** {summary}")
        lines.append("")

    # Source validation summary
    if source_validation:
        terminal_ok = source_validation.get("meets_terminal_threshold", False)
        max_level = source_validation.get("max_hierarchy_level")
        level_str = f" (best level: {max_level})" if max_level is not None else ""
        lines.append(
            f"**Source Hierarchy:** Meets terminal: "
            f"{'Yes' if terminal_ok else 'No'}{level_str}"
        )
        lines.append("")

    return "\n".join(lines)


# ── Provider Guard ───────────────────────────────────────────────────────


def _load_evidence_context(petri_dir: Path, node_id: str, dish_id: str) -> str:
    """Load accumulated evidence from evidence.md for use as iteration context."""
    node_path = _node_dir(petri_dir, node_id, dish_id)
    evidence_path = node_path / "evidence.md"
    if not evidence_path.exists():
        return ""
    content = evidence_path.read_text()
    # Skip the initial template header (everything before the first ---)
    # to avoid feeding back the raw claim text which is already in claim_text
    parts = content.split("\n---\n", 1)
    if len(parts) > 1:
        return parts[1].strip()
    return ""


class NoProviderError(RuntimeError):
    """Raised when the processor is invoked without a InferenceProvider."""

    def __init__(self) -> None:
        super().__init__(
            "No InferenceProvider configured. "
            "Set model and harness in petri.yaml to enable processing."
        )


# ── Phase Runners ────────────────────────────────────────────────────────


_SOCRATIC_STEPS = [
    {
        "step": "clarify",
        "prompt": (
            "SOCRATIC STEP 1 — CLARIFY: What exactly is being claimed? "
            "Identify every key term that needs a precise definition. "
            "What is ambiguous? What could be interpreted multiple ways? "
            "Return JSON with: \"definitions\" (key terms and their meanings), "
            "\"ambiguities\" (what's unclear), \"refined_claim\" (the claim restated precisely). "
            "Set \"verdict\" to exactly \"CLARIFIED\" once the claim has been restated "
            "precisely and every key term has been defined."
        ),
    },
    {
        "step": "challenge_assumptions",
        "prompt": (
            "SOCRATIC STEP 2 — CHALLENGE ASSUMPTIONS: What are we assuming is true "
            "without evidence? List every hidden premise. For each assumption, ask: "
            "Is this actually true, or do we just believe it? What would change if "
            "this assumption were false? "
            "Return JSON with: \"assumptions\" (list of hidden premises), "
            "\"if_false\" (what changes if each assumption fails), "
            "\"strongest_assumption\" (the most load-bearing one). "
            "Set \"verdict\" to exactly \"ASSUMPTIONS_CHALLENGED\" once every hidden "
            "assumption has been surfaced and pressure-tested."
        ),
    },
    {
        "step": "identify_evidence_needed",
        "prompt": (
            "SOCRATIC STEP 3 — MAP EVIDENCE: Based on the "
            "definitions and assumptions identified, what specific evidence would "
            "confirm or deny this claim? What would prove it false (falsification "
            "conditions)? What's the minimum evidence needed for a judgment? "
            "Return JSON with: \"evidence_needed\" (specific evidence to seek), "
            "\"falsification_conditions\" (what would disprove the claim), "
            "\"minimum_threshold\" (least evidence needed for a judgment). "
            "Set \"verdict\" to exactly \"EVIDENCE_MAPPED\" once the evidence plan "
            "and falsification conditions are on the table."
        ),
    },
]


def _run_socratic_phase(
    node_id: str,
    claim_text: str,
    petri_dir: Path,
    dish_id: str,
    iteration: int,
    provider: InferenceProvider,
) -> None:
    """Phase 0: Socratic questioning — clarify, challenge assumptions, identify evidence needed.

    This phase is **idempotent**.  If the node's event log already contains a
    ``verdict_issued`` event from any ``socratic_*`` agent, the phase is a
    no-op: it just advances the queue state to ``research_active`` and
    returns.  This prevents a restarted ``petri grow`` run from re-appending a
    duplicate ``### Socratic Analysis`` block to ``evidence.md``.
    """
    events_file = _events_path(petri_dir, node_id, dish_id)
    queue_file = _queue_path(petri_dir)

    # Idempotency guard — if any prior socratic_* verdict exists for this
    # node, assume the Socratic phase has already run and skip it.
    prior_verdicts = query_events(
        events_file, node_id=node_id, event_type="verdict_issued",
    )
    for prior_event in prior_verdicts:
        prior_agent = prior_event.get("agent", "")
        if isinstance(prior_agent, str) and prior_agent.startswith("socratic_"):
            update_state(
                queue_file, node_id, QueueState.research_active.value,
            )
            return

    lines = ["### Socratic Analysis\n"]

    for step in _SOCRATIC_STEPS:
        step_name = step["step"]
        prompt_text = step["prompt"]

        # Use assess_node with a synthetic "socratic_questioner" role
        context = {
            "iteration": iteration,
            "phase": f"socratic_{step_name}",
            "focused_directive": prompt_text,
        }
        result = provider.assess_node(
            node_id, claim_text, context, "socratic_questioner"
        )

        # Log as verdict_issued
        append_event(
            events_path=events_file,
            node_id=node_id,
            event_type="verdict_issued",
            agent=f"socratic_{step_name}",
            iteration=iteration,
            data=_verdict_data(result),
        )

        # Append to evidence
        summary = _to_str(_get(result, "summary", ""))
        arguments = _to_str(_get(result, "arguments", ""))
        step_title = step_name.replace("_", " ").title()
        lines.append(f"**{step_title}:**")
        if summary:
            lines.append(f"{summary}")
        if arguments:
            lines.append(f"\n{arguments}")
        lines.append("")

    # NOTE: The Socratic Analysis block is intentionally written to
    # ``evidence.md`` only ONCE — after the verification step below.
    # Writing it both before and after verification produced duplicate
    # ``### Socratic Analysis`` headers on a single run.

    # Verify the Socratic analysis was thorough
    verification_context = {
        "iteration": iteration,
        "phase": "socratic_verification",
        "prior_evidence": "\n".join(lines),
        "focused_directive": (
            "SOCRATIC STEP 4 — VERIFY: Verify that the Socratic analysis above "
            "is complete. Check: "
            "1) Were key terms actually DEFINED (not just mentioned)? "
            "2) Were ASSUMPTIONS explicitly identified and challenged? "
            "3) Were specific FALSIFICATION CONDITIONS stated? "
            "If any step was superficial or missing, set \"verdict\" to "
            "\"SOCRATIC_INCOMPLETE\". "
            "If all three prior steps were genuinely performed, set \"verdict\" "
            "to exactly \"VERIFIED\"."
        ),
    }
    verification = provider.assess_node(
        node_id, claim_text, verification_context, "socratic_questioner"
    )
    verification_verdict = _get(verification, "verdict", "VERIFIED")

    append_event(
        events_path=events_file,
        node_id=node_id,
        event_type="verdict_issued",
        agent="socratic_verifier",
        iteration=iteration,
        data=_verdict_data(verification),
    )

    # Append verification to the in-memory Socratic block, then write
    # the complete block to evidence.md exactly once.
    verification_summary = _to_str(_get(verification, "summary", ""))
    if verification_summary:
        lines.append(f"**Socratic Verification:** {verification_verdict}")
        lines.append(f"{verification_summary}\n")
    content = "\n".join(lines)
    _append_evidence(petri_dir, node_id, dish_id, "socratic", iteration, content)

    # Transition to research phase
    update_state(queue_file, node_id, QueueState.research_active.value)


def _run_decomposition_audit(
    node_id: str,
    claim_text: str,
    petri_dir: Path,
    dish_id: str,
    iteration: int,
    provider: InferenceProvider,
) -> None:
    """First-principles re-examination when convergence fails repeatedly.

    Asks whether the decomposition itself is flawed — not just the evidence.
    """
    events_file = _events_path(petri_dir, node_id, dish_id)
    prior_evidence = _load_evidence_context(petri_dir, node_id, dish_id)

    context = {
        "iteration": iteration,
        "phase": "decomposition_audit",
        "prior_evidence": prior_evidence,
        "focused_directive": (
            "This node has FAILED TO CONVERGE after multiple iterations. "
            "The agents could not agree on a verdict. This means the problem "
            "may be in the DECOMPOSITION, not the evidence. "
            "Ask: Are the assumptions for this node actually the RIGHT assumptions? "
            "Should this claim be broken down DIFFERENTLY? "
            "Is the claim itself poorly formed or ambiguous? "
            "In 'arguments', explain what's structurally wrong with how this node "
            "is framed. In 'evidence', suggest how it should be restructured."
        ),
    }
    result = provider.assess_node(
        node_id, claim_text, context, "socratic_questioner"
    )

    suggestion = _to_str(_get(result, "arguments", ""))
    should_restructure = "restructur" in suggestion.lower() or "redefin" in suggestion.lower()

    append_event(
        events_path=events_file,
        node_id=node_id,
        event_type="decomposition_audit",
        agent="decomposition_auditor",
        iteration=iteration,
        data={
            "iteration": iteration,
            "suggestion": suggestion,
            "should_restructure": should_restructure,
        },
    )

    # Append audit to evidence file
    audit_lines = [
        f"### Decomposition Audit (Iteration {iteration})\n",
        f"**Convergence failed after {iteration + 1} iterations.**\n",
        f"**Re-examination:** {_to_str(_get(result, 'summary', ''))}",
        "",
    ]
    if suggestion:
        audit_lines.append(f"**Structural Suggestions:** {suggestion}\n")
    if should_restructure:
        audit_lines.append("**Recommendation:** This node may need to be restructured.\n")

    _append_evidence(
        petri_dir, node_id, dish_id, "decomposition_audit", iteration,
        "\n".join(audit_lines),
    )


def _run_phase1(
    node_id: str,
    claim_text: str,
    petri_dir: Path,
    dish_id: str,
    iteration: int,
    provider: InferenceProvider,
    agent_roles: dict,
    queue_entry: dict,
) -> list[dict]:
    """Phase 1: Research -- agents assigned to the research phase in config."""
    from petri.config import get_research_agents

    events_file = _events_path(petri_dir, node_id, dish_id)
    queue_file = _queue_path(petri_dir)
    phase1_agents = get_research_agents()
    verdicts_collected: list[dict] = []

    prior_evidence = _load_evidence_context(petri_dir, node_id, dish_id)
    context = {
        "iteration": iteration,
        "weakest_link": queue_entry.get("weakest_link"),
        "focused_directive": queue_entry.get("focused_directive"),
        "prior_evidence": prior_evidence,
    }

    for agent_name in phase1_agents:
        result = provider.assess_node(
            node_id, claim_text, context, agent_name
        )

        # Log verdict_issued event
        append_event(
            events_path=events_file,
            node_id=node_id,
            event_type="verdict_issued",
            agent=agent_name,
            iteration=iteration,
            data=_verdict_data(result),
        )
        _log_sources_from_result(events_file, node_id, agent_name, iteration, result)
        verdicts_collected.append(result)

    # Append research findings to evidence file
    content = _format_phase1_evidence(verdicts_collected, iteration)
    _append_evidence(petri_dir, node_id, dish_id, "research", iteration, content)

    # Transition to critique_active
    update_state(queue_file, node_id, QueueState.critique_active.value)
    return verdicts_collected


def _run_phase2(
    node_id: str,
    claim_text: str,
    petri_dir: Path,
    dish_id: str,
    iteration: int,
    provider: InferenceProvider,
    agent_roles: dict,
    debate_pairings: list | None,
    queue_entry: dict,
) -> list[dict]:
    """Phase 2: Critique -- agents assigned to the critique phase in config."""
    from petri.config import get_critique_agents

    events_file = _events_path(petri_dir, node_id, dish_id)
    queue_file = _queue_path(petri_dir)
    phase2_agents = get_critique_agents()
    verdicts_collected: list[dict] = []
    agent_outputs: dict[str, dict] = {}

    prior_evidence = _load_evidence_context(petri_dir, node_id, dish_id)
    context = {
        "iteration": iteration,
        "weakest_link": queue_entry.get("weakest_link"),
        "focused_directive": queue_entry.get("focused_directive"),
        "prior_evidence": prior_evidence,
    }

    for agent_name in phase2_agents:
        result = provider.assess_node(
            node_id, claim_text, context, agent_name
        )

        # Log verdict_issued event
        append_event(
            events_path=events_file,
            node_id=node_id,
            event_type="verdict_issued",
            agent=agent_name,
            iteration=iteration,
            data=_verdict_data(result),
        )
        _log_sources_from_result(events_file, node_id, agent_name, iteration, result)
        verdicts_collected.append(result)
        agent_outputs[agent_name] = result

    # Run debate mediation
    debate_results = []
    pairings = debate_pairings or load_debate_pairings()
    for debate in pairings:
        agent_a_name = debate.pair[0]
        agent_b_name = debate.pair[1]
        agent_a_output = agent_outputs[agent_a_name]
        agent_b_output = agent_outputs[agent_b_name]

        debate_result = mediate_debate(
            agent_a_output=agent_a_output,
            agent_b_output=agent_b_output,
            debate=debate,
            provider=provider,
        )
        log_debate(
            events_path=events_file,
            node_id=node_id,
            iteration=iteration,
            debate_result=debate_result,
        )
        debate_results.append(debate_result)

    # Append critique assessment to evidence file
    content = _format_phase2_evidence(verdicts_collected, debate_results, iteration)
    _append_evidence(petri_dir, node_id, dish_id, "critique", iteration, content)

    # Transition to mediating
    update_state(queue_file, node_id, QueueState.mediating.value)
    return verdicts_collected


def _run_convergence(
    node_id: str,
    claim_text: str,
    petri_dir: Path,
    dish_id: str,
    iteration: int,
    provider: InferenceProvider,
    agent_roles: dict,
    queue_entry: dict,
) -> ConvergenceOutcome:
    """Convergence check -- determines converged, iterate, or stalled."""
    events_file = _events_path(petri_dir, node_id, dish_id)
    queue_file = _queue_path(petri_dir)

    # Gather all verdicts for this node at the current iteration
    verdicts = get_verdicts(events_file, node_id=node_id, iteration=iteration)

    # Check for short circuits first
    short_circuit = evaluate_short_circuits(verdicts, agent_roles)
    if short_circuit:
        sc_type = short_circuit.type
        append_event(
            events_path=events_file,
            node_id=node_id,
            event_type="convergence_checked",
            agent="node_lead",
            iteration=iteration,
            data={
                "converged": False,
                "blocking_verdicts": {"short_circuit": short_circuit.model_dump()},
                "weakest_link": short_circuit.agent,
                "focused_directive": f"Short-circuit: {sc_type}",
            },
        )
        if sc_type == "needs_experiment":
            _update_node_status(petri_dir, node_id, NodeStatus.NEEDS_EXPERIMENT.value)
            update_state(queue_file, node_id, QueueState.stalled.value)
            update_state(queue_file, node_id, QueueState.needs_human.value)
            return ConvergenceOutcome(outcome="short_circuit", type=sc_type)
        elif sc_type == "defer_open":
            _update_node_status(petri_dir, node_id, NodeStatus.DEFER_OPEN.value)
            update_state(queue_file, node_id, QueueState.converged.value)
            update_state(queue_file, node_id, QueueState.deferred_open.value)
            return ConvergenceOutcome(outcome="short_circuit", type=sc_type)

    # Normal convergence check
    convergence = check_convergence(verdicts, agent_roles)

    # Check circuit breaker
    max_iter = queue_entry.get("max_iterations", MAX_ITERATIONS)
    cycle_start = queue_entry.get("cycle_start_iteration", 0)
    breaker_fires = compute_circuit_breaker(iteration, cycle_start, max_iter)

    # Log convergence event
    append_event(
        events_path=events_file,
        node_id=node_id,
        event_type="convergence_checked",
        agent="node_lead",
        iteration=iteration,
        data={
            "converged": convergence.converged,
            "blocking_verdicts": {
                agent_name: result_entry.get("verdict") for agent_name, result_entry in convergence.blocking_results.items()
            } or None,
            "weakest_link": convergence.weakest_link,
        },
    )

    if convergence.converged:
        update_state(queue_file, node_id, QueueState.converged.value)
        return ConvergenceOutcome(outcome="converged")

    # Not converged -- iterate or stall
    if breaker_fires:
        # Before stalling, run a decomposition audit (first-principles re-examination)
        _run_decomposition_audit(
            node_id, claim_text, petri_dir, dish_id, iteration, provider,
        )
        update_state(queue_file, node_id, QueueState.stalled.value)
        update_state(queue_file, node_id, QueueState.needs_human.value)
        _update_node_status(petri_dir, node_id, NodeStatus.STALLED.value)
        return ConvergenceOutcome(outcome="circuit_breaker")

    # Iterate: increment iteration, set weakest link, back to research_active
    # Note: we use set_iteration (not new_cycle) because this is a retry
    # within the same convergence cycle, not a fresh cycle start.
    weakest_link = convergence.weakest_link

    set_iteration(queue_file, node_id, iteration + 1)
    if weakest_link:
        set_weakest_link(queue_file, node_id, weakest_link)
        directive = f"Focus on addressing {weakest_link} concerns"
        set_focused_directive(queue_file, node_id, directive)

    update_state(queue_file, node_id, QueueState.research_active.value)
    return ConvergenceOutcome(outcome="iterate", weakest_link=weakest_link)


def _run_red_team(
    node_id: str,
    claim_text: str,
    petri_dir: Path,
    dish_id: str,
    iteration: int,
    provider: InferenceProvider,
    agent_roles: dict,
) -> dict:
    """Red Team phase -- red_team_lead attempts disproval."""
    events_file = _events_path(petri_dir, node_id, dish_id)
    queue_file = _queue_path(petri_dir)

    update_state(queue_file, node_id, QueueState.red_team_active.value)

    context = {"iteration": iteration, "phase": "red_team"}
    result = provider.assess_node(node_id, claim_text, context, "red_team_lead")

    append_event(
        events_path=events_file,
        node_id=node_id,
        event_type="verdict_issued",
        agent="red_team_lead",
        iteration=iteration,
        data=_verdict_data(result),
    )
    _log_sources_from_result(events_file, node_id, "red_team_lead", iteration, result)

    # Append red team findings to evidence file
    content = _format_red_team_evidence(result, iteration)
    _append_evidence(petri_dir, node_id, dish_id, "red_team", iteration, content)

    update_state(queue_file, node_id, QueueState.evaluating.value)
    return result


def _run_evaluation(
    node_id: str,
    claim_text: str,
    petri_dir: Path,
    dish_id: str,
    iteration: int,
    provider: InferenceProvider,
    agent_roles: dict,
) -> EvaluationResult:
    """Evidence Evaluation -- final verdict based on source hierarchy."""
    events_file = _events_path(petri_dir, node_id, dish_id)
    queue_file = _queue_path(petri_dir)

    # Validate terminal sources
    source_validation = validate_terminal_sources(events_file, node_id)

    context = {
        "iteration": iteration,
        "phase": "evaluation",
        "source_validation": source_validation,
    }
    result = provider.assess_node(
        node_id, claim_text, context, "evidence_evaluator"
    )

    append_event(
        events_path=events_file,
        node_id=node_id,
        event_type="verdict_issued",
        agent="evidence_evaluator",
        iteration=iteration,
        data=_verdict_data(result),
    )
    _log_sources_from_result(events_file, node_id, "evidence_evaluator", iteration, result)

    # Determine final node status from evaluator verdict
    evaluator_verdict = _get(result, "verdict", "EVIDENCE_CONFIRMS")
    if evaluator_verdict == "EVIDENCE_CONFIRMS":
        final_status = NodeStatus.VALIDATED.value
    elif evaluator_verdict == "EVIDENCE_REFUTES":
        final_status = NodeStatus.DISPROVEN.value
    else:
        # EVIDENCE_INCONCLUSIVE or unknown
        final_status = NodeStatus.DEFER_OPEN.value

    # Append evaluation findings to evidence file
    content = _format_evaluation_evidence(result, source_validation, iteration)
    _append_evidence(petri_dir, node_id, dish_id, "evaluation", iteration, content)
    _update_evidence_status(petri_dir, node_id, dish_id, final_status)

    _update_node_status(petri_dir, node_id, final_status)
    update_state(queue_file, node_id, QueueState.done.value)

    return EvaluationResult(verdict=evaluator_verdict, final_status=final_status)


# ── Main Pipeline ────────────────────────────────────────────────────────


def process_node(
    node_id: str,
    petri_dir: Path,
    provider: InferenceProvider,
    agent_roles: dict | None = None,
    debate_pairings: list | None = None,
) -> ProcessNodeResult:
    """Process a single node through the validation pipeline.

    Drives the node from its current queue state through successive phases
    until it reaches a terminal state (done, needs_human, deferred) or a
    graceful stop is requested.

    Returns a status dict:
        {node_id, final_state, iterations, events_logged, ...}
    """
    dish_id = _get_dish_id(petri_dir)
    queue_file = _queue_path(petri_dir)
    events_file = _events_path(petri_dir, node_id, dish_id)

    # Load agent roles if not provided
    if agent_roles is None:
        agent_roles = load_agent_roles()

    # Load node metadata for claim_text
    metadata = _load_node_metadata(petri_dir, node_id, dish_id)
    claim_text = metadata.get("claim_text", "")

    # Track processing stats
    iterations_run = 0
    events_before = len(get_verdicts(events_file, node_id=node_id))

    # Load queue entry
    from petri.storage.queue import load_queue

    queue = load_queue(queue_file)
    if node_id not in queue.get("entries", {}):
        return ProcessNodeResult(
            node_id=node_id,
            final_state="not_in_queue",
            iterations=0,
            events_logged=0,
            error=f"Node {node_id} not found in queue",
        )

    # Drive the state machine until we reach a terminal state
    max_loops = 50  # safety limit to prevent infinite loops
    loop_count = 0

    while loop_count < max_loops:
        loop_count += 1

        # Reload queue entry each loop iteration to get fresh state
        queue = load_queue(queue_file)
        entry = queue["entries"].get(node_id, {})
        current_state = entry.get("queue_state", "done")
        iteration = entry.get("iteration", 0)

        # Check for graceful stop between phases
        if is_stop_requested():
            logger.info("Graceful stop requested for node %s", node_id)
            if current_state not in ("done", "needs_human", "deferred_open", "deferred_closed"):
                try:
                    update_state(queue_file, node_id, QueueState.stalled.value)
                except ValueError:
                    pass  # Already in a non-stallable state
            break

        # Terminal states -- we are done
        if current_state in ("done", "needs_human", "deferred_open", "deferred_closed"):
            break

        # Dispatch based on current state
        if current_state == QueueState.queued.value:
            _update_node_status(petri_dir, node_id, NodeStatus.RESEARCH.value)
            update_state(queue_file, node_id, QueueState.socratic_active.value)

        elif current_state == QueueState.socratic_active.value:
            _run_socratic_phase(
                node_id, claim_text, petri_dir, dish_id,
                iteration, provider,
            )

        elif current_state == QueueState.research_active.value:
            _run_phase1(
                node_id, claim_text, petri_dir, dish_id,
                iteration, provider, agent_roles, entry,
            )
            iterations_run += 1

        elif current_state == QueueState.critique_active.value:
            _run_phase2(
                node_id, claim_text, petri_dir, dish_id,
                iteration, provider, agent_roles, debate_pairings, entry,
            )

        elif current_state == QueueState.mediating.value:
            convergence_outcome = _run_convergence(
                node_id, claim_text, petri_dir, dish_id,
                iteration, provider, agent_roles, entry,
            )
            if convergence_outcome.outcome == "iterate":
                iterations_run += 1

        elif current_state == QueueState.converged.value:
            _update_node_status(petri_dir, node_id, NodeStatus.RED_TEAM.value)
            _run_red_team(
                node_id, claim_text, petri_dir, dish_id,
                iteration, provider, agent_roles,
            )

        elif current_state == QueueState.evaluating.value:
            _update_node_status(petri_dir, node_id, NodeStatus.EVALUATE.value)
            _run_evaluation(
                node_id, claim_text, petri_dir, dish_id,
                iteration, provider, agent_roles,
            )

        elif current_state == QueueState.stalled.value:
            update_state(queue_file, node_id, QueueState.needs_human.value)

        else:
            # Unknown or unhandled state
            logger.warning("Unhandled queue state %s for node %s", current_state, node_id)
            break

    # Compute final state
    queue = load_queue(queue_file)
    final_entry = queue["entries"].get(node_id, {})
    final_state = final_entry.get("queue_state", "unknown")

    events_after = len(get_verdicts(events_file, node_id=node_id))
    events_logged = events_after - events_before

    return ProcessNodeResult(
        node_id=node_id,
        final_state=final_state,
        iterations=iterations_run,
        events_logged=events_logged,
        final_iteration=final_entry.get("iteration", 0),
    )


# ── Eligible Node Discovery ─────────────────────────────────────────────


def find_eligible_nodes(
    petri_dir: Path,
    dish_id: str,
    node_ids: list[str] | None = None,
    colony_filter: str | None = None,
    all_nodes: bool = False,
) -> list[str]:
    """Find nodes eligible for validation.

    Eligible = cell nodes (or nodes whose deps are all VALIDATED) in NEW status.

    When *node_ids* is provided, only those specific nodes are considered.
    When *colony_filter* is provided, only nodes from that colony are scanned.
    When *all_nodes* is True, all colonies are scanned.
    """
    from petri.graph.colony import deserialize_colony

    dishes_dir = petri_dir / "petri-dishes"
    if not dishes_dir.exists():
        return []

    eligible: list[str] = []

    # If specific node IDs are given, return them directly (caller knows best)
    if node_ids:
        return list(node_ids)

    # Scan colonies
    for colony_dir in sorted(dishes_dir.iterdir()):
        if not colony_dir.is_dir():
            continue

        colony_slug = colony_dir.name
        if colony_filter and colony_slug != colony_filter:
            continue

        try:
            graph, colony = deserialize_colony(colony_dir, dish_id)
        except Exception:
            continue

        # Build status map from metadata files
        nodes_status: dict[str, NodeStatus] = {}
        for node in graph.get_nodes():
            nodes_status[node.id] = node.status

        # Get eligible nodes from graph
        for node in graph.get_eligible_for_validation(nodes_status):
            eligible.append(node.id)

        # If not processing all, stop after first colony
        if not all_nodes and not colony_filter:
            break

    return eligible


# ── Queue Processor ──────────────────────────────────────────────────────


def process_queue(
    petri_dir: Path,
    provider: InferenceProvider | None = None,
    max_concurrent: int = MAX_CONCURRENT,
    node_ids: list[str] | None = None,
    colony_filter: str | None = None,
    all_nodes: bool = False,
    dry_run: bool = False,
) -> QueueProcessingResult:
    """Process the queue, running eligible nodes through the pipeline.

    Uses ``ThreadPoolExecutor`` for concurrent processing.

    Returns summary dict:
        {processed, succeeded, failed, stalled, results, ...}
    """
    if not dry_run and provider is None:
        raise NoProviderError()

    reset_stop()

    dish_id = _get_dish_id(petri_dir)
    queue_file = _queue_path(petri_dir)

    # Find eligible nodes
    eligible = find_eligible_nodes(
        petri_dir, dish_id,
        node_ids=node_ids,
        colony_filter=colony_filter,
        all_nodes=all_nodes,
    )

    if dry_run:
        return QueueProcessingResult(
            processed=0,
            succeeded=0,
            failed=0,
            stalled=0,
            dry_run=True,
            would_process=eligible,
        )

    if not eligible:
        return QueueProcessingResult(
            processed=0,
            succeeded=0,
            failed=0,
            stalled=0,
            results=[],
        )

    # Enqueue eligible nodes that are not already in the queue
    existing_queue = {entry["node_id"] for entry in list_queue(queue_file)}
    for nid in eligible:
        if nid not in existing_queue:
            try:
                add_to_queue(queue_file, nid)
            except ValueError:
                pass  # Already in queue (race condition)

    # Load shared config
    agent_roles = load_agent_roles()
    debate_pairings = load_debate_pairings()

    # Process concurrently
    results: list[ProcessNodeResult] = []
    succeeded = 0
    failed = 0
    stalled = 0

    def _process_one(nid: str) -> ProcessNodeResult:
        try:
            return process_node(
                nid, petri_dir,
                provider=provider,
                agent_roles=agent_roles,
                debate_pairings=debate_pairings,
            )
        except Exception as exc:
            logger.exception("Error processing node %s", nid)
            return ProcessNodeResult(
                node_id=nid,
                final_state="error",
                iterations=0,
                events_logged=0,
                error=str(exc),
            )

    from petri.engine.load_balancer import AdaptiveLoadBalancer

    balancer = AdaptiveLoadBalancer(
        max_workers=max_concurrent,
        min_workers=1,
    )
    balancer.start()

    try:
        remaining = list(eligible)
        active_futures: dict = {}

        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            while remaining or active_futures:
                # Submit new work up to the balancer's recommended level
                while remaining and len(active_futures) < balancer.recommended_workers:
                    nid = remaining.pop(0)
                    future = executor.submit(_process_one, nid)
                    active_futures[future] = nid

                # Poll for completed futures
                newly_done = [completed for completed in active_futures if completed.done()]

                for future in newly_done:
                    node_result = future.result()
                    results.append(node_result)
                    del active_futures[future]
                    final = node_result.final_state
                    if final == "done":
                        succeeded += 1
                    elif final in ("needs_human", "stalled"):
                        stalled += 1
                    elif final == "error":
                        failed += 1
                    else:
                        succeeded += 1

                # Brief pause before re-checking
                if not newly_done and (remaining or active_futures):
                    import time
                    time.sleep(0.5)
    finally:
        balancer.stop()

    return QueueProcessingResult(
        processed=len(results),
        succeeded=succeeded,
        failed=failed,
        stalled=stalled,
        results=results,
    )
