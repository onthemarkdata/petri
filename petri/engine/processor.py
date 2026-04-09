"""Pipeline processor for the Petri Research Orchestration Framework.

Processes cells through the full validation pipeline:
research -> critique -> convergence -> red team -> evaluation.

Pure library logic -- no CLI, no UI.  The CLI (``petri grow``) calls this.
"""

from __future__ import annotations

import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

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
    CellStatus,
    ConvergenceOutcome,
    EvaluationResult,
    InferenceProvider,
    ProcessCellResult,
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


# ── Lifecycle Callback Types ────────────────────────────────────────────


@dataclass
class CellProgressEvent:
    """Lifecycle event fired by process_cell at every state transition.

    Consumed by the CLI's MultiSpinner to update per-slot rows in real time.
    The engine itself never reads these events — they're a write-only signal.
    """
    slot_idx: int               # which worker slot owns this cell (-1 if unassigned)
    cell_id: str
    kind: str                   # "started" | "phase" | "agent" | "verdict" | "agent_text" | "finished"
    phase: str | None = None    # set when kind in {"phase", "agent", "verdict", "agent_text"}
    agent: str | None = None    # set when kind in {"agent", "verdict", "agent_text"}
    verdict: str | None = None  # set when kind == "verdict"
    iteration: int | None = None
    error: str | None = None    # set when kind == "finished" with failure
    text: str | None = None     # set when kind == "agent_text" — streaming model chunk


CellProgressCallback = Callable[[CellProgressEvent], None]


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


def _colony_slug(cell_id: str, dish_id: str) -> str:
    """Extract the colony slug from a cell's composite key."""
    dish_prefix = dish_id + "-"
    if cell_id.startswith(dish_prefix):
        colony_and_rest = cell_id[len(dish_prefix):]
        return colony_and_rest.rsplit("-", 2)[0]
    parts = cell_id.split("-")
    return "-".join(parts[:-2])


def _load_cell_paths(petri_dir: Path, cell_id: str, dish_id: str) -> dict[str, str]:
    """Load cell_paths from colony.json for the cell's colony."""
    slug = _colony_slug(cell_id, dish_id)
    colony_json = petri_dir / "petri-dishes" / slug / "colony.json"
    if colony_json.exists():
        data = json.loads(colony_json.read_text())
        return data.get("cell_paths", {})
    return {}


def _cell_dir(petri_dir: Path, cell_id: str, dish_id: str) -> Path:
    """Derive the filesystem path for a cell.

    Looks up the human-readable path from colony.json's cell_paths mapping.
    Falls back to flat {level}-{seq} layout for backwards compatibility.
    """
    slug = _colony_slug(cell_id, dish_id)
    colony_base = petri_dir / "petri-dishes" / slug

    # Try cell_paths from colony.json
    cell_paths = _load_cell_paths(petri_dir, cell_id, dish_id)
    if cell_id in cell_paths:
        return colony_base / cell_paths[cell_id]

    # Fallback: flat layout
    parts = cell_id.split("-")
    return colony_base / f"{parts[-2]}-{parts[-1]}"


def _events_path(petri_dir: Path, cell_id: str, dish_id: str) -> Path:
    return _cell_dir(petri_dir, cell_id, dish_id) / "events.jsonl"


def _metadata_path(petri_dir: Path, cell_id: str, dish_id: str) -> Path:
    return _cell_dir(petri_dir, cell_id, dish_id) / "metadata.json"


def _load_cell_metadata(petri_dir: Path, cell_id: str, dish_id: str) -> dict:
    """Load a cell's metadata.json."""
    path = _metadata_path(petri_dir, cell_id, dish_id)
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


# ── Cell Status Update ──────────────────────────────────────────────────


def _update_cell_status(petri_dir: Path, cell_id: str, new_status: str) -> None:
    """Update a cell's status in its metadata.json file."""
    dish_id = _get_dish_id(petri_dir)
    path = _metadata_path(petri_dir, cell_id, dish_id)
    if not path.exists():
        logger.warning("metadata.json not found for cell %s at %s", cell_id, path)
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


def _source_to_dict(source_entry: object) -> dict | None:
    """Normalize a cited source to a plain dict.

    Accepts:

    * ``SourceCitation`` Pydantic instances (or any object exposing
      ``model_dump``) — returned via ``model_dump(exclude_none=True)``.
    * Raw dicts — returned verbatim.
    * Anything else — returns ``None``.

    **Critical**: the evidence.md formatters (``_format_phase1_evidence``,
    ``_format_phase2_evidence``, ``_format_red_team_evidence``,
    ``_format_evaluation_evidence``) receive ``AssessmentResult``
    Pydantic objects from the phase runners. Their ``sources_cited``
    fields are lists of ``SourceCitation`` Pydantic instances — NOT
    dicts. Before this helper existed, each formatter did
    ``if not isinstance(source, dict): continue`` and silently dropped
    100% of the cited sources on the floor. The evidence.md files
    ended up with zero URLs even though the ``verdict_issued`` events
    on disk had the full source payloads via ``_verdict_data``. Every
    formatter MUST pipe each source through this helper before reading
    fields.
    """
    if source_entry is None:
        return None
    if hasattr(source_entry, "model_dump"):
        try:
            return source_entry.model_dump(exclude_none=True)
        except Exception:
            return None
    if isinstance(source_entry, dict):
        return source_entry
    return None


def _render_source_line(source_index: int, source_dict: dict) -> str:
    """Render one cited source as a single markdown bullet line.

    Shared by every formatter so the citation format is identical
    across phases (research, critique, red team, evaluation).
    """
    level = source_dict.get("hierarchy_level", 6)
    try:
        level_int = int(level)
    except (TypeError, ValueError):
        level_int = 6
    level_name = _LEVEL_NAMES.get(level_int, "Unknown")
    title = source_dict.get("title") or "Unknown source"
    url = source_dict.get("url") or source_dict.get("url_or_name") or ""
    finding = source_dict.get("finding") or ""
    direction_raw = (source_dict.get("supports_or_contradicts") or "supports").lower()
    if direction_raw == "contradicts":
        direction_label = "Contradicts claim."
    elif direction_raw == "supports":
        direction_label = "Supports claim."
    else:
        direction_label = f"{direction_raw.capitalize()}."
    url_part = f" — {url}" if url else ""
    return (
        f"**Source {source_index} (Level {level_int} — {level_name}):** "
        f"{title}{url_part} — {finding} **{direction_label}**"
    )


def _iter_verdict_sources(verdict_entry: object):
    """Yield normalized dicts for every valid source on a verdict entry.

    ``verdict_entry`` may be an ``AssessmentResult`` Pydantic object or
    a dict (the two shapes the formatters receive). Returns an empty
    iterator if there are no sources. Invalid / non-dict-like entries
    are silently dropped at the helper boundary.
    """
    sources = _get(verdict_entry, "sources_cited", [])
    if not isinstance(sources, list):
        return
    for source_entry in sources:
        source_dict = _source_to_dict(source_entry)
        if source_dict is not None:
            yield source_dict


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
    events_file: Path, cell_id: str, agent: str, iteration: int, result: object,
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
            cell_id=cell_id,
            event_type="source_reviewed",
            agent=agent,
            iteration=iteration,
            data=data,
        )


def _append_evidence(
    petri_dir: Path, cell_id: str, dish_id: str,
    phase: str, iteration: int, content: str,
) -> None:
    """Append a section to a cell's evidence.md and log evidence_appended."""
    cell_path = _cell_dir(petri_dir, cell_id, dish_id)
    evidence_path = cell_path / "evidence.md"
    if not evidence_path.exists():
        return

    with open(evidence_path, "a") as f:
        f.write(f"\n\n---\n\n{content}")

    events_file = _events_path(petri_dir, cell_id, dish_id)
    append_event(
        events_path=events_file,
        cell_id=cell_id,
        event_type="evidence_appended",
        agent="cell_lead",
        iteration=iteration,
        data={"summary": f"Appended {phase} findings (iteration {iteration})"},
    )


def _update_evidence_status(
    petri_dir: Path, cell_id: str, dish_id: str, new_status: str,
) -> None:
    """Update the Status line in a cell's evidence.md file."""
    import re as _re

    cell_path = _cell_dir(petri_dir, cell_id, dish_id)
    evidence_path = cell_path / "evidence.md"
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


def _write_summary(
    petri_dir: Path,
    cell_id: str,
    dish_id: str,
    claim_text: str,
    iteration: int,
    provider: "InferenceProvider",
) -> None:
    """Regenerate ``summary.md`` for a cell from the current state of
    ``evidence.md``. Called at the end of every iteration and once more
    at the end of the final evaluation, so the file is always current.

    Swallows errors instead of raising: a failed summary must never
    break the main pipeline. The raw ``evidence.md`` is always
    available as a fallback for debugging.
    """
    cell_path = _cell_dir(petri_dir, cell_id, dish_id)
    evidence_path = cell_path / "evidence.md"
    if not evidence_path.exists():
        return
    try:
        evidence_md = evidence_path.read_text()
    except OSError:
        return
    if not evidence_md.strip():
        return

    try:
        summary_md = provider.summarize_evidence(
            cell_id=cell_id,
            claim_text=claim_text,
            evidence_md=evidence_md,
            iteration=iteration,
        )
    except Exception as exc:  # noqa: BLE001 - summary is best-effort
        logger.warning(
            "summary generation failed for cell %s iteration %d: %s",
            cell_id, iteration, exc,
        )
        return

    try:
        (cell_path / "summary.md").write_text(summary_md)
    except OSError as exc:
        logger.warning(
            "could not write summary.md for cell %s: %s", cell_id, exc,
        )
        return

    try:
        append_event(
            events_path=_events_path(petri_dir, cell_id, dish_id),
            cell_id=cell_id,
            event_type="evidence_summarized",
            agent="cell_lead",
            iteration=iteration,
            data={
                "summary_length": len(summary_md),
                "evidence_length": len(evidence_md),
            },
        )
    except Exception as exc:  # noqa: BLE001 — audit log is best-effort per docstring
        logger.warning(
            "could not log evidence_summarized event for cell %s: %s",
            cell_id, exc,
        )


def _format_phase1_evidence(verdicts: list, iteration: int) -> str:
    """Format Phase 1 (Research) findings as markdown — citation-first.

    ``verdicts`` may contain ``AssessmentResult`` Pydantic objects OR
    dicts; every source is normalized via ``_source_to_dict`` before
    rendering so Pydantic instances can't be silently skipped.

    Sources are deduplicated by URL within this phase block so two
    agents citing the same paper with different titles collapse into
    a single numbered entry, crediting the first agent that surfaced it.
    """
    lines = [f"### Iteration {iteration} — Phase 1 Research\n"]

    # Collect all sources across phase 1 agents into a numbered list.
    source_num = 0
    seen_urls: set[str] = set()
    agent_summaries: dict[str, tuple[str, str]] = {}  # agent -> (verdict, summary)

    for verdict_entry in verdicts:
        agent_name = _get(verdict_entry, "agent", "unknown")
        verdict_value = _get(verdict_entry, "verdict", "")
        summary_text = _get(verdict_entry, "summary", "")
        agent_summaries[agent_name] = (verdict_value, summary_text)

        for source_dict in _iter_verdict_sources(verdict_entry):
            url = source_dict.get("url") or source_dict.get("url_or_name") or ""
            if url and url in seen_urls:
                continue
            if url:
                seen_urls.add(url)
            source_num += 1
            lines.append(_render_source_line(source_num, source_dict))
            lines.append("")

    if source_num == 0:
        lines.append("_No sources cited by research agents._")
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
    """Format Phase 2 (Critique) assessment as markdown — citation-first.

    ``verdicts`` may contain ``AssessmentResult`` Pydantic objects OR
    dicts; every source is normalized via ``_source_to_dict`` before
    rendering. Historically this formatter only produced a verdict
    summary table and never surfaced cited sources, which meant every
    skeptic/champion/pragmatist URL landed in the event log but never
    in evidence.md. Now we render sources too.
    """
    lines = [f"### Iteration {iteration} — Phase 2 Critique\n"]

    # Collect all sources across phase 2 agents into a numbered list.
    # Dedup by URL so multiple agents citing the same paper render once.
    source_num = 0
    seen_urls: set[str] = set()
    for verdict_entry in verdicts:
        for source_dict in _iter_verdict_sources(verdict_entry):
            url = source_dict.get("url") or source_dict.get("url_or_name") or ""
            if url and url in seen_urls:
                continue
            if url:
                seen_urls.add(url)
            source_num += 1
            lines.append(_render_source_line(source_num, source_dict))
            lines.append("")

    if source_num == 0:
        lines.append("_No sources cited by critique agents._")
        lines.append("")

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


def _format_red_team_evidence(result: object, iteration: int) -> str:
    """Format Red Team review as markdown — citation-first.

    ``result`` may be an ``AssessmentResult`` Pydantic object OR a
    dict; sources are normalized via ``_source_to_dict``.
    """
    verdict = _get(result, "verdict", "")
    summary = _to_str(_get(result, "summary", ""))

    lines = [f"### Red Team Review (Iteration {iteration})\n"]

    # Numbered counter-arguments from sources (deduped by URL).
    source_num = 0
    seen_urls: set[str] = set()
    for source_dict in _iter_verdict_sources(result):
        url = source_dict.get("url") or source_dict.get("url_or_name") or ""
        if url and url in seen_urls:
            continue
        if url:
            seen_urls.add(url)
        source_num += 1
        lines.append(_render_source_line(source_num, source_dict))
        lines.append("")

    if source_num == 0:
        lines.append("_No sources cited by red team._")
        lines.append("")

    lines.append(f"**Red Team Verdict:** {verdict} — {summary}")
    lines.append("")

    return "\n".join(lines)


def _format_evaluation_evidence(
    result: object, source_validation: dict, iteration: int,
) -> str:
    """Format Evidence Evaluation as markdown — citation-first.

    ``result`` may be an ``AssessmentResult`` Pydantic object OR a
    dict; sources are normalized via ``_source_to_dict``. The
    evaluation phase also renders the full URL under each table row
    so the final verdict report has every citation inline (this is
    the row the user will read when they're looking up a specific
    claim's source repository).
    """
    verdict = _get(result, "verdict", "")
    summary = _to_str(_get(result, "summary", ""))
    confidence = _to_str(_get(result, "confidence", ""))

    lines = [f"### Evidence Evaluation (Iteration {iteration})\n"]

    # Materialize a deduped source list (by URL) once so the numbered list
    # and the inventory table below render the same entries in the same
    # order. Sources without a URL are always kept (rare but possible).
    deduped_sources: list[dict] = []
    seen_urls: set[str] = set()
    for source_dict in _iter_verdict_sources(result):
        url = source_dict.get("url") or source_dict.get("url_or_name") or ""
        if url and url in seen_urls:
            continue
        if url:
            seen_urls.add(url)
        deduped_sources.append(source_dict)

    # Full source list with URLs (numbered, same format as the other
    # phases so a user can copy-paste citations between phases).
    for source_index, source_dict in enumerate(deduped_sources, 1):
        lines.append(_render_source_line(source_index, source_dict))
        lines.append("")
    source_num = len(deduped_sources)

    if source_num == 0:
        lines.append("_No sources cited by evidence evaluator._")
        lines.append("")

    # Compact source inventory table — one row per source, keyed to
    # the numbered entries above. Great for quick scanning.
    if source_num > 0:
        lines.append("| # | Source | URL | Level | Direction | Finding |")
        lines.append("|---|--------|-----|-------|-----------|---------|")
        for table_index, source_dict in enumerate(deduped_sources, 1):
            title = source_dict.get("title") or "Unknown"
            source_url = (
                source_dict.get("url") or source_dict.get("url_or_name") or ""
            )
            level = source_dict.get("hierarchy_level", "—")
            direction = source_dict.get("supports_or_contradicts") or "—"
            finding = source_dict.get("finding") or "—"
            lines.append(
                f"| {table_index} | {title} | {source_url} | {level} | "
                f"{direction} | {finding} |"
            )
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


def _load_evidence_context(petri_dir: Path, cell_id: str, dish_id: str) -> str:
    """Load accumulated evidence from evidence.md for use as iteration context."""
    cell_path = _cell_dir(petri_dir, cell_id, dish_id)
    evidence_path = cell_path / "evidence.md"
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
    cell_id: str,
    claim_text: str,
    petri_dir: Path,
    dish_id: str,
    iteration: int,
    provider: InferenceProvider,
    fire: Callable[..., None] | None = None,
    text_emitter: Callable[[str, str, str], None] | None = None,
) -> None:
    """Phase 0: Socratic questioning — clarify, challenge assumptions, identify evidence needed.

    This phase is **idempotent**.  If the cell's event log already contains a
    ``verdict_issued`` event from any ``socratic_*`` agent, the phase is a
    no-op: it just advances the queue state to ``research_active`` and
    returns.  This prevents a restarted ``petri grow`` run from re-appending a
    duplicate ``### Socratic Analysis`` block to ``evidence.md``.
    """
    events_file = _events_path(petri_dir, cell_id, dish_id)
    queue_file = _queue_path(petri_dir)

    # Idempotency guard — if any prior socratic_* verdict exists for this
    # cell, assume the Socratic phase has already run and skip it.
    prior_verdicts = query_events(
        events_file, cell_id=cell_id, event_type="verdict_issued",
    )
    for prior_event in prior_verdicts:
        prior_agent = prior_event.get("agent", "")
        if isinstance(prior_agent, str) and prior_agent.startswith("socratic_"):
            update_state(
                queue_file, cell_id, QueueState.research_active.value,
            )
            return

    lines = ["### Socratic Analysis\n"]

    for step in _SOCRATIC_STEPS:
        step_name = step["step"]
        prompt_text = step["prompt"]

        # Use assess_cell with a synthetic "socratic_questioner" role
        context = {
            "iteration": iteration,
            "phase": f"socratic_{step_name}",
            "focused_directive": prompt_text,
        }
        step_agent_name = f"socratic_{step_name}"
        if fire is not None:
            fire("agent", phase="socratic", agent=step_agent_name, iteration=iteration)
        if text_emitter is not None:
            def on_progress_step(chunk: str, _phase="socratic", _agent=step_agent_name) -> None:
                text_emitter(_phase, _agent, chunk)
        else:
            on_progress_step = None
        result = provider.assess_cell(
            cell_id, claim_text, context, "socratic_questioner",
            on_progress=on_progress_step,
        )
        if fire is not None:
            fire(
                "verdict",
                phase="socratic",
                agent=step_agent_name,
                verdict=str(_get(result, "verdict", "")),
                iteration=iteration,
            )

        # Log as verdict_issued
        append_event(
            events_path=events_file,
            cell_id=cell_id,
            event_type="verdict_issued",
            agent=f"socratic_{step_name}",
            iteration=iteration,
            data=_verdict_data(result),
        )
        # Log one source_reviewed event per cited source so the citation
        # repository captures anything the Socratic step grounded in a
        # real URL (e.g. a definition citing a published glossary).
        _log_sources_from_result(
            events_file, cell_id, f"socratic_{step_name}", iteration, result,
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
        # Render any cited sources from this Socratic step inline so
        # URLs show up in evidence.md (previously: sources were
        # persisted to events.jsonl via the provider but never
        # surfaced in the human-readable file).
        for source_dict in _iter_verdict_sources(result):
            lines.append(_render_source_line(len(lines), source_dict))
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
    if fire is not None:
        fire(
            "agent",
            phase="socratic",
            agent="socratic_verifier",
            iteration=iteration,
        )
    if text_emitter is not None:
        def on_progress_verify(chunk: str) -> None:
            text_emitter("socratic", "socratic_verifier", chunk)
    else:
        on_progress_verify = None
    verification = provider.assess_cell(
        cell_id, claim_text, verification_context, "socratic_questioner",
        on_progress=on_progress_verify,
    )
    verification_verdict = _get(verification, "verdict", "VERIFIED")
    if fire is not None:
        fire(
            "verdict",
            phase="socratic",
            agent="socratic_verifier",
            verdict=str(verification_verdict),
            iteration=iteration,
        )

    append_event(
        events_path=events_file,
        cell_id=cell_id,
        event_type="verdict_issued",
        agent="socratic_verifier",
        iteration=iteration,
        data=_verdict_data(verification),
    )
    _log_sources_from_result(
        events_file, cell_id, "socratic_verifier", iteration, verification,
    )

    # Append verification to the in-memory Socratic block, then write
    # the complete block to evidence.md exactly once.
    verification_summary = _to_str(_get(verification, "summary", ""))
    if verification_summary:
        lines.append(f"**Socratic Verification:** {verification_verdict}")
        lines.append(f"{verification_summary}\n")
    # Render any sources the verifier cited (e.g. linking to a
    # terminology reference used during clarification).
    for source_dict in _iter_verdict_sources(verification):
        lines.append(_render_source_line(len(lines), source_dict))
    content = "\n".join(lines)
    _append_evidence(petri_dir, cell_id, dish_id, "socratic", iteration, content)

    # Transition to research phase
    update_state(queue_file, cell_id, QueueState.research_active.value)


def _run_decomposition_audit(
    cell_id: str,
    claim_text: str,
    petri_dir: Path,
    dish_id: str,
    iteration: int,
    provider: InferenceProvider,
) -> None:
    """First-principles re-examination when convergence fails repeatedly.

    Asks whether the decomposition itself is flawed — not just the evidence.
    """
    events_file = _events_path(petri_dir, cell_id, dish_id)
    prior_evidence = _load_evidence_context(petri_dir, cell_id, dish_id)

    context = {
        "iteration": iteration,
        "phase": "decomposition_audit",
        "prior_evidence": prior_evidence,
        "focused_directive": (
            "This cell has FAILED TO CONVERGE after multiple iterations. "
            "The agents could not agree on a verdict. This means the problem "
            "may be in the DECOMPOSITION, not the evidence. "
            "Ask: Are the assumptions for this cell actually the RIGHT assumptions? "
            "Should this claim be broken down DIFFERENTLY? "
            "Is the claim itself poorly formed or ambiguous? "
            "In 'arguments', explain what's structurally wrong with how this cell "
            "is framed. In 'evidence', suggest how it should be restructured."
        ),
    }
    result = provider.assess_cell(
        cell_id, claim_text, context, "socratic_questioner"
    )

    suggestion = _to_str(_get(result, "arguments", ""))
    should_restructure = "restructur" in suggestion.lower() or "redefin" in suggestion.lower()

    append_event(
        events_path=events_file,
        cell_id=cell_id,
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
        audit_lines.append("**Recommendation:** This cell may need to be restructured.\n")

    _append_evidence(
        petri_dir, cell_id, dish_id, "decomposition_audit", iteration,
        "\n".join(audit_lines),
    )


def _run_phase1(
    cell_id: str,
    claim_text: str,
    petri_dir: Path,
    dish_id: str,
    iteration: int,
    provider: InferenceProvider,
    agent_roles: dict,
    queue_entry: dict,
    fire: Callable[..., None] | None = None,
    text_emitter: Callable[[str, str, str], None] | None = None,
) -> list[dict]:
    """Phase 1: Research -- agents assigned to the research phase in config."""
    from petri.config import get_research_agents

    events_file = _events_path(petri_dir, cell_id, dish_id)
    queue_file = _queue_path(petri_dir)
    phase1_agents = get_research_agents()
    verdicts_collected: list[dict] = []

    prior_evidence = _load_evidence_context(petri_dir, cell_id, dish_id)
    context = {
        "iteration": iteration,
        "weakest_link": queue_entry.get("weakest_link"),
        "focused_directive": queue_entry.get("focused_directive"),
        "prior_evidence": prior_evidence,
    }

    for agent_name in phase1_agents:
        if fire is not None:
            fire(
                "agent",
                phase="research",
                agent=agent_name,
                iteration=iteration,
            )
        if text_emitter is not None:
            def on_progress_research(chunk: str, _agent=agent_name) -> None:
                text_emitter("research", _agent, chunk)
        else:
            on_progress_research = None
        result = provider.assess_cell(
            cell_id, claim_text, context, agent_name,
            on_progress=on_progress_research,
        )
        if fire is not None:
            fire(
                "verdict",
                phase="research",
                agent=agent_name,
                verdict=str(_get(result, "verdict", "")),
                iteration=iteration,
            )

        # Log verdict_issued event
        append_event(
            events_path=events_file,
            cell_id=cell_id,
            event_type="verdict_issued",
            agent=agent_name,
            iteration=iteration,
            data=_verdict_data(result),
        )
        _log_sources_from_result(events_file, cell_id, agent_name, iteration, result)
        verdicts_collected.append(result)

    # Append research findings to evidence file
    content = _format_phase1_evidence(verdicts_collected, iteration)
    _append_evidence(petri_dir, cell_id, dish_id, "research", iteration, content)

    # Transition to critique_active
    update_state(queue_file, cell_id, QueueState.critique_active.value)
    return verdicts_collected


def _run_phase2(
    cell_id: str,
    claim_text: str,
    petri_dir: Path,
    dish_id: str,
    iteration: int,
    provider: InferenceProvider,
    agent_roles: dict,
    debate_pairings: list | None,
    queue_entry: dict,
    fire: Callable[..., None] | None = None,
    text_emitter: Callable[[str, str, str], None] | None = None,
) -> list[dict]:
    """Phase 2: Critique -- agents assigned to the critique phase in config."""
    from petri.config import get_critique_agents

    events_file = _events_path(petri_dir, cell_id, dish_id)
    queue_file = _queue_path(petri_dir)
    phase2_agents = get_critique_agents()
    verdicts_collected: list[dict] = []
    agent_outputs: dict[str, dict] = {}

    prior_evidence = _load_evidence_context(petri_dir, cell_id, dish_id)
    context = {
        "iteration": iteration,
        "weakest_link": queue_entry.get("weakest_link"),
        "focused_directive": queue_entry.get("focused_directive"),
        "prior_evidence": prior_evidence,
    }

    for agent_name in phase2_agents:
        if fire is not None:
            fire(
                "agent",
                phase="critique",
                agent=agent_name,
                iteration=iteration,
            )
        if text_emitter is not None:
            def on_progress_critique(chunk: str, _agent=agent_name) -> None:
                text_emitter("critique", _agent, chunk)
        else:
            on_progress_critique = None
        result = provider.assess_cell(
            cell_id, claim_text, context, agent_name,
            on_progress=on_progress_critique,
        )
        if fire is not None:
            fire(
                "verdict",
                phase="critique",
                agent=agent_name,
                verdict=str(_get(result, "verdict", "")),
                iteration=iteration,
            )

        # Log verdict_issued event
        append_event(
            events_path=events_file,
            cell_id=cell_id,
            event_type="verdict_issued",
            agent=agent_name,
            iteration=iteration,
            data=_verdict_data(result),
        )
        _log_sources_from_result(events_file, cell_id, agent_name, iteration, result)
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
            cell_id=cell_id,
            iteration=iteration,
            debate_result=debate_result,
        )
        debate_results.append(debate_result)

    # Append critique assessment to evidence file
    content = _format_phase2_evidence(verdicts_collected, debate_results, iteration)
    _append_evidence(petri_dir, cell_id, dish_id, "critique", iteration, content)

    # Transition to mediating
    update_state(queue_file, cell_id, QueueState.mediating.value)
    return verdicts_collected


def _run_convergence(
    cell_id: str,
    claim_text: str,
    petri_dir: Path,
    dish_id: str,
    iteration: int,
    provider: InferenceProvider,
    agent_roles: dict,
    queue_entry: dict,
    fire: Callable[..., None] | None = None,
    text_emitter: Callable[[str, str, str], None] | None = None,
) -> ConvergenceOutcome:
    """Convergence check -- determines converged, iterate, or stalled."""
    events_file = _events_path(petri_dir, cell_id, dish_id)
    queue_file = _queue_path(petri_dir)

    # Gather all verdicts for this cell at the current iteration
    verdicts = get_verdicts(events_file, cell_id=cell_id, iteration=iteration)

    # Check for short circuits first
    short_circuit = evaluate_short_circuits(verdicts, agent_roles)
    if short_circuit:
        sc_type = short_circuit.type
        append_event(
            events_path=events_file,
            cell_id=cell_id,
            event_type="convergence_checked",
            agent="cell_lead",
            iteration=iteration,
            data={
                "converged": False,
                "blocking_verdicts": {"short_circuit": short_circuit.model_dump()},
                "weakest_link": short_circuit.agent,
                "focused_directive": f"Short-circuit: {sc_type}",
            },
        )
        if sc_type == "needs_experiment":
            _update_cell_status(petri_dir, cell_id, CellStatus.NEEDS_EXPERIMENT.value)
            update_state(queue_file, cell_id, QueueState.stalled.value)
            update_state(queue_file, cell_id, QueueState.needs_human.value)
            return ConvergenceOutcome(outcome="short_circuit", type=sc_type)
        elif sc_type == "defer_open":
            _update_cell_status(petri_dir, cell_id, CellStatus.DEFER_OPEN.value)
            update_state(queue_file, cell_id, QueueState.converged.value)
            update_state(queue_file, cell_id, QueueState.deferred_open.value)
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
        cell_id=cell_id,
        event_type="convergence_checked",
        agent="cell_lead",
        iteration=iteration,
        data={
            "converged": convergence.converged,
            "blocking_verdicts": {
                agent_name: result_entry.get("verdict") for agent_name, result_entry in convergence.blocking_results.items()
            } or None,
            "weakest_link": convergence.weakest_link,
        },
    )

    # Regenerate summary.md now that phase 1 + phase 2 of this iteration
    # have landed in evidence.md. This fires for every outcome path
    # (converged, iterate, circuit_breaker) so the dashboard's Evidence
    # Summary is always current. Best-effort — a failure here logs a
    # warning and continues; it must never break the pipeline.
    _write_summary(petri_dir, cell_id, dish_id, claim_text, iteration, provider)

    if convergence.converged:
        update_state(queue_file, cell_id, QueueState.converged.value)
        return ConvergenceOutcome(outcome="converged")

    # Not converged -- iterate or stall
    if breaker_fires:
        # Before stalling, run a decomposition audit (first-principles re-examination)
        _run_decomposition_audit(
            cell_id, claim_text, petri_dir, dish_id, iteration, provider,
        )
        update_state(queue_file, cell_id, QueueState.stalled.value)
        update_state(queue_file, cell_id, QueueState.needs_human.value)
        _update_cell_status(petri_dir, cell_id, CellStatus.STALLED.value)
        return ConvergenceOutcome(outcome="circuit_breaker")

    # Iterate: increment iteration, set weakest link, back to research_active
    # Note: we use set_iteration (not new_cycle) because this is a retry
    # within the same convergence cycle, not a fresh cycle start.
    weakest_link = convergence.weakest_link

    set_iteration(queue_file, cell_id, iteration + 1)
    if weakest_link:
        set_weakest_link(queue_file, cell_id, weakest_link)
        directive = f"Focus on addressing {weakest_link} concerns"
        set_focused_directive(queue_file, cell_id, directive)

    update_state(queue_file, cell_id, QueueState.research_active.value)
    return ConvergenceOutcome(outcome="iterate", weakest_link=weakest_link)


def _run_red_team(
    cell_id: str,
    claim_text: str,
    petri_dir: Path,
    dish_id: str,
    iteration: int,
    provider: InferenceProvider,
    agent_roles: dict,
    fire: Callable[..., None] | None = None,
    text_emitter: Callable[[str, str, str], None] | None = None,
) -> dict:
    """Red Team phase -- red_team_lead attempts disproval."""
    events_file = _events_path(petri_dir, cell_id, dish_id)
    queue_file = _queue_path(petri_dir)

    update_state(queue_file, cell_id, QueueState.red_team_active.value)

    context = {"iteration": iteration, "phase": "red_team"}
    if fire is not None:
        fire(
            "agent",
            phase="red_team",
            agent="red_team_lead",
            iteration=iteration,
        )
    if text_emitter is not None:
        def on_progress_red_team(chunk: str) -> None:
            text_emitter("red_team", "red_team_lead", chunk)
    else:
        on_progress_red_team = None
    result = provider.assess_cell(
        cell_id, claim_text, context, "red_team_lead",
        on_progress=on_progress_red_team,
    )
    if fire is not None:
        fire(
            "verdict",
            phase="red_team",
            agent="red_team_lead",
            verdict=str(_get(result, "verdict", "")),
            iteration=iteration,
        )

    append_event(
        events_path=events_file,
        cell_id=cell_id,
        event_type="verdict_issued",
        agent="red_team_lead",
        iteration=iteration,
        data=_verdict_data(result),
    )
    _log_sources_from_result(events_file, cell_id, "red_team_lead", iteration, result)

    # Append red team findings to evidence file
    content = _format_red_team_evidence(result, iteration)
    _append_evidence(petri_dir, cell_id, dish_id, "red_team", iteration, content)

    update_state(queue_file, cell_id, QueueState.evaluating.value)
    return result


def _run_evaluation(
    cell_id: str,
    claim_text: str,
    petri_dir: Path,
    dish_id: str,
    iteration: int,
    provider: InferenceProvider,
    agent_roles: dict,
    fire: Callable[..., None] | None = None,
    text_emitter: Callable[[str, str, str], None] | None = None,
) -> EvaluationResult:
    """Evidence Evaluation -- final verdict based on source hierarchy."""
    events_file = _events_path(petri_dir, cell_id, dish_id)
    queue_file = _queue_path(petri_dir)

    # Validate terminal sources
    source_validation = validate_terminal_sources(events_file, cell_id)

    context = {
        "iteration": iteration,
        "phase": "evaluation",
        "source_validation": source_validation,
    }
    if fire is not None:
        fire(
            "agent",
            phase="evaluating",
            agent="evidence_evaluator",
            iteration=iteration,
        )
    if text_emitter is not None:
        def on_progress_evaluator(chunk: str) -> None:
            text_emitter("evaluating", "evidence_evaluator", chunk)
    else:
        on_progress_evaluator = None
    result = provider.assess_cell(
        cell_id, claim_text, context, "evidence_evaluator",
        on_progress=on_progress_evaluator,
    )
    if fire is not None:
        fire(
            "verdict",
            phase="evaluating",
            agent="evidence_evaluator",
            verdict=str(_get(result, "verdict", "")),
            iteration=iteration,
        )

    append_event(
        events_path=events_file,
        cell_id=cell_id,
        event_type="verdict_issued",
        agent="evidence_evaluator",
        iteration=iteration,
        data=_verdict_data(result),
    )
    _log_sources_from_result(events_file, cell_id, "evidence_evaluator", iteration, result)

    # Determine final cell status from evaluator verdict
    evaluator_verdict = _get(result, "verdict", "EVIDENCE_CONFIRMS")
    if evaluator_verdict == "EVIDENCE_CONFIRMS":
        final_status = CellStatus.VALIDATED.value
    elif evaluator_verdict == "EVIDENCE_REFUTES":
        final_status = CellStatus.DISPROVEN.value
    else:
        # EVIDENCE_INCONCLUSIVE or unknown
        final_status = CellStatus.DEFER_OPEN.value

    # Append evaluation findings to evidence file
    content = _format_evaluation_evidence(result, source_validation, iteration)
    _append_evidence(petri_dir, cell_id, dish_id, "evaluation", iteration, content)
    _update_evidence_status(petri_dir, cell_id, dish_id, final_status)

    # Regenerate summary.md one last time now that red team + final
    # evaluation content is in evidence.md. This is the terminal
    # summary the dashboard will show once the cell reaches done.
    _write_summary(petri_dir, cell_id, dish_id, claim_text, iteration, provider)

    _update_cell_status(petri_dir, cell_id, final_status)
    update_state(queue_file, cell_id, QueueState.done.value)

    return EvaluationResult(verdict=evaluator_verdict, final_status=final_status)


# ── Main Pipeline ────────────────────────────────────────────────────────


def process_cell(
    cell_id: str,
    petri_dir: Path,
    provider: InferenceProvider,
    agent_roles: dict | None = None,
    debate_pairings: list | None = None,
    slot_idx: int | None = None,
    on_event: CellProgressCallback | None = None,
) -> ProcessCellResult:
    """Process a single cell through the validation pipeline.

    Drives the cell from its current queue state through successive phases
    until it reaches a terminal state (done, needs_human, deferred) or a
    graceful stop is requested.

    Returns a status dict:
        {cell_id, final_state, iterations, events_logged, ...}
    """
    dish_id = _get_dish_id(petri_dir)
    queue_file = _queue_path(petri_dir)
    events_file = _events_path(petri_dir, cell_id, dish_id)

    # Resolve slot index used in emitted lifecycle events.  ``-1`` means the
    # caller did not participate in slot pooling (e.g. a direct
    # ``process_cell`` call from a unit test).
    resolved_slot_idx = slot_idx if slot_idx is not None else -1

    def fire(kind: str, **fields: object) -> None:
        """Emit a lifecycle event.  UI callbacks MUST NOT break the engine."""
        if on_event is None:
            return
        try:
            on_event(
                CellProgressEvent(
                    slot_idx=resolved_slot_idx,
                    cell_id=cell_id,
                    kind=kind,
                    **fields,  # type: ignore[arg-type]
                )
            )
        except Exception:
            # Swallow UI callback errors silently — the engine must never
            # fail because of a spinner bug.
            pass

    def emit_agent_text(phase_name: str, agent_role: str, chunk: str) -> None:
        """Emit a streaming text chunk for the current agent.

        Fires a ``kind="agent_text"`` ``CellProgressEvent`` so the
        multi-spinner row can show the model's live output as it
        arrives, matching ``petri seed``'s streaming UX. Phase runners
        pass this as ``on_progress`` to ``provider.assess_cell``; the
        provider streams deltas to us, we stamp them with the current
        phase/agent and forward to the UI callback.
        """
        fire(
            "agent_text",
            phase=phase_name,
            agent=agent_role,
            text=chunk,
        )

    # Load agent roles if not provided
    if agent_roles is None:
        agent_roles = load_agent_roles()

    # Load cell metadata for claim_text
    metadata = _load_cell_metadata(petri_dir, cell_id, dish_id)
    claim_text = metadata.get("claim_text", "")

    # Track processing stats
    iterations_run = 0
    events_before = len(get_verdicts(events_file, cell_id=cell_id))

    # Load queue entry
    from petri.storage.queue import load_queue

    queue = load_queue(queue_file)
    if cell_id not in queue.get("entries", {}):
        fire("finished", error=f"Cell {cell_id} not found in queue")
        return ProcessCellResult(
            cell_id=cell_id,
            final_state="not_in_queue",
            iterations=0,
            events_logged=0,
            error=f"Cell {cell_id} not found in queue",
        )

    initial_entry = queue["entries"].get(cell_id, {})
    fire("started", iteration=initial_entry.get("iteration", 0))

    # Drive the state machine until we reach a terminal state
    max_loops = 50  # safety limit to prevent infinite loops
    loop_count = 0

    while loop_count < max_loops:
        loop_count += 1

        # Reload queue entry each loop iteration to get fresh state
        queue = load_queue(queue_file)
        entry = queue["entries"].get(cell_id, {})
        current_state = entry.get("queue_state", "done")
        iteration = entry.get("iteration", 0)

        # Check for graceful stop between phases
        if is_stop_requested():
            logger.info("Graceful stop requested for cell %s", cell_id)
            if current_state not in ("done", "needs_human", "deferred_open", "deferred_closed"):
                try:
                    update_state(queue_file, cell_id, QueueState.stalled.value)
                except ValueError:
                    pass  # Already in a non-stallable state
            break

        # Terminal states -- we are done
        if current_state in ("done", "needs_human", "deferred_open", "deferred_closed"):
            break

        # Dispatch based on current state
        if current_state == QueueState.queued.value:
            _update_cell_status(petri_dir, cell_id, CellStatus.RESEARCH.value)
            update_state(queue_file, cell_id, QueueState.socratic_active.value)

        elif current_state == QueueState.socratic_active.value:
            fire("phase", phase="socratic", iteration=iteration)
            _run_socratic_phase(
                cell_id, claim_text, petri_dir, dish_id,
                iteration, provider,
                fire=fire,
                text_emitter=emit_agent_text,
            )

        elif current_state == QueueState.research_active.value:
            fire("phase", phase="research", iteration=iteration)
            _run_phase1(
                cell_id, claim_text, petri_dir, dish_id,
                iteration, provider, agent_roles, entry,
                fire=fire,
                text_emitter=emit_agent_text,
            )
            iterations_run += 1

        elif current_state == QueueState.critique_active.value:
            fire("phase", phase="critique", iteration=iteration)
            _run_phase2(
                cell_id, claim_text, petri_dir, dish_id,
                iteration, provider, agent_roles, debate_pairings, entry,
                fire=fire,
                text_emitter=emit_agent_text,
            )

        elif current_state == QueueState.mediating.value:
            fire("phase", phase="mediating", iteration=iteration)
            convergence_outcome = _run_convergence(
                cell_id, claim_text, petri_dir, dish_id,
                iteration, provider, agent_roles, entry,
                fire=fire,
                text_emitter=emit_agent_text,
            )
            if convergence_outcome.outcome == "iterate":
                iterations_run += 1

        elif current_state == QueueState.converged.value:
            _update_cell_status(petri_dir, cell_id, CellStatus.RED_TEAM.value)
            fire("phase", phase="red_team", iteration=iteration)
            _run_red_team(
                cell_id, claim_text, petri_dir, dish_id,
                iteration, provider, agent_roles,
                fire=fire,
                text_emitter=emit_agent_text,
            )

        elif current_state == QueueState.evaluating.value:
            _update_cell_status(petri_dir, cell_id, CellStatus.EVALUATE.value)
            fire("phase", phase="evaluating", iteration=iteration)
            _run_evaluation(
                cell_id, claim_text, petri_dir, dish_id,
                iteration, provider, agent_roles,
                fire=fire,
                text_emitter=emit_agent_text,
            )

        elif current_state == QueueState.stalled.value:
            update_state(queue_file, cell_id, QueueState.needs_human.value)

        else:
            # Unknown or unhandled state
            logger.warning("Unhandled queue state %s for cell %s", current_state, cell_id)
            break

    # Compute final state
    queue = load_queue(queue_file)
    final_entry = queue["entries"].get(cell_id, {})
    final_state = final_entry.get("queue_state", "unknown")

    events_after = len(get_verdicts(events_file, cell_id=cell_id))
    events_logged = events_after - events_before

    fire(
        "finished",
        iteration=final_entry.get("iteration", 0),
    )

    return ProcessCellResult(
        cell_id=cell_id,
        final_state=final_state,
        iterations=iterations_run,
        events_logged=events_logged,
        final_iteration=final_entry.get("iteration", 0),
    )


# ── Eligible Cell Discovery ─────────────────────────────────────────────


def find_eligible_cells(
    petri_dir: Path,
    dish_id: str,
    cell_ids: list[str] | None = None,
    colony_filter: str | None = None,
    all_cells: bool = False,
) -> list[str]:
    """Find cells eligible for validation.

    Eligible = leaf cells (or cells whose deps are all VALIDATED) in NEW status.

    When *cell_ids* is provided, only those specific cells are considered.
    When *colony_filter* is provided, only cells from that colony are scanned.
    When *all_cells* is True, all colonies are scanned.
    """
    from petri.graph.colony import deserialize_colony

    dishes_dir = petri_dir / "petri-dishes"
    if not dishes_dir.exists():
        return []

    eligible: list[str] = []

    # If specific cell IDs are given, return them directly (caller knows best)
    if cell_ids:
        return list(cell_ids)

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
        cells_status: dict[str, CellStatus] = {}
        for cell in graph.get_all_cells():
            cells_status[cell.id] = cell.status

        # Get eligible cells from graph
        for cell in graph.get_eligible_for_validation(cells_status):
            eligible.append(cell.id)

        # If not processing all, stop after first colony
        if not all_cells and not colony_filter:
            break

    return eligible


# ── Queue Processor ──────────────────────────────────────────────────────


def process_queue(
    petri_dir: Path,
    provider: InferenceProvider | None = None,
    max_concurrent: int = MAX_CONCURRENT,
    cell_ids: list[str] | None = None,
    colony_filter: str | None = None,
    all_cells: bool = False,
    dry_run: bool = False,
    on_event: CellProgressCallback | None = None,
) -> QueueProcessingResult:
    """Process the queue, running eligible cells through the pipeline.

    Uses ``ThreadPoolExecutor`` for concurrent processing.

    Returns summary dict:
        {processed, succeeded, failed, stalled, results, ...}
    """
    if not dry_run and provider is None:
        raise NoProviderError()

    reset_stop()

    dish_id = _get_dish_id(petri_dir)
    queue_file = _queue_path(petri_dir)

    # Find eligible cells
    eligible = find_eligible_cells(
        petri_dir, dish_id,
        cell_ids=cell_ids,
        colony_filter=colony_filter,
        all_cells=all_cells,
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

    # Enqueue eligible cells that are not already in the queue
    existing_queue = {entry["cell_id"] for entry in list_queue(queue_file)}
    for eligible_cell_id in eligible:
        if eligible_cell_id not in existing_queue:
            try:
                add_to_queue(queue_file, eligible_cell_id)
            except ValueError:
                pass  # Already in queue (race condition)

    # Load shared config
    agent_roles = load_agent_roles()
    debate_pairings = load_debate_pairings()

    # Process concurrently
    results: list[ProcessCellResult] = []
    succeeded = 0
    failed = 0
    stalled = 0

    # Slot pool — a bounded FIFO of slot indices, one per worker.  Each
    # submission pulls an index before calling ``process_cell`` and returns
    # it after.  This gives every in-flight cell a stable slot identity
    # which the CLI's MultiSpinner uses to pick the row to update.
    #
    # NOTE: the stdlib module is aliased to ``_stdlib_queue`` because
    # ``petri.storage.queue`` is already imported at module scope above and
    # the bare name ``queue`` would collide.
    import queue as _stdlib_queue

    slot_pool: _stdlib_queue.Queue[int] = _stdlib_queue.Queue()
    for slot_index in range(max_concurrent):
        slot_pool.put(slot_index)

    def _process_one(worker_cell_id: str) -> ProcessCellResult:
        slot_index = slot_pool.get()
        try:
            return process_cell(
                worker_cell_id, petri_dir,
                provider=provider,
                agent_roles=agent_roles,
                debate_pairings=debate_pairings,
                slot_idx=slot_index,
                on_event=on_event,
            )
        except Exception as exc:
            logger.exception("Error processing cell %s", worker_cell_id)
            if on_event is not None:
                try:
                    on_event(
                        CellProgressEvent(
                            slot_idx=slot_index,
                            cell_id=worker_cell_id,
                            kind="finished",
                            error=str(exc),
                        )
                    )
                except Exception:
                    pass
            return ProcessCellResult(
                cell_id=worker_cell_id,
                final_state="error",
                iterations=0,
                events_logged=0,
                error=str(exc),
            )
        finally:
            slot_pool.put(slot_index)

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
                    next_cell_id = remaining.pop(0)
                    future = executor.submit(_process_one, next_cell_id)
                    active_futures[future] = next_cell_id

                # Poll for completed futures
                newly_done = [completed for completed in active_futures if completed.done()]

                for future in newly_done:
                    cell_result = future.result()
                    results.append(cell_result)
                    del active_futures[future]
                    final = cell_result.final_state
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
