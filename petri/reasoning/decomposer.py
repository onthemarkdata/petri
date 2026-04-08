"""Claim decomposition module for Petri.

Takes a claim, asks clarifying questions, and decomposes it into a colony
DAG of logical premises. All reasoning is delegated to an
``InferenceProvider`` — there is no offline fallback. Callers that cannot
supply a provider must surface that to the user instead of generating a
template-stamped tree.
"""

from __future__ import annotations

import re
from typing import Callable, Optional

from petri.models import (
    Cell,
    ClarifyingQuestion,
    DecompositionResult,
    Edge,
    InferenceProvider,
    build_cell_key,
    validate_slug,
)


# Type alias for the per-cell serialization callback fired by the
# decomposer the moment a new cell is built. ``new_edges`` are the edges
# whose ``to_node`` is the newly-created cell — i.e. the parent links the
# CLI needs to persist alongside the cell itself.
OnCellCreated = Callable[[Cell, "list[Edge]"], None]


# ── Public API ───────────────────────────────────────────────────────────


def generate_clarifying_questions(
    claim: str,
    provider: InferenceProvider | None,
    max_questions: int = 5,
    on_progress: Optional[Callable[[str], None]] = None,
) -> list[ClarifyingQuestion]:
    """Generate claim-specific clarifying questions via the provider.

    Raises ValueError if no provider is supplied — the wizard is fully
    agentic and has no hardcoded question set. ``on_progress`` is forwarded
    to the provider so callers can stream the model's text live.
    """
    if provider is None:
        raise ValueError(
            "generate_clarifying_questions requires an InferenceProvider; "
            "the wizard is fully agentic and has no hardcoded fallback."
        )

    raw = provider.generate_clarifying_questions(
        claim, max_questions, on_progress=on_progress
    )
    return [
        ClarifyingQuestion(
            question=question["question"],
            options=question.get("options", []),
            answer=question.get("answer"),
        )
        for question in raw[:max_questions]
    ]


def decompose_claim(
    claim: str,
    clarifications: list[ClarifyingQuestion],
    dish_id: str,
    colony_name: str,
    provider: InferenceProvider | None,
    guidance: str = "",
    on_progress: Optional[Callable[[str], None]] = None,
    on_cell_created: Optional[OnCellCreated] = None,
    center: Optional[Cell] = None,
) -> DecompositionResult:
    """Decompose a claim into a colony of cells and edges.

    Raises ValueError if no provider is supplied.

    ``guidance`` is optional free-text feedback from a regenerate-with-guidance
    loop and is threaded into the model context.

    ``on_progress`` streams the model's text to the caller during every LLM
    call (initial decomposition + each Five Whys iteration).

    ``on_cell_created`` is fired the moment each new cell is constructed
    from the LLM response — before the next LLM call begins. The CLI uses
    this to serialize each cell + log a ``cell_created`` event incrementally
    so on-disk state always reflects what the agent has done so far. The
    callback is *not* called for the level-0 center cell (the CLI creates
    that itself before invoking ``decompose_claim``).

    ``center`` is an optional caller-supplied level-0 Cell. When provided,
    the decomposer mutates that same object when wiring dependencies — so
    the CLI's in-memory ``center_node`` reference reflects the level-1
    children and the subsequent ``serialize_colony`` writes them to disk.
    When ``center`` is None, the decomposer constructs its own level-0
    cell internally (backwards-compatible default used by unit tests).
    """
    if provider is None:
        raise ValueError(
            "decompose_claim requires an InferenceProvider; "
            "there is no offline decomposition fallback."
        )

    return _provider_decompose(
        claim,
        clarifications,
        dish_id,
        colony_name,
        provider,
        guidance,
        on_progress=on_progress,
        on_cell_created=on_cell_created,
        center=center,
    )


def format_colony_display(result: DecompositionResult) -> str:
    """Format a decomposition result as a text tree for display.

    Groups cells by level and sorts by seq within each level.
    Shows composite IDs and claim text.
    """
    if not result.cells:
        return f"Colony: {result.colony_name} (empty)\n"

    center_id = result.cells[0].id if result.cells else "?"
    lines = [f"Colony: {result.colony_name} (center: {center_id})\n"]

    # Group by level
    by_level: dict[int, list[Cell]] = {}
    for cell in result.cells:
        by_level.setdefault(cell.level, []).append(cell)

    for level in sorted(by_level):
        # Sort cells within a level by their seq (last 3-digit segment)
        cells_at_level = sorted(by_level[level], key=lambda c: c.id)
        for cell in cells_at_level:
            lines.append(
                f"Level {level}: {cell.id} \u2014 {cell.claim_text}"
            )

    return "\n".join(lines) + "\n"


def generate_colony_name(claim: str) -> str:
    """Generate a kebab-case colony name from a claim string.

    Takes the first few significant words, lowercases, joins with hyphens.
    Max 30 chars. Must contain at least one letter and be valid per
    validate_slug.
    """
    # Strip punctuation, lowercase, split into words
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", "", claim.lower())
    words = cleaned.split()

    # Drop common stop words to keep the name meaningful
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "shall", "can",
        "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "as", "into", "through", "during", "before", "after", "and",
        "but", "or", "nor", "not", "so", "yet", "both", "either",
        "neither", "each", "every", "all", "any", "few", "more",
        "most", "other", "some", "such", "no", "only", "own", "same",
        "than", "too", "very", "just", "because", "if", "when",
        "that", "this", "it", "its", "there", "their", "they",
        "what", "which", "who", "whom", "how", "where", "why",
        "about", "up", "out", "off", "over", "under", "again",
        "further", "then", "once",
    }
    significant = [word for word in words if word not in stop_words and word]

    # Fall back to all words if filtering removed everything
    if not significant:
        significant = [word for word in words if word]

    # Still empty -- use a generic fallback
    if not significant:
        return "colony"

    # Build name from significant words, respecting the 30-char limit
    name_parts: list[str] = []
    current_len = 0
    for word in significant:
        needed = len(word) + (1 if name_parts else 0)  # +1 for hyphen
        if current_len + needed > 30:
            break
        name_parts.append(word)
        current_len += needed

    # At least one word
    if not name_parts:
        name_parts.append(significant[0][:30])

    candidate = "-".join(name_parts)

    # Ensure valid slug
    if validate_slug(candidate):
        return candidate

    # Sanitize: keep only lowercase alphanumeric and hyphens
    candidate = re.sub(r"[^a-z0-9-]", "", candidate)
    candidate = re.sub(r"-+", "-", candidate).strip("-")

    if not candidate or not validate_slug(candidate):
        return "colony"

    return candidate


# ── Private Helpers ──────────────────────────────────────────────────────


def _provider_decompose(
    claim: str,
    clarifications: list[ClarifyingQuestion],
    dish_id: str,
    colony_name: str,
    provider: InferenceProvider,
    guidance: str = "",
    on_progress: Optional[Callable[[str], None]] = None,
    on_cell_created: Optional[OnCellCreated] = None,
    center: Optional[Cell] = None,
) -> DecompositionResult:
    """Decompose using iterative Five Whys via an LLM provider.

    1. Ask the provider for Level 1 premises (direct assumptions).
    2. For each non-atomic premise, ask "Why is this true?" to generate
       the next level down.
    3. Repeat until premises are atomic or max_depth is reached.

    Each level is capped at ``get_max_nodes_per_layer()`` total cells.
    The provider prompts ask the model to brainstorm broadly, prioritise,
    and return only the top N most important premises; the decomposer
    additionally hard-truncates as a safety net so the cap holds even if
    the model ignores the instruction.

    Each new cell fires ``on_cell_created`` *immediately* so the CLI can
    persist it to disk and log a ``cell_created`` event before the next
    LLM call begins. ``on_progress`` is forwarded to every provider call.
    """
    from petri.config import get_max_decomposition_depth, get_max_nodes_per_layer

    max_depth = get_max_decomposition_depth()
    max_per_layer = get_max_nodes_per_layer()
    colony_id = f"{dish_id}-{colony_name}"

    # Per-level usage tracker — used to enforce the cap and to compute
    # remaining budget for each Five Whys call so the model can be told
    # how many top-N picks are still available at the target level.
    cells_per_level: dict[int, int] = {0: 1}  # center already exists

    # Step 1: Get initial decomposition (Level 0 + Level 1)
    clarification_dicts = [
        {
            "question": clarification.question,
            "answer": clarification.answer or "",
            "options": clarification.options,
        }
        for clarification in clarifications
    ]
    raw = provider.decompose_claim(
        claim,
        clarification_dicts,
        guidance=guidance,
        max_premises=max_per_layer,
        on_progress=on_progress,
    )

    # Build Level 0 center cell — the CLI is responsible for serializing
    # this one (it's created and persisted before _provider_decompose runs)
    # so we do NOT fire on_cell_created for the center.
    #
    # If the caller passed in an existing ``center`` Cell, mutate that
    # object so the CLI's in-memory reference sees the wired dependencies
    # (the source-of-truth fix for the bottom-up inversion bug). Otherwise
    # construct a local Cell — backwards-compatible for existing callers
    # and unit tests.
    if center is None:
        center_key = build_cell_key(dish_id, colony_name, 0, 0)
        center = Cell(
            id=center_key,
            colony_id=colony_id,
            claim_text=claim,
            level=0,
        )
    else:
        center_key = center.id

    all_cells: list[Cell] = [center]
    all_edges: list[Edge] = []

    # Parse Level 1 cells from initial decomposition. Hard-truncate to
    # the per-layer cap as a safety net (the prompt asks for at most N,
    # but we don't trust the model to obey).
    raw_cells = raw.get("nodes", [])
    level_one_cells: list[Cell] = []
    seq_counter = 1
    for raw_cell in raw_cells:
        if len(level_one_cells) >= max_per_layer:
            break  # cap reached
        raw_level = raw_cell.get("level", 1)
        if raw_level == 0:
            continue  # Skip center — we already have it
        claim_text = raw_cell.get("claim_text", "")
        if not claim_text:
            continue
        cell_key = build_cell_key(dish_id, colony_name, 1, seq_counter)
        cell = Cell(
            id=cell_key,
            colony_id=colony_id,
            claim_text=claim_text,
            level=1,
            dependents=[center_key],
        )
        edge = Edge(from_cell=center_key, to_cell=cell_key)
        level_one_cells.append(cell)
        all_cells.append(cell)
        all_edges.append(edge)
        seq_counter += 1
        cells_per_level[1] = cells_per_level.get(1, 0) + 1
        if on_cell_created is not None:
            on_cell_created(cell, [edge])

    # If no Level 1 cells were produced, fail loudly — there is no
    # hardcoded fallback in the agentic flow.
    if not level_one_cells:
        raise RuntimeError(
            "LLM returned no level-1 premises for this claim; "
            "try regenerating with guidance or refining the claim."
        )

    # Wire center dependencies
    center.dependencies = [cell.id for cell in level_one_cells]

    # Step 2: Iterative Five Whys — drill deeper on non-atomic premises.
    # Each child level is capped at max_per_layer total cells; the cap is
    # enforced by tracking cells_per_level and skipping calls (or
    # truncating returned sub-premises) once the budget is exhausted.
    cells_to_expand: list[tuple[Cell, int]] = [
        (cell, 1) for cell in level_one_cells
    ]

    while cells_to_expand:
        parent_cell, current_level = cells_to_expand.pop(0)
        next_level = current_level + 1

        # Stop if we've reached max depth
        if next_level > max_depth:
            continue

        # Stop if the next level is already at the per-layer cap
        remaining_budget = max_per_layer - cells_per_level.get(next_level, 0)
        if remaining_budget <= 0:
            continue

        # Ask "Why is this true?" — if atomic, skip
        if not hasattr(provider, "decompose_why"):
            continue

        sub_premises = provider.decompose_why(
            parent_cell.claim_text,
            parent_level=current_level,
            parent_seq=0,
            max_premises=remaining_budget,
            on_progress=on_progress,
        )

        if not sub_premises:
            # Premise is atomic — no further decomposition needed
            continue

        child_keys: list[str] = []
        for sub_premise in sub_premises:
            if cells_per_level.get(next_level, 0) >= max_per_layer:
                break  # cap reached mid-iteration
            if not isinstance(sub_premise, dict):
                continue
            sub_claim = sub_premise.get("claim_text", "")
            if not sub_claim:
                continue

            child_seq = cells_per_level.get(next_level, 0) + 1
            child_key = build_cell_key(dish_id, colony_name, next_level, child_seq)
            child_cell = Cell(
                id=child_key,
                colony_id=colony_id,
                claim_text=sub_claim,
                level=next_level,
                dependents=[parent_cell.id],
            )
            child_edge = Edge(from_cell=parent_cell.id, to_cell=child_key)
            all_cells.append(child_cell)
            all_edges.append(child_edge)
            child_keys.append(child_key)
            cells_per_level[next_level] = cells_per_level.get(next_level, 0) + 1
            if on_cell_created is not None:
                on_cell_created(child_cell, [child_edge])

            # If not atomic, queue for further decomposition
            is_atomic = sub_premise.get("is_atomic", False)
            if not is_atomic:
                cells_to_expand.append((child_cell, next_level))

        parent_cell.dependencies = child_keys

    return DecompositionResult(
        cells=all_cells,
        edges=all_edges,
        colony_name=colony_name,
        center_claim=claim,
    )
