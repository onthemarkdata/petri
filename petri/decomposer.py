"""Claim decomposition module for Petri.

Takes a claim, optionally asks clarifying questions, and decomposes it into
a colony DAG of logical premises. Delegates actual reasoning to a
InferenceProvider when available; provides a simple structural fallback
for standalone CLI use without an LLM.
"""

from __future__ import annotations

import re

from petri.models import (
    ClarifyingQuestion,
    DecompositionResult,
    Edge,
    Node,
    InferenceProvider,
    build_node_key,
    validate_slug,
)


# ── Public API ───────────────────────────────────────────────────────────


def generate_clarifying_questions(
    claim: str,
    provider: InferenceProvider | None = None,
    max_questions: int = 5,
) -> list[ClarifyingQuestion]:
    """Generate clarifying questions for a claim.

    If provider is None, returns a default set of generic questions.
    """
    if provider is not None:
        raw = provider.generate_clarifying_questions(claim, max_questions)
        return [
            ClarifyingQuestion(
                question=q["question"],
                options=q.get("options", []),
                answer=q.get("answer"),
            )
            for q in raw[:max_questions]
        ]

    return _default_questions()[:max_questions]


def decompose_claim(
    claim: str,
    clarifications: list[ClarifyingQuestion],
    dish_id: str,
    colony_name: str,
    provider: InferenceProvider | None = None,
) -> DecompositionResult:
    """Decompose a claim into a colony of nodes and edges.

    If provider is None, creates a simple default decomposition
    with 3 premises at level 1 and 2 sub-premises at level 2.
    """
    if provider is not None:
        return _provider_decompose(
            claim, clarifications, dish_id, colony_name, provider
        )

    return _default_decompose(claim, dish_id, colony_name)


def format_colony_display(result: DecompositionResult) -> str:
    """Format a decomposition result as a text tree for display.

    Groups nodes by level and sorts by seq within each level.
    Shows composite IDs and claim text.
    """
    if not result.nodes:
        return f"Colony: {result.colony_name} (empty)\n"

    center_id = result.nodes[0].id if result.nodes else "?"
    lines = [f"Colony: {result.colony_name} (center: {center_id})\n"]

    # Group by level
    by_level: dict[int, list[Node]] = {}
    for node in result.nodes:
        by_level.setdefault(node.level, []).append(node)

    for level in sorted(by_level):
        # Sort nodes within a level by their seq (last 3-digit segment)
        nodes_at_level = sorted(by_level[level], key=lambda n: n.id)
        for node in nodes_at_level:
            lines.append(
                f"Level {level}: {node.id} \u2014 {node.claim_text}"
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
    significant = [w for w in words if w not in stop_words and w]

    # Fall back to all words if filtering removed everything
    if not significant:
        significant = [w for w in words if w]

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


def _default_questions() -> list[ClarifyingQuestion]:
    """Return the standard set of generic clarifying questions."""
    return [
        ClarifyingQuestion(
            question="What is the primary domain or industry?",
        ),
        ClarifyingQuestion(
            question="What geographic scope applies?",
        ),
        ClarifyingQuestion(
            question="What time horizon are you considering?",
            options=["1 year", "3 years", "5 years", "10+ years"],
        ),
        ClarifyingQuestion(
            question="What aspect matters most?",
            options=[
                "Technical feasibility",
                "Economic viability",
                "Regulatory compliance",
                "Market demand",
            ],
        ),
        ClarifyingQuestion(
            question="Are there specific constraints to consider?",
        ),
    ]


def _default_decompose(
    claim: str,
    dish_id: str,
    colony_name: str,
) -> DecompositionResult:
    """Create a simple default decomposition without an LLM.

    Structure:
      Level 0: center (the original claim)
      Level 1: 3 generic premises (evidence, feasibility, context)
      Level 2: 2 sub-premises under the evidence premise
    """
    colony_id = f"{dish_id}-{colony_name}"

    # Level 0 -- center
    center_key = build_node_key(dish_id, colony_name, 0, 0)
    center = Node(
        id=center_key,
        colony_id=colony_id,
        claim_text=claim,
        level=0,
    )

    # Level 1 -- three premises
    evidence_key = build_node_key(dish_id, colony_name, 1, 1)
    feasibility_key = build_node_key(dish_id, colony_name, 1, 2)
    context_key = build_node_key(dish_id, colony_name, 1, 3)

    evidence_node = Node(
        id=evidence_key,
        colony_id=colony_id,
        claim_text=f"Evidence supports that: {claim}",
        level=1,
    )
    feasibility_node = Node(
        id=feasibility_key,
        colony_id=colony_id,
        claim_text=f"It is practically feasible: {claim}",
        level=1,
    )
    context_node = Node(
        id=context_key,
        colony_id=colony_id,
        claim_text=f"The context and conditions are favorable: {claim}",
        level=1,
    )

    # Level 2 -- two sub-premises under evidence
    sub1_key = build_node_key(dish_id, colony_name, 2, 1)
    sub2_key = build_node_key(dish_id, colony_name, 2, 2)

    sub1 = Node(
        id=sub1_key,
        colony_id=colony_id,
        claim_text=f"Sufficient data exists to evaluate: {claim}",
        level=2,
    )
    sub2 = Node(
        id=sub2_key,
        colony_id=colony_id,
        claim_text=f"Data sources are current and reliable for: {claim}",
        level=2,
    )

    # Wire up dependencies: parent depends on children (bottom-up validation)
    center.dependencies = [evidence_key, feasibility_key, context_key]
    evidence_node.dependencies = [sub1_key, sub2_key]

    # Wire up dependents (reverse of dependencies)
    evidence_node.dependents = [center_key]
    feasibility_node.dependents = [center_key]
    context_node.dependents = [center_key]
    sub1.dependents = [evidence_key]
    sub2.dependents = [evidence_key]

    # Edges -- from_node depends on to_node (parent depends on child)
    edges = [
        Edge(from_node=center_key, to_node=evidence_key),
        Edge(from_node=center_key, to_node=feasibility_key),
        Edge(from_node=center_key, to_node=context_key),
        Edge(from_node=evidence_key, to_node=sub1_key),
        Edge(from_node=evidence_key, to_node=sub2_key),
    ]

    nodes = [
        center,
        evidence_node,
        feasibility_node,
        context_node,
        sub1,
        sub2,
    ]

    return DecompositionResult(
        nodes=nodes,
        edges=edges,
        colony_name=colony_name,
        center_claim=claim,
    )


def _provider_decompose(
    claim: str,
    clarifications: list[ClarifyingQuestion],
    dish_id: str,
    colony_name: str,
    provider: InferenceProvider,
) -> DecompositionResult:
    """Decompose using an LLM reasoning provider.

    Calls provider.decompose_claim and converts the raw dict response
    into typed Node/Edge objects with proper composite keys.
    """
    clarification_dicts = [
        {
            "question": q.question,
            "answer": q.answer or "",
            "options": q.options,
        }
        for q in clarifications
    ]

    raw = provider.decompose_claim(claim, clarification_dicts)
    colony_id = f"{dish_id}-{colony_name}"

    nodes: list[Node] = []
    edges: list[Edge] = []

    # Parse nodes from provider response
    raw_nodes = raw.get("nodes", [])
    for raw_node in raw_nodes:
        level = raw_node.get("level", 0)
        seq = raw_node.get("seq", 0)
        node_key = build_node_key(dish_id, colony_name, level, seq)
        nodes.append(
            Node(
                id=node_key,
                colony_id=colony_id,
                claim_text=raw_node.get("claim_text", ""),
                level=level,
                dependencies=[
                    build_node_key(dish_id, colony_name, d["level"], d["seq"])
                    for d in raw_node.get("dependencies", [])
                ],
            )
        )

    # Parse edges from provider response
    raw_edges = raw.get("edges", [])
    for raw_edge in raw_edges:
        from_info = raw_edge.get("from", {})
        to_info = raw_edge.get("to", {})
        edges.append(
            Edge(
                from_node=build_node_key(
                    dish_id,
                    colony_name,
                    from_info.get("level", 0),
                    from_info.get("seq", 0),
                ),
                to_node=build_node_key(
                    dish_id,
                    colony_name,
                    to_info.get("level", 0),
                    to_info.get("seq", 0),
                ),
                edge_type=raw_edge.get("edge_type", "intra_colony"),
            )
        )

    return DecompositionResult(
        nodes=nodes,
        edges=edges,
        colony_name=colony_name,
        center_claim=claim,
    )
