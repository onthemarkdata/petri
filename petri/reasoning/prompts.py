"""Pure prompt builders shared by typed agents and legacy providers.

Builders return the prompt body; legacy callers append a separate JSON-output
suffix. Callers supply configuration and dates explicitly.
"""

import json


def build_assess_claim_substance_prompt(claim: str) -> str:
    """Build the claim-substance assessment prompt body."""
    return (
        "Assess whether the following text is a SUBSTANTIVE research claim that "
        "warrants decomposition, or whether it is placeholder/test input (e.g. 'this "
        "is a test claim', 'asdf', 'hello world', empty).\n\nText: \""
        f"{claim}"
        '"\n\n'
    )


def assess_claim_substance_output_suffix() -> str:
    """Return the legacy JSON-output instructions."""
    return (
        'Return ONLY a JSON object:\n{\n  "is_substantive": true|false,\n  "reason": '
        '"one-sentence explanation",\n  "suggested_rewrite": "tighter rephrasing, or '
        'empty string"\n}\n'
    )


def build_generate_clarifying_questions_prompt(claim: str, max_questions: int = 5) -> str:
    """Build the clarifying-question prompt body."""
    return (
        "Generate "
        f"{max_questions}"
        " CLAIM-SPECIFIC clarifying questions for this research claim. The questions "
        "must be tailored to the actual content of the claim — do not use generic "
        "questions like 'what is the domain?' or 'what is the time horizon?'. Each "
        "question should surface a specific assumption, scope boundary, or "
        'definitional ambiguity in THIS claim.\n\nClaim: "'
        f"{claim}"
        '"\n\nFor each question, optionally provide 2-5 multiple-choice options when '
        "the answer space is naturally bounded; otherwise leave options empty for "
        "free-text input.\n\n"
    )


def generate_clarifying_questions_output_suffix() -> str:
    """Return the legacy JSON-output instructions."""
    return 'Return ONLY a JSON array: [{"question": "...", "options": ["...", "..."]}, ...]'


def build_decompose_claim_prompt(
    claim: str, clarifications: list[dict], guidance: str = "", max_premises: int = 5,
    *, show_reasoning: bool = True,
) -> str:
    """Build the claim decomposition prompt body."""
    clarification_text = ""
    if clarifications:
        lines = [f"Q: {c['question']} A: {c.get('answer', 'N/A')}" for c in clarifications]
        clarification_text = "\nClarifications:\n" + "\n".join(lines)
    guidance_text = ""
    if guidance.strip():
        guidance_text = (
            f"\nRefinement guidance from user (must shape this re-roll):\n{guidance.strip()}\n"
        )
    return (
        'Decompose this claim into first-principles premises.\n\nClaim: "'
        f"{claim}"
        '"'
        f"{clarification_text}"
        f"{guidance_text}"
        + ("\n\nWork through this in four explicit steps. Show your reasoning as you go — "
           "the user is watching this stream live." if show_reasoning else
           "\n\nUse these steps to select premises. Return only the final structured result.")
        + "\n\nSTEP 1 — BRAINSTORM (overgenerate).\n"
        "List candidate premises that, if false, would invalidate the claim. "
        "Cast a wide net. Cover different angles: definitions, mechanisms, scope "
        "boundaries, counterfactuals, hidden assumptions. Number them 1..N. Do NOT "
        "filter at this stage.\n\nSTEP 2 — PRIORITIZE.\nFor each candidate, score on "
        "three dimensions:\n  - LOAD-BEARING: would the claim collapse if this were "
        "false? (HIGH/MED/LOW)\n  - CONTESTABLE: is this actually in question, not "
        "trivially true? (HIGH/MED/LOW)\n  - DISTINCT: does it cover ground the other "
        "top candidates don't?\nBriefly justify each score.\n\nSTEP 3 — SELECT.\n"
        "Let the structure determine the count; do not pad to a quota; fewer is fine. "
        "Select the strongest premises after ranking. Order them by importance (most important first). "
        "Drop duplicates and trivial restatements. Do not pick the first ones you "
        "wrote down — pick the ones that scored highest.\n\n"
    )


def decompose_claim_output_suffix(max_premises: int) -> str:
    """Return the legacy JSON-output instructions."""
    return (
        'STEP 4 — EMIT JSON.\nReturn a JSON object on the final lines:\n{"nodes": '
        '[{"level": 1, "seq": 1, "claim_text": "..."}, ...], "edges": []}\n\nThe JSON '
        "must contain only genuine first-principles premises; fewer is fine. "
        "Return no commentary after the JSON."
    )


def build_decompose_why_prompt(
    premise: str, parent_level: int, parent_seq: int, max_premises: int = 5,
    *, show_reasoning: bool = True,
) -> str:
    """Build the premise decomposition prompt body."""
    return (
        'FIVE WHYS — drill one level deeper into a parent premise.\n\nPremise: "'
        f"{premise}"
        + ('"\n\nWork through this in four explicit steps. Show your reasoning as you go '
           "— the user is watching this stream live." if show_reasoning else
           '"\n\nUse these steps to select sub-premises. Return only the final structured result.')
        + "\n\nSTEP 1 — BRAINSTORM "
        "(overgenerate).\nList candidate sub-premises that must hold for the "
        "parent premise to be true. Number them 1..N. Don't filter yet.\n\nSTEP 2 — "
        "PRIORITIZE.\nFor each candidate, score on:\n  - FOUNDATIONALITY: closer to "
        "bedrock = higher (HIGH/MED/LOW)\n  - INDEPENDENCE: separately verifiable "
        "from the others (HIGH/MED/LOW)\n  - LOAD-BEARING: would the parent collapse "
        "if this sub-premise were false?\nBriefly justify each score.\n\nSTEP 3 — "
        "SELECT.\nLet the structure determine the count; do not pad to a quota; fewer is fine. "
        "Select the strongest sub-premises. Drop redundant or surface-level ones. Do not pick the first "
        "ones you wrote down — pick the ones that scored highest.\n\n"
    )


def decompose_why_output_suffix(max_premises: int) -> str:
    """Return the legacy JSON-output instructions."""
    return (
        "If the parent is already ATOMIC (a directly verifiable fact, definition, or "
        'measurement), skip steps 1-3 and return:\n{"sub_premises": [], "is_atomic": '
        'true, "reason": "why it is atomic"}\n\nSTEP 4 — EMIT JSON.\nReturn: '
        '{"sub_premises": [{"claim_text": "...", "is_atomic": false}, ...], '
        '"is_atomic": false}\n\nLet the structure determine the count; do not pad to a quota; fewer is fine. '
        "Return no commentary after the JSON."
    )


def build_assess_cell_prompt(
    cell_id: str,
    claim_text: str,
    context: dict,
    agent_role: str,
    *,
    valid_verdicts: list[str],
    role_instruction: str,
    today_iso: str,
    require_source_retrieval: bool = True,
) -> str:
    """Build the cell assessment prompt body."""
    role_instruction = role_instruction or "Assess this claim thoroughly."
    verdict_list = ", ".join(valid_verdicts)
    context_parts = []
    if context.get("iteration"):
        context_parts.append(f"Iteration: {context['iteration']}")
    if context.get("weakest_link"):
        context_parts.append(f"Focus area: {context['weakest_link']}")
    if context.get("focused_directive"):
        context_parts.append(f"Directive: {context['focused_directive']}")
    if context.get("phase"):
        context_parts.append(f"Phase: {context['phase']}")
    if context.get("source_validation"):
        context_parts.append(f"Source validation: {json.dumps(context['source_validation'])}")
    context_str = "\n".join(context_parts) if context_parts else "Initial assessment"
    prior_evidence = context.get("prior_evidence", "")
    evidence_section = ""
    if prior_evidence:
        evidence_section = (
            "\n--- Prior Evidence ---\n"
            f"{prior_evidence}"
            "\n--- End Prior Evidence ---\n\nBuild on the evidence above. Focus on new "
            "insights and gaps.\n"
        )
    freshness_directive = (
        "IMPORTANT — Today's date is "
        f"{today_iso}"
        ". Your training data is stale relative to this date. You MUST use the "
        "WebSearch tool to find current sources and WebFetch to verify that each URL "
        "you cite actually exists before including it. Do NOT cite URLs from memory "
        "— they will be treated as fabricated. Prefer sources published within the "
        "last 12 months unless the claim is explicitly about historical events.\n\n"
    )
    if not require_source_retrieval:
        freshness_directive = ("This exploratory phase plans research before source retrieval. "
                               "Use supplied context; do not claim to have browsed. "
                               "When no verified source is supplied, return sources_cited as an empty array.\n\n")
    return (
        f"{freshness_directive}"
        "You are the "
        f"{agent_role}"
        " agent in a research validation pipeline.\n\n"
        f"{role_instruction}"
        "\n\nCell: "
        f"{cell_id}"
        '\nClaim: "'
        f"{claim_text}"
        '"\nContext: '
        f"{context_str}"
        "\n"
        f"{evidence_section}"
        "\nValid verdicts: "
        f"{verdict_list}"
        "\n\n"
    )


def assess_cell_output_suffix(valid_verdicts: list[str]) -> str:
    """Return the legacy JSON-output instructions."""
    verdict_list = ", ".join(valid_verdicts)
    return (
        'Return ONLY a JSON object with:\n- "verdict": one of ['
        f"{verdict_list}"
        ']\n- "sources_cited": REQUIRED array. Each source must have:\n  - "url": full '
        'URL (https://...) to the source\n  - "title": "Publication, Article Title '
        '(Year)"\n  - "hierarchy_level": 1-6 (1=direct measurement, 2=authoritative '
        "docs, 3=derived calculation, 4=expert consensus, 5=single expert, "
        '6=community report)\n  - "finding": 1-2 sentence finding from this source\n  '
        '- "supports_or_contradicts": "supports" or "contradicts"\n- "summary": 1-3 '
        "concise sentences. Enumerate dimensions, cite specific numbers.\n- "
        '"confidence": "HIGH", "MEDIUM", or "LOW"\n\nRULES:\n- Every claim MUST be '
        "backed by at least one source with a valid URL.\n- Keep summary TERSE — "
        'citations are the evidence.\n- Do NOT include "arguments" or "evidence" '
        "fields.\n\nReturn ONLY the JSON."
    )


def build_summarize_evidence_prompt(
    cell_id: str, claim_text: str, evidence_md: str, iteration: int
) -> str:
    """Build the evidence summary prompt body."""
    return (
        "You are cell_lead, the orchestrator for this research cell. Your job right "
        "now is to read the accumulated evidence below and produce a concise "
        "markdown summary (no more than ~400 words) that a human can scan in 30 "
        "seconds.\n\n## Cell claim\n"
        f"{claim_text}"
        "\n\n## Iteration\n"
        f"{iteration}"
        "\n\n## Accumulated evidence\n"
        f"{evidence_md}"
        "\n\n## Output format\nReturn ONLY markdown with these sections, in this order:\n"
        "1. **Current state** — one sentence describing where the cell stands right "
        "now.\n2. **Key findings** — 3-6 bullet points, each anchored to a specific "
        "source or agent verdict. Be concrete, not generic.\n3. **Open questions** — "
        "1-3 bullets of what is still unresolved, or 'None' if the cell has "
        "converged.\n4. **Confidence** — a short phrase (e.g. 'high, primary sources "
        "agree', 'low, contradicting sources', 'medium, pending red team').\n\nDo not "
        "repeat the phase-by-phase breakdown from evidence.md — that's the raw log. "
        "Synthesize across phases."
    )



def build_match_evidence_prompt(content: str, candidates: list[dict]) -> str:
    """Compare evidence with explicitly supplied cell claims."""
    return (
        "Match this evidence to the candidate cells. Return only relevant candidates, "
        "using their exact cell IDs. Score relevance from 0 (unrelated) to 1 (directly relevant) "
        "and explain each match. An empty list means no candidate is relevant.\n\n"
        f"Evidence:\n{content}\n\nCandidates:\n{json.dumps(candidates)}"
    )
