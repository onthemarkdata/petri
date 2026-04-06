"""Debate mediation for Petri validation pipeline.

Structures adversarial exchanges between agent pairs (e.g. skeptic vs.
champion) and logs the results as ``debate_mediated`` events.  Without an
LLM provider the module formats existing agent outputs into a structured
exchange -- no API calls required.
"""

from __future__ import annotations

import math
from pathlib import Path

import yaml

from petri.event_log import append_event
from petri.models import Debate, InferenceProvider

# ── Defaults ──────────────────────────────────────────────────────────────

_DEFAULTS_DIR = Path(__file__).parent / "defaults"


# ── Config Loaders ────────────────────────────────────────────────────────


def load_debate_pairings(config_path: Path | None = None) -> list[Debate]:
    """Load debate pairings from a YAML file.

    Falls back to ``petri/defaults/petri.yaml`` when *config_path* is None
    or points to a file that does not exist.

    Returns a list of ``Debate`` models.
    """
    path = config_path if config_path and config_path.exists() else _DEFAULTS_DIR / "petri.yaml"
    raw = yaml.safe_load(path.read_text())
    debates: list[Debate] = []
    for entry in raw.get("debates", []):
        pair = tuple(entry["pair"])
        debates.append(
            Debate(
                pair=pair,
                rounds=entry["rounds"],
                purpose=entry["purpose"],
            )
        )
    return debates


# ── Debate Mediation ──────────────────────────────────────────────────────


def mediate_debate(
    agent_a_output: dict,
    agent_b_output: dict,
    debate: Debate,
    provider: InferenceProvider | None = None,
) -> dict:
    """Mediate a debate between two agents.

    Parameters
    ----------
    agent_a_output:
        Dict with ``{agent, verdict, summary, arguments}`` for the first
        agent in the debate pair.
    agent_b_output:
        Dict with ``{agent, verdict, summary, arguments}`` for the second
        agent in the debate pair.
    debate:
        The ``Debate`` model describing the pairing.
    provider:
        Optional ``InferenceProvider`` for generating rebuttals.  When
        ``None`` the exchange is constructed purely from existing outputs.

    Returns
    -------
    dict with keys: pair, rounds, purpose, exchanges, summary.
    """
    agent_a = agent_a_output.get("agent", debate.pair[0])
    agent_b = agent_b_output.get("agent", debate.pair[1])

    exchanges: list[dict] = []

    # ── Round 1: agent_a presents, agent_b responds ───────────────────
    a_content = _build_presentation(agent_a_output, provider, opponent_output=None)
    exchanges.append({"speaker": agent_a, "content": a_content, "round": 1})

    b_content = _build_response(agent_b_output, agent_a_output, provider)
    exchanges.append({"speaker": agent_b, "content": b_content, "round": 1})

    # ── Round 1.5: agent_a gets a final rebuttal ──────────────────────
    if debate.rounds > 1:
        rebuttal = _build_rebuttal(agent_a_output, agent_b_output, provider)
        exchanges.append({"speaker": agent_a, "content": rebuttal, "round": 1.5})

    # ── Summary ───────────────────────────────────────────────────────
    summary = _build_summary(agent_a, agent_b, agent_a_output, agent_b_output, debate)

    return {
        "pair": (agent_a, agent_b),
        "rounds": debate.rounds,
        "purpose": debate.purpose,
        "exchanges": exchanges,
        "summary": summary,
    }


# ── Event Logging ─────────────────────────────────────────────────────────


def log_debate(
    events_path: Path,
    node_id: str,
    iteration: int,
    debate_result: dict,
) -> None:
    """Write a ``debate_mediated`` event to the event log."""
    pair = debate_result.get("pair", ("", ""))
    exchange_lines = []
    for ex in debate_result.get("exchanges", []):
        exchange_lines.append(f"[Round {ex['round']}] {ex['speaker']}: {ex['content']}")
    exchange_summary = " | ".join(exchange_lines) if exchange_lines else debate_result.get("summary", "")

    append_event(
        events_path=events_path,
        node_id=node_id,
        event_type="debate_mediated",
        agent="node_lead",
        iteration=iteration,
        data={
            "from_agent": pair[0] if pair else "",
            "to_agent": pair[1] if len(pair) > 1 else "",
            "exchange_summary": exchange_summary,
        },
    )


# ── Held Messages ─────────────────────────────────────────────────────────


def get_held_messages(
    debates: list[dict],
    current_phase: int,
) -> list[dict]:
    """Filter debate results for messages that should be held for the next iteration.

    Cross-phase messages (Phase 2 -> Phase 1) are held until the next
    iteration's Phase 1 begins.

    Returns a list of dicts with ``{from_agent, to_agent, content,
    hold_until_phase}``.
    """
    held: list[dict] = []

    for debate_result in debates:
        for ex in debate_result.get("exchanges", []):
            speaker = ex.get("speaker", "")
            # Identify the other participant in the debate.
            pair = debate_result.get("pair", ())
            if len(pair) == 2:
                to_agent = pair[1] if speaker == pair[0] else pair[0]
            else:
                to_agent = ""

            # A cross-phase message: Phase 2 agent sending feedback that
            # should reach Phase 1 agents in the next iteration.
            if current_phase == 2:
                held.append(
                    {
                        "from_agent": speaker,
                        "to_agent": to_agent,
                        "content": ex.get("content", ""),
                        "hold_until_phase": 1,
                    }
                )

    return held


# ── Internal Helpers ──────────────────────────────────────────────────────


def _build_presentation(
    output: dict,
    provider: InferenceProvider | None,
    opponent_output: dict | None,
) -> str:
    """Build the opening presentation for an agent."""
    if provider is not None and opponent_output is not None:
        # With a provider we could generate a richer opening, but for the
        # initial presentation we always use the agent's own arguments.
        pass

    summary = output.get("summary", "")
    arguments = output.get("arguments", "")
    verdict = output.get("verdict", "")

    parts = []
    if verdict:
        parts.append(f"Verdict: {verdict}.")
    if summary:
        parts.append(summary)
    if arguments:
        parts.append(f"Arguments: {arguments}")

    return " ".join(parts) if parts else "(no arguments provided)"


def _build_response(
    responder_output: dict,
    presenter_output: dict,
    provider: InferenceProvider | None,
) -> str:
    """Build the response from agent_b to agent_a's presentation."""
    if provider is not None:
        # With a provider we could ask the LLM to craft a rebuttal.
        # For now, fall through to the static formatter.
        pass

    summary = responder_output.get("summary", "")
    arguments = responder_output.get("arguments", "")
    verdict = responder_output.get("verdict", "")

    parts = []
    if verdict:
        parts.append(f"Verdict: {verdict}.")
    if summary:
        parts.append(summary)
    if arguments:
        parts.append(f"Counter-arguments: {arguments}")

    return " ".join(parts) if parts else "(no response provided)"


def _build_rebuttal(
    original_output: dict,
    opponent_output: dict,
    provider: InferenceProvider | None,
) -> str:
    """Build the final rebuttal for a 1.5-round debate."""
    if provider is not None:
        # With a provider we could generate a targeted rebuttal that
        # directly addresses the opponent's response.
        pass

    verdict = original_output.get("verdict", "")
    arguments = original_output.get("arguments", "")

    opp_verdict = opponent_output.get("verdict", "")
    opp_summary = opponent_output.get("summary", "")

    parts = []
    if opp_verdict:
        parts.append(f"Responding to {opp_verdict}.")
    if opp_summary:
        parts.append(f"Regarding \"{opp_summary}\":")
    if arguments:
        parts.append(f"Rebuttal: {arguments}")
    elif verdict:
        parts.append(f"Maintaining position: {verdict}.")

    return " ".join(parts) if parts else "(no rebuttal provided)"


def _build_summary(
    agent_a: str,
    agent_b: str,
    agent_a_output: dict,
    agent_b_output: dict,
    debate: Debate,
) -> str:
    """Build a concise summary of the debate exchange."""
    a_verdict = agent_a_output.get("verdict", "unknown")
    b_verdict = agent_b_output.get("verdict", "unknown")
    rounds_label = f"{debate.rounds} round{'s' if debate.rounds != 1 else ''}"

    return (
        f"{debate.purpose}: {agent_a} ({a_verdict}) vs "
        f"{agent_b} ({b_verdict}), {rounds_label}."
    )
