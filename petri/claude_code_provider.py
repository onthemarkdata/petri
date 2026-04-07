"""Claude Code harness-based InferenceProvider for Petri.

Routes inference through the ``claude`` CLI in print mode.
Claude Code handles all authentication and model routing
(cloud models via API, local models via Ollama).
See: https://docs.ollama.com/integrations/claude-code
"""

from __future__ import annotations

import json
import logging
import re
import subprocess

from petri.config import LLM_INFERENCE_MODEL

logger = logging.getLogger(__name__)





def _extract_json(text: str) -> dict | None:
    """Try to extract a JSON object from LLM output."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass

    json_match = re.search(r"```(?:json)?\s*\n?(\{.*?\})\s*\n?```", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Greedy match for nested JSON
    json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def _coerce_str(value: object) -> str:
    """Coerce LLM output to string. Handles lists/dicts returned instead of strings."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                parts.append("; ".join(f"{k}: {v}" for k, v in item.items()))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    if isinstance(value, dict):
        return json.dumps(value)
    return str(value) if value else ""


def _parse_verdict(text: str, valid_verdicts: list[str]) -> str:
    upper = text.upper()
    for verdict in valid_verdicts:
        if verdict in upper:
            return verdict
    return valid_verdicts[0]


class ClaudeCodeProvider:
    """InferenceProvider that routes through the claude CLI."""

    def __init__(self, model: str = LLM_INFERENCE_MODEL):
        self.model = model

    def _ask(self, prompt: str) -> str:
        """Send a prompt to claude CLI in print mode."""
        result = subprocess.run(
            [
                "claude",
                "--print",
                "--model", self.model,
                prompt,
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            logger.warning("claude CLI error: %s", result.stderr[:500])
        return result.stdout.strip()

    def generate_clarifying_questions(
        self, claim: str, max_questions: int = 5
    ) -> list[dict]:
        prompt = (
            f"Generate {max_questions} clarifying questions for this research claim:\n"
            f"\"{claim}\"\n\n"
            f"Return ONLY a JSON array: [{{\"question\": \"...\", \"options\": [...]}}]"
        )
        raw = self._ask(prompt)
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return parsed[:max_questions]
        except (json.JSONDecodeError, TypeError):
            pass
        return []

    def decompose_claim(self, claim: str, clarifications: list[dict]) -> dict:
        clarification_text = ""
        if clarifications:
            lines = [f"Q: {c['question']} A: {c.get('answer', 'N/A')}" for c in clarifications]
            clarification_text = "\nClarifications:\n" + "\n".join(lines)

        prompt = (
            "Decompose this claim using FIRST PRINCIPLES thinking.\n"
            "Break it down to fundamental, irreducible truths that can be independently verified.\n\n"
            "DO NOT just rephrase the claim. Ask 'WHY?' repeatedly until you reach bedrock facts.\n"
            "Identify key DEFINITIONS, ASSUMPTIONS, and EVIDENCE needed.\n\n"
            f"Claim: \"{claim}\"{clarification_text}\n\n"
            "Return JSON with 'nodes' and 'edges' arrays.\n"
            "Each node: {level, seq, claim_text, dependencies: [{level, seq}]}\n"
            "Each edge: {from: {level, seq}, to: {level, seq}}\n"
            "Level 0 = original claim. Level 1 = core assumptions. "
            "Level 2 = fundamental facts.\n\n"
            "Return ONLY the JSON object."
        )
        raw = self._ask(prompt)
        parsed = _extract_json(raw)
        if parsed and "nodes" in parsed:
            return parsed
        return {"nodes": [], "edges": []}

    def decompose_why(self, premise: str, parent_level: int, parent_seq: int) -> list[dict]:
        """Five Whys: ask 'Why is this true?' for a single premise.

        Returns a list of sub-premise dicts: [{claim_text, is_atomic}].
        If the premise is already atomic (directly verifiable), returns
        an empty list.
        """
        prompt = (
            "FIVE WHYS DECOMPOSITION — one level deeper.\n\n"
            f"Premise: \"{premise}\"\n\n"
            "Ask: WHY is this true? What must be true for this premise to hold?\n\n"
            "If this premise is already ATOMIC (a directly verifiable fact, a "
            "definition, or a measurement that can be looked up), return:\n"
            '{"sub_premises": [], "is_atomic": true, "reason": "why it is atomic"}\n\n'
            "Otherwise, break it into 2-4 sub-premises that are more fundamental. "
            "Each sub-premise should be closer to a bedrock, independently verifiable truth.\n"
            "Return:\n"
            '{"sub_premises": [{"claim_text": "...", "is_atomic": false}], "is_atomic": false}\n\n'
            "Return ONLY the JSON object."
        )
        raw = self._ask(prompt)
        parsed = _extract_json(raw)
        if parsed and parsed.get("is_atomic", False):
            return []
        if parsed and "sub_premises" in parsed:
            return parsed["sub_premises"]
        return []

    def assess_node(
        self, node_id: str, claim_text: str, context: dict, agent_role: str
    ) -> "AssessmentResult":
        from petri.config import get_agent_verdicts, get_agent_instruction, get_agents_with_sources
        from petri.models import AssessmentResult, SourceCitation

        valid_verdicts = get_agent_verdicts(agent_role) or ["PASS"]
        verdict_list = ", ".join(valid_verdicts)
        role_instruction = get_agent_instruction(agent_role) or "Assess this claim thoroughly."

        sources_schema = ""
        if agent_role in get_agents_with_sources():
            sources_schema = (
                '- "sources_cited": array of {"url_or_name": "...", '
                '"hierarchy_level": 1-6, "finding": "...", '
                '"supports_or_contradicts": "supports"|"contradicts"}\n'
            )

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
                f"\n--- Prior Evidence ---\n{prior_evidence}\n--- End Prior Evidence ---\n\n"
                f"Build on the evidence above. Focus on new insights and gaps.\n"
            )

        prompt = (
            f"You are the {agent_role} agent in a research validation pipeline.\n\n"
            f"{role_instruction}\n\n"
            f"Node: {node_id}\n"
            f"Claim: \"{claim_text}\"\n"
            f"Context: {context_str}\n"
            f"{evidence_section}\n"
            f"Valid verdicts: {verdict_list}\n\n"
            f"Return ONLY a JSON object with:\n"
            f'- "verdict": one of [{verdict_list}]\n'
            f'- "summary": detailed 3-5 sentence assessment\n'
            f'- "arguments": thorough analysis as a single string\n'
            f'- "evidence": specific facts and data as a single string\n'
            f'- "confidence": low/medium/high with justification\n'
            f"{sources_schema}\n"
            f"All fields must be strings (not arrays). Return ONLY the JSON."
        )

        raw = self._ask(prompt)
        parsed = _extract_json(raw)

        if parsed and "verdict" in parsed:
            validated_verdict = parsed["verdict"].upper()
            if validated_verdict not in valid_verdicts:
                validated_verdict = _parse_verdict(raw, valid_verdicts)

            # Coerce sources_cited to SourceCitation models
            raw_sources = parsed.get("sources_cited", [])
            sources = []
            if isinstance(raw_sources, list):
                for source_entry in raw_sources:
                    if isinstance(source_entry, dict):
                        sources.append(SourceCitation(**{
                            k: v for k, v in source_entry.items()
                            if k in SourceCitation.model_fields
                        }))

            return AssessmentResult(
                agent=agent_role,
                verdict=validated_verdict,
                summary=_coerce_str(parsed.get("summary", "")),
                arguments=_coerce_str(parsed.get("arguments", "")),
                evidence=_coerce_str(parsed.get("evidence", "")),
                confidence=_coerce_str(parsed.get("confidence", "")),
                sources_cited=sources,
            )

        verdict = _parse_verdict(raw, valid_verdicts)
        return AssessmentResult(
            agent=agent_role,
            verdict=verdict,
            summary=raw[:500].strip(),
        )
