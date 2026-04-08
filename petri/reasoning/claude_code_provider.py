"""Claude Code harness-based InferenceProvider for Petri.

Routes inference through the ``claude`` CLI in print mode.
Claude Code handles authentication and model routing via the Anthropic API.
"""

from __future__ import annotations

import json
import logging
import random
import re
import shutil
import subprocess
import time
from typing import Callable, Optional

from petri.config import AGENT_TOOLS, LLM_INFERENCE_MODEL

logger = logging.getLogger(__name__)


# ── Retry policy for transient claude CLI failures ──────────────────────
# Up to (1 + _MAX_RETRIES) attempts per _ask call. Backoff is exponential
# with bounded jitter so concurrent workers don't synchronise their
# retries against the same upstream rate limit.
_MAX_RETRIES = 2
_RETRY_BASE_DELAY_SECONDS = 1.5
_RETRY_JITTER_SECONDS = 0.5


def _is_transient_failure(stderr: str) -> bool:
    """Decide whether a claude CLI failure looks worth retrying.

    Permanent markers win over transient ones — a stderr that mentions
    both 'auth' and 'rate limit' is treated as permanent (the auth
    issue won't fix itself). Empty/unknown stderr is treated as
    transient (claude CLI sometimes exits 1 with no diagnostic on
    network blips and rate limits).
    """
    text = (stderr or "").lower()

    # Permanent: don't waste retries on these.
    permanent_markers = (
        "unauthorized", "401", "403", "forbidden",
        "auth",  # auth, authentication, unauthenticated
        "model", "not found", "404",
        "context", "too long", "exceeds", "max tokens",
        "invalid api key",
        "billing",
    )
    for marker in permanent_markers:
        if marker in text:
            return False

    # Transient: worth retrying.
    transient_markers = (
        "rate limit", "rate-limit", "too many requests", "429",
        "timeout", "timed out",
        "connection reset", "connection refused", "broken pipe",
        "network",
        "service unavailable", "503", "502", "504",
        "internal server error", "500",
        "temporarily unavailable",
        "overloaded",
    )
    for marker in transient_markers:
        if marker in text:
            return True

    # Empty/unknown stderr — claude CLI sometimes fails this way on
    # transient issues. Retry once rather than fail hard.
    return not text.strip()


def _retry_delay_seconds(attempt: int) -> float:
    """Exponential backoff with bounded jitter. attempt is 1-indexed:
    attempt=1 → ~1.5s, attempt=2 → ~3s, attempt=3 → ~6s."""
    base = _RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1))
    return base + random.uniform(0, _RETRY_JITTER_SECONDS)


def _extract_text_delta(event: dict) -> str:
    """Extract assistant-text delta from a stream-json event.

    Defensive: returns "" for any event shape we don't recognise so the
    spinner stays empty rather than crashing if the format ever changes.
    Handles the common Anthropic streaming shapes:
      - {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "..."}}
      - {"type": "stream_event", "event": {<the above>}}
      - {"type": "assistant", "message": {"content": [{"type": "text", "text": "..."}]}}
    """
    if not isinstance(event, dict):
        return ""

    # Direct content_block_delta
    if event.get("type") == "content_block_delta":
        delta = event.get("delta") or {}
        if isinstance(delta, dict) and delta.get("type") == "text_delta":
            text = delta.get("text", "")
            return text if isinstance(text, str) else ""

    # Wrapped stream_event
    if event.get("type") == "stream_event":
        return _extract_text_delta(event.get("event") or {})

    # Full assistant message (sent as one event in some modes)
    if event.get("type") == "assistant":
        message = event.get("message") or {}
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text", "")
                    if isinstance(text, str):
                        parts.append(text)
            return "".join(parts)

    return ""





def _process_stream_lines(
    lines, on_progress: Callable[[str], None]
) -> str:
    """Process claude ``--output-format stream-json`` output lines.

    Accumulates text deltas from JSON events into a single buffer and
    feeds the most recent text line to ``on_progress`` after each delta.
    Returns the full accumulated text.

    Non-JSON lines are **dropped silently** rather than being treated
    as model output. claude CLI in stream-json mode emits structured
    JSON events for everything model-related; any plain-text line on
    stdout is almost always error noise (auth banner, rate-limit
    message, debug print) and should NOT pollute the response buffer
    or the spinner. Dropped lines are logged at DEBUG level for
    postmortem visibility.

    ``lines`` is any iterable of strings — a real ``proc.stdout`` in
    production, a list in tests.
    """
    buffer: list[str] = []
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            logger.debug(
                "Discarding non-JSON stream line: %r", line[:200]
            )
            continue
        chunk = _extract_text_delta(event)
        if not chunk:
            continue
        buffer.append(chunk)
        joined = "".join(buffer)
        last_line = joined.rsplit("\n", 1)[-1]
        if last_line:
            on_progress(last_line)
    return "".join(buffer)


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


class ClaudeCLIError(RuntimeError):
    """Raised when the ``claude`` subprocess exits with a non-zero code.

    Carries the exit code, captured stderr, and any partial stdout so
    callers can build informative error messages instead of silently
    swallowing the failure.
    """

    def __init__(
        self,
        *,
        exit_code: int,
        stderr: str,
        stdout: str = "",
    ) -> None:
        self.exit_code = exit_code
        self.stderr = stderr
        self.stdout = stdout
        stderr_preview = stderr.strip()[:300] or "(empty)"
        super().__init__(
            f"claude CLI exited {exit_code}. stderr: {stderr_preview}"
        )


def _parse_verdict(text: str, valid_verdicts: list[str]) -> str:
    upper = text.upper()
    for verdict in valid_verdicts:
        if verdict in upper:
            return verdict
    raise ValueError(
        f"Model output did not contain any recognized verdict. "
        f"Expected one of {valid_verdicts}; got: {text[:200]!r}"
    )


class ClaudeCodeProvider:
    """InferenceProvider that routes through the claude CLI.

    ``allowed_tools`` is the explicit allowlist passed to ``claude
    --allowedTools``. It defaults to ``AGENT_TOOLS`` (read from
    petri.yaml). Petri NEVER passes
    ``--allow-dangerously-skip-permissions`` — every tool grant is
    explicit and named, so adding new tools to Claude Code in the future
    cannot silently widen the agents' permissions.
    """

    def __init__(
        self,
        model: str = LLM_INFERENCE_MODEL,
        *,
        allowed_tools: list[str] | None = None,
    ):
        self.model = model
        # If allowed_tools is None, fall back to the global AGENT_TOOLS
        # default. An explicit empty list means "no tools" — text-only
        # reasoning, no web access — and is honored as-is.
        self.allowed_tools = (
            list(AGENT_TOOLS) if allowed_tools is None else list(allowed_tools)
        )
        if shutil.which("claude") is None:
            raise FileNotFoundError(
                "Claude Code CLI ('claude') not found on PATH. "
                "Petri requires it for inference.\n"
                "Install: https://docs.anthropic.com/en/docs/claude-code\n"
                "Run 'petri inspect' to check all prerequisites."
            )

    def _sleep(self, seconds: float) -> None:
        """Sleep for ``seconds``. Overridable so tests can drive the
        retry loop without actually waiting on the wall clock."""
        time.sleep(seconds)

    def _build_claude_command(
        self, prompt: str, *, streaming: bool
    ) -> list[str]:
        """Build the argv for a single claude CLI invocation.

        Centralised so both ``_ask_oneshot`` and ``_ask_streaming`` use
        the same flag set, including the explicit ``--allowedTools``
        allowlist. The prompt is always the final positional argument.
        """
        command = ["claude", "--print", "--model", self.model]
        if self.allowed_tools:
            # IMPORTANT: use the ``--allowedTools=value`` equals form,
            # NOT the space-separated ``--allowedTools value`` form.
            # claude CLI declares the flag as ``<tools...>`` (variadic),
            # so a space-separated value causes the parser to consume
            # every following positional argument — including our prompt
            # — as additional tool names, producing the cryptic error
            # ``Error: Input must be provided either through stdin or
            # as a prompt argument when using --print``. The equals form
            # binds the value to the flag unambiguously.
            command.append(
                f"--allowedTools={','.join(self.allowed_tools)}"
            )
        else:
            # Explicit empty list means "no tools at all" — text-only
            # reasoning. ``--tools=`` (equals form, empty value) is
            # claude CLI's documented way to disable the entire
            # built-in tool set, distinct from simply omitting
            # --allowedTools (which would inherit whatever the user's
            # settings.json grants). Use the equals form for the same
            # variadic-parser reason as above.
            command.append("--tools=")
        if streaming:
            command.extend(
                [
                    "--output-format", "stream-json",
                    "--include-partial-messages",
                    "--verbose",
                ]
            )
        command.append(prompt)
        return command

    def _ask(
        self,
        prompt: str,
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Send a prompt to claude CLI in print mode.

        If ``on_progress`` is supplied, streams output via stream-json so the
        caller can display the model's text as it's generated. Otherwise uses
        the simpler one-shot path.
        """
        if on_progress is None:
            return self._ask_oneshot(prompt)
        return self._ask_streaming(prompt, on_progress)

    def _ask_oneshot(self, prompt: str) -> str:
        """One-shot subprocess call with retry on transient failures.

        Retries up to ``_MAX_RETRIES`` times for transient failures
        (rate limits, network blips, server errors). Permanent failures
        (auth, model not found, context too long) raise immediately so
        we don't burn budget on a doomed call.
        """
        for attempt in range(1, _MAX_RETRIES + 2):
            try:
                return self._oneshot_attempt(prompt)
            except ClaudeCLIError as cli_error:
                if attempt > _MAX_RETRIES:
                    raise
                if not _is_transient_failure(cli_error.stderr):
                    raise
                delay = _retry_delay_seconds(attempt)
                logger.warning(
                    "Transient claude CLI failure (exit %d, attempt %d/%d), "
                    "retrying in %.1fs. stderr: %r",
                    cli_error.exit_code,
                    attempt,
                    _MAX_RETRIES + 1,
                    delay,
                    (cli_error.stderr or "")[:200],
                )
                self._sleep(delay)
        # Unreachable: the loop either returns or raises.
        raise RuntimeError(  # pragma: no cover
            "_ask_oneshot retry loop fell through"
        )

    def _oneshot_attempt(self, prompt: str) -> str:
        """A single one-shot subprocess call. No retries.

        Raises ``ClaudeCLIError`` on non-zero exit so the caller (the
        retry wrapper) can decide whether to back off or give up.
        """
        try:
            result = subprocess.run(
                self._build_claude_command(prompt, streaming=False),
                capture_output=True,
                text=True,
                timeout=300,
            )
        except FileNotFoundError:
            raise FileNotFoundError(
                "Claude Code CLI ('claude') not found on PATH. "
                "Install: https://docs.anthropic.com/en/docs/claude-code"
            ) from None
        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            stdout = (result.stdout or "").strip()
            logger.warning(
                "claude CLI error (exit %d). stderr=%r stdout=%r",
                result.returncode,
                stderr[:500],
                stdout[:500],
            )
            if "model" in stderr.lower() and "not found" in stderr.lower():
                logger.warning(
                    "Model '%s' may not be available. "
                    "Verify the model name and your Claude Code authentication.",
                    self.model,
                )
            raise ClaudeCLIError(
                exit_code=result.returncode,
                stderr=stderr,
                stdout=stdout,
            )
        return result.stdout.strip()

    def _ask_streaming(
        self, prompt: str, on_progress: Callable[[str], None]
    ) -> str:
        """Stream claude output line-by-line with retry on transient failures.

        Each retry restarts the subprocess from scratch — partial output
        from the failed attempt is discarded. ``on_progress`` may be
        called multiple times across retries (the spinner will simply
        update twice).
        """
        for attempt in range(1, _MAX_RETRIES + 2):
            try:
                return self._streaming_attempt(prompt, on_progress)
            except ClaudeCLIError as cli_error:
                if attempt > _MAX_RETRIES:
                    raise
                if not _is_transient_failure(cli_error.stderr):
                    raise
                delay = _retry_delay_seconds(attempt)
                logger.warning(
                    "Transient claude CLI streaming failure "
                    "(exit %d, attempt %d/%d), retrying in %.1fs. stderr: %r",
                    cli_error.exit_code,
                    attempt,
                    _MAX_RETRIES + 1,
                    delay,
                    (cli_error.stderr or "")[:200],
                )
                self._sleep(delay)
        # Unreachable: the loop either returns or raises.
        raise RuntimeError(  # pragma: no cover
            "_ask_streaming retry loop fell through"
        )

    def _streaming_attempt(
        self, prompt: str, on_progress: Callable[[str], None]
    ) -> str:
        """A single streaming subprocess call. No retries.

        Uses ``--output-format stream-json --include-partial-messages`` so
        each line of stdout is a JSON event. Assistant-text deltas are
        accumulated into a buffer; on each new chunk we feed the most recent
        line of accumulated text to ``on_progress``. Returns the full
        accumulated text on completion. Raises ``ClaudeCLIError`` on
        non-zero exit so the retry wrapper can decide whether to back off.
        """
        cmd = self._build_claude_command(prompt, streaming=True)
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except FileNotFoundError:
            raise FileNotFoundError(
                "Claude Code CLI ('claude') not found on PATH. "
                "Install: https://docs.anthropic.com/en/docs/claude-code"
            ) from None

        accumulated_text = ""
        assert proc.stdout is not None
        try:
            accumulated_text = _process_stream_lines(proc.stdout, on_progress)
        finally:
            try:
                proc.wait(timeout=300)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

        if proc.returncode != 0:
            stderr = ""
            if proc.stderr is not None:
                try:
                    stderr = proc.stderr.read() or ""
                except Exception:
                    stderr = ""
            partial_stdout = accumulated_text.strip()
            logger.warning(
                "claude CLI streaming error (exit %d). stderr=%r stdout=%r",
                proc.returncode,
                stderr.strip()[:500],
                partial_stdout[:500],
            )
            raise ClaudeCLIError(
                exit_code=proc.returncode,
                stderr=stderr.strip(),
                stdout=partial_stdout,
            )

        return accumulated_text.strip()

    def assess_claim_substance(
        self,
        claim: str,
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> dict:
        """Decide whether a claim is substantive enough to warrant a wizard.

        The model flags placeholder/test/empty input ("this is a test claim",
        "asdf", "hello world") as non-substantive so the CLI can short-circuit
        the clarifying-question wizard.

        Returns a dict with keys:
            is_substantive (bool)
            reason (str)               -- one-sentence explanation
            suggested_rewrite (str)    -- optional; "" if none
        """
        prompt = (
            "Assess whether the following text is a SUBSTANTIVE research claim "
            "that warrants decomposition, or whether it is placeholder/test "
            "input (e.g. 'this is a test claim', 'asdf', 'hello world', empty).\n\n"
            f"Text: \"{claim}\"\n\n"
            "Return ONLY a JSON object:\n"
            "{\n"
            '  "is_substantive": true|false,\n'
            '  "reason": "one-sentence explanation",\n'
            '  "suggested_rewrite": "tighter rephrasing, or empty string"\n'
            "}\n"
        )
        try:
            raw = self._ask(prompt, on_progress=on_progress)
        except ClaudeCLIError:
            # Subprocess failed — fall through to the wizard rather than
            # block the user. The seed flow will surface the error if
            # subsequent calls also fail.
            return {"is_substantive": True, "reason": "", "suggested_rewrite": ""}
        parsed = _extract_json(raw)
        if not isinstance(parsed, dict):
            # Treat parse failure as substantive — fall through to the wizard
            # rather than block the user on a model glitch.
            return {"is_substantive": True, "reason": "", "suggested_rewrite": ""}
        return {
            "is_substantive": bool(parsed.get("is_substantive", True)),
            "reason": _coerce_str(parsed.get("reason", "")),
            "suggested_rewrite": _coerce_str(parsed.get("suggested_rewrite", "")),
        }

    def generate_clarifying_questions(
        self,
        claim: str,
        max_questions: int = 5,
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> list[dict]:
        prompt = (
            f"Generate {max_questions} CLAIM-SPECIFIC clarifying questions for this research claim. "
            "The questions must be tailored to the actual content of the claim — do not use generic "
            "questions like 'what is the domain?' or 'what is the time horizon?'. Each question should "
            "surface a specific assumption, scope boundary, or definitional ambiguity in THIS claim.\n\n"
            f"Claim: \"{claim}\"\n\n"
            "For each question, optionally provide 2-5 multiple-choice options when the answer space is "
            "naturally bounded; otherwise leave options empty for free-text input.\n\n"
            "Return ONLY a JSON array: "
            '[{"question": "...", "options": ["...", "..."]}, ...]'
        )
        try:
            raw = self._ask(prompt, on_progress=on_progress)
        except ClaudeCLIError:
            # Subprocess failed — return no questions and let the seed
            # flow proceed without the wizard step.
            return []
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return parsed[:max_questions]
        except (json.JSONDecodeError, TypeError):
            pass
        # Try the more lenient extractor for fenced/wrapped output
        wrapped = _extract_json(raw)
        if isinstance(wrapped, list):
            return wrapped[:max_questions]
        return []

    def decompose_claim(
        self,
        claim: str,
        clarifications: list[dict],
        guidance: str = "",
        max_premises: int = 5,
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> dict:
        clarification_text = ""
        if clarifications:
            lines = [f"Q: {c['question']} A: {c.get('answer', 'N/A')}" for c in clarifications]
            clarification_text = "\nClarifications:\n" + "\n".join(lines)

        guidance_text = ""
        if guidance.strip():
            guidance_text = (
                "\nRefinement guidance from user (must shape this re-roll):\n"
                f"{guidance.strip()}\n"
            )

        prompt = (
            "Decompose this claim into first-principles premises.\n\n"
            f"Claim: \"{claim}\"{clarification_text}{guidance_text}\n\n"
            "Work through this in four explicit steps. Show your reasoning as "
            "you go — the user is watching this stream live.\n\n"
            "STEP 1 — BRAINSTORM (overgenerate).\n"
            "List 10-15 candidate premises that, if false, would invalidate "
            "the claim. Cast a wide net. Cover different angles: definitions, "
            "mechanisms, scope boundaries, counterfactuals, hidden assumptions. "
            "Number them 1..N. Do NOT filter at this stage.\n\n"
            "STEP 2 — PRIORITIZE.\n"
            "For each candidate, score on three dimensions:\n"
            "  - LOAD-BEARING: would the claim collapse if this were false? (HIGH/MED/LOW)\n"
            "  - CONTESTABLE: is this actually in question, not trivially true? (HIGH/MED/LOW)\n"
            "  - DISTINCT: does it cover ground the other top candidates don't?\n"
            "Briefly justify each score.\n\n"
            "STEP 3 — SELECT.\n"
            f"Pick the TOP {max_premises} premises after ranking. Order them by "
            "importance (most important first). Drop duplicates and trivial restatements. "
            "Do not pick the first ones you wrote down — pick the ones that scored highest.\n\n"
            "STEP 4 — EMIT JSON.\n"
            "Return a JSON object on the final lines:\n"
            '{"nodes": [{"level": 1, "seq": 1, "claim_text": "..."}, ...], "edges": []}\n\n'
            f"The JSON must contain AT MOST {max_premises} nodes (fewer is fine if "
            "fewer genuine first-principles premises exist). Return no commentary "
            "after the JSON."
        )
        raw = self._ask(prompt, on_progress=on_progress)
        parsed = _extract_json(raw)
        if parsed and "nodes" in parsed:
            return parsed
        return {"nodes": [], "edges": []}

    def decompose_why(
        self,
        premise: str,
        parent_level: int,
        parent_seq: int,
        max_premises: int = 5,
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> list[dict]:
        """Five Whys: ask 'Why is this true?' for a single premise.

        Returns a list of sub-premise dicts: [{claim_text, is_atomic}].
        If the premise is already atomic (directly verifiable), returns
        an empty list.
        """
        prompt = (
            "FIVE WHYS — drill one level deeper into a parent premise.\n\n"
            f"Premise: \"{premise}\"\n\n"
            "Work through this in four explicit steps. Show your reasoning "
            "as you go — the user is watching this stream live.\n\n"
            "STEP 1 — BRAINSTORM (overgenerate).\n"
            "List 6-10 candidate sub-premises that must hold for the parent "
            "premise to be true. Number them 1..N. Don't filter yet.\n\n"
            "STEP 2 — PRIORITIZE.\n"
            "For each candidate, score on:\n"
            "  - FOUNDATIONALITY: closer to bedrock = higher (HIGH/MED/LOW)\n"
            "  - INDEPENDENCE: separately verifiable from the others (HIGH/MED/LOW)\n"
            "  - LOAD-BEARING: would the parent collapse if this sub-premise were false?\n"
            "Briefly justify each score.\n\n"
            "STEP 3 — SELECT.\n"
            f"Pick the TOP {max_premises} sub-premises. Drop redundant or "
            "surface-level ones. Do not pick the first ones you wrote down — "
            "pick the ones that scored highest.\n\n"
            "If the parent is already ATOMIC (a directly verifiable fact, "
            "definition, or measurement), skip steps 1-3 and return:\n"
            '{"sub_premises": [], "is_atomic": true, "reason": "why it is atomic"}\n\n'
            "STEP 4 — EMIT JSON.\n"
            'Return: {"sub_premises": [{"claim_text": "...", "is_atomic": false}, ...], "is_atomic": false}\n\n'
            f"The sub_premises list must contain AT MOST {max_premises} items. "
            "Return no commentary after the JSON."
        )
        raw = self._ask(prompt, on_progress=on_progress)
        parsed = _extract_json(raw)
        if parsed and parsed.get("is_atomic", False):
            return []
        if parsed and "sub_premises" in parsed:
            return parsed["sub_premises"]
        return []

    def assess_cell(
        self,
        cell_id: str,
        claim_text: str,
        context: dict,
        agent_role: str,
        *,
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> "AssessmentResult":
        from petri.config import get_agent_verdicts, get_agent_instruction
        from petri.models import AssessmentResult, SourceCitation

        valid_verdicts = get_agent_verdicts(agent_role)
        if not valid_verdicts:
            raise ValueError(
                f"Unknown agent role {agent_role!r}: not declared in agents.yaml. "
                "Add it to .petri/defaults/petri.yaml under 'agents:' with explicit "
                "verdicts_pass and verdicts_block lists."
            )
        verdict_list = ", ".join(valid_verdicts)
        role_instruction = get_agent_instruction(agent_role) or "Assess this claim thoroughly."

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
            f"Cell: {cell_id}\n"
            f"Claim: \"{claim_text}\"\n"
            f"Context: {context_str}\n"
            f"{evidence_section}\n"
            f"Valid verdicts: {verdict_list}\n\n"
            f"Return ONLY a JSON object with:\n"
            f'- "verdict": one of [{verdict_list}]\n'
            f'- "sources_cited": REQUIRED array. Each source must have:\n'
            f'  - "url": full URL (https://...) to the source\n'
            f'  - "title": "Publication, Article Title (Year)"\n'
            f'  - "hierarchy_level": 1-6 (1=direct measurement, 2=authoritative docs, '
            f'3=derived calculation, 4=expert consensus, 5=single expert, 6=community report)\n'
            f'  - "finding": 1-2 sentence finding from this source\n'
            f'  - "supports_or_contradicts": "supports" or "contradicts"\n'
            f'- "summary": 1-3 concise sentences. Enumerate dimensions, cite specific numbers.\n'
            f'- "confidence": "HIGH", "MEDIUM", or "LOW"\n\n'
            f"RULES:\n"
            f"- Every claim MUST be backed by at least one source with a valid URL.\n"
            f"- Keep summary TERSE — citations are the evidence.\n"
            f"- Do NOT include \"arguments\" or \"evidence\" fields.\n\n"
            f"Return ONLY the JSON."
        )

        try:
            raw = self._ask(prompt, on_progress=on_progress)
        except ClaudeCLIError as cli_error:
            # Subprocess failure — surface the real stderr in the summary
            # so the user can see WHY claude failed (auth, rate limit,
            # model name, prompt too long, etc.) instead of a generic
            # "execution error". Truncate aggressively to keep the row
            # readable in the multi-spinner UI.
            stderr_excerpt = (cli_error.stderr or "").strip()[:400] or "(empty)"
            return AssessmentResult(
                agent=agent_role,
                verdict="EXECUTION_ERROR",
                summary=(
                    f"claude CLI failed (exit {cli_error.exit_code}). "
                    f"stderr: {stderr_excerpt}"
                ),
            )
        parsed = _extract_json(raw)

        if parsed and "verdict" in parsed:
            raw_verdict = _coerce_str(parsed["verdict"]).upper()
            if raw_verdict in valid_verdicts:
                validated_verdict = raw_verdict
            else:
                # JSON had a verdict field but its value isn't valid for
                # this agent — try to recover from the raw text. If THAT
                # also fails, surface as EXECUTION_ERROR rather than
                # silently returning the first pass verdict.
                try:
                    validated_verdict = _parse_verdict(raw, valid_verdicts)
                except ValueError:
                    return AssessmentResult(
                        agent=agent_role,
                        verdict="EXECUTION_ERROR",
                        summary=(
                            f"Model returned unrecognized verdict {raw_verdict!r}. "
                            f"Raw output: {raw[:300]}"
                        ),
                    )

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
                confidence=_coerce_str(parsed.get("confidence", "")),
                sources_cited=sources,
            )

        # No JSON could be extracted — total failure path. Try to salvage
        # a verdict from raw text; otherwise surface EXECUTION_ERROR.
        try:
            verdict = _parse_verdict(raw, valid_verdicts)
        except ValueError:
            return AssessmentResult(
                agent=agent_role,
                verdict="EXECUTION_ERROR",
                summary=(
                    f"Model output could not be parsed as JSON or matched "
                    f"against a valid verdict. Raw output: {raw[:300]}"
                ),
            )
        return AssessmentResult(
            agent=agent_role,
            verdict=verdict,
            summary=raw[:500].strip(),
        )
