"""Unit tests for ``petri.reasoning.claude_code_provider``.

Focus: verdict parsing and ``assess_node`` error-handling semantics. The
critical invariants tested here are:

  1. ``_parse_verdict`` raises ``ValueError`` on no-match (it must NOT
     silently return the first pass verdict — that was the old bug).
  2. ``assess_node`` surfaces ``EXECUTION_ERROR`` when the model output
     is unparseable or returns an unknown verdict value.
  3. ``assess_node`` raises ``ValueError`` for undeclared agent roles
     rather than falling back to a default ``["PASS"]`` verdict list.

These tests stub out ``_ask`` so they never touch the real ``claude``
CLI subprocess. The stub mirrors the ``FakeProvider`` pattern used
elsewhere in the test suite (see ``tests/conftest.py``).
"""

from __future__ import annotations

import pytest

from petri.reasoning.claude_code_provider import (
    ClaudeCLIError,
    ClaudeCodeProvider,
    _parse_verdict,
)


class StubProvider(ClaudeCodeProvider):
    """ClaudeCodeProvider subclass that returns a canned ``_ask`` response.

    Avoids the ``claude`` CLI dependency check in ``__init__`` by skipping
    the parent's ``__init__`` entirely — we only need the instance methods.
    """

    def __init__(self, canned_response: str) -> None:
        # Intentionally skip ClaudeCodeProvider.__init__ so we don't need
        # the real claude CLI on PATH. Set the minimal attributes the
        # instance methods touch.
        self.model = "test-model"
        self._canned_response = canned_response
        self.last_prompt: str | None = None

    def _ask(self, prompt, on_progress=None):  # type: ignore[override]
        self.last_prompt = prompt
        return self._canned_response


# ── _parse_verdict ────────────────────────────────────────────────────────


def test_parse_verdict_returns_match_when_present():
    """Happy path: the first recognized verdict in the text wins."""
    valid_verdicts = ["EVIDENCE_SUFFICIENT", "NEEDS_MORE_EVIDENCE"]
    raw_text = "After analysis the verdict is EVIDENCE_SUFFICIENT for this claim."
    assert _parse_verdict(raw_text, valid_verdicts) == "EVIDENCE_SUFFICIENT"


def test_parse_verdict_raises_when_no_match():
    """Critical: must raise ValueError, NOT silently return valid_verdicts[0]."""
    valid_verdicts = ["EVIDENCE_SUFFICIENT", "NEEDS_MORE_EVIDENCE"]
    raw_text = "The model panicked and returned gibberish."
    with pytest.raises(ValueError) as exception_info:
        _parse_verdict(raw_text, valid_verdicts)
    error_message = str(exception_info.value)
    assert "did not contain any recognized verdict" in error_message
    assert "EVIDENCE_SUFFICIENT" in error_message


# ── assess_node error handling ────────────────────────────────────────────


def test_assess_node_returns_execution_error_on_unparseable_output():
    """When ``_ask`` returns unparseable text, verdict must be EXECUTION_ERROR.

    Regression guard against the old silent-PASS bug where failing calls
    became the strongest PASS verdict.
    """
    provider = StubProvider("Execution error")
    result = provider.assess_node(
        node_id="test-dish-colony-001-001",
        claim_text="A sample claim",
        context={},
        agent_role="investigator",
    )
    assert result.verdict == "EXECUTION_ERROR"
    assert result.agent == "investigator"
    # The summary should mention the raw output to aid debugging.
    assert "Execution error" in result.summary
    # And it must NOT be the first pass verdict (what the old code returned).
    assert result.verdict != "EVIDENCE_SUFFICIENT"


def test_assess_node_returns_execution_error_on_invalid_verdict_field():
    """When JSON has a verdict field with a bogus value, surface EXECUTION_ERROR."""
    provider = StubProvider('{"verdict": "MAYBE", "summary": "x"}')
    result = provider.assess_node(
        node_id="test-dish-colony-001-001",
        claim_text="A sample claim",
        context={},
        agent_role="investigator",
    )
    assert result.verdict == "EXECUTION_ERROR"
    assert "MAYBE" in result.summary


def test_assess_node_raises_on_unknown_agent_role():
    """Unknown agent roles must raise, not fall back to a PASS sentinel."""
    provider = StubProvider('{"verdict": "EVIDENCE_SUFFICIENT", "summary": "ok"}')
    with pytest.raises(ValueError) as exception_info:
        provider.assess_node(
            node_id="test-dish-colony-001-001",
            claim_text="A sample claim",
            context={},
            agent_role="not_a_real_agent",
        )
    error_message = str(exception_info.value)
    assert "not_a_real_agent" in error_message
    assert "agents.yaml" in error_message


def test_assess_node_returns_first_pass_verdict_when_model_says_so():
    """Positive control: the refactor didn't break the happy path."""
    provider = StubProvider(
        '{"verdict": "EVIDENCE_SUFFICIENT", '
        '"summary": "Three sources confirm.", '
        '"confidence": "HIGH", '
        '"sources_cited": []}'
    )
    result = provider.assess_node(
        node_id="test-dish-colony-001-001",
        claim_text="A sample claim",
        context={},
        agent_role="investigator",
    )
    assert result.verdict == "EVIDENCE_SUFFICIENT"
    assert result.summary == "Three sources confirm."
    assert result.confidence == "HIGH"


def test_assess_node_recovers_from_invalid_json_verdict_via_raw_text():
    """If JSON verdict is bogus but raw text still contains a valid verdict,
    the raw-text salvage path should succeed (no EXECUTION_ERROR).
    """
    provider = StubProvider(
        '{"verdict": "MAYBE", "summary": "x"} '
        "Final decision: EVIDENCE_SUFFICIENT based on three primary sources."
    )
    result = provider.assess_node(
        node_id="test-dish-colony-001-001",
        claim_text="A sample claim",
        context={},
        agent_role="investigator",
    )
    assert result.verdict == "EVIDENCE_SUFFICIENT"


def test_assess_node_accepts_socratic_questioner_role():
    """``socratic_questioner`` was added to defaults/petri.yaml; verify the
    config loader sees it so the Socratic phase doesn't trip the new
    "unknown agent" error.
    """
    provider = StubProvider(
        '{"verdict": "CLARIFIED", "summary": "terms defined"}'
    )
    result = provider.assess_node(
        node_id="test-dish-colony-001-001",
        claim_text="A sample claim",
        context={"phase": "socratic_clarify"},
        agent_role="socratic_questioner",
    )
    assert result.verdict == "CLARIFIED"


# ── _process_stream_lines (stream-json line handling) ──────────────────


def _make_text_delta_event(text: str) -> str:
    """Build a stream-json text_delta event line as claude CLI emits it."""
    import json as _json
    return _json.dumps({
        "type": "content_block_delta",
        "delta": {"type": "text_delta", "text": text},
    })


def test_process_stream_lines_accumulates_text_deltas():
    """Happy path: a sequence of text_delta events is concatenated into
    the returned buffer and each chunk fires on_progress."""
    from petri.reasoning.claude_code_provider import _process_stream_lines

    progress_calls: list[str] = []
    lines = [
        _make_text_delta_event("Hello "),
        _make_text_delta_event("world"),
        _make_text_delta_event("!"),
    ]
    result = _process_stream_lines(lines, progress_calls.append)
    assert result == "Hello world!"
    # on_progress fires after each delta with the latest line.
    assert progress_calls == ["Hello ", "Hello world", "Hello world!"]


def test_process_stream_lines_drops_non_json_lines():
    """Critical: non-JSON lines (error noise, debug prints) must NOT
    appear in the response buffer or in on_progress calls. They are
    dropped silently — the spinner is for model output only."""
    from petri.reasoning.claude_code_provider import _process_stream_lines

    progress_calls: list[str] = []
    lines = [
        "Error: rate limit exceeded",      # plain-text noise
        _make_text_delta_event("real "),
        "claude CLI debug: connecting...",  # plain-text noise
        _make_text_delta_event("output"),
        "Some other plain line",            # plain-text noise
    ]
    result = _process_stream_lines(lines, progress_calls.append)
    # Buffer contains ONLY model text — no error noise.
    assert result == "real output"
    # on_progress NEVER receives the noise lines.
    assert "Error: rate limit exceeded" not in progress_calls
    assert "claude CLI debug: connecting..." not in progress_calls
    assert "Some other plain line" not in progress_calls
    # And it DOES receive the real chunks.
    assert progress_calls == ["real ", "real output"]


def test_process_stream_lines_skips_empty_lines():
    """Whitespace-only lines are silently skipped before JSON parsing."""
    from petri.reasoning.claude_code_provider import _process_stream_lines

    progress_calls: list[str] = []
    lines = [
        "",
        "   ",
        "\n",
        _make_text_delta_event("hello"),
        "",
    ]
    result = _process_stream_lines(lines, progress_calls.append)
    assert result == "hello"
    assert progress_calls == ["hello"]


def test_process_stream_lines_handles_events_without_text():
    """Events that aren't text_delta (e.g. message_start, ping) yield no
    text and don't fire on_progress."""
    from petri.reasoning.claude_code_provider import _process_stream_lines
    import json as _json

    progress_calls: list[str] = []
    lines = [
        _json.dumps({"type": "message_start"}),
        _make_text_delta_event("real text"),
        _json.dumps({"type": "ping"}),
    ]
    result = _process_stream_lines(lines, progress_calls.append)
    assert result == "real text"
    assert progress_calls == ["real text"]


def test_process_stream_lines_uses_last_line_for_progress():
    """When a delta contains a newline, on_progress receives only the
    text after the most recent newline (the 'last visible line')."""
    from petri.reasoning.claude_code_provider import _process_stream_lines

    progress_calls: list[str] = []
    lines = [
        _make_text_delta_event("first line\nsecond line"),
        _make_text_delta_event(" continued"),
    ]
    result = _process_stream_lines(lines, progress_calls.append)
    assert result == "first line\nsecond line continued"
    # First call: only "second line" (after the newline).
    # Second call: "second line continued" (still after the newline).
    assert progress_calls == ["second line", "second line continued"]


def test_process_stream_lines_does_not_pollute_buffer_on_noise():
    """Regression for the bug that surfaced 'claude CLI error (exit 1):'
    on the spinner: even if the noise line happens to look like a
    sentence the model might have produced, it must not end up in the
    response buffer that downstream parsers see."""
    from petri.reasoning.claude_code_provider import _process_stream_lines

    progress_calls: list[str] = []
    lines = [
        "claude CLI error (exit 1):",  # the exact string the user saw
        _make_text_delta_event('{"verdict": "EVIDENCE_SUFFICIENT"}'),
    ]
    result = _process_stream_lines(lines, progress_calls.append)
    # The noise is dropped; only the real model output remains.
    assert result == '{"verdict": "EVIDENCE_SUFFICIENT"}'
    assert "claude CLI error" not in result
    # And the spinner never sees the noise either.
    assert all("claude CLI error" not in call for call in progress_calls)


# ── _is_transient_failure classification ────────────────────────────────


def test_is_transient_failure_classifies_rate_limit():
    from petri.reasoning.claude_code_provider import _is_transient_failure

    assert _is_transient_failure("Error: rate limit exceeded") is True
    assert _is_transient_failure("HTTP 429 Too Many Requests") is True
    assert _is_transient_failure("Request was throttled") is False  # not in our list


def test_is_transient_failure_classifies_server_errors():
    from petri.reasoning.claude_code_provider import _is_transient_failure

    assert _is_transient_failure("503 Service Unavailable") is True
    assert _is_transient_failure("HTTP 502 bad gateway") is True
    assert _is_transient_failure("Internal server error") is True
    assert _is_transient_failure("temporarily unavailable") is True
    assert _is_transient_failure("overloaded") is True


def test_is_transient_failure_classifies_network_errors():
    from petri.reasoning.claude_code_provider import _is_transient_failure

    assert _is_transient_failure("connection reset by peer") is True
    assert _is_transient_failure("network error: timeout") is True
    assert _is_transient_failure("Request timed out after 60s") is True


def test_is_transient_failure_classifies_permanent_failures():
    from petri.reasoning.claude_code_provider import _is_transient_failure

    assert _is_transient_failure("Error: unauthorized (401)") is False
    assert _is_transient_failure("HTTP 403 forbidden") is False
    assert _is_transient_failure("Authentication failed") is False
    assert _is_transient_failure("Invalid API key") is False
    assert _is_transient_failure("Model 'claude-foo' not found") is False
    assert _is_transient_failure("Prompt exceeds max context (200k tokens)") is False
    assert _is_transient_failure("Billing issue: please update payment") is False


def test_is_transient_failure_permanent_wins_over_transient():
    """A stderr that mentions both 'auth' and 'rate limit' is permanent
    — the auth issue won't fix itself, no point burning retries."""
    from petri.reasoning.claude_code_provider import _is_transient_failure

    assert _is_transient_failure(
        "auth error after multiple rate limit retries"
    ) is False


def test_is_transient_failure_empty_stderr_is_transient():
    """Empty stderr is the case the user actually hit — claude CLI
    sometimes exits 1 with no diagnostic on rate limits or network blips.
    Retry once rather than fail hard."""
    from petri.reasoning.claude_code_provider import _is_transient_failure

    assert _is_transient_failure("") is True
    assert _is_transient_failure("   \n  ") is True
    assert _is_transient_failure(None) is True  # type: ignore[arg-type]


# ── _ask retry behavior ──────────────────────────────────────────────────


class _FlakyOneshotProvider(ClaudeCodeProvider):
    """Provider whose ``_oneshot_attempt`` returns/raises a scripted
    sequence and whose ``_sleep`` is a no-op.

    Used to drive the retry loop without touching real subprocesses or
    the wall clock. Records the wait amounts that would have happened.
    """

    def __init__(self, sequence: list) -> None:
        self.model = "test-model"
        self.allowed_tools = []
        self._sequence = list(sequence)
        self.attempt_count = 0
        self.sleep_calls: list[float] = []

    def _oneshot_attempt(self, prompt: str) -> str:  # type: ignore[override]
        self.attempt_count += 1
        action = self._sequence.pop(0)
        if isinstance(action, ClaudeCLIError):
            raise action
        return action  # str — success

    def _sleep(self, seconds: float) -> None:  # type: ignore[override]
        # No-op: record the requested duration but don't actually wait,
        # so the test runs at full speed even with retries.
        self.sleep_calls.append(seconds)


def test_ask_oneshot_succeeds_on_first_attempt():
    provider = _FlakyOneshotProvider(sequence=["the model response"])
    result = provider._ask_oneshot("hello")
    assert result == "the model response"
    assert provider.attempt_count == 1
    assert provider.sleep_calls == []  # no retry, no sleep


def test_ask_oneshot_retries_transient_then_succeeds():
    provider = _FlakyOneshotProvider(
        sequence=[
            ClaudeCLIError(exit_code=1, stderr="rate limit exceeded", stdout=""),
            "the model response",
        ]
    )
    result = provider._ask_oneshot("hello")
    assert result == "the model response"
    assert provider.attempt_count == 2
    # One retry → one sleep call.
    assert len(provider.sleep_calls) == 1
    # First retry delay should be in the [_RETRY_BASE_DELAY_SECONDS,
    # _RETRY_BASE_DELAY_SECONDS + _RETRY_JITTER_SECONDS] window.
    assert 1.5 <= provider.sleep_calls[0] <= 2.0


def test_ask_oneshot_retries_then_gives_up():
    """After _MAX_RETRIES retries (3 attempts total), give up and raise
    the most recent ClaudeCLIError."""
    provider = _FlakyOneshotProvider(
        sequence=[
            ClaudeCLIError(exit_code=1, stderr="rate limit", stdout=""),
            ClaudeCLIError(exit_code=1, stderr="rate limit", stdout=""),
            ClaudeCLIError(exit_code=1, stderr="rate limit", stdout=""),
        ]
    )
    with pytest.raises(ClaudeCLIError):
        provider._ask_oneshot("hello")
    assert provider.attempt_count == 3
    # Two retries → two sleep calls (the third attempt isn't followed
    # by a sleep — it raises straight to the caller).
    assert len(provider.sleep_calls) == 2
    # Backoff grows.
    assert provider.sleep_calls[0] < provider.sleep_calls[1]


def test_ask_oneshot_does_not_retry_permanent_failure():
    """Auth failure should NOT trigger any retries — fail fast."""
    provider = _FlakyOneshotProvider(
        sequence=[
            ClaudeCLIError(
                exit_code=1, stderr="Error: unauthorized (401)", stdout=""
            ),
        ]
    )
    with pytest.raises(ClaudeCLIError, match="exited 1"):
        provider._ask_oneshot("hello")
    assert provider.attempt_count == 1  # no retries
    assert provider.sleep_calls == []  # no backoff


def test_ask_oneshot_retries_empty_stderr_once():
    """Empty stderr is ambiguous → treated as transient → retried."""
    provider = _FlakyOneshotProvider(
        sequence=[
            ClaudeCLIError(exit_code=1, stderr="", stdout=""),
            "recovered",
        ]
    )
    result = provider._ask_oneshot("hello")
    assert result == "recovered"
    assert provider.attempt_count == 2
    assert len(provider.sleep_calls) == 1


def test_retry_delay_grows_exponentially():
    from petri.reasoning.claude_code_provider import _retry_delay_seconds

    delays = [_retry_delay_seconds(attempt) for attempt in (1, 2, 3)]
    # Each successive delay is at least roughly 2x the prior. With
    # bounded jitter (+0..0.5s) this ordering is reliable.
    assert delays[0] < delays[1] < delays[2]
    # Attempt 1 should be in [base, base + jitter] = [1.5, 2.0]
    assert 1.5 <= delays[0] <= 2.0
    # Attempt 2 should be in [3.0, 3.5]
    assert 3.0 <= delays[1] <= 3.5


# ── ClaudeCLIError surfacing ────────────────────────────────────────────


class _RaisingStubProvider(ClaudeCodeProvider):
    """ClaudeCodeProvider whose _ask raises ClaudeCLIError instead of
    returning a string. Used to verify the high-level methods produce
    informative error reports rather than corrupt buffers."""

    def __init__(
        self, *, exit_code: int = 1, stderr: str = "rate limit hit"
    ) -> None:
        self.model = "test-model"
        self.allowed_tools = []
        self._exit_code = exit_code
        self._stderr = stderr

    def _ask(self, prompt, on_progress=None):  # type: ignore[override]
        raise ClaudeCLIError(
            exit_code=self._exit_code, stderr=self._stderr, stdout=""
        )


def test_assess_node_returns_execution_error_with_stderr_on_cli_failure():
    """When the claude subprocess fails, assess_node must surface the
    actual stderr in the AssessmentResult.summary so users can see WHY
    (rate limit, auth, model name, etc.) instead of generic 'execution
    error'."""
    provider = _RaisingStubProvider(
        exit_code=1, stderr="Error: rate limit exceeded (HTTP 429)"
    )
    result = provider.assess_node(
        node_id="test-dish-colony-001-001",
        claim_text="A sample claim",
        context={"phase": "research"},
        agent_role="investigator",
    )
    assert result.verdict == "EXECUTION_ERROR"
    assert "exit 1" in result.summary
    assert "rate limit" in result.summary.lower()


def test_assess_node_returns_execution_error_with_empty_stderr():
    """Empty stderr (which is what the user actually saw — exit 1 with
    no diagnostic) must produce a result that flags the empty case
    rather than a confusing blank summary."""
    provider = _RaisingStubProvider(exit_code=1, stderr="")
    result = provider.assess_node(
        node_id="test-dish-colony-001-001",
        claim_text="A claim",
        context={"phase": "research"},
        agent_role="investigator",
    )
    assert result.verdict == "EXECUTION_ERROR"
    assert "exit 1" in result.summary
    assert "(empty)" in result.summary


def test_assess_claim_substance_falls_back_when_cli_fails():
    """The seed wizard's substance check must not crash on CLI failure;
    fall through to is_substantive=True so the user can still proceed."""
    provider = _RaisingStubProvider(exit_code=1, stderr="auth expired")
    result = provider.assess_claim_substance("a real claim")
    assert result == {
        "is_substantive": True,
        "reason": "",
        "suggested_rewrite": "",
    }


def test_generate_clarifying_questions_returns_empty_when_cli_fails():
    """Clarifying-question generation falls through to an empty list on
    CLI failure so the seed wizard can proceed without the wizard step."""
    provider = _RaisingStubProvider(exit_code=1, stderr="model not found")
    questions = provider.generate_clarifying_questions("a claim")
    assert questions == []


def test_claude_cli_error_message_includes_exit_code_and_stderr():
    """ClaudeCLIError's __str__ must include both the exit code and a
    stderr preview so logs surface the full context."""
    error = ClaudeCLIError(
        exit_code=2,
        stderr="Error: prompt exceeds max context (200000 tokens)",
        stdout="partial",
    )
    text = str(error)
    assert "exited 2" in text
    assert "200000" in text


def test_claude_cli_error_empty_stderr_shows_empty_marker():
    error = ClaudeCLIError(exit_code=1, stderr="", stdout="")
    text = str(error)
    assert "exited 1" in text
    assert "(empty)" in text


# ── allowed_tools / _build_claude_command ───────────────────────────────


class _ToolsStubProvider(ClaudeCodeProvider):
    """ClaudeCodeProvider that skips the claude-CLI presence check.

    Used to exercise the ``_build_claude_command`` argv shape without
    requiring ``claude`` on PATH.
    """

    def __init__(self, *, allowed_tools: list[str] | None = None) -> None:
        self.model = "test-model"
        from petri.config import AGENT_TOOLS

        self.allowed_tools = (
            list(AGENT_TOOLS) if allowed_tools is None
            else list(allowed_tools)
        )


def test_build_command_includes_allowed_tools_when_non_empty():
    """The --allowedTools flag is passed in equals form (--allowedTools=X,Y,Z).

    The equals form is critical: claude CLI declares --allowedTools as
    variadic (``<tools...>``), so the space-separated form
    ``--allowedTools value`` would let the parser consume every
    following positional arg — including our prompt — as additional
    tool names, producing ``Error: Input must be provided either
    through stdin or as a prompt argument when using --print``.
    """
    provider = _ToolsStubProvider(
        allowed_tools=["WebSearch", "WebFetch", "Read"]
    )
    cmd = provider._build_claude_command("hello", streaming=False)
    expected_flag = "--allowedTools=WebSearch,WebFetch,Read"
    assert expected_flag in cmd
    # The flag+value is ONE element, not two — no bare --allowedTools.
    assert "--allowedTools" not in cmd
    # The prompt is always the final positional argument.
    assert cmd[-1] == "hello"
    # The dangerous bypass flag must NEVER be present.
    assert "--allow-dangerously-skip-permissions" not in cmd


def test_build_command_disables_all_tools_when_empty_list():
    """An explicit empty list means 'no tools at all' — passes
    ``--tools=`` (equals form, empty value), which is claude CLI's
    documented way to disable the entire built-in tool set. The equals
    form matters for the same variadic-parser reason as --allowedTools:
    ``--tools ""`` with a separate empty arg would still let the parser
    consume the prompt as a tool name."""
    provider = _ToolsStubProvider(allowed_tools=[])
    cmd = provider._build_claude_command("hello", streaming=False)
    # The equals form is a single element.
    assert "--tools=" in cmd
    # No bare --tools flag (which would be variadic).
    assert "--tools" not in cmd
    # No allowedTools when disabled.
    assert not any(
        element.startswith("--allowedTools") for element in cmd
    )
    assert cmd[-1] == "hello"


def test_build_command_streaming_adds_stream_json_flags():
    """Streaming mode adds the stream-json output format and verbose."""
    provider = _ToolsStubProvider(allowed_tools=["WebSearch"])
    cmd = provider._build_claude_command("hello", streaming=True)
    assert "--output-format" in cmd
    output_index = cmd.index("--output-format")
    assert cmd[output_index + 1] == "stream-json"
    assert "--include-partial-messages" in cmd
    assert "--verbose" in cmd
    # The prompt is still last.
    assert cmd[-1] == "hello"
    # And tools are still passed in equals form.
    assert "--allowedTools=WebSearch" in cmd


def test_build_command_prompt_is_unambiguous_positional():
    """Regression for the 'Input must be provided' bug: the prompt
    must be the LAST element of argv, and NO preceding element can be
    a bare variadic flag that would consume it. This test iterates
    over every combination (streaming on/off, various tool sets)
    and asserts the invariant."""
    for tool_set in (
        ["WebSearch"],
        ["WebSearch", "WebFetch", "Read", "Glob", "Grep"],
        [],  # empty list -> --tools=
    ):
        provider = _ToolsStubProvider(allowed_tools=tool_set)
        for streaming_flag in (True, False):
            cmd = provider._build_claude_command(
                "the prompt text", streaming=streaming_flag
            )
            # 1. Prompt is the last element.
            assert cmd[-1] == "the prompt text"
            # 2. NO bare variadic flag precedes it.
            # The only tool-related elements should be the equals form.
            assert "--allowedTools" not in cmd  # no bare flag
            assert "--tools" not in cmd  # no bare flag
            # 3. Every --allowedTools / --tools reference is an
            #    equals-form single element.
            for element in cmd:
                if element.startswith("--allowedTools"):
                    assert "=" in element, (
                        f"--allowedTools must use equals form, got {element!r}"
                    )
                if element.startswith("--tools"):
                    assert "=" in element, (
                        f"--tools must use equals form, got {element!r}"
                    )


def test_get_agent_tools_default_when_key_missing():
    """A petri.yaml without an agent_tools key falls back to a
    research-focused default that includes WebSearch and WebFetch."""
    from petri.config import get_agent_tools

    tools = get_agent_tools(config={"name": "petri"})
    assert "WebSearch" in tools
    assert "WebFetch" in tools


def test_get_agent_tools_explicit_empty_list():
    """An explicit empty list is honored (means: disable all tools)."""
    from petri.config import get_agent_tools

    tools = get_agent_tools(config={"name": "petri", "agent_tools": []})
    assert tools == []


def test_get_agent_tools_explicit_list():
    """An explicit list is returned verbatim, in order."""
    from petri.config import get_agent_tools

    tools = get_agent_tools(
        config={"name": "petri", "agent_tools": ["WebSearch", "Read"]}
    )
    assert tools == ["WebSearch", "Read"]


def test_get_agent_tools_rejects_non_list():
    """A non-list value in agent_tools surfaces a TypeError early
    rather than producing weird argv at run time."""
    from petri.config import get_agent_tools

    with pytest.raises(TypeError, match="must be a list"):
        get_agent_tools(config={"name": "petri", "agent_tools": "WebSearch"})


def test_default_agent_tools_includes_web_search_and_fetch():
    """Regression: the shipped petri.yaml MUST grant WebSearch and
    WebFetch by default — without them the research agents fabricate
    citation URLs from training data instead of validating live."""
    from petri.config import AGENT_TOOLS

    assert "WebSearch" in AGENT_TOOLS
    assert "WebFetch" in AGENT_TOOLS


def test_build_command_never_passes_dangerous_skip_flag():
    """Regression: petri MUST NEVER pass --allow-dangerously-skip-permissions
    regardless of what's in allowed_tools. Every grant is explicit."""
    for tool_set in (
        None,  # default
        [],
        ["WebSearch"],
        ["WebSearch", "WebFetch", "Bash", "Edit", "Write"],
    ):
        provider = _ToolsStubProvider(allowed_tools=tool_set)
        for streaming_flag in (True, False):
            cmd = provider._build_claude_command(
                "x", streaming=streaming_flag
            )
            assert "--allow-dangerously-skip-permissions" not in cmd
            assert "--dangerously-skip-permissions" not in cmd
