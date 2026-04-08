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
    """The --allowedTools flag is passed with a comma-joined list."""
    provider = _ToolsStubProvider(
        allowed_tools=["WebSearch", "WebFetch", "Read"]
    )
    cmd = provider._build_claude_command("hello", streaming=False)
    assert "--allowedTools" in cmd
    flag_index = cmd.index("--allowedTools")
    assert cmd[flag_index + 1] == "WebSearch,WebFetch,Read"
    # The prompt is always the final positional argument.
    assert cmd[-1] == "hello"
    # The dangerous bypass flag must NEVER be present.
    assert "--allow-dangerously-skip-permissions" not in cmd


def test_build_command_disables_all_tools_when_empty_list():
    """An explicit empty list means 'no tools at all' — passes --tools ""
    rather than --allowedTools, because claude CLI's --tools "" is the
    documented way to disable the entire built-in tool set. Omitting
    --allowedTools would silently fall through to the user's settings."""
    provider = _ToolsStubProvider(allowed_tools=[])
    cmd = provider._build_claude_command("hello", streaming=False)
    assert "--allowedTools" not in cmd
    assert "--tools" in cmd
    tools_index = cmd.index("--tools")
    assert cmd[tools_index + 1] == ""
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
    # And tools are still passed.
    assert "--allowedTools" in cmd


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
