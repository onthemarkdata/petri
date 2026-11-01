"""Factory behavior through real pydantic-ai validation and offline model doubles."""

import json
from pathlib import Path

import pytest
from pydantic_ai import capture_run_messages
from pydantic_ai.messages import (
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel

from petri.agents.config import load_agent_config
from petri.agents.contract import AGENT_SPECS, AgentName, BlockingMode, Phase
from petri.agents.factory import AgentRunError, build_agents, run_agent
from petri.agents.outputs import SourceCitation

DEFAULTS = Path(__file__).parents[2] / "petri/defaults"


@pytest.fixture
def config():
    return load_agent_config(DEFAULTS / "petri.yaml")


def payload(verdict="EVIDENCE_SUFFICIENT"):
    return {
        "verdict": verdict,
        "summary": "The source supports the claim.",
        "sources_cited": [{"url": "https://example.org/source", "hierarchy_level": 2}],
    }


def test_roster_names_instructions_and_structure(config):
    agents = build_agents(config, constitution="Preserve evidence.", model=TestModel())
    assert list(agents) == list(AgentName)
    assert len({a.name for a in agents.values()}) == 13
    for name, agent in agents.items():
        assert any("Preserve evidence." in item.instruction for item in agent._instructions)
        assert any(
            config.agents[name].instruction in item.instruction for item in agent._instructions
        )
    assert AGENT_SPECS[AgentName.TRIAGE].blocking_mode == BlockingMode.CONDITIONAL
    assert AGENT_SPECS[AgentName.TRIAGE].redirect_on_block == "DEFER_OPEN"
    assert AGENT_SPECS[AgentName.INVESTIGATOR].phase == Phase.RESEARCH
    assert AGENT_SPECS[AgentName.CELL_LEAD].phase == Phase.LEAD
    assert config.debates[0].rounds == 1.5
    assert config.agents[AgentName.SIMPLIFIER].verdicts_block == ["OVERCOMPLICATED"]
    assert AGENT_SPECS[AgentName.SIMPLIFIER].blocking_mode == BlockingMode.NON_BLOCKING


@pytest.mark.parametrize("broken", ["pair", "vocabulary"])
def test_eager_failure_before_model_resolution(config, broken):
    raw = config.model_dump(mode="json")
    if broken == "pair":
        raw["debates"][0]["pair"][0] = "unknown_debater"
    else:
        raw["agents"]["triage"]["verdicts_pass"] = ["TYPO"]
    calls = []
    with pytest.raises(ValueError, match="unknown_debater|TYPO"):
        build_agents(raw, constitution="Evidence first.", model_resolver=lambda c: calls.append(c))
    assert calls == []


def test_per_agent_model_override(config):
    config.agents[AgentName.TRIAGE].model = "test:triage-specialist"
    calls = []

    def resolver(raw):
        calls.append(raw)
        return TestModel(model_name=raw["harness"].get("model") or "default")

    agents = build_agents(config, constitution="Evidence first.", model_resolver=resolver)
    assert agents[AgentName.TRIAGE].model.model_name == "test:triage-specialist"
    assert agents[AgentName.INVESTIGATOR].model.model_name == "default"
    assert len(calls) == 13


@pytest.mark.asyncio
async def test_valid_tool_output(config):
    def respond(messages, info):
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, payload())])

    agents = build_agents(
        config, constitution="Evidence first.", model=FunctionModel(respond), output_mode="tool"
    )
    result = await run_agent(agents, AgentName.INVESTIGATOR, "Assess this claim.", config)
    assert result.output.verdict == "EVIDENCE_SUFFICIENT"
    assert result.output.agent == AgentName.INVESTIGATOR
    assert isinstance(result.output.sources_cited[0], SourceCitation)


@pytest.mark.asyncio
async def test_prompted_output(config):
    def respond(messages, info):
        return ModelResponse(parts=[TextPart(json.dumps(payload()))])

    agents = build_agents(config, constitution="Evidence first.", model=FunctionModel(respond))
    result = await run_agent(agents, AgentName.INVESTIGATOR, "Assess this claim.", config)
    assert result.output.verdict == "EVIDENCE_SUFFICIENT"


@pytest.mark.asyncio
async def test_invalid_verdict_retries_then_raises(config):
    def respond(messages, info):
        return ModelResponse(
            parts=[ToolCallPart(info.output_tools[0].name, payload("NOT_A_VERDICT"))]
        )

    agents = build_agents(
        config,
        constitution="Evidence first.",
        model=FunctionModel(respond),
        output_mode="tool",
        retries=1,
    )
    with capture_run_messages() as messages, pytest.raises(AgentRunError):
        await run_agent(agents, AgentName.INVESTIGATOR, "Assess.", config)
    assert any(
        (
            isinstance(part, RetryPromptPart)
            or (isinstance(part, ToolReturnPart) and part.outcome == "retried")
        )
        for msg in messages
        for part in msg.parts
    )


@pytest.mark.asyncio
async def test_constitution_is_resent_as_instructions(config):
    seen = []

    def respond(messages, info):
        seen.append(info.instructions)
        return ModelResponse(parts=[TextPart(json.dumps({"verdict": "PIPELINE_COMPLETE"}))])

    agents = build_agents(
        config, constitution="Always preserve sources.", model=FunctionModel(respond)
    )
    first = await run_agent(agents, AgentName.CELL_LEAD, "First run.", config)
    await run_agent(
        agents, AgentName.CELL_LEAD, "Next run.", config, message_history=first.all_messages()
    )
    assert len(seen) == 2
    assert all("Always preserve sources." in instructions for instructions in seen)
    assert all(
        part.part_kind != "system-prompt" for msg in first.all_messages() for part in msg.parts
    )
    protocol = config.debates[0].protocol
    assert protocol.rounds == 1
    assert protocol.final_word == AgentName.SKEPTIC


def test_factory_import_does_not_load_content_or_resolve_models(tmp_path):
    import os
    import subprocess
    import sys

    code = """
import pydantic_ai
import petri
from unittest.mock import patch
with patch('pathlib.Path.read_text', side_effect=AssertionError('file read')), patch('socket.socket.connect', side_effect=AssertionError('network')):
    import petri.agents.factory
"""
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(DEFAULTS.parents[1])},
        check=True,
    )
