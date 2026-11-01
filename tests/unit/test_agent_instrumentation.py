"""Factory agents inherit native instrumentation and preserve parent context."""

import importlib
from pathlib import Path
from unittest.mock import patch

import pytest
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.usage import RequestUsage

from petri.agents.config import load_agent_config
from petri.agents.contract import AgentName
from petri.agents.factory import build_agents, run_agent
from petri.cli._bootstrap import bootstrap_telemetry
from petri.telemetry import init_telemetry, shutdown_telemetry
from petri.telemetry.instrument_agents import instrument_agents

DEFAULTS = Path(__file__).parents[2] / "petri/defaults/petri.yaml"


@pytest.fixture
def tracing(monkeypatch):
    module = importlib.import_module("petri.telemetry.instrument_agents")
    monkeypatch.setattr(module, "_active_provider", None)
    shutdown_telemetry()
    provider = init_telemetry({}, dish_id="fixture")
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    instrument_agents(provider)
    yield provider, exporter
    Agent.instrument_all(False)
    shutdown_telemetry()


@pytest.mark.asyncio
async def test_factory_agents_emit_run_model_and_usage_spans(tracing):
    provider, exporter = tracing
    config = load_agent_config(DEFAULTS)

    def respond(messages, info):
        return ModelResponse(
            parts=[TextPart('{"verdict":"ARGUMENT_WITHSTANDS_CRITIQUE"}')],
            usage=RequestUsage(input_tokens=20, output_tokens=8),
        )

    agents = build_agents(
        config, constitution="A private fixture prompt.", model=FunctionModel(respond)
    )
    with provider.get_tracer("test").start_as_current_span("phase") as parent:
        await run_agent(agents, AgentName.SKEPTIC, "A private claim.", config)
    spans = exporter.get_finished_spans()
    agent = next(s for s in spans if s.attributes.get("gen_ai.operation.name") == "invoke_agent")
    request = next(s for s in spans if s.attributes.get("gen_ai.operation.name") == "chat")
    assert agent.parent.span_id == parent.get_span_context().span_id
    assert request.parent.span_id == agent.context.span_id
    assert agent.attributes["gen_ai.aggregated_usage.input_tokens"] == 20
    assert request.attributes["gen_ai.usage.output_tokens"] == 8
    assert "A private claim" not in str([s.attributes for s in spans])
    assert "A private fixture prompt" not in str([s.attributes for s in spans])


@pytest.mark.asyncio
async def test_new_agents_and_registered_tools_inherit_instrumentation(tracing):
    _, exporter = tracing
    calls = 0

    def respond(messages, info):
        nonlocal calls
        calls += 1
        parts = [ToolCallPart("lookup", {})] if calls == 1 else [TextPart("done")]
        return ModelResponse(parts=parts, usage=RequestUsage(input_tokens=4, output_tokens=2))

    agent = Agent(FunctionModel(respond), name="decomposer_fixture")

    @agent.tool_plain
    def lookup() -> str:
        return "fixture"

    await agent.run("Use the fixture.")
    spans = exporter.get_finished_spans()
    run = next(s for s in spans if s.attributes.get("gen_ai.operation.name") == "invoke_agent")
    tool = next(s for s in spans if s.attributes.get("gen_ai.operation.name") == "execute_tool")
    assert tool.parent.span_id == run.context.span_id
    assert tool.attributes["gen_ai.tool.name"] == "lookup"


def test_bootstrap_instruments_once_and_changes_provider_for_another_dish(tmp_path, monkeypatch):
    module = importlib.import_module("petri.telemetry.instrument_agents")
    monkeypatch.setattr(module, "_active_provider", None)
    first_dir = tmp_path / "first/.petri"
    second_dir = tmp_path / "second/.petri"
    with patch.object(Agent, "instrument_all") as install:
        first = bootstrap_telemetry(first_dir)
        assert bootstrap_telemetry(first_dir) is first
        assert install.call_count == 1
        second = bootstrap_telemetry(second_dir)
        assert second is not first
        assert install.call_count == 2
        assert second.resource.attributes["petri.dish_id"] == "second"
    shutdown_telemetry()


def test_no_logfire_or_external_exporter_is_configured(tracing):
    provider, _ = tracing
    processors = provider._active_span_processor._span_processors
    assert len(processors) == 1
    assert isinstance(processors[0].span_exporter, InMemorySpanExporter)


@pytest.mark.asyncio
async def test_entire_roster_uses_shared_instrumentation(tracing):
    _, exporter = tracing
    config = load_agent_config(DEFAULTS)

    def respond(messages, info):
        import json

        name = next(name for name in AgentName if f"Act as {name.value}." in info.instructions)
        return ModelResponse(
            parts=[
                TextPart(
                    json.dumps(
                        {
                            "verdict": config.agents[name].verdicts_pass[0],
                            "sources_cited": [
                                {"url": "https://example.org/fixture", "hierarchy_level": 1}
                            ],
                        }
                    )
                )
            ],
            usage=RequestUsage(input_tokens=4, output_tokens=2),
        )

    agents = build_agents(
        config, constitution="Use fixture evidence.", model=FunctionModel(respond)
    )
    for name in AgentName:
        await run_agent(agents, name, "Fixture.", config)
    runs = [
        s
        for s in exporter.get_finished_spans()
        if s.attributes.get("gen_ai.operation.name") == "invoke_agent"
    ]
    assert {s.attributes["gen_ai.agent.name"] for s in runs} == set(AgentName)
