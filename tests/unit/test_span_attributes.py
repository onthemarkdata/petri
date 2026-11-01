"""Domain spans use explicit identity, real model usage, and optional configured prices."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.usage import RequestUsage, RunUsage
from typer.testing import CliRunner

from petri.agents.config import load_agent_config
from petri.agents.contract import AgentName
from petri.agents.factory import build_agents, run_agent
from petri.cli import app
from petri.cli._bootstrap import bootstrap_telemetry
from petri.execution.models import CellPipelineState
from petri.execution.pipeline import (
    ConvergenceDecision,
    EvaluationVerdict,
    PhaseResult,
    PipelineDeps,
    RedTeamResult,
    run_pipeline,
)
from petri.models import build_cell_key
from petri.telemetry import init_telemetry, shutdown_telemetry
from petri.telemetry.attributes import (
    CellSpanContext,
    agent_attributes,
    cell_attributes,
    phase_span,
    root_command_span,
    set_verdict_attributes,
    verdict_attributes,
)
from petri.telemetry.config import ModelPrice


@pytest.fixture(autouse=True)
def cleanup_telemetry():
    shutdown_telemetry()
    yield
    Agent.instrument_all(False)
    shutdown_telemetry()


def attach(provider):
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return exporter


def test_attribute_maps_are_pure_and_use_the_composite_key():
    context = CellSpanContext("dish-with-hyphens", "dish-with-hyphens-colony-name", 2, 3)
    with patch("builtins.open", side_effect=AssertionError("file read")):
        first = cell_attributes(context)
        second = cell_attributes(context)
        assert first == second
        first["petri.cell_id"] = "changed"
        assert second["petri.cell_id"] == build_cell_key("dish-with-hyphens", "colony-name", 2, 3)
        assert agent_attributes(AgentName.SKEPTIC) == {
            "petri.agent_role": "skeptic",
            "petri.agent_blocking": True,
        }


def test_usage_and_prices_are_explicit():
    prices = {"fixture": ModelPrice(input_per_million=2, output_per_million=5)}
    usage = RunUsage(input_tokens=100, output_tokens=20, requests=1)
    attrs = verdict_attributes("PASS", usage=usage, model="fixture", prices=prices)
    assert attrs["petri.tokens_input"] == 100
    assert attrs["petri.tokens_output"] == 20
    assert attrs["petri.cost_usd"] == pytest.approx(0.0003)
    assert "petri.cost_usd" not in verdict_attributes("PASS", usage=usage)
    assert "petri.tokens_input" not in verdict_attributes("PASS", prices=prices, model="fixture")
    unmeasured = RunUsage(requests=1, unmeasured_requests=1)
    assert "petri.tokens_input" not in verdict_attributes("PASS", usage=unmeasured)
    partial = RunUsage(input_tokens=100, output_tokens=20, requests=2, unmeasured_requests=1)
    assert "petri.cost_usd" not in verdict_attributes(
        "PASS", usage=partial, model="fixture", prices=prices
    )


@pytest.mark.asyncio
async def test_real_typed_pipeline_emits_phase_identity_verdict_and_usage():
    context = CellSpanContext("fixture", "fixture-colony", 1, 1)
    exporter = attach(init_telemetry({}, dish_id=context.dish_id))
    config = load_agent_config(Path(__file__).parents[2] / "petri/defaults/petri.yaml")

    def respond(messages, info):
        name = next(n for n in AgentName if f"Act as {n.value}." in info.instructions)
        payload = {
            "verdict": config.agents[name].verdicts_pass[0],
            "sources_cited": [{"url": "https://example.org/fixture", "hierarchy_level": 1}],
        }
        return ModelResponse(
            parts=[TextPart(json.dumps(payload))],
            usage=RequestUsage(input_tokens=9, output_tokens=4),
        )

    agents = build_agents(
        config, constitution="Fixture evidence only.", model=FunctionModel(respond)
    )

    def specialist(phase, agent, output):
        async def call(value):
            with phase_span(phase, context, iteration=value.iteration, agent=agent) as span:
                result = await run_agent(agents, agent, "Fixture.", config)
                set_verdict_attributes(span, result.output.verdict, usage=result.usage)
                return output

        return call

    async def convergence(value):
        with phase_span("convergence", context, iteration=value.iteration) as span:
            set_verdict_attributes(span, "CONVERGED", outcome="converged")
        return ConvergenceDecision(outcome="converged")

    async def sync_check(value):
        return None

    deps = PipelineDeps(
        specialist("socratic", AgentName.TRIAGE, PhaseResult()),
        specialist("research", AgentName.INVESTIGATOR, PhaseResult()),
        specialist("critique", AgentName.SKEPTIC, PhaseResult()),
        convergence,
        specialist("red_team", AgentName.RED_TEAM_LEAD, RedTeamResult()),
        specialist("evaluate", AgentName.EVIDENCE_EVALUATOR, EvaluationVerdict.VALIDATED),
        sync_check,
    )
    result = await run_pipeline(CellPipelineState(cell_id=context.cell_id), deps)
    assert result.status == "validated"
    spans = exporter.get_finished_spans()
    phases = [s for s in spans if "petri.phase" in s.attributes]
    assert {s.attributes["petri.phase"] for s in phases} == {
        "socratic",
        "research",
        "critique",
        "convergence",
        "red_team",
        "evaluate",
    }
    for span in phases:
        assert span.attributes["petri.cell_id"] == context.cell_id
        assert span.attributes["petri.verdict"]
        if span.attributes["petri.phase"] == "convergence":
            assert span.attributes["petri.iteration"] == 0
            assert span.attributes["petri.outcome"] == "converged"
        else:
            assert span.attributes["petri.agent_role"] in {role.value for role in AgentName}
            assert span.attributes["petri.tokens_input"] == 9
            assert span.attributes["petri.tokens_output"] == 4


def test_grow_dry_run_has_root_identity_without_inferred_usage(seeded_petri_dir):
    exporter = attach(bootstrap_telemetry(seeded_petri_dir["petri_dir"]))
    result = CliRunner().invoke(app, ["grow", "--dry-run"])
    assert result.exit_code == 0, result.output
    root = next(s for s in exporter.get_finished_spans() if s.name == "petri.grow")
    assert root.attributes["petri.command"] == "grow"
    assert root.attributes["petri.dish_id"] == seeded_petri_dir["dish_id"]
    assert "petri.tokens_input" not in root.attributes


@pytest.mark.parametrize("command", ["scan", "feed"])
def test_scan_and_feed_root_spans(command, petri_env_with_colony, monkeypatch):
    env = petri_env_with_colony
    monkeypatch.chdir(env["tmp_path"])
    exporter = attach(bootstrap_telemetry(env["petri_dir"]))
    args = [command]
    if command == "feed":
        source = env["tmp_path"] / "evidence.txt"
        source.write_text("A supplied fixture document.")
        args.append(str(source))
    result = CliRunner().invoke(app, args)
    assert result.exception is None or isinstance(result.exception, SystemExit), result.exception
    root = next(s for s in exporter.get_finished_spans() if s.name == f"petri.{command}")
    assert root.attributes["petri.command"] == command
    assert root.attributes["petri.dish_id"] == "test-dish"


def test_redecomposition_can_use_same_root_helper():
    exporter = attach(init_telemetry({}))
    with root_command_span("re-decompose", "fixture"):
        pass
    span = exporter.get_finished_spans()[0]
    assert span.name == "petri.re-decompose"
    assert span.attributes["petri.command"] == "re-decompose"


def test_strict_agent_config_accepts_explicit_model_prices():
    config = load_agent_config(Path(__file__).parents[2] / "petri/defaults/petri.yaml")
    raw = config.model_dump(mode="json")
    raw["prices"] = {"local:fixture": {"input_per_million": 0, "output_per_million": 0}}
    validated = type(config).model_validate(raw)
    attrs = verdict_attributes(
        "PASS",
        usage=RunUsage(input_tokens=10, output_tokens=3),
        model="local:fixture",
        prices=validated.prices,
    )
    assert attrs["petri.cost_usd"] == 0
