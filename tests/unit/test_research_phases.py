"""Run research phases through typed agents, SQLite events, and the graph."""

import json
import pickle
from contextlib import ExitStack, closing
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import ValidationError
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage
from pydantic_ai.exceptions import UnexpectedModelBehavior

from petri.agents.config import load_agent_config
from petri.agents.contract import AgentName
from petri.agents.factory import build_agents
from petri.execution.evidence import (
    RecordedAssessment,
    debate_summary,
    format_critique,
    format_research,
    render_evidence,
)
from petri.execution.models import CellPipelineState
from petri.execution.phases import (
    PhaseServices,
    build_research_dependencies,
    build_socratic_agent,
    run_research,
    run_socratic,
)
from petri.execution.pipeline import PipelineInput, Validated, run_pipeline
from petri.models import validate_event_data
from petri.storage import database
from tests.pipeline_stubs import StubPhases

DEFAULTS = Path(__file__).parents[2] / "petri/defaults"
SOURCE = {
    "url": "https://example.org/evidence",
    "title": "Measured result (2026)",
    "hierarchy_level": 2,
    "finding": "The measurement supports the claim.",
    "supports_or_contradicts": "supports",
    "confidence": "HIGH",
}


@pytest.mark.asyncio
async def test_failed_model_output_is_durable_without_a_verdict(phase_services):
    services, signals = phase_services
    cell_id = database.list_cells(services.db_path)[0]["id"]

    def invalid(messages, info):
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name,
                             {"verdict": "NOT_A_VALID_VERDICT", "summary": "Invalid model output"})])

    services.socratic_agent = build_socratic_agent(
        services.config, constitution="Preserve source attribution.", model=FunctionModel(invalid))
    value = PipelineInput.model_validate(CellPipelineState(cell_id=cell_id).model_dump())
    with pytest.raises(UnexpectedModelBehavior):
        await run_socratic(value, services)
    failures = database.read_events(services.db_path, cell_id=cell_id, event_type="agent_execution_failed")
    assert len(failures) == 1
    detail = failures[0]["data"]["failure_detail"]
    assert "NOT_A_VALID_VERDICT" in detail["partial_output"]
    assert detail["last_stream_event"]["validation_feedback"]
    assert not database.read_events(services.db_path, cell_id=cell_id, event_type="verdict_issued")
    assert database.get_cell(services.db_path, cell_id)["status"] == "NEW"


@pytest.mark.asyncio
async def test_permanent_harness_failure_keeps_backend_diagnostics(phase_services):
    from petri.harness.errors import FailureDetail, PermanentHarnessError

    services, signals = phase_services
    cell_id = database.list_cells(services.db_path)[0]["id"]

    def failed(messages, info):
        raise PermanentHarnessError("Local transport failed", detail=FailureDetail(
            stderr_tail="backend diagnostic", partial_output="unfinished model output",
            last_stream_event={"kind": "incomplete"}))

    services.socratic_agent = build_socratic_agent(services.config,
        constitution="Preserve source attribution.", model=FunctionModel(failed))
    with pytest.raises(PermanentHarnessError):
        await run_socratic(inputs(cell_id), services)
    detail = database.read_events(services.db_path, event_type="agent_execution_failed")[0]["data"]["failure_detail"]
    assert detail["stderr_tail"] == "backend diagnostic"
    assert detail["partial_output"] == "unfinished model output"
    assert detail["last_stream_event"]["backend_event"] == {"kind": "incomplete"}


@pytest.fixture
def phase_services(tmp_path, canonical_colony):
    config = load_agent_config(DEFAULTS / "petri.yaml")
    agents = build_agents(
        config, constitution="Preserve source attribution.", model=TestModel(), output_mode="tool"
    )
    signals = []

    def socratic_response(messages, info):
        prompt = str(messages[-1].parts)
        verdict = (
            "VERIFIED"
            if "SOCRATIC STEP 4" in prompt
            else "EVIDENCE_MAPPED"
            if "SOCRATIC STEP 3" in prompt
            else "ASSUMPTIONS_CHALLENGED"
            if "SOCRATIC STEP 2" in prompt
            else "CLARIFIED"
        )
        return ModelResponse(
            parts=[
                ToolCallPart(
                    info.output_tools[0].name,
                    {
                        "verdict": verdict,
                        "summary": "The claim is precisely defined.",
                        "sources_cited": [SOURCE],
                    },
                )
            ]
        )

    questioner = build_socratic_agent(
        config, constitution="Preserve source attribution.", model=FunctionModel(socratic_response)
    )
    path = tmp_path / "petri.sqlite"
    for cell in canonical_colony["graph"].get_all_cells():
        database.put_cell(
            path,
            {
                "id": cell.id,
                "colony": cell.colony_id,
                "level": cell.level,
                "seq": int(cell.id.rsplit("-", 1)[1]),
                "claim_text": cell.claim_text,
                "status": cell.status.value,
                "created_at": "2026-01-01T00:00:00Z",
            },
        )
    services = PhaseServices(
        path,
        1,
        "2026-11-01",
        config,
        agents,
        questioner,
        dish_id=canonical_colony["dish_id"],
        on_progress=signals.append,
    )
    with ExitStack() as stack:
        for name, agent in agents.items():
            stack.enter_context(
                agent.override(
                    model=TestModel(
                        custom_output_args={
                            "agent": name.value,
                            "verdict": config.agents[name].verdicts_pass[0],
                            "summary": f"{name.value} reviewed the evidence.",
                            "sources_cited": [SOURCE],
                        }
                    )
                )
            )
        yield services, signals


def deps(services):
    stubs = StubPhases()
    return build_research_dependencies(
        services,
        convergence=stubs.convergence,
        red_team=stubs.red_team,
        evaluate=stubs.evaluate,
        sync_check=stubs.sync_check,
    )


def inputs(cell_id):
    return PipelineInput.model_validate(CellPipelineState(cell_id=cell_id).model_dump())


@pytest.mark.asyncio
async def test_research_handoff_contains_current_investigator_findings(phase_services):
    services, signals = phase_services
    prompts = {}

    def respond(messages, info):
        name = next(name for name in AgentName if f"Act as {name.value}." in info.instructions)
        prompts[name] = str(messages)
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {
            "verdict": services.config.agents[name].verdicts_pass[0],
            "summary": "CURRENT_INVESTIGATOR_FINDING" if name == AgentName.INVESTIGATOR else name.value,
            "arguments": "FULL_ARGUMENT_HANDOFF",
            "evidence": "FULL_EVIDENCE_HANDOFF",
        })])

    services.agents = build_agents(services.config, constitution="Preserve source attribution.",
                                  model=FunctionModel(respond), output_mode="tool")
    cell_id = database.list_cells(services.db_path)[0]["id"]
    await run_research(inputs(cell_id), services)
    assert "CURRENT_INVESTIGATOR_FINDING" in prompts[AgentName.FRESHNESS_CHECKER]
    assert "CURRENT_INVESTIGATOR_FINDING" in prompts[AgentName.DEPENDENCY_AUDITOR]
    assert "FULL_ARGUMENT_HANDOFF" in prompts[AgentName.FRESHNESS_CHECKER]
    assert "FULL_EVIDENCE_HANDOFF" in prompts[AgentName.DEPENDENCY_AUDITOR]


@pytest.mark.asyncio
async def test_socratic_steps_receive_previous_analysis(phase_services):
    services, signals = phase_services
    prompts = []

    def respond(messages, info):
        prompts.append(str(messages))
        verdict = ["CLARIFIED", "ASSUMPTIONS_CHALLENGED", "EVIDENCE_MAPPED", "VERIFIED"][len(prompts) - 1]
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {
            "verdict": verdict, "summary": f"PREVIOUS_STEP_{len(prompts)}",
        })])

    services.socratic_agent = build_socratic_agent(services.config,
        constitution="Preserve source attribution.", model=FunctionModel(respond))
    await run_socratic(inputs(database.list_cells(services.db_path)[0]["id"]), services)
    assert "PREVIOUS_STEP_1" in prompts[1]
    assert "PREVIOUS_STEP_1" in prompts[2] and "PREVIOUS_STEP_2" in prompts[2]
    assert "PREVIOUS_STEP_3" in prompts[3]


@pytest.mark.asyncio
async def test_diamond_graph_runs_real_phases_and_replays_idempotently(
    phase_services, canonical_colony, monkeypatch
):
    services, signals = phase_services
    for cell in canonical_colony["graph"].get_all_cells():
        current = replace(services, usage=RunUsage())
        state = CellPipelineState(cell_id=cell.id)
        assert isinstance(await run_pipeline(state, deps(current)), Validated)
        assert len(pickle.dumps(state)) < 64 * 1024
        assert set(state.model_dump()) == {
            "cell_id",
            "iteration",
            "weakest_link",
            "focused_directive",
            "cycle_start_iteration",
            "max_iterations",
        }
    before = database.read_events(services.db_path)
    assert {event["type"] for event in before} >= {
        "verdict_issued",
        "source_reviewed",
        "debate_mediated",
    }
    for event in before:
        assert set(event) == {"id", "cell_id", "timestamp", "type", "agent", "iteration", "data"}
        validate_event_data(event["type"], event["data"])
        if event["type"] == "debate_mediated":
            detail = json.loads(event["data"]["exchange_summary"])
            assert {turn["speaker"] for turn in detail["exchanges"]} == {
                event["data"]["from_agent"],
                event["data"]["to_agent"],
            }
            assert len(detail["exchanges"]) >= 2
            assert detail["usage"]["requests"] == len(detail["exchanges"])
    async def unexpected_request(*args, **kwargs):
        raise AssertionError("A completed phase must reuse its committed response")

    for agent in [*services.agents.values(), services.socratic_agent]:
        monkeypatch.setattr(agent, "run", unexpected_request)
    for cell in canonical_colony["graph"].get_all_cells():
        replay = replace(services, usage=RunUsage())
        await run_pipeline(CellPipelineState(cell_id=cell.id), deps(replay))
        assert replay.usage.requests == 0
    assert database.read_events(services.db_path) == before
    assert {signal.kind for signal in signals} == {
        "phase_start",
        "phase_end",
        "agent_start",
        "agent_end",
    }
    assert {signal.phase for signal in signals} == {"socratic", "research", "critique"}


@pytest.mark.asyncio
async def test_evidence_regenerates_from_committed_events(
    phase_services, canonical_colony
):

    services, _ = phase_services
    cell_id = canonical_colony["cell1"].id
    await run_pipeline(CellPipelineState(cell_id=cell_id), deps(services))
    events = database.read_events(services.db_path, cell_id=cell_id)
    research = []
    critique = []
    from petri.agents.contract import AGENT_SPECS, Phase

    for event in events:
        if event["type"] == "verdict_issued" and event["agent"] in AgentName._value2member_map_:
            value = RecordedAssessment.model_validate({"agent": event["agent"], **event["data"]})
            (
                research
                if AGENT_SPECS[AgentName(value.agent)].phase == Phase.RESEARCH
                else critique
            ).append(value)
    debates = [
        debate_summary(event["data"]) for event in events if event["type"] == "debate_mediated"
    ]
    evidence = render_evidence(events)
    assert evidence.count("### Socratic Analysis") == 1
    assert format_research(research, 0) in evidence
    assert format_critique(critique, debates, 0) in evidence
    assert render_evidence(json.loads(json.dumps(events))) == evidence
    assert not list(services.db_path.parent.rglob("evidence.md"))


@pytest.mark.asyncio
async def test_partial_source_write_recovers_from_committed_verdict(
    phase_services, canonical_colony, monkeypatch
):
    from petri.execution import phases

    services, _ = phase_services
    value = inputs(canonical_colony["cell1"].id)
    original = phases.append_event

    def interrupt(target, cell_id, event_type, *args, **kwargs):
        if event_type == "source_reviewed":
            raise RuntimeError("crash after verdict commit")
        return original(target, cell_id, event_type, *args, **kwargs)

    monkeypatch.setattr(phases, "append_event", interrupt)
    with pytest.raises(RuntimeError, match="crash after verdict"):
        await run_research(value, services)
    assert services.usage.requests == 1
    monkeypatch.setattr(phases, "append_event", original)
    await run_research(value, services)
    events = database.read_events(services.db_path, cell_id=value.cell_id)
    investigator = [event for event in events if event["agent"] == "investigator"]
    assert [event["type"] for event in investigator] == ["verdict_issued", "source_reviewed"]
    # The first agent's committed response was reused, not requested again.
    assert services.usage.requests == 3


def test_unknown_pairing_fails_before_graph_construction(phase_services):
    services, _ = phase_services
    raw = services.config.model_dump(mode="json")
    raw["debates"][0]["pair"][0] = "unknown_agent"
    with pytest.raises(ValidationError):
        services.config.model_validate(raw)
    with pytest.raises(ValueError, match="missing agents"):
        replace(
            services,
            agents={
                name: agent for name, agent in services.agents.items() if name != AgentName.SKEPTIC
            },
        )


@pytest.mark.asyncio
async def test_each_phase_result_and_state_remain_small(phase_services, canonical_colony):
    services, _ = phase_services
    value = inputs(canonical_colony["cell1"].id)
    for function in [run_socratic, run_research]:
        result = await function(value, services)
        assert len(pickle.dumps(result)) < 64 * 1024
        assert len(pickle.dumps(value)) < 64 * 1024
    with closing(database.connect(services.db_path)) as conn:
        assert conn.execute("SELECT count(*) FROM events").fetchone()[0] > 0


@pytest.mark.asyncio
async def test_large_evidence_stays_out_of_each_graph_checkpoint(phase_services, canonical_colony):
    services, _ = phase_services
    state = CellPipelineState(cell_id=canonical_colony["cell1"].id)
    checkpoints = []

    def record_checkpoint(signal):
        if signal.kind == "phase_end":
            checkpoints.append(pickle.dumps(state))

    services.on_progress = record_checkpoint
    with services.agents[AgentName.INVESTIGATOR].override(
        model=TestModel(
            custom_output_args={
                "agent": "investigator",
                "verdict": "EVIDENCE_SUFFICIENT",
                "summary": "z" * 70_000,
                "sources_cited": [SOURCE],
            }
        )
    ):
        await run_pipeline(state, deps(services))
    assert len(checkpoints) == 3
    assert all(len(checkpoint) < 64 * 1024 for checkpoint in checkpoints)
    assert (
        len(render_evidence(database.read_events(services.db_path, cell_id=state.cell_id)))
        > 64 * 1024
    )


@pytest.mark.asyncio
async def test_shared_debate_budget_parks_and_preserves_partial_turns(
    phase_services, canonical_colony
):
    from petri.execution.pipeline import NeedsHuman

    services, _ = phase_services
    services.config.limits.debate.request_limit = 1
    state = CellPipelineState(cell_id=canonical_colony["cell1"].id)
    result = await run_pipeline(state, deps(services))
    assert isinstance(result, NeedsHuman)
    events = database.read_events(services.db_path, cell_id=state.cell_id)
    debates = [event for event in events if event["type"] == "debate_mediated"]
    assert len(debates) == 1
    detail = json.loads(debates[0]["data"]["exchange_summary"])
    assert detail["status"] == "limit_exceeded"
    assert len(detail["exchanges"]) == detail["usage"]["requests"] == 1
    replay = replace(services, usage=RunUsage())
    assert isinstance(
        await run_pipeline(CellPipelineState(cell_id=state.cell_id), deps(replay)), NeedsHuman
    )
    assert replay.usage.requests == 0
    assert database.read_events(services.db_path, cell_id=state.cell_id) == events


@pytest.mark.asyncio
async def test_actual_phase_spans_attribute_usage_once_per_call(phase_services):
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from pydantic_ai.usage import RequestUsage
    from petri.telemetry import init_telemetry, shutdown_telemetry
    from petri.telemetry.config import ModelPrice
    from petri.telemetry.span_store import SQLiteSpanExporter
    from petri.execution.decisions import run_convergence

    services, _ = phase_services
    cell_id = database.list_cells(services.db_path)[0]["id"]
    provider = init_telemetry({}, dish_id=services.dish_id)
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    provider.add_span_processor(SimpleSpanProcessor(SQLiteSpanExporter(services.db_path)))
    services.config = services.config.model_copy(update={"prices": {
        services.config.model: ModelPrice(input_per_million=2, output_per_million=5)}})
    def measured(messages, info):
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name,
            {"verdict": "VERIFIED", "summary": "A precise claim", "sources_cited": [SOURCE]})],
            usage=RequestUsage(input_tokens=10, output_tokens=4))
    value = inputs(cell_id)
    try:
        with services.socratic_agent.override(model=FunctionModel(measured)):
            await run_socratic(value, services)
            await run_socratic(value, services)
        await run_convergence(value, services)
        provider.force_flush()
        phases = [s for s in exporter.get_finished_spans() if s.name == "petri.phase.socratic"]
        measured_spans = [s for s in phases if "petri.tokens_input" in s.attributes]
        assert len(phases) == 8 and len(measured_spans) == 4
        assert all(s.attributes["petri.cell_id"] == cell_id for s in phases)
        assert all(s.attributes["petri.agent_role"].startswith("socratic_") for s in phases)
        assert all(s.attributes["petri.verdict"] == "VERIFIED" for s in phases)
        assert all(s.attributes["petri.tokens_input"] == 10 for s in measured_spans)
        assert sum(s.attributes["petri.tokens_output"] for s in measured_spans) == 16
        assert sum(s.attributes["petri.cost_usd"] for s in measured_spans) == pytest.approx(0.00016)
        convergence = next(s for s in exporter.get_finished_spans() if s.name == "petri.phase.convergence")
        assert convergence.attributes["petri.iteration"] == value.iteration
        assert convergence.attributes["petri.outcome"]
        with closing(database.connect(services.db_path)) as conn:
            rows = conn.execute("SELECT cell_id,SUM(tokens_input),SUM(tokens_output) FROM usage GROUP BY cell_id").fetchall()
        assert [tuple(row) for row in rows] == [(cell_id, 40, 16)]
    finally:
        shutdown_telemetry()
