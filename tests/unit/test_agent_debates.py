"""Real model hand-off semantics using offline FunctionModel calls and SQLite."""

import json
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.usage import RequestUsage

from petri.agents.config import load_agent_config
from petri.agents.contract import AgentName
from petri.agents.debates import DebateStatus, log_debate, run_debate
from petri.agents.factory import build_agents
from petri.storage.database import SQLiteEventLog, connect
from petri.storage.event_log import load_events

DEFAULTS = Path(__file__).parents[2] / "petri/defaults/petri.yaml"


def setup_debate():
    config = load_agent_config(DEFAULTS)
    calls = []

    def respond(messages, info):
        name = next(n for n in AgentName if f"Act as {n.value}." in info.instructions)
        calls.append((name, list(messages)))
        return ModelResponse(
            parts=[
                ToolCallPart(
                    info.output_tools[0].name,
                    {
                        "verdict": config.agents[name].verdicts_pass[0],
                        "summary": f"{name.value} argument {len(calls)}",
                    },
                )
            ],
            usage=RequestUsage(input_tokens=5, output_tokens=5),
        )

    agents = build_agents(
        config,
        constitution="Use only supplied evidence.",
        model=FunctionModel(respond),
        output_mode="tool",
    )
    return config, agents, calls


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "index,order",
    [
        (0, [AgentName.SKEPTIC, AgentName.CHAMPION, AgentName.SKEPTIC]),
        (1, [AgentName.SKEPTIC, AgentName.PRAGMATIST]),
    ],
)
async def test_real_handoff_and_shared_usage(index, order):
    config, agents, calls = setup_debate()
    protocol = config.debates[index].protocol
    result = await run_debate(protocol, agents, "A fixture claim.", config.limits.debate)
    assert result.status == DebateStatus.COMPLETE
    assert [item.speaker for item in result.exchanges] == order
    assert result.usage.requests == len(order)
    assert result.usage.input_tokens == result.usage.output_tokens == 5 * len(order)
    assert "skeptic argument 1" in str(calls[1][1])
    assert len(calls[1][1]) > len(calls[0][1])
    if index == 0:
        assert protocol.rounds == 2
        assert protocol.final_word == AgentName.SKEPTIC
        assert "champion argument 2" in str(calls[2][1])


@pytest.mark.asyncio
async def test_debate_receives_research_and_per_speaker_tool_context():
    config, agents, calls = setup_debate()
    contexts = []

    def context(name):
        contexts.append(name)
        return {"speaker": name.value, "cell": "fixture-cell"}

    result = await run_debate(config.debates[0].protocol, agents, "A fixture claim.",
                             config.limits.debate, prior_evidence="VERIFIED_RESEARCH_FINDING",
                             claim_context="Unverified ancestor context: CONTEXT_IS_NOT_EVIDENCE",
                             deps_factory=context)
    assert result.status == DebateStatus.COMPLETE
    assert "VERIFIED_RESEARCH_FINDING" in str(calls[0][1])
    assert "VERIFIED_RESEARCH_FINDING" in str(calls[1][1])
    assert all("CONTEXT_IS_NOT_EVIDENCE" in str(call[1]) for call in calls)
    assert contexts == [AgentName.SKEPTIC, AgentName.CHAMPION, AgentName.SKEPTIC]


@pytest.mark.asyncio
async def test_limit_returns_partial_exchanges():
    config, agents, _ = setup_debate()
    config.limits.debate.total_tokens_limit = 15
    result = await run_debate(config.debates[0].protocol, agents, "A claim.", config.limits.debate)
    assert result.status == DebateStatus.LIMIT_EXCEEDED
    assert len(result.exchanges) == 1
    assert result.usage.requests == 2
    assert result.message_history
    assert "token" in result.detail.lower()


@pytest.mark.asyncio
async def test_failed_debate_output_and_feedback_are_durable(tmp_path):
    config, agents, _ = setup_debate()

    def invalid(messages, info):
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name,
                             {"verdict": "UNKNOWN_VERDICT", "summary": "Unvalidated output."})])

    with agents[AgentName.SKEPTIC].override(model=FunctionModel(invalid)):
        result = await run_debate(config.debates[0].protocol, agents, "A claim.", config.limits.debate)
    assert result.status == DebateStatus.EXECUTION_ERROR and not result.exchanges
    event = log_debate(SQLiteEventLog(tmp_path / "petri.sqlite"), "test-colony-0-1", 0, result)
    payload = json.loads(event.data["exchange_summary"])
    assert "UNKNOWN_VERDICT" in payload["failure_detail"]["partial_output"]
    assert payload["failure_detail"]["last_stream_event"]["validation_feedback"]
    assert not payload["exchanges"]


@pytest.mark.asyncio
async def test_history_survives_next_iteration_and_event_append(tmp_path):
    config, agents, calls = setup_debate()
    protocol = config.debates[1].protocol
    result = await run_debate(protocol, agents, "First iteration.", config.limits.debate)
    second = await run_debate(
        protocol,
        agents,
        "Second iteration.",
        config.limits.debate,
        message_history=result.message_history,
    )
    assert "skeptic argument 1" in str(calls[2][1])
    assert len(second.message_history) > len(result.message_history)
    target = SQLiteEventLog(tmp_path / "petri.sqlite")
    event = log_debate(target, "test-colony-0-1", 1, result)
    assert set(event.model_dump()) == {
        "id",
        "cell_id",
        "timestamp",
        "type",
        "agent",
        "iteration",
        "data",
    }
    log_debate(target, "test-colony-0-1", 2, second)
    stored = load_events(target)
    assert len(stored) == 2
    assert stored[0]["type"] == "debate_mediated"
    summary = json.loads(stored[0]["data"]["exchange_summary"])
    assert summary["status"] == "complete"
    assert len(summary["exchanges"]) == 2
    with connect(target.db_path) as conn:
        assert conn.execute("SELECT count(*) FROM events").fetchone()[0] == 2


@pytest.mark.asyncio
async def test_pairing_and_round_identify_distinct_replayable_events(tmp_path):
    config, agents, _ = setup_debate()
    target = SQLiteEventLog(tmp_path / "petri.sqlite")
    first = await run_debate(config.debates[0].protocol, agents, "Claim.", config.limits.debate)
    second = await run_debate(config.debates[1].protocol, agents, "Claim.", config.limits.debate)
    event = log_debate(target, "test-colony-0-1", 1, first, round_number=1)
    replay = log_debate(target, "test-colony-0-1", 1, first, round_number=1)
    other_pair = log_debate(target, "test-colony-0-1", 1, second, round_number=1)
    other_round = log_debate(target, "test-colony-0-1", 1, first, round_number=2)
    assert replay == event
    assert len({event.id, other_pair.id, other_round.id}) == 3
    assert len(load_events(target)) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("error_name", ["TransientHarnessError", "AuthExpiredError"])
async def test_harness_failure_preserves_completed_turns(error_name):
    from petri.harness import errors

    config, agents, _ = setup_debate()

    def fail(messages, info):
        raise getattr(errors, error_name)("Local backend unavailable.")

    failed_agents = build_agents(
        config,
        constitution="Use only supplied evidence.",
        model=FunctionModel(fail),
        output_mode="tool",
    )
    agents[AgentName.CHAMPION] = failed_agents[AgentName.CHAMPION]
    result = await run_debate(config.debates[0].protocol, agents, "Claim.", config.limits.debate)
    assert result.status == DebateStatus.EXECUTION_ERROR
    assert len(result.exchanges) == 1
    assert result.exchanges[0].speaker == AgentName.SKEPTIC
    assert "skeptic argument 1" in str(result.message_history)
    assert result.detail == "Local backend unavailable."
