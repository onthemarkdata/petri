"""Evidence policy retries through offline pydantic-ai model responses."""

from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from petri.agents.config import load_agent_config
from petri.agents.contract import AgentName
from petri.agents.factory import AgentRunError, build_agents, run_agent
from petri.agents.source_policy import check_terminal_sources

DEFAULTS = Path(__file__).parents[2] / "petri/defaults"


@pytest.mark.parametrize(
    "sources,threshold,strongest,passes",
    [
        ([], 4, None, False),
        ([{}], 4, None, False),
        ([{"hierarchy_level": 6}, {"hierarchy_level": 2}], 4, 2, True),
        ([{"hierarchy_level": 5}, {"hierarchy_level": 6}], 4, 5, False),
        ([{"hierarchy_level": 3}], 2, 3, False),
        ([{"hierarchy_level": 3}], 3, 3, True),
        ([{"hierarchy_level": 0}, {"hierarchy_level": True}], 4, None, False),
    ],
)
def test_policy_is_pure(sources, threshold, strongest, passes):
    with patch("builtins.open", side_effect=AssertionError("file access")):
        result = check_terminal_sources(sources, threshold)
    assert result.strongest_level == strongest
    assert result.passes is passes


async def evaluate(payloads, *, threshold=4, retries=1):
    config = load_agent_config(DEFAULTS / "petri.yaml")
    config.source_hierarchy.minimum_terminal_level = threshold
    for level, content in config.source_hierarchy.levels.items():
        content.terminal_eligible = level <= threshold
    calls = []

    def respond(messages, info):
        calls.append(list(messages))
        payload = payloads[min(len(calls) - 1, len(payloads) - 1)]
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, payload)])

    agents = build_agents(
        config,
        constitution="Use verifiable evidence.",
        model=FunctionModel(respond),
        output_mode="tool",
        retries=retries,
    )
    result = await run_agent(agents, AgentName.EVIDENCE_EVALUATOR, "Evaluate.", config)
    return result, calls


def terminal(level=2, url="https://example.org/source"):
    return {
        "verdict": "EVIDENCE_CONFIRMS",
        "sources_cited": [{"url": url, "hierarchy_level": level}],
    }


@pytest.mark.asyncio
async def test_weak_sources_reflect_then_accept_strong_source():
    result, calls = await evaluate([terminal(5), terminal(2)])
    assert len(calls) == 2
    assert "Source hierarchy policy" in str(calls[1])
    assert result.output.verdict == "EVIDENCE_CONFIRMS"


@pytest.mark.asyncio
async def test_level_two_passes_first_try():
    result, calls = await evaluate([terminal(2)])
    assert len(calls) == 1
    assert result.output.sources_cited[0].hierarchy_level == 2


@pytest.mark.asyncio
async def test_inconclusive_needs_no_sources():
    result, calls = await evaluate([{"verdict": "EVIDENCE_INCONCLUSIVE", "sources_cited": []}])
    assert len(calls) == 1
    assert result.output.verdict == "EVIDENCE_INCONCLUSIVE"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source", [{"hierarchy_level": 2}, {"url": "not a URL", "hierarchy_level": 2}]
)
async def test_missing_or_invalid_url_causes_reflection(source):
    result, calls = await evaluate(
        [{"verdict": "EVIDENCE_CONFIRMS", "sources_cited": [source]}, terminal()]
    )
    assert len(calls) == 2
    assert "url" in str(calls[1])
    assert result.output.sources_cited[0].url == "https://example.org/source"


@pytest.mark.asyncio
async def test_exhaustion_is_typed_execution_failure():
    with pytest.raises(AgentRunError) as error:
        await evaluate([terminal(6)])
    assert error.value.agent == AgentName.EVIDENCE_EVALUATOR
    assert error.value.verdict == "EXECUTION_ERROR"
    assert "retries" in error.value.detail


@pytest.mark.asyncio
async def test_threshold_comes_from_configuration():
    result, _ = await evaluate([terminal(3)], threshold=3)
    assert result.output.verdict == "EVIDENCE_CONFIRMS"
    with pytest.raises(AgentRunError):
        await evaluate([terminal(3)], threshold=2)
