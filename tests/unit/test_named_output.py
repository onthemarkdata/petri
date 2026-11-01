"""Known result envelopes retain strict fields, verdicts, and raw messages."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from pydantic_ai import capture_run_messages
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from petri.agents.config import load_agent_config
from petri.agents.contract import AgentName
from petri.agents.factory import build_agents
from petri.agents.outputs import output_model
from petri.reasoning.decomposition_schemas import WhyExpansion


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["prompted", "tool"])
async def test_known_output_name_is_unwrapped_with_raw_message_preserved(mode):
    config = load_agent_config(Path(__file__).parents[2] / "petri/defaults/petri.yaml")
    wrapped = {"investigatorAssessment": {"verdict": "EVIDENCE_SUFFICIENT", "summary": "Supplied evidence."}}

    def respond(messages, info):
        return ModelResponse(parts=[TextPart(json.dumps(wrapped))] if mode == "prompted" else
                             [ToolCallPart(info.output_tools[0].name, wrapped)])

    agent = build_agents(config, constitution="Preserve attribution.",
                         model=FunctionModel(respond), output_mode=mode)[AgentName.INVESTIGATOR]
    with capture_run_messages() as messages:
        result = await agent.run("Assess the supplied evidence.")
    assert result.output.verdict == "EVIDENCE_SUFFICIENT"
    assert "investigatorAssessment" in str(messages)


@pytest.mark.parametrize("value", [
    {"unknownAssessment": {"verdict": "EVIDENCE_SUFFICIENT"}},
    {"InvestigatorAssessment": {"verdict": "INVENTED"}},
    {"InvestigatorAssessment": {"verdict": "EVIDENCE_SUFFICIENT", "unknown": True}},
    {"InvestigatorAssessment": {"verdict": "EVIDENCE_SUFFICIENT"}, "unknown": True},
    {"InvestigatorAssessment": {"InvestigatorAssessment": {"verdict": "EVIDENCE_SUFFICIENT"}}},
])
def test_unwrapping_rejects_unknown_fields_verdicts_and_nested_envelopes(value):
    config = load_agent_config(Path(__file__).parents[2] / "petri/defaults/petri.yaml")
    schema = output_model(AgentName.INVESTIGATOR, config.agents[AgentName.INVESTIGATOR])
    with pytest.raises(ValidationError):
        schema.model_validate(value)


def test_decomposition_envelope_still_requires_atomic_reason():
    with pytest.raises(ValidationError, match="bedrock_reason"):
        WhyExpansion.model_validate({"WhyExpansion": {"is_atomic": True, "sub_premises": []}})
