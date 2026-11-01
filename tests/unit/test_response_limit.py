"""Response caps reach native HTTP requests without accepting truncated output."""

import json
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError
from pydantic_ai import capture_run_messages
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from petri.agents.config import AgentConfig, load_agent_config
from petri.agents.contract import AgentName
from petri.agents.factory import build_agents
from petri.execution.phases import build_socratic_agent
from petri.harness.diagnostics import model_failure_detail


@pytest.mark.asyncio
@pytest.mark.parametrize("socratic", [False, True])
@pytest.mark.parametrize("truncated", [False, True])
async def test_actual_provider_request_has_response_cap(socratic, truncated):
    raw = load_agent_config(Path(__file__).parents[2] / "petri/defaults/petri.yaml").model_dump(mode="json")
    raw["harness"] = {"backend": "provider", "model": "openai-chat:local-test"}
    raw["model_settings"] = {"max_tokens": 123}
    config = AgentConfig.model_validate(raw)
    requests = []

    def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        message = {"role": "assistant", "content": '{"verdict":'} if truncated else {
            "role": "assistant", "tool_calls": [{"id": "call-output", "type": "function", "function": {
                "name": payload["tools"][0]["function"]["name"],
                "arguments": json.dumps({"verdict": "CLARIFIED" if socratic else "EVIDENCE_SUFFICIENT"}),
            }}],
        }
        return httpx.Response(200, json={"id": "local-response", "object": "chat.completion",
            "created": 0, "model": "local-test", "choices": [{"index": 0, "message": message,
            "finish_reason": "length" if truncated else "tool_calls"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 123, "total_tokens": 133}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        model = OpenAIChatModel("local-test", provider=OpenAIProvider(
            base_url="http://local.invalid/v1", api_key="local-test", http_client=client))
        agent = build_socratic_agent(config, constitution="Preserve sources.", model=model) if socratic else (
            build_agents(config, constitution="Preserve sources.", model=model)[AgentName.INVESTIGATOR])
        with capture_run_messages() as messages:
            if truncated:
                with pytest.raises(UnexpectedModelBehavior) as failed:
                    await agent.run("Assess the supplied claim.")
                assert model_failure_detail(failed.value, messages).partial_output == '{"verdict":'
            else:
                assert (await agent.run("Assess the supplied claim.")).output.verdict
    assert requests and all(request.get("max_completion_tokens") == 123 for request in requests)


def test_cli_response_cap_cannot_be_silently_ignored():
    raw = load_agent_config(Path(__file__).parents[2] / "petri/defaults/petri.yaml").model_dump(mode="json")
    raw["model_settings"] = {"max_tokens": 123}
    with pytest.raises(ValidationError, match="requires the provider backend"):
        AgentConfig.model_validate(raw)
