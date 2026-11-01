"""Native prompted roles can search, fetch, and validate a strict final result."""

import json
import re
from pathlib import Path

import httpx
import pytest
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from petri.agents.config import AgentConfig, load_agent_config
from petri.agents.contract import AgentName
from petri.agents.factory import build_agents
from petri.research.web import RetrievedPage, SearchHit


@pytest.mark.asyncio
@pytest.mark.parametrize("name", [name for name in AgentName
                                 if name not in {AgentName.DECOMPOSITION_LEAD, AgentName.CELL_LEAD}])
async def test_native_role_searches_then_fetches_before_strict_final_output(name):
    raw = load_agent_config(Path(__file__).parents[2] / "petri/defaults/petri.yaml").model_dump(mode="json")
    raw["harness"] = {"backend": "provider", "model": "local-test", "output_mode": "prompted"}
    raw["research"] = {"public_web": True}
    raw["model_settings"] = {"max_tokens": 321}
    config = AgentConfig.model_validate(raw)
    requests, calls = [], []
    url = "https://example.org/primary"

    class Sources:
        async def search(self, query, *, limit=5):
            calls.append(("search", query))
            return [SearchHit(url=url, title="Primary source")]

        async def fetch(self, source_url):
            calls.append(("fetch", source_url))
            return RetrievedPage(url=url, title="Primary source", text="Measured finding.",
                retrieved_at="2026-11-01T00:00:00Z", content_sha256="a" * 64, truncated=False)

    def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        if len(requests) <= 2:
            tool = "web_search" if len(requests) == 1 else "web_fetch"
            arguments = {"query": "primary evidence"} if len(requests) == 1 else {"url": url}
            message = {"role": "assistant", "tool_calls": [{"id": tool, "type": "function",
                "function": {"name": tool, "arguments": json.dumps(arguments)}}]}
        else:
            message = {"role": "assistant", "content": json.dumps({
                "verdict": "NOT_ALLOWED" if len(requests) == 3 else config.agents[name].verdicts_pass[0],
                "sources_cited": [{"url": url, "hierarchy_level": 2, "finding": "Measured finding."}],
            })}
        return httpx.Response(200, json={"id": "fixture", "object": "chat.completion", "created": 0,
            "model": "local-test", "choices": [{"index": 0, "message": message,
            "finish_reason": "tool_calls" if "tool_calls" in message else "stop"}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        model = OpenAIChatModel("local-test", provider=OpenAIProvider(
            base_url="http://local.invalid/v1", api_key="local-test", http_client=http))
        agent = build_agents(config, constitution="Preserve attribution.", model=model,
                             research_client=Sources())[name]
        result = await agent.run("Assess the public fixture claim.")
    assert calls == [("search", "primary evidence"), ("fetch", url)]
    assert len(requests) == 4
    assert all("response_format" not in request and request["max_completion_tokens"] == 321 for request in requests)
    assert all({tool["function"]["name"] for tool in request["tools"]} == {"web_search", "web_fetch"}
               for request in requests)
    instructions = "\n".join(message["content"] for message in requests[2]["messages"]
                             if isinstance(message.get("content"), str))
    assert re.search(r'"required"\s*:\s*\[\s*"verdict"\s*\]', instructions)
    assert '"title": "SourceCitation"' not in instructions
    assert result.output.verdict == config.agents[name].verdicts_pass[0]
    assert result.output.sources_cited[0].url == url
