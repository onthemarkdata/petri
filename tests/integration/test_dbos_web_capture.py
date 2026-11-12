"""Registered web tools retain content and ancestry under native DBOS steps."""

import json
import os
import subprocess
import sys

import pytest

SCRIPT = r"""
import asyncio, json, socket, sys
from pathlib import Path
import httpx
from dbos import DBOS
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from petri.agents.config import AgentConfig, load_agent_config
from petri.agents.contract import AgentName
from petri.agents.factory import build_agents
from petri.execution.otel import DBOSTelemetry
from petri.harness.pi.model import PiModel
from petri.research.web import PublicWebClient
from petri.telemetry import shutdown_telemetry
from petri.telemetry.span_store import query_spans

attempts = []
def deny_network(*args, **kwargs):
    attempts.append(True)
    raise AssertionError("No network in the DBOS web capture test")
socket.socket.connect = deny_network
directory = Path(sys.argv[1])
backend = sys.argv[2]
(directory / "petri.yaml").write_text("name: capture\ntelemetry:\n  capture_content: true\n")
bridge = DBOSTelemetry(directory)
raw = load_agent_config(Path("petri/defaults/petri.yaml")).model_dump(mode="json")
raw["harness"] = {"backend": backend, "model": "offline", "output_mode": "prompted"}
raw["research"] = {"public_web": True}
config = AgentConfig.model_validate(raw)
retrievals, requests = [], []
steps = [("web_search", {"query": "Public evidence"}), ("web_fetch", {"url": "https://example.org/source"})]
final = {"verdict": "NEEDS_MORE_EVIDENCE", "sources_cited": []}

def retrieve(request):
    retrievals.append(str(request.url))
    text = ('<a class="result__a" href="https://example.org/source">Public source</a>'
            if request.url.host == "html.duckduckgo.com" else '<p>Public independent result password=CANARY_dbos</p>')
    return httpx.Response(200, headers={"content-type": "text/html"}, text=text)

async def resolve(host):
    return ["93.184.216.34"]

def respond(request):
    requests.append(json.loads(request.content))
    index = len(requests) - 1
    if index < len(steps):
        tool, args = steps[index]
        message = {"role": "assistant", "tool_calls": [{"id": tool, "type": "function",
            "function": {"name": tool, "arguments": json.dumps(args)}}]}
    else:
        message = {"role": "assistant", "content": json.dumps(final)}
    return httpx.Response(200, json={"id": "fixture", "object": "chat.completion", "created": 0,
        "model": "offline", "choices": [{"index": 0, "message": message,
            "finish_reason": "tool_calls" if "tool_calls" in message else "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}})

class FakePiTransport:
    async def request(self, request):
        if request["type"] != "prompt":
            return {}
        requests.append(request)
        index = len(requests) - 1
        text = json.dumps({steps[index][0]: [steps[index][1]]}) if index < len(steps) else json.dumps(final)
        return {"messages": [{"role": "assistant", "stopReason": "stop", "content": [{"type": "text", "text": text}],
            "usage": {"input": 1, "output": 1, "cacheRead": 0, "cacheWrite": 0}}]}
    async def aclose(self):
        pass

http = httpx.AsyncClient(transport=httpx.MockTransport(respond))
model = (PiModel("offline", transport=FakePiTransport()) if backend == "pi" else
    OpenAIChatModel("offline", provider=OpenAIProvider(base_url="http://local.invalid/v1", api_key="offline", http_client=http)))
agent = build_agents(config, constitution="Public evidence.", model=model,
    research_client=PublicWebClient(transport=httpx.MockTransport(retrieve), resolve_host=resolve))[AgentName.INVESTIGATOR]

@DBOS.step(name="capture.web_step")
async def research_step():
    result = await agent.run("Public claim.")
    return result.output.verdict

@DBOS.workflow(name="capture.web_workflow")
async def research_workflow():
    return await research_step()

DBOS(config={"name": "web-capture", "system_database_url": "sqlite:///" + str(directory / "dbos.sqlite"),
    "application_version": "test", "use_listen_notify": False, "log_level": "WARNING", **bridge.dbos_config()})
bridge.bind()
DBOS.launch()
try:
    assert asyncio.run(research_workflow()) == "NEEDS_MORE_EVIDENCE"
finally:
    DBOS.destroy()
    bridge.close()
    asyncio.run(http.aclose())
    asyncio.run(model.aclose()) if backend == "pi" else None
rows = query_spans(directory / "petri-dishes/capture/petri.sqlite", limit=None)
step = next(row for row in rows if row["name"] == "capture.web_step")
workflow = next(row for row in rows if row["name"] == "capture.web_workflow")
run = next(row for row in rows if row["attributes"].get("gen_ai.operation.name") == "invoke_agent")
tools = [row for row in rows if row["attributes"].get("gen_ai.operation.name") == "execute_tool"]
assert step["parent_span_id"] == workflow["span_id"]
assert run["parent_span_id"] == step["span_id"]
assert len(tools) == len(retrievals) == 2 and len(requests) == 3
assert {row["attributes"]["gen_ai.tool.name"] for row in tools} == {"web_search", "web_fetch"}
assert all(row["parent_span_id"] == run["span_id"] and row["trace_id"] == workflow["trace_id"] for row in tools)
assert all(row["attributes"]["gen_ai.tool.call.id"] for row in tools)
assert len({row["span_id"] for row in rows}) == len(rows)
assert "Public independent result" in str(tools) and "CANARY_dbos" not in str(rows)
assert run["attributes"]["gen_ai.aggregated_usage.input_tokens"] == (3 if backend == "pi" else 15)
assert not attempts
shutdown_telemetry()
print(json.dumps({"tool_spans": len(tools), "retrievals": len(retrievals), "parent_is_dbos_step": True, "network_attempts": len(attempts)}))
"""


@pytest.mark.parametrize("backend", ["provider", "pi"])
def test_web_tool_parentage_and_local_content_under_dbos(tmp_path, backend):
    env = {
        key: value for key, value in os.environ.items() if not key.startswith(("OTEL_", "LOGFIRE_"))
    }
    result = subprocess.run(
        [sys.executable, "-c", SCRIPT, str(tmp_path), backend],
        check=False,
        capture_output=True,
        text=True,
        timeout=45,
        env=env,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout.splitlines()[-1]) == {
        "tool_spans": 2,
        "retrievals": 2,
        "parent_is_dbos_step": True,
        "network_attempts": 0,
    }
