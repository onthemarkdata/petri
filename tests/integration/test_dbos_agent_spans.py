"""A real DBOS step supplies parent context to native agent instrumentation."""

import json
import os
import subprocess
import sys

import pytest

pytest.importorskip("dbos", reason="Install dbos==3.4.0 for the DBOS integration test.")

SCRIPT = """
import asyncio, json, socket, sys
from dbos import DBOS
from opentelemetry import trace
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic_ai import Agent
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.usage import RequestUsage
from petri.telemetry import init_telemetry
from petri.telemetry.instrument_agents import instrument_agents

attempts = []
def deny_network(*args, **kwargs):
    attempts.append(True)
    raise AssertionError("No network in the DBOS instrumentation test")
socket.socket.connect = deny_network
provider = init_telemetry({}, dish_id="dbos-fixture")
exporter = InMemorySpanExporter()
provider.add_span_processor(SimpleSpanProcessor(exporter))
# DBOS uses the process-global provider. This isolated process supplies it explicitly.
trace.set_tracer_provider(provider)
instrument_agents(provider)

def respond(messages, info):
    return ModelResponse(parts=[TextPart("fixture result")], usage=RequestUsage(input_tokens=11, output_tokens=4))
agent = Agent(FunctionModel(respond), name="step_agent")

@DBOS.step()
async def research_step():
    result = await agent.run("Use the fixture.")
    return result.output

@DBOS.workflow()
async def research_workflow():
    return await research_step()

DBOS(config={
    "name": "petri-agent-spans", "system_database_url": "sqlite:///" + sys.argv[1],
    "application_version": "test-1", "use_listen_notify": False,
    "enable_otlp": True, "otlp_traces_endpoints": [], "otlp_logs_endpoints": [],
    "log_level": "WARNING", "otel_attribute_format": "semconv",
})
DBOS.launch()
try:
    assert asyncio.run(research_workflow()) == "fixture result"
finally:
    DBOS.destroy()
spans = exporter.get_finished_spans()
agent_span = next(s for s in spans if s.attributes.get("gen_ai.operation.name") == "invoke_agent")
step_span = next(s for s in spans if s.attributes.get("dbos.operation.type") == "step")
assert agent_span.parent.span_id == step_span.context.span_id
assert agent_span.attributes["gen_ai.aggregated_usage.input_tokens"] == 11
assert not attempts
print(json.dumps({"agent_parent_is_dbos_step": True, "network_attempts": len(attempts), "input_tokens": 11}))
provider.shutdown()
"""


def test_agent_span_is_child_of_real_dbos_step(tmp_path):
    env = {
        key: value for key, value in os.environ.items() if not key.startswith(("OTEL_", "LOGFIRE_"))
    }
    env["PYDANTIC_AI_NO_BANNER"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", SCRIPT, str(tmp_path / "dbos.sqlite")],
        check=False,
        capture_output=True,
        text=True,
        timeout=45,
        env=env,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout.splitlines()[-1])
    assert report == {"agent_parent_is_dbos_step": True, "network_attempts": 0, "input_tokens": 11}
