"""Reproduce both instrumentation experiments in isolated, network-blocked processes."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[2] / "scripts/spikes/agent_instrumentation.py"


@pytest.mark.parametrize("mechanism", ["native", "logfire"])
def test_instrumentation_spike(mechanism):
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), mechanism],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    report = json.loads(completed.stdout)
    assert report["network_connection_attempts"] == 0
    assert report["exporter"] == "InMemorySpanExporter"
    assert report["logfire_sdk_imported"] is (mechanism == "logfire")
    spans = report["spans"]
    operations = [span["attributes"]["gen_ai.operation.name"] for span in spans]
    assert operations.count("invoke_agent") == 1
    assert operations.count("chat") == 2
    assert operations.count("execute_tool") == 1
    agent = next(
        span for span in spans if span["attributes"]["gen_ai.operation.name"] == "invoke_agent"
    )
    assert agent["attributes"]["gen_ai.aggregated_usage.input_tokens"] == 12
    assert agent["attributes"]["gen_ai.aggregated_usage.output_tokens"] == 5
    for span in spans:
        if span is not agent:
            assert span["parent_span_id"] == agent["span_id"]
