"""Compare native OTel and local-only Logfire instrumentation without network access."""

from __future__ import annotations

import argparse
import asyncio
import importlib.abc
import importlib.metadata
import json
import os
import socket
import sys
import tempfile
from pathlib import Path


class BlockLogfire(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "logfire" or fullname.startswith("logfire."):
            raise ModuleNotFoundError(
                "Logfire is blocked for this native OTel probe.", name="logfire"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mechanism", choices=["native", "logfire"])
    args = parser.parse_args()
    os.environ["PYDANTIC_AI_NO_BANNER"] = "1"
    for key in list(os.environ):
        if key.startswith(("OTEL_", "LOGFIRE_")):
            os.environ.pop(key)
    attempts = []

    def deny_network(*args, **kwargs):
        attempts.append("connection attempted")
        raise AssertionError("The instrumentation spike must not access the network.")

    socket.socket.connect = deny_network
    socket.create_connection = deny_network
    if args.mechanism == "native":
        sys.meta_path.insert(0, BlockLogfire())

    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from pydantic_ai import Agent
    from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
    from pydantic_ai.models.function import FunctionModel
    from pydantic_ai.models.instrumented import InstrumentationSettings
    from pydantic_ai.usage import RequestUsage

    exporter = InMemorySpanExporter()
    processor = SimpleSpanProcessor(exporter)
    with tempfile.TemporaryDirectory(prefix="petri-otel-spike-") as directory:
        if args.mechanism == "native":
            provider = TracerProvider()
            provider.add_span_processor(processor)
            Agent.instrument_all(
                InstrumentationSettings(
                    tracer_provider=provider,
                    include_content=False,
                    include_binary_content=False,
                    include_model_request_parameters=False,
                )
            )
            shutdown = provider.shutdown
        else:
            import logfire

            local = logfire.configure(
                local=True,
                send_to_logfire=False,
                console=False,
                metrics=False,
                data_dir=Path(directory),
                additional_span_processors=[processor],
            )
            local.instrument_pydantic_ai(include_content=False, include_binary_content=False)
            shutdown = local.shutdown
        calls = 0

        def respond(messages, info):
            nonlocal calls
            calls += 1
            if calls == 1:
                return ModelResponse(
                    parts=[ToolCallPart("lookup", {"key": "fixture"})],
                    usage=RequestUsage(input_tokens=5, output_tokens=3),
                )
            return ModelResponse(
                parts=[TextPart("The fixture contains three successful runs.")],
                usage=RequestUsage(input_tokens=7, output_tokens=2),
            )

        agent = Agent(FunctionModel(respond), name="instrumentation_probe")

        @agent.tool_plain
        def lookup(key: str) -> str:
            """Read a supplied fixture without contacting an external service."""
            return f"{key}: three successful runs"

        result = asyncio.run(agent.run("Check the supplied fixture."))
        spans = exporter.get_finished_spans()
        report = {
            "mechanism": args.mechanism,
            "pydantic_ai_version": importlib.metadata.version("pydantic-ai"),
            "opentelemetry_sdk_version": importlib.metadata.version("opentelemetry-sdk"),
            "logfire_sdk_imported": "logfire" in sys.modules,
            "network_connection_attempts": len(attempts),
            "exporter": type(exporter).__name__,
            "requests": result.usage.requests,
            "input_tokens": result.usage.input_tokens,
            "output_tokens": result.usage.output_tokens,
            "spans": [
                {
                    "name": span.name,
                    "span_id": span.context.span_id,
                    "parent_span_id": span.parent.span_id if span.parent else None,
                    "attributes": {
                        key: value
                        for key, value in (span.attributes or {}).items()
                        if key.startswith("gen_ai.")
                        and ("usage" in key or key.endswith((".name", ".model")))
                    },
                }
                for span in spans
            ],
        }
        shutdown()
        assert not attempts
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
