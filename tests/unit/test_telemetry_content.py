"""Content capture is explicit, bounded, and scrubbed before both exports."""

import importlib
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Status, StatusCode
from pydantic import ValidationError
from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel

from petri.cli._bootstrap import bootstrap_telemetry
from petri.execution.otel import DBOSTelemetry
from petri.telemetry import init_telemetry, shutdown_telemetry
from petri.telemetry.config import TelemetryConfig
from petri.telemetry.instrument_agents import instrument_agents
from petri.telemetry.redaction import MAX_TEXT
from petri.telemetry.span_store import aggregate_usage, query_spans


@pytest.fixture
def sinks(tmp_path, monkeypatch, request):
    shutdown_telemetry()
    monkeypatch.setattr(
        importlib.import_module("petri.telemetry.instrument_agents"), "_active_provider", None
    )
    monkeypatch.setenv("PETRI_TEST_SECRET", "CANARY_runtime_value")
    payloads = []

    def create_exporter(**kwargs):
        exporter = OTLPSpanExporter(**kwargs)

        def receive(data):
            payloads.append(ExportTraceServiceRequest.FromString(data))
            return SimpleNamespace(success=True)

        monkeypatch.setattr(exporter._client, "export", receive)
        return exporter

    cfg = TelemetryConfig(
        capture_content=getattr(request, "param", True),
        redact_env_vars=["PETRI_TEST_SECRET"],
        otlp={
            "traces_endpoint": "http://collector.invalid/v1/traces",
            "headers": {"Authorization": "Bearer CANARY_header_value"},
        },
    )
    with patch(
        "opentelemetry.exporter.otlp.proto.http.trace_exporter.OTLPSpanExporter",
        side_effect=create_exporter,
    ):
        provider = init_telemetry(cfg, dish_id="fixture", petri_dir=tmp_path)
    memory = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(memory))
    instrument_agents(provider)
    yield provider, memory, payloads, tmp_path, cfg
    Agent.instrument_all(False)
    shutdown_telemetry()


def serialized(sinks):
    provider, _memory, payloads, directory, _ = sinks
    assert provider.force_flush()
    rows = query_spans(directory / "petri-dishes/fixture/petri.sqlite", limit=None)
    assert rows and payloads
    return rows, json.dumps(rows), str(payloads)


@pytest.mark.asyncio
async def test_prompt_output_settings_events_status_and_nested_content_are_scrubbed(sinks):
    provider, memory, _, _, _ = sinks
    output = "Public measured output CANARY_runtime_value Authorization: Basic CANARY_basic_value"

    def respond(messages, info):
        return ModelResponse(parts=[TextPart(output)])

    agent = Agent(
        FunctionModel(respond),
        name="diagnostic_agent",
        model_settings={"temperature": 0.25, "max_tokens": 123},
    )
    with provider.get_tracer("test").start_as_current_span("dbos.step") as parent:
        parent.set_attributes(
            {
                "petri.tokens_input": 7,
                "petri.tokens_output": 3,
                "nested": json.dumps(
                    {
                        "api_key": "CANARY_json_value",
                        "data": json.dumps({"password": "CANARY_nested_value"}),
                        "url": "https://example.org/source?api_key=CANARY_url_value&paper=42",
                    }
                ),
                "headers": json.dumps(
                    {"Authorization": "Bearer CANARY_auth_value", "x-api-key": "CANARY_api_value"}
                ),
                "echo": "CANARY_header_value",
                "petri.cell_id": "dish-colony-001-001",
                "http.request.header.x-api-key": "CANARY_flat_header",
                "environment_like": json.dumps({"OPENAI_API_KEY": "CANARY_env_key"}),
                "escaped_error": r"Public error {\"api_key\":\"CANARY_escaped_value\"}",
            }
        )
        parent.add_event(
            "diagnostic",
            {"error": "Public error password=CANARY_event_value", "api_key": "CANARY_event_key"},
        )
        parent.record_exception(
            ValueError("Public exception Bearer CANARY_exception_value CANARY_runtime_value")
        )
        parent.set_status(Status(StatusCode.ERROR, "Public failure password=CANARY_status_value"))
        await agent.run("Public prompt CANARY_runtime_value api_key=CANARY_prompt_value")
    rows, local, remote = serialized(sinks)
    for data in (
        local,
        remote,
        str([dict(span.attributes) for span in memory.get_finished_spans()]),
    ):
        assert "CANARY_" not in data
        assert "Public" in data
    for text in (
        "Public prompt",
        "Public measured output",
        "Public error",
        "Public failure",
        "https://example.org/source",
        "paper=42",
        "diagnostic_agent",
    ):
        assert text in local and text in remote
    assert any(
        "temperature" in str(row["attributes"]) and "0.25" in str(row["attributes"]) for row in rows
    )
    row = next(row for row in rows if row["name"] == "dbos.step")
    assert len(row["attributes"]["petri.telemetry.events"]) == 2
    nested = json.loads(row["attributes"]["nested"])
    assert nested["api_key"] == "[REDACTED]"
    assert json.loads(nested["data"])["password"] == "[REDACTED]"
    for span in memory.get_finished_spans():
        for key, value in span.attributes.items():
            if key in {
                "gen_ai.input.messages",
                "gen_ai.output.messages",
                "gen_ai.system_instructions",
            }:
                json.loads(value)
    assert (
        aggregate_usage(sinks[3] / "petri-dishes/fixture/petri.sqlite", group_by="phase")[0][
            "tokens_input"
        ]
        == 7
    )
    remote_spans = [
        span
        for batch in sinks[2]
        for resource in batch.resource_spans
        for scope in resource.scope_spans
        for span in scope.spans
    ]
    assert {row["span_id"] for row in rows} == {span.span_id.hex() for span in remote_spans}
    assert len(remote_spans) == len(rows)


def test_bounds_and_sdk_drops_are_visible(sinks):
    provider = sinks[0]
    with provider.get_tracer("test").start_as_current_span("bounded") as span:
        span.set_attribute("long", "public " * MAX_TEXT + " CANARY_runtime_value")
        for index in range(140):
            span.add_event("event", {"index": index})
    rows, local, remote = serialized(sinks)
    assert rows[0]["attributes"]["petri.telemetry.bounded_values"] >= 1
    assert rows[0]["attributes"]["petri.telemetry.dropped_events"] == 12
    assert "[TRUNCATED]" in local and "[TRUNCATED]" in remote
    assert "CANARY_" not in local + remote


@pytest.mark.parametrize("value", ["yes", 1, None, {}])
def test_capture_requires_an_explicit_boolean(value):
    with pytest.raises(ValidationError, match="capture_content"):
        TelemetryConfig(capture_content=value)


def test_runtime_secrets_rotate_without_entering_cache_keys(sinks, monkeypatch):
    provider, _, _, directory, cfg = sinks
    monkeypatch.setenv("PETRI_TEST_SECRET", "CANARY_rotated_value")
    assert init_telemetry(cfg, dish_id="fixture", petri_dir=directory) is provider
    with provider.get_tracer("test").start_as_current_span("rotation") as span:
        span.set_attribute("evidence", "Public CANARY_runtime_value CANARY_rotated_value")
    _, local, remote = serialized(sinks)
    assert "CANARY_" not in local + remote
    setup = importlib.import_module("petri.telemetry.setup")
    assert "CANARY_" not in str(setup._providers.keys())
    assert "CANARY_" not in repr(cfg)


def test_cli_dbos_provider_reuse_dish_and_disabled_capture(tmp_path):
    shutdown_telemetry()
    for name, capture, enabled in (
        ("one", True, True),
        ("two", False, True),
        ("three", True, False),
    ):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "petri.yaml").write_text(
            f"name: {name}\ntelemetry:\n  capture_content: {str(capture).lower()}\n  enabled: {str(enabled).lower()}\n"
        )
        provider = bootstrap_telemetry(directory)
        bridge = DBOSTelemetry(directory)
        assert bridge.provider is provider
        settings = instrument_agents(provider)
        assert settings.include_content == (capture and enabled)
        assert settings.include_model_request_parameters == (capture and enabled)
        assert not settings.include_binary_content
        with provider.get_tracer("test").start_as_current_span("enabled") as span:
            assert span.is_recording() == enabled
        bridge.close()
    Agent.instrument_all(False)
    shutdown_telemetry()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["provider", "pi"])
@pytest.mark.parametrize(
    "case", ["success", "http_failure", "invalid_arguments", "retry", "cancellation"]
)
async def test_registered_web_tools_capture_without_gateway_replay(
    sinks, fake_pi, monkeypatch, backend, case
):
    import asyncio
    from contextlib import AsyncExitStack
    from pathlib import Path

    import httpx
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    from petri.agents.config import AgentConfig, load_agent_config
    from petri.agents.contract import AgentName
    from petri.agents.factory import build_agents
    from petri.harness.pi.model import PiModel
    from petri.harness.pi.transport import PiRpcTransport
    from petri.research.web import PublicWebClient

    provider = sinks[0]
    raw = load_agent_config(Path(__file__).parents[2] / "petri/defaults/petri.yaml").model_dump(
        mode="json"
    )
    raw["harness"] = {"backend": backend, "model": "offline-model", "output_mode": "prompted"}
    raw["research"] = {"public_web": True}
    config = AgentConfig.model_validate(raw)
    url = "https://example.org/source?paper=42&api_key=CANARY_url_value"
    final = {
        "verdict": "NEEDS_MORE_EVIDENCE",
        "sources_cited": [],
        "summary": "Public model output CANARY_runtime_value",
    }
    steps = [
        ("web_search", {"query": "Public query CANARY_runtime_value"}, "search-1"),
        ("web_fetch", {"url": url}, "fetch-1"),
    ]
    if case == "invalid_arguments":
        steps.insert(0, ("web_search", {"query": {"api_key": "CANARY_invalid_value"}}, "invalid-1"))
    if case == "cancellation":
        steps = [steps[-1]]
    requests, retrievals = [], []
    waiting = asyncio.Event()
    retried = False

    async def resolve_host(host):
        return ["93.184.216.34"]

    def retrieve(request):
        retrievals.append(request)
        if case == "http_failure":
            return httpx.Response(503)
        if request.url.host == "html.duckduckgo.com":
            return httpx.Response(
                200,
                headers={"content-type": "text/html", "x-api-key": "CANARY_response_header"},
                text='<a class="result__a" href="https://example.org/source?paper=42">Public source CANARY_runtime_value</a>',
            )
        return httpx.Response(
            200,
            headers={"content-type": "text/html", "x-api-key": "CANARY_response_header"},
            text="<title>Public source</title><p>Public retained result CANARY_runtime_value password=CANARY_result_value</p>",
        )

    async def respond(request):
        nonlocal retried
        payload = json.loads(request.content)
        requests.append(payload)
        if case == "retry" and len(requests) == 2:
            retried = True
            return httpx.Response(
                503,
                headers={"retry-after-ms": "1"},
                json={
                    "error": {
                        "message": "Public failure Bearer CANARY_retry_value",
                        "type": "server_error",
                    }
                },
            )
        index = len(requests) - 1 - int(retried)
        if index < len(steps):
            tool, arguments, call_id = steps[index]
            message = {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": tool, "arguments": json.dumps(arguments)},
                    }
                ],
            }
        else:
            message = {"role": "assistant", "content": json.dumps(final)}
        return httpx.Response(
            200,
            json={
                "id": f"response-{index}",
                "object": "chat.completion",
                "created": 0,
                "model": "offline-model",
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": "tool_calls" if "tool_calls" in message else "stop",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            },
        )

    async with AsyncExitStack() as stack:
        client = PublicWebClient(transport=httpx.MockTransport(retrieve), resolve_host=resolve_host)
        if backend == "provider":
            http = await stack.enter_async_context(
                httpx.AsyncClient(transport=httpx.MockTransport(respond))
            )
            model = OpenAIChatModel(
                "offline-model",
                provider=OpenAIProvider(
                    base_url="http://local.invalid/v1",
                    api_key="CANARY_header_value",
                    http_client=http,
                ),
            )
        else:
            scenarios = [{"text": json.dumps({tool: [arguments]})} for tool, arguments, _ in steps]
            if case == "retry":
                scenarios.insert(
                    1, {"kind": "rate_limit", "error": "rate limit Bearer CANARY_retry_value"}
                )
            fake_pi.scenario(*scenarios, {"text": json.dumps(final)})
            from petri.harness.errors import RateLimitedError

            transport = PiRpcTransport(binary=str(fake_pi.path))
            original_request = transport.request

            async def pi_request(request):
                if request["type"] == "prompt":
                    requests.append(request)
                try:
                    return await original_request(request)
                except RateLimitedError:
                    # The real transport closes a failed process. The next fake process
                    # must resume the script after the transient failure.
                    fake_pi.scenario(*scenarios[2:], {"text": json.dumps(final)})
                    raise

            monkeypatch.setattr(transport, "request", pi_request)
            from petri.harness.retry import RetryPolicy

            model = await stack.enter_async_context(
                PiModel(
                    "offline-model", transport=transport, retry_policy=RetryPolicy(base=0, jitter=0)
                )
            )
        if case == "cancellation":
            original_model_request = model.request

            async def hold_after_result(messages, settings, parameters):
                from pydantic_ai.messages import ToolReturnPart

                if any(
                    isinstance(part, ToolReturnPart)
                    for message in messages
                    for part in message.parts
                ):
                    waiting.set()
                    await asyncio.Event().wait()
                return await original_model_request(messages, settings, parameters)

            monkeypatch.setattr(model, "request", hold_after_result)
        agent = build_agents(
            config,
            constitution="Public evidence CANARY_runtime_value",
            model=model,
            research_client=client,
        )[AgentName.INVESTIGATOR]
        with provider.get_tracer("test").start_as_current_span("offline.step") as step:
            task = asyncio.create_task(agent.run("Public claim CANARY_runtime_value"))
            if case == "cancellation":
                await asyncio.wait_for(waiting.wait(), timeout=5)
                # The native tool span has ended before a second response exists.
                assert any(
                    span.attributes.get("gen_ai.operation.name") == "execute_tool"
                    for span in sinks[1].get_finished_spans()
                )
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                result = await task
                assert result.output.verdict == "NEEDS_MORE_EVIDENCE"
                assert result.output.sources_cited == []
    rows, local, remote = serialized(sinks)
    assert "CANARY_" not in local + remote
    runs = [row for row in rows if row["attributes"].get("gen_ai.operation.name") == "invoke_agent"]
    tools = [
        row for row in rows if row["attributes"].get("gen_ai.operation.name") == "execute_tool"
    ]
    chats = [row for row in rows if row["attributes"].get("gen_ai.operation.name") == "chat"]
    assert len(runs) == 1 and runs[0]["parent_span_id"] == f"{step.get_span_context().span_id:016x}"
    assert all(
        row["parent_span_id"] == runs[0]["span_id"] and row["trace_id"] == runs[0]["trace_id"]
        for row in tools + chats
    )
    expected_tools = 1 if case == "cancellation" else 2
    assert len(retrievals) == expected_tools
    assert len(tools) == expected_tools + int(case == "invalid_arguments")
    assert len({row["attributes"]["gen_ai.tool.call.id"] for row in tools}) == len(tools)
    assert len({row["span_id"] for row in rows}) == len(rows)
    assert all(row["name"].startswith("execute_tool ") for row in tools)
    assert {row["attributes"]["gen_ai.tool.name"] for row in tools} == (
        {"web_fetch"} if case == "cancellation" else {"web_search", "web_fetch"}
    )
    assert all(row["attributes"].get("gen_ai.tool.call.id") for row in tools)
    if backend == "provider":
        expected_responses = {
            f"response-{index}" for index in range(1 if case == "cancellation" else len(steps) + 1)
        }
        assert {
            row["attributes"]["gen_ai.response.id"]
            for row in chats
            if "gen_ai.response.id" in row["attributes"]
        } == expected_responses
        assert all(response_id in remote for response_id in expected_responses)
        assert {row["attributes"]["gen_ai.tool.call.id"] for row in tools} == {
            call_id for _, _, call_id in steps
        }
    if case == "http_failure":
        assert "Public retrieval failed: HTTPStatusError." in str(
            [row["attributes"] for row in tools]
        )
        # The registered tools return an error object; they do not raise an exception.
        assert all(row["status_code"] == "UNSET" for row in tools)
    else:
        assert "Public retained result" in str([row["attributes"] for row in tools])
    if case == "invalid_arguments":
        assert "validation" in local.lower() or "string_type" in local
        rejected = [row for row in tools if row["status_code"] == "ERROR"]
        assert len(rejected) == 1
        assert rejected[0]["attributes"]["gen_ai.tool.name"] == "web_search"
        assert "string_type" in str(rejected[0])
        if backend == "provider":
            assert rejected[0]["attributes"]["gen_ai.tool.call.id"] == "invalid-1"
    if case == "cancellation":
        assert len(requests) == 1  # No result was replayed to either inference transport.
        assert len(chats) == 2
        cancelled = max(chats, key=lambda row: row["start_time_ns"])
        # Pinned native instrumentation leaves cancellation UNSET; it reports no output or usage.
        assert cancelled["status_code"] == "UNSET"
        assert "gen_ai.output.messages" not in cancelled["attributes"]
        assert "gen_ai.usage.input_tokens" not in cancelled["attributes"]
        assert "Public retained result" in local and "Public retained result" in remote
        # Only the completed first request contributes measured native request usage.
        assert sum(row["attributes"].get("gen_ai.usage.input_tokens", 0) for row in chats) == (
            5 if backend == "provider" else 1
        )
    else:
        assert len(requests) == len(steps) + 1 + int(case == "retry")
        measured = sum(row["attributes"].get("gen_ai.usage.input_tokens", 0) for row in chats)
        assert measured == (len(requests) - int(case == "retry")) * (
            5 if backend == "provider" else 1
        )
        assert runs[0]["attributes"]["gen_ai.aggregated_usage.input_tokens"] == measured
    # Native usage is diagnostic; it must not create duplicate Petri usage rows.
    assert aggregate_usage(sinks[3] / "petri-dishes/fixture/petri.sqlite", group_by="phase") == []


@pytest.mark.asyncio
@pytest.mark.parametrize("sinks", [False], indirect=True)
async def test_capture_off_keeps_prompt_and_output_out_of_both_sinks(sinks):
    def respond(messages, info):
        return ModelResponse(parts=[TextPart("Private output text.")])

    await Agent(FunctionModel(respond), instructions="Private constitution text.").run(
        "Private claim text."
    )
    _, local, remote = serialized(sinks)
    assert "Private" not in local + remote
    for span in sinks[1].get_finished_spans():
        for key in ("gen_ai.input.messages", "gen_ai.output.messages"):
            if key in span.attributes:
                for message in json.loads(span.attributes[key]):
                    assert all("content" not in part for part in message.get("parts", []))


def test_sdk_length_limits_drop_partial_secrets_and_report_omissions(tmp_path, monkeypatch):
    monkeypatch.setenv("OTEL_SPAN_ATTRIBUTE_VALUE_LENGTH_LIMIT", "20")
    monkeypatch.setenv("OTEL_ATTRIBUTE_VALUE_LENGTH_LIMIT", "20")
    provider = init_telemetry(
        {},
        dish_id="short",
        petri_dir=tmp_path,
        protected_secret_values=("CANARY_this_secret_is_longer_than_the_sdk_limit",),
    )
    with provider.get_tracer("test").start_as_current_span("limits") as span:
        span.set_attribute("private", "CANARY_this_secret_is_longer_than_the_sdk_limit")
        span.add_event(
            "exception", {"exception.message": "CANARY_this_secret_is_longer_than_the_sdk_limit"}
        )
    assert provider.force_flush()
    rows = query_spans(tmp_path / "petri-dishes/short/petri.sqlite")
    assert "CANARY" not in json.dumps(rows)
    assert rows[0]["attributes"]["petri.telemetry.bounded_values"] == 2
    assert rows[0]["attributes"]["private"] == "[TRUNCATED: SDK length limit]"
    shutdown_telemetry()


def test_redaction_failure_drops_span_without_exporting_payload(sinks, monkeypatch, caplog):
    provider, memory, payloads, directory, _ = sinks
    redactor = provider._active_span_processor.redactor
    monkeypatch.setattr(
        redactor, "span", lambda *args: (_ for _ in ()).throw(ValueError("CANARY_failure_value"))
    )
    with provider.get_tracer("test").start_as_current_span("CANARY_span_name"):
        pass
    assert provider.force_flush()
    assert memory.get_finished_spans() == () and payloads == []
    assert query_spans(directory / "petri-dishes/fixture/petri.sqlite") == []
    assert "span dropped during redaction (ValueError)" in caplog.text
    assert "CANARY" not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("canary", ['CANARY_"quoted"_value', r"CANARY_\folder\leaf", "CANARY_雪_𝄞"])
@pytest.mark.parametrize("encoding_levels", [1, 2])
@pytest.mark.parametrize("ensure_ascii", [False, True])
async def test_json_string_scalars_are_scrubbed_after_both_sink_decodes(
    sinks, canary, encoding_levels, ensure_ascii
):
    from copy import deepcopy

    from google.protobuf.json_format import MessageToDict
    from pydantic_ai.messages import UserPromptPart

    provider, _, payloads, directory, cfg = sinks
    assert (
        init_telemetry(
            cfg, dish_id="fixture", petri_dir=directory, protected_secret_values=(canary,)
        )
        is provider
    )
    encoded = canary
    for _ in range(encoding_levels):
        encoded = json.dumps(encoded, ensure_ascii=ensure_ascii)
    attributes = {
        "scalar": encoded,
        "object": json.dumps(
            {"result": encoded, "public": "Public evidence", "id": "response-stable"}
        ),
        "array": json.dumps([encoded, "Public evidence"]),
    }
    prompt = json.dumps(attributes)
    received = []
    snapshots = []

    def respond(messages, info):
        received.extend(messages)
        snapshots.extend(deepcopy(messages))
        assert [
            part.content
            for message in messages
            for part in message.parts
            if isinstance(part, UserPromptPart)
        ] == [prompt]
        return ModelResponse(parts=[TextPart(prompt)], provider_response_id="response-stable")

    with provider.get_tracer("test").start_as_current_span("scalar-readback") as span:
        span.set_attributes(attributes)
        result = await Agent(FunctionModel(respond), name="scalar_agent").run(prompt)
    rows, _, _ = serialized(sinks)

    def decode(value):
        if isinstance(value, str) and value.lstrip().startswith(('"', "{", "[")):
            try:
                return decode(json.loads(value))
            except ValueError:
                pass
        if isinstance(value, dict):
            return {key: decode(item) for key, item in value.items()}
        if isinstance(value, list):
            return [decode(item) for item in value]
        return value

    def leaves(value):
        if isinstance(value, dict):
            return [leaf for item in value.values() for leaf in leaves(item)]
        if isinstance(value, list):
            return [leaf for item in value for leaf in leaves(item)]
        return [value] if isinstance(value, str) else []

    local = decode(rows)
    remote = decode([MessageToDict(batch, preserving_proto_field_name=True) for batch in payloads])
    for readback in (local, remote):
        strings = leaves(readback)
        assert all(canary not in value and "CANARY_" not in value for value in strings)
        assert "[REDACTED]" in strings
        assert "Public evidence" in strings and "response-stable" in strings
    local_span = next(row for row in local if row["name"] == "scalar-readback")
    assert local_span["attributes"]["scalar"] == "[REDACTED]"
    assert local_span["attributes"]["object"] == {
        "result": "[REDACTED]",
        "public": "Public evidence",
        "id": "response-stable",
    }
    assert local_span["span_id"] == f"{span.get_span_context().span_id:016x}"
    assert local_span["trace_id"] == f"{span.get_span_context().trace_id:032x}"
    assert received == snapshots
    assert result.output == prompt
    assert attributes["scalar"] == encoded


def test_embedded_escaped_credential_drops_ambiguous_tail_in_both_sinks(sinks):
    from google.protobuf.json_format import MessageToDict

    secret = 'CANARY_first"CANARY_second\\CANARY_third雪'
    escaped = json.dumps(json.dumps({"password": secret}, ensure_ascii=True))[1:-1]
    original = "Public error " + escaped + " trailing diagnostic"
    with sinks[0].get_tracer("test").start_as_current_span("escaped-assignment") as span:
        span.set_attribute("diagnostic", original)
    rows, _, _ = serialized(sinks)
    stored = rows[0]["attributes"]["diagnostic"]
    remote = json.dumps([MessageToDict(batch) for batch in sinks[2]])
    assert "CANARY_" not in stored + remote
    assert "Public error" in stored and "[REDACTED]" in stored
    assert "[TRUNCATED: escaped credential tail]" in stored
    assert rows[0]["attributes"]["petri.telemetry.bounded_values"] >= 1
    assert "CANARY_second" in original and "trailing diagnostic" in original


@pytest.fixture
def elevated_sinks(tmp_path, monkeypatch, request):
    """Exercise the real shared redactor and both owned exporters without networking."""
    shutdown_telemetry()
    monkeypatch.setattr(
        importlib.import_module("petri.telemetry.instrument_agents"), "_active_provider", None
    )
    payloads = []

    def create_exporter(**kwargs):
        exporter = OTLPSpanExporter(**kwargs)

        def receive(data):
            payloads.append(ExportTraceServiceRequest.FromString(data))
            return SimpleNamespace(success=True)

        monkeypatch.setattr(exporter._client, "export", receive)
        return exporter

    cfg = TelemetryConfig(
        capture_content=True, capture_limits=getattr(request, "param", {}),
        otlp={"traces_endpoint": "http://collector.invalid/v1/traces"},
    )
    import socket

    monkeypatch.setattr(socket.socket, "connect", lambda *a, **k: pytest.fail("Network is forbidden."))
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: pytest.fail("DNS is forbidden."))
    monkeypatch.setattr(
        "opentelemetry.exporter.otlp.proto.http.trace_exporter.OTLPSpanExporter", create_exporter
    )
    provider = init_telemetry(
        cfg, dish_id="fixture", petri_dir=tmp_path,
        protected_secret_values=("CANARY_boundary", 'CANARY_"雪\\tail'),
    )
    memory = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(memory))
    instrument_agents(provider)
    try:
        yield provider, memory, payloads, tmp_path, cfg
    finally:
        Agent.instrument_all(False)
        shutdown_telemetry()


def _remote_spans(sinks):
    return [span for batch in sinks[2] for resource in batch.resource_spans
            for scope in resource.scope_spans for span in scope.spans]


def _remote_attributes(span):
    def decode(value):
        kind = value.WhichOneof("value")
        if kind == "array_value":
            return [decode(item) for item in value.array_value.values]
        if kind == "kvlist_value":
            return {item.key: decode(item.value) for item in value.kvlist_value.values}
        return getattr(value, kind)

    return {item.key: decode(item.value) for item in span.attributes}


@pytest.mark.asyncio
async def test_elevated_function_model_preserves_complete_multiturn_capture(elevated_sinks):
    from copy import deepcopy

    from pydantic_ai.messages import ToolCallPart, UserPromptPart
    from pydantic_ai.usage import RequestUsage

    provider, memory, _, _, _ = elevated_sinks
    # Cross the old cutoff and the retained 208296-byte sizing sample.
    prompt = "CANARY_boundary " + "p" * 65_520 + " CANARY_boundary " + "q" * 150_000
    public_prompt = prompt.replace("CANARY_boundary", "[REDACTED]")
    received, snapshots, responses, response_copies, settings = [], [], [], [], []

    def respond(messages, info):
        received.append(messages)
        snapshots.append(deepcopy(messages))
        settings.append(deepcopy(info.model_settings))
        if len(responses) == 0:
            response = ModelResponse(
                parts=[ToolCallPart("lookup", {"query": "Public CANARY_boundary"}, "call-1")],
                provider_response_id="response-1", finish_reason="tool_call",
                usage=RequestUsage(input_tokens=31, output_tokens=7, cache_read_tokens=3),
            )
        else:
            response = ModelResponse(
                parts=[TextPart("Public output CANARY_boundary")],
                provider_response_id="response-2", finish_reason="stop",
                usage=RequestUsage(input_tokens=41, output_tokens=9, cache_read_tokens=5),
            )
        responses.append(response)
        response_copies.append(deepcopy(response))
        return response

    agent = Agent(FunctionModel(respond, model_name="capture-fixture"), name="complete_agent",
                  model_settings={"temperature": 0.25, "max_tokens": 123})

    @agent.tool_plain
    def lookup(query: str) -> str:
        assert query == "Public CANARY_boundary"
        return "Public tool result CANARY_boundary"

    with provider.get_tracer("test").start_as_current_span("workflow") as parent:
        parent.set_attributes({"petri.workflow_id": "workflow-exact", "petri.tokens_input": 72,
                               "petri.tokens_output": 16, "petri.cell_id": "cell-exact"})
        result = await agent.run(prompt)
    rows, local, remote = serialized(elevated_sinks)
    assert len(prompt.encode()) > 208_296
    assert result.output == "Public output CANARY_boundary"
    assert received == snapshots
    from dataclasses import replace

    # FunctionModel/Agent annotate these four fields after the callback returns.
    for response, snapshot in zip(responses, response_copies, strict=True):
        assert replace(response, model_name=snapshot.model_name, model_id=snapshot.model_id,
                       run_id=snapshot.run_id, conversation_id=snapshot.conversation_id) == snapshot
    assert settings == [{"temperature": 0.25, "max_tokens": 123}] * 2
    assert [part.content for msg in received[0] for part in msg.parts
            if isinstance(part, UserPromptPart)] == [prompt]
    assert "CANARY_" not in local + remote
    remotes = {span.span_id.hex(): span for span in _remote_spans(elevated_sinks)}
    spans = {f"{span.context.span_id:016x}": span for span in memory.get_finished_spans()}
    assert set(remotes) == set(spans) == {row["span_id"] for row in rows}
    model_rows = [row for row in rows if row["attributes"].get("gen_ai.response.id")]
    assert {row["attributes"]["gen_ai.response.id"] for row in model_rows} == {"response-1", "response-2"}
    for row in rows:
        attrs = row["attributes"]
        remote_span, original = remotes[row["span_id"]], spans[row["span_id"]]
        assert remote_span.trace_id.hex() == row["trace_id"]
        assert (remote_span.parent_span_id.hex() or None) == row["parent_span_id"]
        assert remote_span.start_time_unix_nano == row["start_time_ns"] == original.start_time
        assert remote_span.end_time_unix_nano == row["end_time_ns"] == original.end_time
        assert row["name"] == remote_span.name
        assert not any("bounded" in key or "dropped" in key for key in attrs)
        remote_attrs = _remote_attributes(remote_span)
        for key, value in attrs.items():
            if key != "petri.telemetry.events":
                assert remote_attrs[key] == value
            if key in {"gen_ai.input.messages", "gen_ai.output.messages", "all_messages"}:
                json.loads(value)
        if "gen_ai.input.messages" in attrs:
            inputs = json.loads(attrs["gen_ai.input.messages"])
            assert public_prompt in json.dumps(inputs)
    assert "Public tool result [REDACTED]" in local
    assert {row["attributes"]["gen_ai.usage.input_tokens"] for row in model_rows} == {31, 41}
    assert {row["attributes"]["gen_ai.usage.output_tokens"] for row in model_rows} == {7, 9}
    assert {row["attributes"]["gen_ai.usage.cache_read.input_tokens"] for row in model_rows} == {3, 5}
    usage = aggregate_usage(elevated_sinks[3] / "petri-dishes/fixture/petri.sqlite", group_by="phase")
    assert sum(item["tokens_input"] or 0 for item in usage) == 72


def test_elevated_redacts_canaries_across_boundaries_and_span_fields(elevated_sinks):
    from opentelemetry.trace import Link, SpanContext, TraceFlags

    provider = elevated_sinks[0]
    raw = "CANARY_boundary " + "a" * 65_520 + " CANARY_boundary "
    raw += "b" * (1_048_500 - len(raw)) + " CANARY_boundary end"
    nested = json.dumps(json.dumps({"public": "keep", "password": 'CANARY_"雪\\tail'}))
    tracer = provider.get_tracer("Public CANARY_boundary", "CANARY_boundary", attributes={
        "public": "scope CANARY_boundary", "api_key": "CANARY_scope"})
    link = Link(SpanContext(123, 456, False, TraceFlags(1)), {"public": "link CANARY_boundary"})
    with tracer.start_as_current_span("Public CANARY_boundary", links=[link]) as span:
        span.set_attributes({"long": raw, "nested": nested,
                             "CANARY_boundary-key": "Public value",
                             "url": "https://user:CANARY_user@example.org?api_key=CANARY_url&keep=42"})
        span.add_event("Public CANARY_boundary", {"api_key": "CANARY_event", "keep": "yes"})
        span.set_status(Status(StatusCode.ERROR, "Public CANARY_boundary"))
    rows, local, remote = serialized(elevated_sinks)
    assert "CANARY_" not in local + remote
    attrs = rows[0]["attributes"]
    assert attrs["long"] == raw.replace("CANARY_boundary", "[REDACTED]")
    assert not any("bounded" in key or "dropped" in key for key in attrs)
    assert json.loads(json.loads(attrs["nested"])) == {"public": "keep", "password": "[REDACTED]"}
    assert "keep=42" in attrs["url"]
    assert _remote_attributes(_remote_spans(elevated_sinks)[0])["long"] == attrs["long"]


@pytest.mark.parametrize("elevated_sinks", [{"max_text_chars": 100}], indirect=True)
def test_elevated_text_exact_boundary_and_structured_overflow(elevated_sinks):
    provider = elevated_sinks[0]
    with provider.get_tracer("test").start_as_current_span("boundary") as span:
        span.set_attributes({"exact": "x" * 100, "over": "y" * 101,
                             "structured": json.dumps({"text": "z" * 100})})
    rows, local, remote = serialized(elevated_sinks)
    attrs = rows[0]["attributes"]
    assert attrs["exact"] == "x" * 100
    assert attrs["over"] == "y" * 100 + "[TRUNCATED]"
    assert json.loads(attrs["structured"]) == "[TRUNCATED: text limit]"
    assert attrs["petri.telemetry.bounded_values"] == 2
    assert "[TRUNCATED]" in local + remote


def test_elevated_collection_and_depth_boundaries(elevated_sinks):
    provider = elevated_sinks[0]
    within = {"public": list(range(512))}
    deep = "public"
    for _ in range(40):
        deep = [deep]
    with provider.get_tracer("test").start_as_current_span("collections") as span:
        span.set_attributes({"exact": json.dumps(within), "over": json.dumps(list(range(513))),
                             "deep": json.dumps(deep)})
    rows, _, _ = serialized(elevated_sinks)
    for attrs in (rows[0]["attributes"], _remote_attributes(_remote_spans(elevated_sinks)[0])):
        assert json.loads(attrs["exact"]) == within
        assert json.loads(attrs["over"]) == list(range(512))
        assert json.loads(attrs["deep"]) == deep
        assert attrs["petri.telemetry.bounded_values"] == 1


@pytest.mark.parametrize("policy,attributes", [
    ({"max_nodes": 100}, {"wide": json.dumps(list(range(100)))}),
    ({"max_input_bytes": 100}, {"one": "a" * 60, "two": "b" * 60}),
    ({"max_span_bytes": 1000}, {"one": "a" * 600, "two": "b" * 600}),
    ({}, {"one": "a" * 1_048_576, "two": "b" * 1_048_576,
           "three": "c" * 1_048_576, "four": "d" * 1_048_576}),
    ({}, {"deep": "[" * 65 + "0" + "]" * 65}),
])
def test_elevated_aggregate_overflow_drops_before_both_sinks(elevated_sinks, caplog, policy, attributes):
    provider, memory, payloads, directory, cfg = elevated_sinks
    # Each case owns a distinct provider, rather than mutating an active policy.
    cfg = TelemetryConfig(**{**cfg.model_dump(), "capture_limits": policy})
    provider = init_telemetry(cfg, dish_id="fixture", petri_dir=directory)
    provider.add_span_processor(SimpleSpanProcessor(memory))
    with provider.get_tracer("test").start_as_current_span("over-budget") as span:
        span.set_attributes(attributes)
    assert provider.force_flush()
    assert not memory.get_finished_spans() and payloads == []
    assert query_spans(directory / "petri-dishes/fixture/petri.sqlite") == []
    assert "span dropped during redaction (CaptureBudgetExceeded)" in caplog.text
    assert "aaaa" not in caplog.text


def test_elevated_budget_boundaries_and_json_preflight(monkeypatch):
    from petri.telemetry.config import CaptureLimits
    from petri.telemetry.redaction import CaptureBudgetExceeded, Redactor, _Budget

    budget = _Budget(CaptureLimits(max_nodes=2, max_input_bytes=4, max_span_bytes=5))
    budget.visit()
    budget.visit()
    with pytest.raises(CaptureBudgetExceeded, match="node"):
        budget.visit()
    budget.text("雪a")
    assert budget.input_bytes == 4
    with pytest.raises(CaptureBudgetExceeded, match="input"):
        budget.text("b")
    assert budget.serialize("abc") == '"abc"'
    with pytest.raises(CaptureBudgetExceeded, match="serialized"):
        budget.serialize("abcd")
    redactor = Redactor(limits=CaptureLimits(max_nodes=10))
    with patch("petri.telemetry.redaction.json.loads", side_effect=AssertionError("parsed")):
        with pytest.raises(CaptureBudgetExceeded, match="decoded node"):
            redactor.clean("[" + "0," * 100_000 + "0]", {"bounded": 0, "redacted": 0,
                           "_budget": _Budget(redactor.limits)})
    for depth, bounded in ((64, 0), (65, 1)):
        counts = {"bounded": 0, "redacted": 0, "_budget": _Budget(CaptureLimits())}
        Redactor(limits=CaptureLimits()).clean("public", counts, depth=depth)
        assert counts["bounded"] == bounded


def test_elevated_sdk_count_and_length_safeguards(elevated_sinks, monkeypatch):
    provider = elevated_sinks[0]
    with provider.get_tracer("test").start_as_current_span("sdk-counts") as span:
        for index in range(140):
            span.set_attribute(f"attr-{index}", index)
            span.add_event("event", {"index": index})
    rows, _, _ = serialized(elevated_sinks)
    attrs = rows[0]["attributes"]
    assert attrs["petri.telemetry.dropped_attributes"] == 12
    assert attrs["petri.telemetry.dropped_events"] == 12
    monkeypatch.setenv("OTEL_ATTRIBUTE_VALUE_LENGTH_LIMIT", "20")
    cfg = elevated_sinks[4]
    other = init_telemetry(cfg, dish_id="length", petri_dir=elevated_sinks[3])
    with other.get_tracer("test").start_as_current_span("sdk-length") as span:
        span.set_attribute("value", "x" * 15 + "CANARY_boundary")
    assert other.force_flush()
    rows = query_spans(elevated_sinks[3] / "petri-dishes/length/petri.sqlite")
    assert rows[0]["attributes"]["value"] == "[TRUNCATED: SDK length limit]"
    remote = next(span for span in _remote_spans(elevated_sinks) if span.name == "sdk-length")
    assert _remote_attributes(remote)["value"] == "[TRUNCATED: SDK length limit]"


@pytest.mark.parametrize("field,exact", [("max_nodes", 6), ("max_input_bytes", 3), ("max_span_bytes", None)])
def test_elevated_whole_span_aggregate_exact_and_one_over(elevated_sinks, field, exact, caplog):
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import ReadableSpan
    from opentelemetry.trace import SpanContext, TraceFlags

    from petri.telemetry.redaction import Redactor

    raw = ReadableSpan(
        name="n", context=SpanContext(123, 456, False, TraceFlags(1)),
        resource=Resource({}), attributes={"a": "x"}, start_time=1, end_time=2,
    )
    # Six content visits: attributes, key, value, resource, schema URL and name.
    # The key, value and name each contribute one input byte.
    if exact is None:
        exact = len(json.dumps(Redactor._payload(raw)).encode())
    memory, payloads, directory = elevated_sinks[1:4]
    for limit in (exact, exact - 1):
        cfg = TelemetryConfig(capture_content=True, capture_limits={field: limit},
                              otlp=elevated_sinks[4].otlp)
        provider = init_telemetry(cfg, dish_id=f"{field}-{limit}", petri_dir=directory)
        provider.add_span_processor(SimpleSpanProcessor(memory))
        provider._active_span_processor.on_end(raw)
        assert provider.force_flush()
        rows = query_spans(directory / f"petri-dishes/{field}-{limit}/petri.sqlite")
        assert len(rows) == (1 if limit == exact else 0)
    assert len(memory.get_finished_spans()) == 1
    assert len(_remote_spans(elevated_sinks)) == 1
    assert payloads and "CaptureBudgetExceeded" in caplog.text


def test_elevated_encoded_depth_boundary_is_visible_in_both_sinks(elevated_sinks):
    values = []
    for levels in (62, 63):
        value = "Public"
        for _ in range(levels):
            value = [value]
        values.append(value)
    with elevated_sinks[0].get_tracer("test").start_as_current_span("depth-boundary") as span:
        span.set_attributes({"exact": json.dumps(values[0]), "over": json.dumps(values[1])})
    rows, _, _ = serialized(elevated_sinks)
    for attrs in (rows[0]["attributes"], _remote_attributes(_remote_spans(elevated_sinks)[0])):
        assert json.loads(attrs["exact"]) == values[0]
        over = json.loads(attrs["over"])
        for _ in range(63):
            over = over[0]
        assert over == "[TRUNCATED: nesting limit]"
        assert attrs["petri.telemetry.bounded_values"] == 1


def test_elevated_resource_and_scope_canaries_never_reach_either_sink(elevated_sinks):
    cfg = elevated_sinks[4].model_copy(update={"service_name": "Public CANARY_boundary"})
    provider = init_telemetry(cfg, dish_id="resource", petri_dir=elevated_sinks[3],
                              protected_secret_values=("CANARY_boundary",))
    tracer = provider.get_tracer("Public CANARY_boundary", "CANARY_boundary",
                                 schema_url="https://example.org/CANARY_boundary",
                                 attributes={"api_key": "CANARY_scope", "public": "Public CANARY_boundary"})
    with tracer.start_as_current_span("resource-canaries"):
        pass
    assert provider.force_flush()
    rows = query_spans(elevated_sinks[3] / "petri-dishes/resource/petri.sqlite")
    assert rows[0]["resource"]["service.name"] == "Public [REDACTED]"
    assert "CANARY_" not in json.dumps(rows) + str(elevated_sinks[2])
    assert _remote_spans(elevated_sinks)[0].name == "resource-canaries"


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", ["wide", "deep"])
async def test_elevated_function_model_structures_exceed_legacy_bounds(elevated_sinks, shape):
    from copy import deepcopy

    value = list(range(200)) if shape == "wide" else "Public CANARY_boundary"
    if shape == "deep":
        for _ in range(40):
            value = [value]
    prompt = json.dumps(value)
    expected = prompt.replace("CANARY_boundary", "[REDACTED]")
    received, snapshots = [], []

    def respond(messages, info):
        received.append(messages)
        snapshots.append(deepcopy(messages))
        return ModelResponse(parts=[TextPart(prompt)], provider_response_id=f"response-{shape}")

    result = await Agent(FunctionModel(respond), name=f"structured_{shape}").run(prompt)
    rows, _, _ = serialized(elevated_sinks)
    assert received == snapshots and result.output == prompt
    remote = {span.span_id.hex(): _remote_attributes(span) for span in _remote_spans(elevated_sinks)}
    found_input = found_output = False
    for row in rows:
        attrs = row["attributes"]
        assert not any("bounded" in key or "dropped" in key for key in attrs)
        for key in ("gen_ai.input.messages", "gen_ai.output.messages"):
            if key not in attrs:
                continue
            assert attrs[key] == remote[row["span_id"]][key]
            messages = json.loads(attrs[key])
            texts = [part["content"] for message in messages for part in message["parts"]
                     if part.get("type") == "text"]
            assert expected in texts
            for text in texts:
                assert json.loads(text) == json.loads(expected)
            found_input |= key == "gen_ai.input.messages"
            found_output |= key == "gen_ai.output.messages"
    assert found_input and found_output
