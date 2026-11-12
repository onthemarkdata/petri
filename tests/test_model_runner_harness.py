"""Keep optional model readiness checks fail-closed without running inference."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "model_runner_harness", Path(__file__).parents[1] / "scripts/e2e/model_runner.py"
)
harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness)


@pytest.fixture
def client():
    return harness.ModelRunner(
        "http://127.0.0.1:12434",
        {
            "model": "test/gemma:Q4_0",
            "dmr_id": "sha256:expected",
            "context_size": 32768,
        },
    )


def test_changed_artifact_is_rejected(client, monkeypatch):
    monkeypatch.setattr(client, "request", lambda *args: {"id": "sha256:changed"})
    with pytest.raises(RuntimeError, match="digest differs"):
        client.verify()


@pytest.mark.parametrize("endpoint", ["https://api.example.com", "http://example.com"])
def test_remote_inference_is_rejected(endpoint):
    with pytest.raises(ValueError, match="local HTTP"):
        harness.ModelRunner(endpoint, {})


def test_truncated_response_cannot_pass(client, monkeypatch):
    monkeypatch.setattr(
        client,
        "request",
        lambda *args: {
            "choices": [{"finish_reason": "length", "message": {"content": "READY"}}],
        },
    )
    with pytest.raises(RuntimeError, match="Incomplete"):
        client.complete([])


def test_download_does_not_count_as_readiness(client, monkeypatch):
    monkeypatch.setattr(client, "verify", lambda: None)
    monkeypatch.setattr(client, "complete", lambda *args: {"message": {"content": "Loading"}})
    with pytest.raises(RuntimeError, match="exactly READY"):
        client.check(1)


def test_pi_config_preserves_existing_file(client, tmp_path):
    path = tmp_path / "models.json"
    path.write_text("existing settings")
    with pytest.raises(FileExistsError):
        client.pi_config(tmp_path)
    assert path.read_text() == "existing settings"


@pytest.fixture
def endpoint():
    from scripts.e2e.local_endpoint import LocalEndpoint

    return LocalEndpoint(
        "http://127.0.0.1:8000/gateway/openai/v1", "gateway-alias", 32768, 4096,
        {"artifact": "test/model", "revision": "declared-revision", "provider": "test-server"},
        no_auth=True,
    )


@pytest.fixture
def worker():
    worker_spec = importlib.util.spec_from_file_location(
        "local_grow_worker", Path(__file__).parent / "helpers/local_grow_worker.py"
    )
    module = importlib.util.module_from_spec(worker_spec)
    worker_spec.loader.exec_module(module)
    return module


def completion(model="gateway-alias", content="READY", finish_reason="stop", **message):
    return {"model": model, "choices": [{"finish_reason": finish_reason,
            "message": {"role": "assistant", "content": content, **message}}]}


def mock_open(monkeypatch, endpoint, data, status=200):
    import io
    from types import SimpleNamespace

    requests = []

    class Response(io.BytesIO):
        def __init__(self, body):
            super().__init__(body)
            self.headers = {"Content-Type": "application/json"}

    def open_request(request, **kwargs):
        requests.append(request)
        response = Response(harness.json.dumps(data).encode())
        response.status = status
        return response

    monkeypatch.setattr(endpoint, "opener", SimpleNamespace(open=open_request))
    return requests


def test_generic_prefix_and_identity_do_not_require_dmr(endpoint, monkeypatch):
    requests = mock_open(monkeypatch, endpoint, completion())
    runner = harness.ModelRunner(endpoint=endpoint)
    assert runner.verify()["verification"].startswith("declared")
    assert requests == []
    assert runner.complete([])["response_model"] == "gateway-alias"
    assert requests[0].full_url == "http://127.0.0.1:8000/gateway/openai/v1/chat/completions"
    assert harness.json.loads(requests[0].data)["model"] == "gateway-alias"
    assert "dmr_id" not in endpoint.identity()["provenance"]
    with pytest.raises(ValueError, match="only for the DMR"):
        runner.configure()


@pytest.mark.parametrize("url", [
    "http://user:password@127.0.0.1:8000/v1", "http://localhost@evil.example/v1",
    "http://127.0.0.1.evil.example/v1", "https://127.0.0.1/v1", "http://0.0.0.0/v1",
    "http://127.0.0.1/v1?token=hidden", "http://127.0.0.1/v1#fragment",
    "http://127.0.0.1:bad/v1", "http://127.0.0.1:0/v1", "http://[::]/v1",
    "http://127.0.0.1/v1/../remote", "http://127.0.0.1/v1/%2e%2e/remote",
    "http://127.0.0.1\\@evil.example/v1", "http://127.0.0.1/\nremote",
])
def test_unsafe_endpoints_are_rejected_without_echoing_input(url):
    from scripts.e2e.local_endpoint import loopback_url

    with pytest.raises(ValueError) as error:
        loopback_url(url)
    assert url not in str(error.value)
    assert "password" not in str(error.value)


@pytest.mark.parametrize("url", ["http://localhost:8000/v1", "http://127.0.0.2/v1", "http://[::1]:8000/v1"])
def test_loopback_endpoints(url):
    from scripts.e2e.local_endpoint import loopback_url

    assert loopback_url(url + "/") == url


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_redirects_never_reach_a_remote_endpoint(endpoint, monkeypatch, status):
    from scripts.e2e.local_endpoint import NoRedirect

    with pytest.raises(RuntimeError, match="redirects are disabled"):
        NoRedirect().redirect_request(None, None, status, "redirect", {}, "https://remote.example")
    mock_open(monkeypatch, endpoint, {}, status)
    with pytest.raises(RuntimeError, match="redirects are disabled"):
        endpoint.request("/chat/completions", {})


def test_dmr_adapter_keeps_digest_and_api_paths(client, monkeypatch):
    requests = mock_open(monkeypatch, client.endpoint, {"id": "sha256:expected"})
    assert client.verify()["id"] == "sha256:expected"
    assert requests[0].full_url == "http://127.0.0.1:12434/models/test/gemma:Q4_0"
    mock_open(monkeypatch, client.endpoint, completion(model=client.model))
    assert client.complete([])["response_model"] == client.model


@pytest.mark.parametrize("model", [None, "different-model"])
def test_missing_or_substituted_completion_identity_fails(endpoint, monkeypatch, model):
    mock_open(monkeypatch, endpoint, completion(model=model))
    with pytest.raises(RuntimeError, match="identity is missing or differs"):
        harness.ModelRunner(endpoint=endpoint).complete([])


def test_declared_response_alias_is_separate_from_artifact(monkeypatch):
    from scripts.e2e.local_endpoint import LocalEndpoint

    endpoint = LocalEndpoint("http://127.0.0.1:8000/v1", "gateway-alias", 32768, 4096,
                             {"artifact": "test/weights", "revision": "revision",
                              "response_model": "upstream-served-id"}, no_auth=True)
    mock_open(monkeypatch, endpoint, completion(model="upstream-served-id"))
    assert harness.ModelRunner(endpoint=endpoint).complete([])["response_model"] == "upstream-served-id"
    assert endpoint.identity()["provenance"]["artifact"] == "test/weights"
    assert "dmr_id" not in endpoint.identity()["provenance"]


def test_capabilities_retain_failure_and_replay_outcome(endpoint, monkeypatch):
    responses = iter([
        completion(), completion(content='{"status":"ready","count":3}'),
        completion(content=None, finish_reason="tool_calls", tool_calls=[{
            "id": "call-1", "type": "function", "function": {
                "name": "lookup_source", "arguments": '{"source_id":"wrong"}',
            },
        }]),
    ])
    monkeypatch.setattr(endpoint, "request", lambda *args, **kwargs: next(responses))
    runner = harness.ModelRunner(endpoint=endpoint)
    with pytest.raises(RuntimeError, match="Tool arguments"):
        runner.check(1)
    assert [record["outcome"] for record in runner.records] == ["passed", "passed", "failed", "blocked"]
    assert runner.records[2]["response"]["choices"][0]["message"]["tool_calls"]


def test_schema_failure_remains_visible_when_tools_pass(endpoint, monkeypatch):
    responses = iter([
        completion(), completion(content="schema unsupported"),
        completion(content=None, finish_reason="tool_calls", tool_calls=[{
            "id": "call-1", "type": "function", "function": {
                "name": "lookup_source", "arguments": '{"source_id":"alpha"}',
            },
        }]), completion(content="PETRI42"),
    ])
    monkeypatch.setattr(endpoint, "request", lambda *args, **kwargs: next(responses))
    runner = harness.ModelRunner(endpoint=endpoint)
    with pytest.raises(RuntimeError):
        runner.check(1)
    assert [record["outcome"] for record in runner.records] == ["passed", "failed", "passed", "passed"]


def test_all_readiness_capabilities_must_pass(endpoint, monkeypatch):
    responses = iter([
        completion(), completion(content='{"status":"ready","count":3}'),
        completion(content=None, finish_reason="tool_calls", tool_calls=[{
            "id": "call-1", "type": "function", "function": {
                "name": "lookup_source", "arguments": '{"source_id":"alpha"}',
            },
        }]), completion(content="PETRI42"),
    ])
    monkeypatch.setattr(endpoint, "request", lambda *args, **kwargs: next(responses))
    assert all(record["outcome"] == "passed" for record in harness.ModelRunner(endpoint=endpoint).check(1))


def test_basic_auth_is_retained_only_in_memory(tmp_path, monkeypatch):
    import base64

    from scripts.e2e.local_endpoint import LocalEndpoint

    credentials = tmp_path / "credentials.toml"
    credentials.write_text('username = "synthetic-user"\npassword = "synthetic-secret"\n')
    credentials.chmod(0o600)
    endpoint = LocalEndpoint("http://127.0.0.1:8000/prefix/v1", "model", 32768, 4096,
                             {"artifact": "test/model", "revision": "revision"},
                             basic_auth_file=credentials)
    requests = mock_open(monkeypatch, endpoint, {"ok": True})
    endpoint.request("/chat/completions", {})
    encoded = base64.b64encode(b"synthetic-user:synthetic-secret").decode()
    assert requests[0].get_header("Authorization") == "Basic " + encoded
    endpoint.pi_config(tmp_path / "agent")
    text = (tmp_path / "agent/models.json").read_text() + harness.json.dumps(endpoint.identity())
    for secret in ("synthetic-secret", "synthetic-user", encoded):
        assert secret not in text
        if secret != "synthetic-user":
            assert secret not in endpoint.redact("echoed " + secret)
    assert "synthetic-user" not in endpoint.redact('{"username":"synthetic-user"}')
    assert "PETRI_LOCAL_AUTHORIZATION" in text
    credentials.chmod(0o644)
    with pytest.raises(ValueError, match="private Basic auth"):
        LocalEndpoint("http://127.0.0.1/v1", "model", 32768, 4096,
                      {"artifact": "test/model", "revision": "revision"}, basic_auth_file=credentials)


def test_bearer_environment_and_pi_configuration(endpoint, tmp_path, monkeypatch):
    from scripts.e2e.local_endpoint import LocalEndpoint, environment_values

    monkeypatch.setenv("ROUTE_KEY", "synthetic-key")
    endpoint = LocalEndpoint(endpoint.base_url, endpoint.model, 32768, 4096,
                             {"artifact": "test/model", "revision": "revision"}, api_key_env="ROUTE_KEY")
    requests = mock_open(monkeypatch, endpoint, {})
    endpoint.request("/chat/completions", {})
    assert requests[0].get_header("Authorization") == "Bearer synthetic-key"
    document = endpoint.pi_document()
    provider = document["providers"]["petri-local"]
    assert provider["apiKey"] == "ROUTE_KEY"
    assert provider["baseUrl"] == endpoint.base_url
    assert provider["models"][0]["id"] == endpoint.model
    assert provider["models"][0]["contextWindow"] == 32768
    assert provider["models"][0]["maxTokens"] == 4096
    assert "synthetic-key" not in harness.json.dumps(document)
    monkeypatch.setenv("OPENAI_API_KEY", "original-key")
    with pytest.raises(RuntimeError), environment_values({"OPENAI_API_KEY": "temporary-key"}):
        raise RuntimeError
    assert harness.os.environ["OPENAI_API_KEY"] == "original-key"


@pytest.mark.parametrize("field,value", [
    ("model", "other"), ("harness", "pi"), ("case", "shared"), ("configuration_hash", "changed"),
    ("base_url", "http://127.0.0.1:9000/v1"), ("max_tokens", 8192),
])
def test_resume_rejects_identity_changes(tmp_path, field, value):
    from scripts.e2e.local_endpoint import save_identity

    path = tmp_path / "run-identity.json"
    identity = {"model": "original", "harness": "provider", "case": "isolated",
                "configuration_hash": "original", "base_url": "http://127.0.0.1:8000/v1", "max_tokens": 4096}
    save_identity(path, identity, resume=False)
    save_identity(path, identity, resume=True)
    with pytest.raises(ValueError, match="Saved run identity differs"):
        save_identity(path, {**identity, field: value}, resume=True)
    assert harness.json.loads(path.read_text()) == identity


def test_pi_config_drift_is_rejected(worker, endpoint, tmp_path):
    endpoint.pi_config(tmp_path)
    path = tmp_path / "models.json"
    worker.validate_pi_config(path, endpoint)
    original = path.read_text()
    path.write_text(original.replace("gateway-alias", "different-model"))
    with pytest.raises(ValueError, match="configuration differs"):
        worker.validate_pi_config(path, endpoint)


def research_event(worker, kind, cell="grow-live-isolated-000-000", round_number=1, **data):
    if kind == "source_reviewed":
        payload = {"retrieval_method": "public-web", "url": "https://example.org/source",
                   "retrieved_at": "2026-11-11T00:00:00Z", "content_sha256": "a" * 64,
                   "finding": "retrieved source content", **data}
        identity = payload["url"] + ":" + payload["content_sha256"]
    else:
        payload = {"query": "test query", "sources_found": 1,
                   "source_urls": ["https://example.org/source"], **data}
        identity = worker.hashlib.sha256(harness.json.dumps(
            [payload["query"], payload["source_urls"]], sort_keys=True
        ).encode()).hexdigest()
    return {"id": worker.database.stable_event_id(cell, "public-web", round_number, "research",
            "investigator", 0, kind, identity), "cell_id": cell, "type": kind,
            "agent": "investigator", "iteration": 0, "data": payload}


@pytest.mark.parametrize("seed_ok,grow_exit", [(False, 0), (True, 1), (True, None)])
def test_nonzero_or_missing_commands_cannot_pass(worker, seed_ok, grow_exit):
    cell = "grow-live-isolated-000-000"
    events = [research_event(worker, kind) for kind in ("source_reviewed", "search_executed")]
    summary = worker.diagnostic_summary(events, [{"id": cell, "status": "VALIDATED"}], cell, 1,
                                        {}, seed_ok=seed_ok, grow_exit=grow_exit)
    assert not worker.diagnostic_passed(summary, "VALIDATED")


def test_other_colony_and_old_round_cannot_satisfy_run(worker):
    cell = "grow-live-isolated-000-000"
    events = [research_event(worker, kind, cell=other, round_number=round_number)
              for other, round_number in [(cell, 1), ("grow-live-shared-000-000", 2)]
              for kind in ("source_reviewed", "search_executed")]
    summary = worker.diagnostic_summary(events, [{"id": "other-root", "status": "VALIDATED"},
                                        {"id": cell, "status": "NEW"}], cell, 2, {}, seed_ok=True, grow_exit=0)
    assert summary["center_status"] == "NEW"
    assert summary["sources"] == summary["searches"] == []
    assert not worker.diagnostic_passed(summary, "VALIDATED")


def test_other_cell_reopen_cannot_hide_failure(worker):
    cell = "grow-live-isolated-000-000"
    failure = {"id": "failed", "cell_id": cell, "type": "agent_execution_failed",
               "agent": "investigator", "iteration": 0, "data": {"message": "failed"}}
    events = [failure, {"cell_id": "other", "type": "cell_reopened"},
              research_event(worker, "source_reviewed"), research_event(worker, "search_executed")]
    summary = worker.diagnostic_summary(events, [{"id": cell, "status": "VALIDATED"}], cell, 1,
                                        {}, seed_ok=True, grow_exit=0)
    assert summary["current_round_failure_count"] == 1
    assert summary["failures"] == [failure]
    assert not worker.diagnostic_passed(summary, "VALIDATED")


def test_failed_history_is_retained_after_selected_cell_reopens(worker):
    cell = "grow-live-isolated-000-000"
    failure = {"id": "old-failure", "cell_id": cell, "type": "agent_execution_failed",
               "agent": "investigator", "iteration": 0, "data": {"message": "old failure"}}
    events = [failure, {"cell_id": cell, "type": "cell_reopened"},
              research_event(worker, "source_reviewed", round_number=2),
              research_event(worker, "search_executed", round_number=2)]
    summary = worker.diagnostic_summary(events, [{"id": cell, "status": "VALIDATED"}], cell, 2,
                                        {}, seed_ok=True, grow_exit=0)
    assert summary["failures"] == [failure]
    assert summary["current_round_failure_count"] == 0
    assert worker.diagnostic_passed(summary, "VALIDATED")


@pytest.mark.parametrize("changes", [{"sources_found": 0, "source_urls": []}, {"finding": ""},
                                      {"content_sha256": ""}, {"retrieved_at": ""}])
def test_empty_search_or_incomplete_source_cannot_pass(worker, changes):
    cell = "grow-live-isolated-000-000"
    events = [research_event(worker, kind, **changes) for kind in ("source_reviewed", "search_executed")]
    summary = worker.diagnostic_summary(events, [{"id": cell, "status": "VALIDATED"}], cell, 1,
                                        {}, seed_ok=True, grow_exit=0)
    assert not worker.diagnostic_passed(summary, "VALIDATED")


def test_guarded_route_forwards_only_selected_model_and_never_auth_artifacts(endpoint, monkeypatch):
    import io
    from types import SimpleNamespace

    import scripts.e2e.local_endpoint as module

    servers = []

    class Server:
        server_port = 34567
        daemon_threads = False

        def __init__(self, address, handler):
            self.handler = handler
            servers.append(self)

        def serve_forever(self):
            pass

        def shutdown(self):
            pass

        def server_close(self):
            pass

    monkeypatch.setattr(module, "ThreadingHTTPServer", Server)
    forwarded = []

    def raw_request(url, payload=None, **kwargs):
        forwarded.append((url, harness.json.loads(payload)))
        return 200, "application/json", harness.json.dumps(completion()).encode()

    monkeypatch.setattr(endpoint, "raw_request", raw_request)
    with endpoint.guarded_route() as (base, token):
        assert base == "http://127.0.0.1:34567/gateway/openai/v1"
        assert token not in harness.json.dumps(endpoint.pi_document(base))
        assert token not in endpoint.redact("Bearer " + token)

        def request(path, model="gateway-alias", key=token, max_tokens=4096):
            handler = object.__new__(servers[0].handler)
            handler.headers = {"Authorization": "Bearer " + key}
            handler.path = path
            body = harness.json.dumps({"model": model, "max_tokens": max_tokens}).encode()
            handler.headers["Content-Length"] = str(len(body))
            handler.rfile, handler.wfile = io.BytesIO(body), io.BytesIO()
            result = SimpleNamespace(status=None)
            handler.send_response = lambda status: setattr(result, "status", status)
            handler.send_error = lambda status: setattr(result, "status", status)
            handler.send_header = lambda *args: None
            handler.end_headers = lambda: None
            handler.do_POST()
            return result.status, handler.wfile.getvalue()

        route = "/gateway/openai/v1/chat/completions"
        assert request(route)[0] == 200
        assert forwarded[0][0] == endpoint.base_url + "/chat/completions"
        assert request(route, model="wrong")[0] == 502
        assert request(route, max_tokens=8192)[0] == 502
        assert request(route, key="wrong")[0] == 403
        assert request("/v1/chat/completions")[0] == 404
        assert len(forwarded) == 1
        monkeypatch.setattr(endpoint, "raw_request", lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("Local endpoint redirects are disabled.")))
        status, body = request(route)
        assert status == 502
        assert "redirects are disabled" in body.decode()


def test_streamed_response_identity_is_checked(endpoint):
    endpoint.validate_response_identity(
        b'data: {"model":"gateway-alias","choices":[{"index":0,"finish_reason":"stop"}]}\n\n'
        b'data: [DONE]\n\n', "text/event-stream"
    )
    with pytest.raises(RuntimeError, match="identity"):
        endpoint.validate_response_identity(sse_body([completion(model="other")]), "text/event-stream")
    with pytest.raises(RuntimeError, match="identity"):
        endpoint.validate_response_identity(b"{}", "application/json")


def test_http_unsupported_capability_is_explicit(endpoint, monkeypatch):
    from scripts.e2e.local_endpoint import EndpointHTTPError

    def respond(path, payload=None, **kwargs):
        if payload.get("response_format"):
            raise EndpointHTTPError(501, "JSON schema is unsupported")
        if payload.get("tool_choice") == "required":
            return completion(content=None, finish_reason="tool_calls", tool_calls=[{
                "id": "call-1", "function": {"name": "lookup_source", "arguments": '{"source_id":"alpha"}'},
            }])
        return completion(content="PETRI42" if payload.get("tools") else "READY")

    monkeypatch.setattr(endpoint, "request", respond)
    runner = harness.ModelRunner(endpoint=endpoint)
    with pytest.raises(RuntimeError, match="unsupported"):
        runner.check(1)
    assert runner.records[1]["outcome"] == "unsupported"
    assert runner.records[1]["http_status"] == 501
    assert runner.records[-1]["outcome"] == "passed"


@pytest.mark.parametrize("failure", ["exit", "length", "missing_model", "wrong_path", "missing_replay",
                                      "missing_reason", "unknown_reason", "toolUse", None])
def test_pi_readiness_checks_command_identity_arguments_and_replay(endpoint, monkeypatch, failure):
    import contextlib
    from types import SimpleNamespace

    @contextlib.contextmanager
    def route():
        yield "http://127.0.0.1:34567/gateway/openai/v1", "temporary-key"

    monkeypatch.setattr(endpoint, "guarded_route", route)
    messages = [
        {"role": "assistant", "model": "gateway-alias", "stopReason": "toolUse", "content": [
            {"type": "toolCall", "name": "read", "id": "read-1", "arguments": {"path": "source.txt"}},
        ]},
        {"role": "toolResult", "toolCallId": "read-1", "isError": False,
         "content": [{"type": "text", "text": "The verification code is PETRI42."}]},
        {"role": "assistant", "model": "gateway-alias", "stopReason": "stop",
         "content": [{"type": "text", "text": "PETRI42"}]},
    ]
    if failure == "length":
        messages[-1]["stopReason"] = "length"
    if failure == "missing_reason":
        messages[-1].pop("stopReason")
    if failure == "unknown_reason":
        messages[-1]["stopReason"] = "unknown"
    if failure == "toolUse":
        messages[-1]["stopReason"] = "toolUse"
    if failure == "missing_model":
        messages[-1].pop("model")
    if failure == "wrong_path":
        messages[0]["content"][0]["arguments"]["path"] = "other.txt"
    if failure == "missing_replay":
        messages.pop(1)

    def run(command, **kwargs):
        document = harness.json.loads((kwargs["cwd"] / "agent/models.json").read_text())
        provider = document["providers"]["petri-local"]
        assert provider["baseUrl"].endswith("/gateway/openai/v1")
        assert provider["apiKey"] == "PETRI_LOCAL_RELAY_KEY"
        assert kwargs["env"]["PETRI_LOCAL_RELAY_KEY"] == "temporary-key"
        assert "temporary-key" not in harness.json.dumps(document)
        return SimpleNamespace(returncode=1 if failure == "exit" else 0, stderr="diagnostic stderr",
                               stdout=harness.json.dumps({"type": "agent_end", "messages": messages}))

    monkeypatch.setattr(harness.subprocess, "run", run)
    runner = harness.ModelRunner(endpoint=endpoint)
    if failure:
        with pytest.raises(RuntimeError):
            runner.pi_check("fake-pi")
        assert runner.records[0]["stderr"] == "diagnostic stderr"
    else:
        assert runner.pi_check("fake-pi")["outcome"] == "passed"


@pytest.mark.parametrize("seed_exit,grow_exit", [(1, 0), (0, 1), (0, 0)])
def test_worker_command_results_and_resume_are_preserved(worker, endpoint, monkeypatch, tmp_path,
                                                         seed_exit, grow_exit):
    import contextlib
    from types import SimpleNamespace

    root = tmp_path / "run"
    cell_id = "grow-live-isolated-000-000"
    monkeypatch.setenv("PETRI_RUN_LOCAL_GROW", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "preserved-upstream-key")
    monkeypatch.setattr(worker, "endpoint_from_args", lambda args: endpoint)
    monkeypatch.setattr(worker, "source_sha", lambda: "source-sha")

    @contextlib.contextmanager
    def route():
        yield "http://127.0.0.1:34567/gateway/openai/v1", "temporary-key"

    monkeypatch.setattr(endpoint, "guarded_route", route)
    commands = []

    def materialize():
        dish = root / ".petri"
        path = worker.dish_db_path(dish, "grow-live")
        center = worker.Cell(id=cell_id, colony_id="grow-live-isolated", claim_text=worker.CASES["isolated"][0],
                             level=0, created_at=worker.now())
        colony = worker.Colony(id="grow-live-isolated", dish="grow-live", center_claim=center.claim_text,
                               center_cell_id=cell_id)
        worker.reserve_colony(path, colony)
        graph = worker.ColonyGraph(colony_id=colony.id)
        graph.add_cell(center)
        worker.save_colony(path, graph, colony)

    def invoke(app, command):
        commands.append(command)
        if command[0] == "seed":
            materialize()
        assert harness.os.environ["OPENAI_API_KEY"] == "temporary-key"
        return SimpleNamespace(exit_code=seed_exit if command[0] == "seed" else grow_exit,
                               output="preserved command output", exception=None)

    monkeypatch.setattr(worker, "CliRunner", lambda: SimpleNamespace(invoke=invoke))
    monkeypatch.setattr(worker.database, "read_events", lambda *args, **kwargs: [
        research_event(worker, "source_reviewed"), research_event(worker, "search_executed"),
    ])
    monkeypatch.setattr(worker.database, "list_cells", lambda *args: [{"id": cell_id, "status": "VALIDATED"}])
    arguments = ["local_grow_worker.py", str(root), "--harness", "provider", "--case", "isolated", "--seed"]
    monkeypatch.setattr(worker.sys, "argv", arguments)
    assert worker.main() == (0 if seed_exit == grow_exit == 0 else 1)
    assert [command[0] for command in commands] == (["seed"] if seed_exit else ["seed", "grow"])
    assert harness.os.environ["OPENAI_API_KEY"] == "preserved-upstream-key"
    report_path = next(root.glob("attempts/*/result.json"))
    first_report = report_path.read_text()
    report = harness.json.loads(first_report)
    assert report["passed"] == (seed_exit == grow_exit == 0)
    assert report["complete_cycle_passed"] is False
    monkeypatch.setattr(worker.sys, "argv", [*arguments[:-1], "--resume"])
    assert worker.main() == (0 if seed_exit == grow_exit == 0 else 1)
    assert report_path.read_text() == first_report
    assert len(list(root.glob("attempts/*/result.json"))) == 2
    # A different requested backend is rejected before another CLI command runs.
    command_count = len(commands)
    changed = list(worker.sys.argv)
    changed[changed.index("provider")] = "pi"
    monkeypatch.setattr(worker.sys, "argv", changed)
    with pytest.raises(SystemExit) as error:
        worker.main()
    assert error.value.code == 2
    assert len(commands) == command_count


@pytest.mark.parametrize("context_size", [131072, 262144])
def test_context_maximum_is_independent_of_output_cap(context_size):
    from scripts.e2e.local_endpoint import LocalEndpoint

    endpoint = LocalEndpoint("http://127.0.0.1:8000/v1", "model", context_size, 4096,
                             {"artifact": "test/model", "revision": "revision",
                              "max_model_len": context_size}, no_auth=True)
    selected = endpoint.pi_document()["providers"]["petri-local"]["models"][0]
    assert selected["contextWindow"] == context_size
    assert selected["maxTokens"] == 4096
    assert endpoint.identity()["context_size"] == context_size


def test_cli_generic_auth_is_explicit_and_manifest_secrets_are_not_copied(tmp_path, monkeypatch):
    import argparse

    from scripts.e2e.local_endpoint import connection_arguments, endpoint_from_args

    manifest = tmp_path / "provenance.json"
    manifest.write_text(harness.json.dumps({"artifact": "test/model", "revision": "revision",
                                          "private_note": "must-not-appear"}))
    parser = argparse.ArgumentParser()
    connection_arguments(parser)
    argv = ["--base-url", "http://127.0.0.1:8000/prefix/v1", "--model", "model",
            "--context-size", "262144", "--provenance", str(manifest)]
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="Select --no-auth"):
        endpoint_from_args(parser.parse_args(argv))
    endpoint = endpoint_from_args(parser.parse_args([*argv, "--no-auth"]))
    assert endpoint.context_size == 262144
    assert "must-not-appear" not in harness.json.dumps(endpoint.identity())


@pytest.fixture
def guarded_client(monkeypatch):
    """Exercise the relay handler without opening a socket or invoking a model."""
    import io
    from contextlib import contextmanager
    from types import SimpleNamespace

    import scripts.e2e.local_endpoint as module

    servers = []

    class Server:
        server_port = 34567

        def __init__(self, address, handler):
            self.handler = handler
            servers.append(self)

        def serve_forever(self):
            pass

        def shutdown(self):
            pass

        def server_close(self):
            pass

    monkeypatch.setattr(module, "ThreadingHTTPServer", Server)

    @contextmanager
    def client(endpoint, **options):
        with endpoint.guarded_route(**options) as (base, token):
            def post(*, headers=None):
                handler = object.__new__(servers[-1].handler)
                payload = harness.json.dumps({"model": endpoint.model}).encode()
                handler.headers = {"Authorization": "Bearer " + token,
                                   "Content-Length": str(len(payload)), **(headers or {})}
                handler.path = "/gateway/openai/v1/chat/completions"
                handler.rfile, handler.wfile = io.BytesIO(payload), io.BytesIO()
                result = SimpleNamespace(status=None, headers={})
                handler.send_response = lambda status: setattr(result, "status", status)
                handler.send_error = lambda status: setattr(result, "status", status)
                handler.send_header = lambda key, value: result.headers.__setitem__(key, value)
                handler.end_headers = lambda: None
                handler.do_POST()
                return result.status, handler.wfile.getvalue()
            post.base_url, post.token = base, token
            yield post

    return client


def credential_endpoint(tmp_path, monkeypatch, secret, *, username="test-user", bearer=False):
    from scripts.e2e.local_endpoint import LocalEndpoint

    options = {}
    if bearer:
        monkeypatch.setenv("SYNTHETIC_ROUTE_KEY", secret)
        options["api_key_env"] = "SYNTHETIC_ROUTE_KEY"
    else:
        credential = tmp_path / "credential.toml"
        credential.write_text(f'username = {harness.json.dumps(username, ensure_ascii=False)}\n'
                              f'password = {harness.json.dumps(secret, ensure_ascii=False)}\n')
        credential.chmod(0o600)
        options["basic_auth_file"] = credential
    return LocalEndpoint("http://127.0.0.1:8000/gateway/openai/v1", "gateway-model-alias", 262144, 4096,
                         {"artifact": "test/model", "revision": "revision"}, **options)


def assert_no_canary(value, secret):
    """Check decoded values, including JSON nested inside diagnostic strings."""
    if isinstance(value, dict):
        for child in value.values():
            assert_no_canary(child, secret)
    elif isinstance(value, list):
        for child in value:
            assert_no_canary(child, secret)
    elif isinstance(value, str):
        assert secret not in value
        try:
            decoded = harness.json.loads(value)
        except ValueError:
            return
        if decoded != value:
            assert_no_canary(decoded, secret)


@pytest.mark.parametrize("secret", ['quoted-"canary', r'backslash-\canary', 'unicode-雪-café'])
@pytest.mark.parametrize("bearer", [False, True])
def test_diagnostic_values_handle_escaped_nested_credentials(tmp_path, monkeypatch, secret, bearer):
    endpoint = credential_endpoint(tmp_path, monkeypatch, secret, bearer=bearer)
    original = {"model": endpoint.model, "content": "ordinary model content remains",
                "nested": [{"error": secret, "arguments": harness.json.dumps({
                    "request": harness.json.dumps({"value": secret, "ordinary": "keep me"}),
                })}]}
    assert_no_canary(harness.json.loads(endpoint.diagnostic_json(original)), secret)
    assert original["nested"][0]["error"] == secret
    assert endpoint.diagnostic(original)["content"] == original["content"]
    text = 'stderr response: ' + harness.json.dumps(original) + '\nordinary diagnostic text'
    safe_text = endpoint.redact(text)
    assert "ordinary diagnostic text" in safe_text
    decoded, _ = harness.json.JSONDecoder().raw_decode(safe_text[len('stderr response: '):])
    assert_no_canary(decoded, secret)


@pytest.mark.parametrize("username", ["model", "content", "gateway-model"])
@pytest.mark.parametrize("streaming", [False, True])
def test_username_substrings_cannot_mutate_transport(tmp_path, monkeypatch, guarded_client,
                                                    username, streaming):
    import io
    from types import SimpleNamespace

    endpoint = credential_endpoint(tmp_path, monkeypatch, "private-password", username=username)
    response = completion(model=endpoint.model, content="The model content is ordinary research text.")
    body = (sse_body([response]) if streaming else harness.json.dumps(response, indent=1).encode())

    class Response(io.BytesIO):
        status = 200

        def __init__(self):
            super().__init__(body)
            self.headers = {"Content-Type": "text/event-stream" if streaming else "application/json"}

    monkeypatch.setattr(endpoint, "opener", SimpleNamespace(open=lambda *args, **kwargs: Response()))
    assert endpoint.raw_request(endpoint.base_url + "/chat/completions", b"{}")[2] == body
    with guarded_client(endpoint) as post:
        status, forwarded = post()
    assert status == 200
    assert forwarded == body
    assert endpoint.diagnostic(response)["model"] == endpoint.model
    assert endpoint.diagnostic(response)["choices"][0]["message"]["content"] == response["choices"][0]["message"]["content"]
    assert_no_canary(endpoint.diagnostic({"username": username}), username)


def sse_body(responses, *, done=True, newline="\n"):
    lines = ["data: " + harness.json.dumps(response) + newline + newline for response in responses]
    if done:
        lines.append("data: [DONE]" + newline + newline)
    return "".join(lines).encode()


@pytest.mark.parametrize("secret", ['quoted-"canary', r'backslash-\canary', 'unicode-雪-café'])
@pytest.mark.parametrize("streaming", [False, True])
def test_credential_echo_is_rejected_with_safe_diagnostic_copy(tmp_path, monkeypatch, guarded_client,
                                                              secret, streaming):
    from scripts.e2e.local_endpoint import ResponseRejected

    endpoint = credential_endpoint(tmp_path, monkeypatch, secret)
    response = completion(model=endpoint.model, content="ordinary diagnostic text " + secret,
                          tool_calls=[{"function": {"arguments": harness.json.dumps({
                              "nested": harness.json.dumps({"value": secret}),
                          })}}])
    original = sse_body([response]) if streaming else harness.json.dumps(response).encode()
    content_type = "text/event-stream" if streaming else "application/json"
    monkeypatch.setattr(endpoint, "raw_request", lambda *args, **kwargs: (200, content_type, original))
    if not streaming:
        with pytest.raises(ResponseRejected, match="echoed protected") as error:
            endpoint.request("/chat/completions", {})
        assert_no_canary(error.value.diagnostic, secret)
    with guarded_client(endpoint) as post:
        status, body = post()
    assert status == 502
    decoded = harness.json.loads(body)
    assert "echoed protected" in decoded["error"]
    assert_no_canary(decoded, secret)
    assert b"ordinary diagnostic text" in body
    assert secret in response["choices"][0]["message"]["content"]


@pytest.mark.parametrize("secret", ['quoted-"canary', r'backslash-\canary', 'unicode-雪-café'])
def test_http_error_diagnostics_decode_credentials(tmp_path, monkeypatch, guarded_client, secret):
    import io
    import urllib.error
    from types import SimpleNamespace

    from scripts.e2e.local_endpoint import EndpointHTTPError

    endpoint = credential_endpoint(tmp_path, monkeypatch, secret, bearer=True)
    original = harness.json.dumps({"error": {"message": secret, "context": harness.json.dumps({
        "secret": secret, "ordinary": "preserve this error detail",
    })}}).encode()

    def failed(request, **kwargs):
        raise urllib.error.HTTPError(request.full_url, 400, "bad request",
                                     {"Content-Type": "application/json"}, io.BytesIO(original))

    monkeypatch.setattr(endpoint, "opener", SimpleNamespace(open=failed))
    assert endpoint.raw_request(endpoint.base_url)[2] == original
    with pytest.raises(EndpointHTTPError) as error:
        endpoint.request("/chat/completions", {})
    assert "preserve this error detail" in str(error.value)
    with guarded_client(endpoint) as post:
        status, body = post()
    assert status == 400
    assert_no_canary(harness.json.loads(body), secret)


@pytest.mark.parametrize("failure", ["partial", "missing_done", "length", "error", "abort", "unknown"])
def test_sse_must_prove_successful_terminal_completion(endpoint, monkeypatch, guarded_client, failure):
    from scripts.e2e.local_endpoint import ResponseRejected

    terminal = None if failure == "partial" else failure if failure != "missing_done" else "stop"
    responses = [{"model": endpoint.model, "choices": [{"index": 0, "delta": {"content": "READY"},
                                                        "finish_reason": terminal}]}]
    if failure == "error":
        responses = [{"error": {"message": "synthetic upstream failure"}}]
    body = sse_body(responses, done=failure not in {"partial", "missing_done"})
    with pytest.raises(ResponseRejected) as error:
        endpoint.validate_response_identity(body, "text/event-stream")
    assert "READY" in error.value.diagnostic or "synthetic upstream failure" in error.value.diagnostic
    monkeypatch.setattr(endpoint, "raw_request", lambda *args, **kwargs: (200, "text/event-stream", body))
    with guarded_client(endpoint) as post:
        status, rejected = post()
    assert status == 502
    assert "diagnostic_response" in harness.json.loads(rejected)


@pytest.mark.parametrize("newline", ["\n", "\r\n", "\r"])
@pytest.mark.parametrize("terminal", ["stop", "tool_calls"])
def test_complete_sse_preserves_usage_and_framing(endpoint, monkeypatch, guarded_client, newline, terminal):
    responses = [{"model": endpoint.model, "choices": [{"index": 0, "delta": {"content": "READY"},
                                                        "finish_reason": None}]},
                 {"model": endpoint.model, "choices": [{"index": 0, "delta": {}, "finish_reason": terminal}]},
                 {"choices": [], "usage": {"completion_tokens": 2}}]
    body = (": keepalive" + newline + "event: message" + newline).encode() + sse_body(responses, newline=newline)
    endpoint.validate_response_identity(body, "text/event-stream; charset=utf-8")
    monkeypatch.setattr(endpoint, "raw_request", lambda *args, **kwargs: (200, "text/event-stream", body))
    with guarded_client(endpoint) as post:
        status, forwarded = post()
    assert status == 200
    assert forwarded == body


def test_relay_correlation_allowlist_and_protected_auth(tmp_path, monkeypatch, guarded_client):
    import io
    from types import SimpleNamespace

    from scripts.e2e.local_endpoint import CORRELATION_HEADERS

    endpoint = credential_endpoint(tmp_path, monkeypatch, "protected-upstream-secret")
    requests = []

    class Response(io.BytesIO):
        status = 200

        def __init__(self):
            super().__init__(harness.json.dumps(completion(model=endpoint.model)).encode())
            self.headers = {"Content-Type": "application/json"}

    def open_request(request, **kwargs):
        requests.append(request)
        return Response()

    monkeypatch.setattr(endpoint, "opener", SimpleNamespace(open=open_request))
    supplied = {name: "caller-identity-" + str(index) for index, name in enumerate(sorted(CORRELATION_HEADERS))}
    supplied.update({"X-Custom-Run": "caller-custom-run", "Cookie": "private-cookie",
                     "X-Api-Key": "client-api-secret", "X-Ignore": "unrelated"})
    with guarded_client(endpoint, correlation_allowlist={"X-Custom-Run", "Authorization", "Cookie", "X-Api-Key"}) as post:
        assert post(headers=supplied)[0] == 200
    upstream = {name.lower(): value for name, value in requests[0].header_items()}
    for name in CORRELATION_HEADERS:
        assert upstream[name] == supplied[name]
    assert upstream["x-custom-run"] == "caller-custom-run"
    assert upstream["authorization"] == endpoint.authorization
    assert not {"cookie", "x-api-key", "x-ignore"} & upstream.keys()
    assert not any("hq" in name for name in CORRELATION_HEADERS)
    with guarded_client(endpoint) as post:
        assert post(headers={"X-Custom-Run": "not-opted-in"})[0] == 200
    assert "x-custom-run" not in {name.lower() for name, _ in requests[-1].header_items()}


@pytest.mark.parametrize("secret", ['quoted-"canary', r'backslash-\canary', 'unicode-雪-café'])
@pytest.mark.parametrize("status", [200, 400])
def test_readiness_saved_artifacts_have_decoded_safe_values(tmp_path, monkeypatch, capsys, secret, status):
    endpoint = credential_endpoint(tmp_path, monkeypatch, secret, bearer=True)
    response = completion(model=endpoint.model, content="ordinary diagnostic text " + secret,
                          tool_calls=[{"function": {"arguments": harness.json.dumps({
                              "nested": harness.json.dumps({"value": secret}),
                          })}}])
    body = harness.json.dumps(response).encode()
    monkeypatch.setattr(endpoint, "raw_request", lambda *args, **kwargs: (status, "application/json", body))
    monkeypatch.setattr(harness, "endpoint_from_args", lambda args: endpoint)
    monkeypatch.setattr(harness, "source_sha", lambda: "source-sha")
    output = tmp_path / "readiness.json"
    monkeypatch.setattr(harness.sys, "argv", ["model_runner.py", "check", "--runs", "1", "--output", str(output)])
    assert harness.main() == 1
    saved = harness.json.loads(output.read_text())
    assert saved["exit_code"] == 1
    assert "ordinary diagnostic text" in output.read_text()
    assert_no_canary(saved, secret)
    assert_no_canary(harness.json.loads(capsys.readouterr().out), secret)
    assert response["model"] == endpoint.model
    assert secret in response["choices"][0]["message"]["content"]


@pytest.mark.parametrize("secret", ['quoted-"canary', r'backslash-\canary', 'unicode-雪-café'])
def test_pi_saved_diagnostics_sanitize_nested_json(tmp_path, monkeypatch, capsys, secret):
    import contextlib
    from types import SimpleNamespace

    endpoint = credential_endpoint(tmp_path, monkeypatch, secret)

    @contextlib.contextmanager
    def route():
        yield "http://127.0.0.1:34567/gateway/openai/v1", "temporary-key"

    messages = [
        {"role": "assistant", "model": endpoint.model, "stopReason": "toolUse", "content": [
            {"type": "toolCall", "name": "read", "id": "read-1", "arguments": {"path": "source.txt"}},
        ]},
        {"role": "toolResult", "toolCallId": "read-1", "isError": False,
         "content": [{"type": "text", "text": "The verification code is PETRI42."}]},
        {"role": "assistant", "model": endpoint.model, "stopReason": "stop",
         "content": [{"type": "text", "text": "PETRI42"}]},
    ]
    stdout = harness.json.dumps({"type": "agent_end", "messages": messages,
                                 "diagnostic": harness.json.dumps({"credential": secret})})
    monkeypatch.setattr(endpoint, "guarded_route", route)
    monkeypatch.setattr(harness, "endpoint_from_args", lambda args: endpoint)
    monkeypatch.setattr(harness, "source_sha", lambda: "source-sha")
    monkeypatch.setattr(harness.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(
        returncode=0, stdout=stdout, stderr='ordinary stderr: ' + harness.json.dumps({"echo": secret})))
    output = tmp_path / "pi-readiness.json"
    monkeypatch.setattr(harness.sys, "argv", ["model_runner.py", "pi-check", "--output", str(output)])
    assert harness.main() == 0
    assert_no_canary(harness.json.loads(output.read_text()), secret)
    assert_no_canary(harness.json.loads(capsys.readouterr().out), secret)
    assert "ordinary stderr" in output.read_text()


@pytest.mark.parametrize("valid", [True, False])
def test_pi_client_boundary_requires_complete_guarded_stream(endpoint, monkeypatch, guarded_client, valid):
    from contextlib import nullcontext
    from types import SimpleNamespace

    responses = [{"model": endpoint.model, "choices": [{"index": 0, "delta": {"content": "PETRI42"},
                                                        "finish_reason": "stop" if valid else None}]}]
    body = sse_body(responses, done=valid)
    monkeypatch.setattr(endpoint, "raw_request", lambda *args, **kwargs: (200, "text/event-stream", body))
    messages = [
        {"role": "assistant", "model": endpoint.model, "stopReason": "toolUse", "content": [
            {"type": "toolCall", "name": "read", "id": "read-1", "arguments": {"path": "source.txt"}},
        ]},
        {"role": "toolResult", "toolCallId": "read-1", "isError": False,
         "content": [{"type": "text", "text": "PETRI42"}]},
        {"role": "assistant", "model": endpoint.model, "stopReason": "stop",
         "content": [{"type": "text", "text": "PETRI42"}]},
    ]
    with guarded_client(endpoint) as post:
        monkeypatch.setattr(endpoint, "guarded_route", lambda: nullcontext((post.base_url, post.token)))

        def run(*args, **kwargs):
            status, response = post()
            return SimpleNamespace(returncode=0 if status == 200 else 1,
                                   stdout=harness.json.dumps({"type": "agent_end", "messages": messages}),
                                   stderr="" if status == 200 else response.decode())

        monkeypatch.setattr(harness.subprocess, "run", run)
        runner = harness.ModelRunner(endpoint=endpoint)
        if valid:
            assert runner.pi_check("fake-pi")["outcome"] == "passed"
        else:
            with pytest.raises(RuntimeError, match="pi exited"):
                runner.pi_check("fake-pi")
            assert "SSE ended" in runner.records[0]["stderr"]


def test_sse_multiline_data_and_usage_after_finish(endpoint):
    body = (b': comment\r\nevent: message\r\ndata: {"model":"gateway-alias",\r\n'
            b'data: "choices":[{"index":0,"delta":{"content":"READY"},"finish_reason":"stop"}]}\r\n\r\n'
            b'data: {"choices":[],"usage":{"completion_tokens":1}}\r\n\r\n'
            b'data: [DONE]\r\n\r\n')
    endpoint.validate_response_identity(body, "text/event-stream")


@pytest.mark.parametrize(('selected', 'declarations', 'valid'), [
    (262144, {'max_model_len': 262144}, True),
    (131072, {'context_size': 131072}, True),
    (262144, {'max_model_len': 262144, 'context_size': 262144}, True),
    (262144, {'max_model_len': 262144, 'context_size': 131072}, False),
    (131072, {'max_model_len': 262144, 'context_size': 131072}, False),
    (262144, {'max_model_len': 131072, 'context_size': 262144}, False),
    (131072, {'max_model_len': 131072, 'context_size': 262144}, False),
    (8192, {'max_model_len': 8192, 'context_size': 8192}, True),
    (8192, {'max_model_len': None, 'context_size': 8192}, False),
])
def test_every_supplied_context_declaration_must_match(selected, declarations, valid):
    from scripts.e2e.local_endpoint import LocalEndpoint

    manifest = {'artifact': 'test/model', 'revision': 'revision', **declarations}
    if valid:
        endpoint = LocalEndpoint('http://127.0.0.1:8000/v1', 'model', selected, 4096,
                                 manifest, no_auth=True)
        assert endpoint.identity()['provenance']['context_declarations'] == declarations
        assert endpoint.pi_document()['providers']['petri-local']['models'][0]['contextWindow'] == selected
    else:
        with pytest.raises(ValueError, match='every supplied manifest context declaration'):
            LocalEndpoint('http://127.0.0.1:8000/v1', 'model', selected, 4096,
                          manifest, no_auth=True)


def interleaved_stream(model, text, arguments, *, newline='\n', done=True):
    """Interleave choices and tool indexes, splitting every logical string character."""
    responses = []
    for offset in range(max(len(text), len(arguments))):
        for index in (1, 0):
            delta = {}
            if offset < len(text):
                delta['content'] = text[offset] if index == 0 else '.'
            calls = []
            for tool_index in (1, 0):
                if offset < len(arguments):
                    call = {'index': tool_index, 'function': {
                        'arguments': arguments[offset] if (index, tool_index) == (0, 0) else ' ',
                    }}
                    if offset == 0:
                        call.update(id=f'call-{index}-{tool_index}', type='function')
                        call['function']['name'] = 'read'
                    calls.append(call)
            if calls:
                delta['tool_calls'] = calls
            responses.append({'id': 'response-original', 'model': model, 'choices': [
                {'index': index, 'delta': delta, 'finish_reason': None},
            ]})
    responses.append({'id': 'response-original', 'model': model, 'choices': [
        {'index': index, 'delta': {}, 'finish_reason': 'tool_calls'} for index in (0, 1)
    ]})
    responses.append({'choices': [], 'usage': {'completion_tokens': 99}})
    return (': keepalive' + newline + 'event: message' + newline).encode() + sse_body(
        responses, newline=newline, done=done)


@pytest.mark.parametrize('secret', ['quoted-"canary', r'backslash-\canary', 'unicode-雪-café'])
@pytest.mark.parametrize('field', ['content', 'arguments'])
@pytest.mark.parametrize('done', [True, False])
def test_split_interleaved_sse_credentials_rejected_without_diagnostic_fragments(
        tmp_path, monkeypatch, guarded_client, secret, field, done):
    from scripts.e2e.local_endpoint import ResponseRejected

    endpoint = credential_endpoint(tmp_path, monkeypatch, secret, bearer=field == 'content')
    text = 'ordinary content ' + secret if field == 'content' else 'ordinary safe content'
    arguments = harness.json.dumps({'nested': harness.json.dumps({'value': secret})}) if field == 'arguments' else '{}'
    body = interleaved_stream(endpoint.model, text, arguments, done=done)
    with pytest.raises(ResponseRejected, match='echoed protected') as error:
        endpoint.validate_response_identity(body, 'text/event-stream')
    diagnostic = error.value.diagnostic
    assert_no_canary(diagnostic, secret)
    assert secret not in ''.join(all_diagnostic_strings(diagnostic))
    assert 'delta' not in diagnostic
    assert '[REDACTED]' in harness.json.dumps(diagnostic)
    monkeypatch.setattr(endpoint, 'raw_request', lambda *args, **kwargs: (200, 'text/event-stream', body))
    with guarded_client(endpoint) as post:
        status, response = post()
    assert status == 502
    saved = harness.json.loads(response)
    assert_no_canary(saved, secret)
    assert secret not in ''.join(all_diagnostic_strings(saved))
    assert 'delta' not in response.decode()
    assert 'ordinary' in response.decode()
    output = tmp_path / 'rejected-stream.json'
    output.write_text(endpoint.diagnostic_json(saved))
    assert_no_canary(harness.json.loads(output.read_text()), secret)


def all_diagnostic_strings(value):
    """Collect logical diagnostic strings, decoding JSON nested in strings."""
    if isinstance(value, dict):
        for item in value.values():
            yield from all_diagnostic_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from all_diagnostic_strings(item)
    elif isinstance(value, str):
        yield value
        try:
            decoded = harness.json.loads(value)
        except ValueError:
            return
        if decoded != value:
            yield from all_diagnostic_strings(decoded)


@pytest.mark.parametrize('newline', ['\n', '\r\n', '\r'])
def test_safe_interleaved_sse_is_forwarded_unchanged(endpoint, monkeypatch, guarded_client, newline):
    body = interleaved_stream(endpoint.model, 'ordinary research content', '{"path":"source.txt"}', newline=newline)
    monkeypatch.setattr(endpoint, 'raw_request', lambda *args, **kwargs: (200, 'text/event-stream', body))
    with guarded_client(endpoint) as post:
        status, forwarded = post()
    assert status == 200
    assert forwarded == body
    assert b'response-original' in forwarded and b'call-0-0' in forwarded
    assert b'completion_tokens' in forwarded


def test_tool_id_deltas_resolve_to_their_choice_index(tmp_path, monkeypatch, guarded_client):
    endpoint = credential_endpoint(tmp_path, monkeypatch, 'split-protected-password')
    responses = [{'model': endpoint.model, 'choices': [{'index': 0, 'delta': {'tool_calls': [
        {'index': 1, 'id': 'call-one', 'function': {'arguments': 'split-protected-'}},
        {'index': 0, 'id': 'call-zero', 'function': {'arguments': 'ordinary safe tool'}},
    ]}, 'finish_reason': None}]}, {'model': endpoint.model, 'choices': [{'index': 0, 'delta': {'tool_calls': [
        {'id': 'call-zero', 'function': {'arguments': ' arguments'}},
        {'id': 'call-one', 'function': {'arguments': 'password'}},
    ]}, 'finish_reason': 'tool_calls'}]}]
    body = sse_body(responses)
    monkeypatch.setattr(endpoint, 'raw_request', lambda *args, **kwargs: (200, 'text/event-stream', body))
    with guarded_client(endpoint) as post:
        status, diagnostic = post()
    assert status == 502
    assert b'split-protected-' not in diagnostic and b'password' not in diagnostic
    assert b'ordinary safe tool arguments' in diagnostic


def test_split_sse_failure_before_logical_assembly_omits_raw_fragments(endpoint, monkeypatch, guarded_client):
    endpoint.secret_values = ['protected-secret']
    body = sse_body([{'model': endpoint.model, 'choices': [{'index': 0, 'delta': {'content': part},
                                                         'finish_reason': None}]} for part in ['protected-', 'secret']],
                    done=False) + b'data: invalid-json\n\n'
    monkeypatch.setattr(endpoint, 'raw_request', lambda *args, **kwargs: (200, 'text/event-stream', body))
    with guarded_client(endpoint) as post:
        status, diagnostic = post()
    assert status == 502
    assert b'protected-' not in diagnostic and b'secret' not in diagnostic
    assert b'Raw SSE omitted' in diagnostic


@pytest.mark.parametrize('separate_tools', [False, True])
def test_sse_does_not_join_fragments_from_distinct_identities(endpoint, monkeypatch, guarded_client, separate_tools):
    endpoint.secret_values = ['protected-secret']
    choices = []
    for index, part in enumerate(['protected-', 'secret']):
        delta = {'tool_calls': [{'index': index, 'id': f'call-{index}',
                                 'function': {'arguments': part}}]} if separate_tools else {'content': part}
        choices.append({'index': 0 if separate_tools else index, 'delta': delta,
                        'finish_reason': None})
    responses = [{'model': endpoint.model, 'choices': [choice]} for choice in choices]
    responses.append({'model': endpoint.model, 'choices': [
        {'index': index, 'delta': {}, 'finish_reason': 'tool_calls' if separate_tools else 'stop'}
        for index in ([0] if separate_tools else [0, 1])
    ]})
    body = sse_body(responses)
    monkeypatch.setattr(endpoint, 'raw_request', lambda *args, **kwargs: (200, 'text/event-stream', body))
    with guarded_client(endpoint) as post:
        status, response = post()
    assert status == 200
    assert response == body


def test_pi_retains_only_safe_diagnostics_for_split_stream(tmp_path, monkeypatch, guarded_client, capsys):
    from contextlib import nullcontext
    from types import SimpleNamespace

    secret = 'split-"password-雪'
    endpoint = credential_endpoint(tmp_path, monkeypatch, secret)
    body = interleaved_stream(endpoint.model, 'ordinary content ' + secret, '{}')
    monkeypatch.setattr(endpoint, 'raw_request', lambda *args, **kwargs: (200, 'text/event-stream', body))
    monkeypatch.setattr(harness, 'endpoint_from_args', lambda args: endpoint)
    monkeypatch.setattr(harness, 'source_sha', lambda: 'source-sha')
    with guarded_client(endpoint) as post:
        monkeypatch.setattr(endpoint, 'guarded_route', lambda: nullcontext((post.base_url, post.token)))

        def run(*args, **kwargs):
            status, response = post()
            assert status == 502
            return SimpleNamespace(returncode=1, stdout='', stderr=response.decode())

        monkeypatch.setattr(harness.subprocess, 'run', run)
        output = tmp_path / 'pi-split-stream.json'
        monkeypatch.setattr(harness.sys, 'argv', ['model_runner.py', 'pi-check', '--output', str(output)])
        assert harness.main() == 1
    saved = harness.json.loads(output.read_text())
    assert saved['exit_code'] == 1
    assert_no_canary(saved, secret)
    assert secret not in ''.join(all_diagnostic_strings(saved))
    assert 'delta' not in output.read_text()
    assert 'ordinary content' in output.read_text()
    assert_no_canary(harness.json.loads(capsys.readouterr().out), secret)
