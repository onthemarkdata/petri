"""Single-agent CLI runs through typed validation and measured usage."""

import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.usage import RequestUsage
from typer.testing import CliRunner

from petri.agents.contract import AgentName
from petri.cli import app

RUNNER = CliRunner()
DEFAULTS = Path(__file__).parents[2] / "petri/defaults"


@pytest.fixture
def v2_dish(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    destination = tmp_path / ".petri/defaults"
    destination.mkdir(parents=True)
    for filename in ("petri.yaml", "constitution.md"):
        shutil.copyfile(DEFAULTS / filename, destination / filename)
    return destination


def scripted(monkeypatch, payloads):
    calls = []
    configs = []

    def respond(messages, info):
        calls.append(messages)
        payload = payloads[min(len(calls) - 1, len(payloads) - 1)]
        return ModelResponse(
            parts=[TextPart(json.dumps(payload))],
            usage=RequestUsage(input_tokens=41, output_tokens=19),
        )

    model = FunctionModel(respond)

    def resolve(config):
        configs.append(config)
        return model

    monkeypatch.setattr("petri.agents.factory._resolve", resolve)
    return calls, configs


def run(agent="skeptic", *flags):
    return RUNNER.invoke(
        app,
        [
            "agents",
            "run",
            agent,
            "--claim",
            "A testable claim.",
            "--output-mode",
            "prompted",
            *flags,
        ],
    )


def test_full_cli_reports_typed_verdict_and_usage(v2_dish, monkeypatch):
    calls, _ = scripted(
        monkeypatch, [{"verdict": "ARGUMENT_WITHSTANDS_CRITIQUE", "summary": "No conflict."}]
    )
    result = run()
    assert result.exit_code == 0, (result.output, result.exception)
    assert "skeptic: ARGUMENT_WITHSTANDS_CRITIQUE" in result.output
    assert "Summary: No conflict." in result.output
    assert "Sources:" in result.output
    assert "requests=1, input_tokens=41, output_tokens=19" in result.output
    assert "$" not in result.output
    assert len(calls) == 1
    assert "Continue?" not in result.output


def test_evaluator_retries_weak_evidence(v2_dish, monkeypatch):
    weak = {"verdict": "EVIDENCE_CONFIRMS", "sources_cited": []}
    strong = {
        "verdict": "EVIDENCE_CONFIRMS",
        "sources_cited": [{"url": "https://example.org/study", "hierarchy_level": 2}],
    }
    calls, _ = scripted(monkeypatch, [weak, strong])
    result = run("evidence_evaluator", "--verbose")
    assert result.exit_code == 0, result.output
    assert len(calls) == 2
    assert "Retries: 1" in result.output
    assert "[level 2] https://example.org/study" in result.output
    assert "requests=2, input_tokens=82, output_tokens=38" in result.output


def test_request_cap_retains_partial_output_and_usage(v2_dish, monkeypatch):
    calls, _ = scripted(monkeypatch, [{"verdict": "EVIDENCE_CONFIRMS", "sources_cited": []}])
    result = run("evidence_evaluator", "--max-requests", "1", "--verbose")
    assert result.exit_code == 1, result.output
    assert len(calls) == 1
    assert "Over-budget" in result.output
    assert "Partial model output (not a validated verdict)" in result.output
    assert "EVIDENCE_CONFIRMS" in result.output
    assert "requests=1, input_tokens=41, output_tokens=19" in result.output


def test_token_cap_is_checked_after_response(v2_dish, monkeypatch):
    scripted(monkeypatch, [{"verdict": "ARGUMENT_WITHSTANDS_CRITIQUE"}])
    result = run("skeptic", "--max-tokens", "10")
    assert result.exit_code == 1
    assert "Over-budget" in result.output
    assert "output_tokens=19" in result.output


def test_model_override_reaches_resolver(v2_dish, monkeypatch):
    _, configs = scripted(monkeypatch, [{"verdict": "ARGUMENT_WITHSTANDS_CRITIQUE"}])
    result = run("skeptic", "--model", "openai:local-test")
    assert result.exit_code == 0, result.output
    selected = [cfg for cfg in configs if cfg["model"] == "openai:local-test"]
    assert selected
    assert selected[0]["harness"] == {"backend": "provider", "model": "openai:local-test"}


def test_unknown_agent_lists_valid_values(v2_dish):
    result = run("invented")
    assert result.exit_code == 1
    for name in AgentName:
        assert name in result.output


@pytest.mark.parametrize(
    "yes,answer,expected_calls", [(False, "n\n", 0), (False, "y\n", 1), (True, "", 1)]
)
def test_interactive_warning(v2_dish, monkeypatch, yes, answer, expected_calls):
    calls, _ = scripted(monkeypatch, [{"verdict": "ARGUMENT_WITHSTANDS_CRITIQUE"}])
    monkeypatch.setattr(
        "petri.cli.agent_run.sys", SimpleNamespace(stdin=SimpleNamespace(isatty=lambda: True))
    )
    result = RUNNER.invoke(
        app,
        ["agents", "run", "skeptic", "--claim", "A claim.", *(["--yes"] if yes else [])],
        input=answer,
    )
    assert len(calls) == expected_calls, result.output
    assert ("may incur model costs" in result.output) is (not yes)
    assert result.exit_code == (0 if expected_calls else 1)


def test_existing_cell_resolution(v2_dish, monkeypatch):
    calls, _ = scripted(monkeypatch, [{"verdict": "ARGUMENT_WITHSTANDS_CRITIQUE"}])
    graph = SimpleNamespace(get_cell=lambda cell: SimpleNamespace(claim_text="A stored claim."))
    monkeypatch.setattr("petri.cli.agent_run.load_colonies", lambda *_: [(graph, None)])
    result = RUNNER.invoke(app, ["agents", "run", "skeptic", "--cell", "example-cell"])
    assert result.exit_code == 0, result.output
    assert "A stored claim." in str(calls[0])


def test_exhausted_validation_is_execution_failure(v2_dish, monkeypatch):
    scripted(monkeypatch, [{"verdict": "invented"}])
    result = run("skeptic", "--verbose")
    assert result.exit_code == 1
    assert "Execution failed" in result.output
    assert "requests=3" in result.output


@pytest.mark.live
@pytest.mark.skipif(
    os.environ.get("PETRI_LIVE_GEMMA") != "1",
    reason="Set PETRI_LIVE_GEMMA=1 for local Docker Model Runner inference.",
)
def test_live_local_gemma_cli(v2_dish, monkeypatch):
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    model = OpenAIChatModel(
        "huggingface.co/google/gemma-4-e2b-it-qat-q4_0-gguf:Q4_0",
        provider=OpenAIProvider(
            base_url="http://127.0.0.1:12434/engines/llama.cpp/v1", api_key="local"
        ),
        settings={"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
    )
    monkeypatch.setattr("petri.agents.factory._resolve", lambda _: model)
    result = RUNNER.invoke(
        app,
        [
            "agents",
            "run",
            "skeptic",
            "--claim",
            (
                "Synthetic fixture: the provided record says all 3 test runs passed. "
                "There is no external evidence to retrieve. Assess only this record."
            ),
            "--yes",
            "--output-mode",
            "prompted",
            "--max-requests",
            "3",
            "--max-tokens",
            "12000",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Usage: requests=" in result.output
    assert "input_tokens=0" not in result.output
    assert "$" not in result.output


def test_harness_failure_reports_partial_output_and_measured_usage(v2_dish, monkeypatch):
    from petri.harness.errors import FailureDetail, TransientHarnessError

    calls = 0

    def respond(messages, info):
        nonlocal calls
        calls += 1
        if calls == 1:
            return ModelResponse(
                parts=[TextPart('{"verdict":"EVIDENCE_CONFIRMS","sources_cited":[]}')],
                usage=RequestUsage(input_tokens=41, output_tokens=19),
            )
        raise TransientHarnessError(
            "Local model process disconnected.",
            detail=FailureDetail(partial_output="Unfinished reply"),
        )

    monkeypatch.setattr("petri.agents.factory._resolve", lambda _: FunctionModel(respond))
    result = run("evidence_evaluator", "--verbose")
    assert result.exit_code == 1
    assert "Execution failed: Local model process disconnected." in result.output
    assert "Partial backend output (not a validated verdict)" in result.output
    assert "Unfinished reply" in result.output
    assert "input_tokens=41, output_tokens=19" in result.output
    assert "Retries: 1" in result.output
