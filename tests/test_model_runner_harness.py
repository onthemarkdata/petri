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
