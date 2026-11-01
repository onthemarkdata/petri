"""Dish overlays select an adapter without inheriting another adapter's options."""

import pytest
import yaml

from petri.config import load_config, load_dish_config
from petri.harness.pi.model import PiModel
from petri.harness.resolve import resolve_model


def load_override(tmp_path, values):
    path = tmp_path / "defaults" / "petri.yaml"
    path.parent.mkdir(exist_ok=True)
    path.write_text(yaml.safe_dump(values))
    config = load_dish_config(tmp_path)
    return config, resolve_model(config.model_dump(mode="json"))


def test_documented_provider_override_drops_packaged_cli_options(tmp_path, monkeypatch):
    import petri.config as module
    from petri.agents.config import HarnessContent

    defaults = load_config().model_copy(update={"harness": HarnessContent(
        backend="claude-code", model="claude-sonnet-4-6", provider="anthropic",
        tools=["Read"], output_mode="prompted",
    )})
    monkeypatch.setattr(module, "load_config", lambda: defaults)
    config, model = load_override(tmp_path, {
        "harness": {"backend": "provider", "model": "openai-chat:local-gemma"},
    })
    assert model == "openai-chat:local-gemma"
    assert config.harness.provider is None
    assert config.harness.tools == []
    assert config.harness.output_mode is None


@pytest.mark.parametrize("backend,top_level,expected", [
    ("provider", "openai-chat:local-gemma", "openai-chat:local-gemma"),
    ("pi", "openai:gpt-test", "gpt-test"),
])
def test_omitted_harness_model_uses_top_level(tmp_path, backend, top_level, expected):
    config, model = load_override(tmp_path, {
        "model": top_level, "harness": {"backend": backend},
    })
    assert config.harness.model is None
    assert config.harness.provider is None
    if backend == "provider":
        assert model == expected
    else:
        assert isinstance(model, PiModel)
        assert model.model_name == expected
        assert model.system == "openai"


def test_explicit_pi_provider_and_native_identifier_are_preserved(tmp_path):
    config, model = load_override(tmp_path, {
        "harness": {"backend": "pi", "provider": "ollama", "model": "gemma:local", "tools": ["read"]},
    })
    assert isinstance(model, PiModel)
    assert model.system == "ollama"
    assert model.model_name == "gemma:local"
    assert model.transport.tools == ("read",)
    assert config.harness.provider == "ollama"


def test_partial_harness_and_other_nested_overrides_remain_supported(tmp_path):
    defaults = load_config()
    config, model = load_override(tmp_path, {
        "harness": {"model": "native-model"},
        "execution": {"worker_concurrency": 2},
        "telemetry": {"local": {"enabled": False}},
    })
    assert config.harness.backend == defaults.harness.backend
    assert isinstance(model, PiModel)
    assert model.model_name == "native-model"
    assert config.execution.worker_concurrency == 2
    assert config.execution.polling_interval_seconds == defaults.execution.polling_interval_seconds
    assert config.telemetry.local.enabled is False
    assert config.telemetry.enabled == defaults.telemetry.enabled
    assert config.agents == defaults.agents


@pytest.mark.parametrize("harness,match", [
    ({"backend": "provider", "provider": "anthropic"}, "harness.provider"),
    ({"backend": "provider", "tools": ["read"]}, "harness.tools"),
    ({"backend": "provider", "tools": None}, "tools"),
    ({"backend": "provider", "model": " "}, "model"),
    ({"backend": "pi", "output_mode": "tool"}, "CLI backends"),
    ({"backend": "claude-code", "provider": "openai"}, "harness.provider"),
    ({"backend": "provider", "surprise": True}, "surprise"),
])
def test_explicit_invalid_fields_are_not_discarded(tmp_path, harness, match):
    with pytest.raises(ValueError, match=match):
        load_override(tmp_path, {"harness": harness})


def test_absent_harness_keeps_packaged_selection(tmp_path):
    config = load_dish_config(tmp_path)
    assert config.harness == load_config().harness
