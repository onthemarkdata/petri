"""Backend selection uses only the supplied mapping and launches no processes."""

import pytest
from pydantic import ValidationError

from petri.harness.claude_code.model import ClaudeCodeModel
from petri.harness.pi.model import PiModel
from petri.harness.resolve import resolve_model


def test_pi_inherits_canonical_model_without_caching():
    config = {"model": "anthropic:claude-sonnet-4-6", "harness": {"backend": "pi"}}
    model = resolve_model(config)
    assert isinstance(model, PiModel)
    assert model.model_name == "claude-sonnet-4-6"
    assert model.system == "anthropic"
    assert resolve_model(config) is not model
    config["model"] = "openai:gpt-test"
    second = resolve_model(config)
    assert second.system == "openai"
    assert second.model_name == "gpt-test"


def test_explicit_pi_identifier_and_tools():
    model = resolve_model({"harness": {
        "backend": "pi", "provider": "local", "model": "huggingface.co/model:Q4_0", "tools": ["read"],
    }})
    assert isinstance(model, PiModel)
    assert model.model_name == "huggingface.co/model:Q4_0"
    assert model.system == "local"
    assert model.transport.tools == ("read",)


def test_claude_model_and_allowlist():
    model = resolve_model({"model": "anthropic:claude-sonnet-4-6", "harness": {
        "backend": "claude-code", "tools": ["Read", "WebSearch"],
    }})
    assert isinstance(model, ClaudeCodeModel)
    assert model.model_name == "claude-sonnet-4-6"
    assert model.allowed_tools == ("Read", "WebSearch")


def test_provider_passthrough_is_exact():
    name = "custom-provider:model:variant"
    assert resolve_model({"harness": {"backend": "provider", "model": name}}) is name
    assert resolve_model({"model": name, "harness": {"backend": "provider"}}) is name


@pytest.mark.parametrize(("harness", "key"), [
    ({"backend": "wrong"}, "backend"),
    ({"backend": "pi", "model": 1}, "model"),
    ({"backend": "pi", "provider": []}, "provider"),
    ({"backend": "pi", "tools": "read"}, "tools"),
    ({"backend": "pi", "tools": [42]}, "tools"),
    ({"backend": "pi", "surprise": True}, "surprise"),
])
def test_invalid_harness_names_key(harness, key):
    with pytest.raises(ValidationError) as error:
        resolve_model({"model": "test", "harness": harness})
    assert any(item["loc"][:2] == ("harness", key) for item in error.value.errors())


@pytest.mark.parametrize(("config", "key"), [
    ({"harness": {"backend": "pi"}}, "model"),
    ({"model": "x", "harness": {"backend": "pi", "model": " "}}, "harness.model"),
    ({"model": "x", "harness": {"backend": "pi", "provider": " "}}, "harness.provider"),
    ({"model": "x", "harness": {"backend": "pi", "tools": [" "]}}, "harness.tools"),
    ({"model": "x", "harness": {"backend": "provider", "tools": ["read"]}}, "harness.tools"),
    ({"model": "x", "harness": {"backend": "provider", "provider": "x"}}, "harness.provider"),
    ({"model": "openai:x", "harness": {"backend": "claude-code"}}, "harness.provider"),
    ({"model": "openai:x", "harness": {"backend": "pi", "provider": "anthropic"}}, "harness.provider"),
    ({"model": "anthropic:", "harness": {"backend": "pi"}}, "harness.model"),
])
def test_incompatible_settings_fail_explicitly(config, key):
    with pytest.raises(ValueError, match=key):
        resolve_model(config)
