"""Strict v2 content validation without network or model construction."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from petri.agents.config import AgentConfig, AgentConfigError, AgentContent, load_agent_config
from petri.agents.contract import AgentName

DEFAULTS = Path(__file__).parents[2] / "petri/defaults"


@pytest.fixture
def content():
    return yaml.safe_load((DEFAULTS / "petri.yaml").read_text())


def test_shipped_defaults_and_usage_limits():
    cfg = load_agent_config(DEFAULTS / "petri.yaml")
    assert set(cfg.agents) == set(AgentName)
    assert all(isinstance(value, AgentContent) for value in cfg.agents.values())
    assert cfg.limits.run.usage_kwargs() == {
        "request_limit": 25,
        "total_tokens_limit": 200000,
        "tool_calls_limit": 10,
    }
    assert cfg.limits.debate.usage_kwargs() == {"request_limit": 6, "total_tokens_limit": 50000}
    assert AgentConfig.model_validate(cfg.model_dump()) == cfg
    assert cfg.debates[0].rounds == 1.5
    assert cfg.harness.backend == "pi"


def test_missing_path_raises_typed_error(tmp_path):
    with pytest.raises(AgentConfigError, match="missing.yaml"):
        load_agent_config(tmp_path / "missing.yaml")


def test_v1_shape_rejected():
    with pytest.raises(AgentConfigError, match="schema_version"):
        load_agent_config(DEFAULTS / "legacy-petri.yaml")


@pytest.mark.parametrize(
    "kind,expected",
    [
        ("typo", "investigatr"),
        ("missing", "investigator"),
        ("verdict", "MADE_UP_VERDICT"),
        ("swapped", "EVIDENCE_SUFFICIENT"),
    ],
)
def test_invalid_roster_and_verdicts(content, kind, expected):
    if kind == "typo":
        content["agents"]["investigatr"] = content["agents"].pop("investigator")
    elif kind == "missing":
        del content["agents"]["investigator"]
    elif kind == "verdict":
        content["agents"]["investigator"]["verdicts_pass"].append("MADE_UP_VERDICT")
    else:
        value = content["agents"]["investigator"]["verdicts_pass"].pop(0)
        content["agents"]["investigator"]["verdicts_block"].append(value)
    with pytest.raises(ValueError, match=expected):
        AgentConfig.model_validate(content)


@pytest.mark.parametrize(
    "key", ["is_lead", "phase", "blocking", "scope", "redirect_on_block", "display_name"]
)
def test_structural_keys_name_migration_note(content, key):
    content["agents"]["investigator"][key] = None
    with pytest.raises(ValueError, match="ARCHITECTURE-V2.md#v2-agent-content-schema"):
        AgentConfig.model_validate(content)


def test_duplicate_yaml_keys_rejected(tmp_path):
    path = tmp_path / "duplicate.yaml"
    path.write_text("model: one\nmodel: two\n")
    with pytest.raises(AgentConfigError, match="Duplicate YAML key"):
        load_agent_config(path)


def test_explicit_load_is_not_cached(content, tmp_path):
    path = tmp_path / "petri.yaml"
    path.write_text(yaml.safe_dump(content))
    assert load_agent_config(path).max_iterations == 3
    content["max_iterations"] = 7
    path.write_text(yaml.safe_dump(content))
    assert load_agent_config(path).max_iterations == 7


def test_import_does_not_read_configuration(tmp_path):
    code = """
from pathlib import Path
from unittest.mock import patch
read_text = Path.read_text
def guarded_read(path, *args, **kwargs):
    if path.suffix in {'.yaml', '.yml'}:
        raise AssertionError('unexpected configuration read')
    return read_text(path, *args, **kwargs)
with patch('pathlib.Path.read_text', guarded_read):
    import petri.agents
    import petri.agents.config
"""
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env={
            **os.environ,
            "PYTHONPATH": str(DEFAULTS.parents[1]),
            "PETRI_CONFIG": str(tmp_path / "missing.yaml"),
        },
        check=True,
    )


@pytest.mark.parametrize("value", [0, -1, True, "25"])
def test_usage_caps_reject_invalid_values(content, value):
    content["limits"]["run"]["request_limit"] = value
    with pytest.raises(ValueError, match="request_limit"):
        AgentConfig.model_validate(content)


@pytest.mark.parametrize("value", [None, False, True])
@pytest.mark.parametrize("inherited", [False, True])
def test_openai_chat_profile_boolean_and_model_precedence(content, value, inherited):
    content["model"] = "openai-chat:inherited:variant" if inherited else "anthropic:other"
    content["harness"] = {"backend": "provider", "openai_chat_supports_multiple_system_messages": value}
    if not inherited:
        content["harness"]["model"] = "openai-chat:selected:variant"
    config = AgentConfig.model_validate(content)
    assert config.harness.openai_chat_supports_multiple_system_messages is value


def test_openai_chat_profile_is_unset_in_packaged_defaults(content):
    assert "openai_chat_supports_multiple_system_messages" not in content["harness"]
    assert AgentConfig.model_validate(content).harness.openai_chat_supports_multiple_system_messages is None


@pytest.mark.parametrize("value", ["false", "true", "", 0, 1, [], {}, [False]])
def test_openai_chat_profile_rejects_non_booleans(content, value):
    content["harness"] = {"backend": "provider", "model": "openai-chat:fixture",
                          "openai_chat_supports_multiple_system_messages": value}
    with pytest.raises(ValueError, match="openai_chat_supports_multiple_system_messages"):
        AgentConfig.model_validate(content)


@pytest.mark.parametrize("harness,top_level,match", [
    ({"backend": "pi", "model": "openai-chat:fixture"}, "openai-chat:fallback", "backend=provider"),
    ({"backend": "claude-code"}, "openai-chat:fallback", "backend=provider"),
    ({"backend": "provider"}, "openai:responses", "openai-chat:"),
    ({"backend": "provider"}, "openai-responses:responses", "openai-chat:"),
    ({"backend": "provider"}, "anthropic:fixture", "openai-chat:"),
    ({"backend": "provider", "model": "custom:fixture"}, "openai-chat:fallback", "openai-chat:"),
    ({"backend": "provider", "model": "openai-chat:"}, "openai-chat:fallback", "nonblank"),
    ({"backend": "provider"}, "openai-chat: ", "nonblank"),
    ({"backend": "provider", "provider": "openai"}, "openai-chat:fallback", "harness.provider"),
    ({"backend": "provider", "tools": ["read"]}, "openai-chat:fallback", "harness.tools"),
    ({"backend": "provider", "unexpected": False}, "openai-chat:fallback", "unexpected"),
])
@pytest.mark.parametrize("value", [False, True])
def test_openai_chat_profile_rejects_incompatible_selection(content, harness, top_level, match, value):
    content["model"] = top_level
    content["harness"] = {**harness, "openai_chat_supports_multiple_system_messages": value}
    with pytest.raises(ValueError, match=match):
        AgentConfig.model_validate(content)
