"""Telemetry configuration stays explicit and external export stays opt-in."""

import importlib
from unittest.mock import Mock, patch

import pytest
import yaml
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import NoOpTracerProvider
from pydantic import ValidationError

from petri.config import get_telemetry_config
from petri.telemetry import TelemetryConfig, get_tracer_provider, init_telemetry, shutdown_telemetry


@pytest.fixture(autouse=True)
def clean_telemetry(monkeypatch):
    for name in ("OTEL_EXPORTER_OTLP_ENDPOINT", "OTEL_EXPORTER_OTLP_HEADERS"):
        monkeypatch.delenv(name, raising=False)
    shutdown_telemetry()
    yield
    shutdown_telemetry()


def write_config(tmp_path, value):
    path = tmp_path / "defaults/petri.yaml"
    path.parent.mkdir(exist_ok=True)
    path.write_text(yaml.safe_dump(value))


def test_missing_section_uses_local_defaults(tmp_path):
    write_config(tmp_path, {"name": "fixture"})
    cfg = get_telemetry_config(tmp_path)
    assert cfg.enabled and cfg.local.enabled
    assert cfg.sample_ratio == 1
    assert cfg.otlp.traces_endpoint is None
    assert cfg.otlp.protocol == "http/protobuf"


def test_yaml_overrides_are_validated_and_not_cached(tmp_path):
    write_config(tmp_path, {"telemetry": {"sample_ratio": 0.25, "local": {"retention_days": 7}}})
    cfg = get_telemetry_config(tmp_path)
    assert cfg.sample_ratio == 0.25
    assert cfg.local.retention_days == 7
    write_config(tmp_path, {"telemetry": {"sample_ratio": 0.5}})
    assert get_telemetry_config(tmp_path).sample_ratio == 0.5
    write_config(tmp_path, {"telemetry": {"sample_ratio": "lots"}})
    with pytest.raises(ValidationError, match="sample_ratio"):
        get_telemetry_config(tmp_path)


def test_endpoint_and_headers_environment_override_yaml(tmp_path, monkeypatch):
    write_config(
        tmp_path, {"telemetry": {"otlp": {"traces_endpoint": "http://yaml:4318/v1/traces"}}}
    )
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4318/")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_HEADERS", "authorization=local%20token,x-test=a%3Db")
    cfg = get_telemetry_config(tmp_path)
    assert cfg.otlp.traces_endpoint == "http://collector:4318/v1/traces"
    assert cfg.otlp.logs_endpoint == "http://collector:4318/v1/logs"
    assert cfg.otlp.headers == {"authorization": "local token", "x-test": "a=b"}
    assert "local token" not in repr(cfg)


def test_default_provider_has_resource_but_no_exporter_or_network():
    with (
        patch("socket.socket.connect", side_effect=AssertionError("network")),
        patch("opentelemetry.exporter.otlp.proto.http.trace_exporter.OTLPSpanExporter") as exporter,
    ):
        provider = init_telemetry(TelemetryConfig(), dish_id="fixture-dish")
        assert isinstance(provider, TracerProvider)
        assert not provider._active_span_processor._span_processors
        assert provider.resource.attributes["service.name"] == "petri"
        assert provider.resource.attributes["service.version"]
        assert provider.resource.attributes["petri.dish_id"] == "fixture-dish"
        with provider.get_tracer("test").start_as_current_span("local") as span:
            assert span.is_recording()
        exporter.assert_not_called()
        assert get_tracer_provider() is provider


def test_endpoint_adds_exactly_one_http_exporter():
    cfg = TelemetryConfig.model_validate(
        {"otlp": {"traces_endpoint": "http://collector:4318/v1/traces"}}
    )
    with (
        patch("opentelemetry.exporter.otlp.proto.http.trace_exporter.OTLPSpanExporter") as exporter,
        patch("opentelemetry.sdk.trace.export.BatchSpanProcessor") as processor,
    ):
        processor.return_value = Mock()
        first = init_telemetry(cfg)
        second = init_telemetry(cfg)
        assert first is second
        exporter.assert_called_once_with(endpoint="http://collector:4318/v1/traces", headers={})
        assert len(first._active_span_processor._span_processors) == 1
        processor.assert_called_once_with(exporter.return_value)


def test_disabled_provider_is_nonrecording():
    provider = init_telemetry({"enabled": False})
    assert isinstance(provider, NoOpTracerProvider)
    with provider.get_tracer("test").start_as_current_span("disabled") as span:
        assert not span.is_recording()


def test_setup_failure_warns_and_returns_noop():
    with (
        patch("petri.telemetry.setup.SDKTracerProvider", side_effect=RuntimeError("failure")),
        pytest.warns(RuntimeWarning, match="continuing without tracing"),
    ):
        provider = init_telemetry({})
    assert isinstance(provider, NoOpTracerProvider)


def test_import_does_not_load_config_or_require_dish(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with (
        patch("petri.config.load_dish_config", side_effect=AssertionError("config read")),
        patch("pathlib.Path.read_text", side_effect=AssertionError("file read")),
    ):
        import petri.telemetry

        importlib.reload(petri.telemetry)


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan")])
def test_sample_ratio_bounds(value):
    with pytest.raises(ValidationError, match="sample_ratio"):
        TelemetryConfig(sample_ratio=value)


def test_changing_dish_or_configuration_selects_a_distinct_provider():
    first = init_telemetry({}, dish_id="first")
    second = init_telemetry({}, dish_id="second")
    assert first is not second
    assert second.resource.attributes["petri.dish_id"] == "second"
    assert get_tracer_provider() is second
    changed = init_telemetry({"sample_ratio": 0}, dish_id="second")
    assert changed is not second
    with changed.get_tracer("test").start_as_current_span("unsampled") as span:
        assert not span.is_recording()
    assert init_telemetry({}, dish_id="first") is first
