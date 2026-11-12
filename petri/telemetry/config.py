"""Validated telemetry settings with no environment or filesystem access."""

from typing import Annotated, Literal

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, TypeAdapter, field_validator, model_validator


class OtlpConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    traces_endpoint: str | None = None
    logs_endpoint: str | None = None
    headers: dict[str, str] = Field(default_factory=dict, repr=False)
    protocol: Literal["http/protobuf"] = "http/protobuf"

    @field_validator("traces_endpoint", "logs_endpoint")
    @classmethod
    def validate_endpoint(cls, value: str | None) -> str | None:
        if value is not None:
            TypeAdapter(AnyHttpUrl).validate_python(value)
        return value


class LocalTelemetryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    retention_days: Annotated[int, Field(strict=True, gt=0)] = 30


class CaptureLimits(BaseModel):
    """Finite opt-in capture ceilings; all limits apply to each completed span."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_text_chars: Annotated[int, Field(strict=True, gt=0, le=1_048_576)] = 1_048_576
    max_collection_items: Annotated[int, Field(strict=True, gt=0, le=512)] = 512
    max_depth: Annotated[int, Field(strict=True, gt=0, le=64)] = 64
    max_nodes: Annotated[int, Field(strict=True, gt=0, le=16_384)] = 16_384
    max_input_bytes: Annotated[int, Field(strict=True, gt=0, le=8_388_608)] = 8_388_608
    max_span_bytes: Annotated[int, Field(strict=True, gt=0, le=4_194_304)] = 4_194_304


class TelemetryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    capture_content: Annotated[bool, Field(strict=True)] = False
    capture_limits: CaptureLimits | None = None
    redact_env_vars: list[Annotated[str, Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")]] = Field(default_factory=list)
    otlp: OtlpConfig = Field(default_factory=OtlpConfig)
    local: LocalTelemetryConfig = Field(default_factory=LocalTelemetryConfig)
    sample_ratio: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)] = 1.0
    service_name: Annotated[str, Field(min_length=1)] = "petri"

    @model_validator(mode="after")
    def require_explicit_capture(self):
        if self.capture_limits is not None and not self.capture_content:
            raise ValueError("capture_limits requires capture_content: true")
        return self


class ModelPrice(BaseModel):
    """Explicit USD rates per million reported input and output tokens."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    input_per_million: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    output_per_million: Annotated[float, Field(ge=0, allow_inf_nan=False)]
