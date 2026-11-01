"""Validated telemetry settings with no environment or filesystem access."""

from typing import Annotated, Literal

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, TypeAdapter, field_validator


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


class TelemetryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    otlp: OtlpConfig = Field(default_factory=OtlpConfig)
    local: LocalTelemetryConfig = Field(default_factory=LocalTelemetryConfig)
    sample_ratio: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)] = 1.0
    service_name: Annotated[str, Field(min_length=1)] = "petri"
