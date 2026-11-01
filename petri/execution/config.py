"""Validated execution settings shared by config loading and runtime bootstrap."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class QueueLimiter(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    limit: Annotated[int, Field(strict=True, gt=0)]
    period: Annotated[float, Field(gt=0, allow_inf_nan=False)]


class ExecutionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    backend: Literal["v1", "dbos"] = "v1"
    worker_concurrency: Annotated[int, Field(strict=True, gt=0)] = 4
    limiter: QueueLimiter | None = None
    polling_interval_seconds: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 0.05

    scan_cron: str | None = None

    @field_validator("scan_cron")
    @classmethod
    def valid_scan_cron(cls, value: str | None) -> str | None:
        from croniter import croniter

        if value is not None and not croniter.is_valid(value, second_at_beginning=True):
            raise ValueError("execution.scan_cron must be a valid cron expression")
        return value
