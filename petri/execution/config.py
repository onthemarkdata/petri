"""Validated execution settings shared by config loading and runtime bootstrap."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


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
