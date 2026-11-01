"""Execution state shared by CLI adapters and the typed pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib

from pydantic import BaseModel, ConfigDict, Field

from petri.models import QueueProcessingResult


class PipelineOutcome(str, Enum):
    converged = "converged"
    iterate = "iterate"
    circuit_breaker = "circuit_breaker"
    short_circuit = "short_circuit"


class CellPipelineState(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    cell_id: str
    iteration: int = Field(default=0, ge=0, frozen=True)
    weakest_link: str | None = None
    focused_directive: str | None = None
    cycle_start_iteration: int = Field(default=0, ge=0)
    max_iterations: int = Field(default=5, ge=1)


@dataclass
class ProgressEvent:
    slot_idx: int
    cell_id: str
    kind: str
    phase: str | None = None
    agent: str | None = None
    verdict: str | None = None
    iteration: int | None = None
    error: str | None = None
    text: str | None = None


@dataclass
class GrowOutcome:
    reason: str
    final_states: dict[str, int] = field(default_factory=dict)
    last_result: QueueProcessingResult | None = None
    passes_run: int = 0


@dataclass
class PreflightCheck:
    name: str
    passed: bool
    message: str


class NoProviderError(RuntimeError):
    """Execution requires an inference provider that has not been configured."""


@dataclass(frozen=True)
class ReopenRequest:
    cell_id: str
    request_id: str
    today_iso: str
    trigger: str = "human_revalidation"
    max_iterations: int = 5

    @property
    def workflow_id(self):
        request = hashlib.sha256(self.request_id.encode()).hexdigest()
        return f"{self.cell_id}:reopen:{request}"


@dataclass(frozen=True)
class FeedRequest:
    cell_id: str
    request_id: str
    source: str
    today_iso: str
    max_iterations: int = 5

    def __post_init__(self):
        if len(self.request_id) != 64 or any(c not in "0123456789abcdef" for c in self.request_id):
            raise ValueError("Evidence request IDs must be SHA-256 hex digests")

    @property
    def workflow_id(self):
        return f"{self.cell_id}:feed:{self.request_id}"
