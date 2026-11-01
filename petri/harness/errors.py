"""Typed failures with bounded diagnostics for domain events."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from pydantic import BaseModel, ConfigDict, JsonValue, field_validator

DIAGNOSTIC_LIMIT = 4096


class FailureDetail(BaseModel):
    """JSON-safe diagnostics; text fields retain their final 4096 characters."""

    model_config = ConfigDict(strict=True, frozen=True, allow_inf_nan=False)

    exit_code: int | None = None
    stderr_tail: str = ""
    stdout_tail: str = ""
    last_stream_event: JsonValue = None
    partial_output: str = ""

    @field_validator("stderr_tail", "stdout_tail", "partial_output")
    @classmethod
    def bound_text(cls, value: str) -> str:
        return value[-DIAGNOSTIC_LIMIT:]

    @field_validator("last_stream_event")
    @classmethod
    def bound_event(cls, value: JsonValue) -> JsonValue:
        encoded = json.dumps(value, allow_nan=False)
        if len(encoded) > DIAGNOSTIC_LIMIT * 4:
            return {"truncated": True, "tail": encoded[-DIAGNOSTIC_LIMIT:]}
        return value


class HarnessError(Exception):
    """An inference failure that may carry event-log diagnostics."""

    def __init__(self, message: str, *, detail: FailureDetail | None = None):
        super().__init__(message)
        self.detail = detail


class TransientHarnessError(HarnessError):
    """A failure that may succeed when retried."""


class PermanentHarnessError(HarnessError):
    """A failure requiring a configuration, input, or protocol correction."""


class RateLimitedError(TransientHarnessError):
    """A rate limit with an optional minimum wait in seconds."""

    def __init__(
        self,
        message: str,
        *,
        retry_after_seconds: float | None = None,
        detail: FailureDetail | None = None,
    ):
        if retry_after_seconds is not None and (
            not math.isfinite(retry_after_seconds) or retry_after_seconds < 0
        ):
            raise ValueError("retry_after_seconds must be finite and nonnegative")
        super().__init__(message, detail=detail)
        self.retry_after_seconds = retry_after_seconds


class AuthExpiredError(PermanentHarnessError):
    """Authentication must be renewed before another attempt."""

    def __init__(
        self,
        message: str = "Authentication expired. Sign in to the inference backend and retry.",
        *,
        detail: FailureDetail | None = None,
    ):
        super().__init__(message, detail=detail)


@dataclass(frozen=True)
class FailureClassifier:
    """Backend-specific text markers and structured codes, with permanent precedence.

    Call only for failed operations, including in-band failures with exit code zero.
    Unknown diagnostics are permanent; empty diagnostics with a nonzero exit code
    are transient. Backends can override that fallback.
    """

    auth_markers: tuple[str, ...] = (
        "unauthorized", "unauthenticated", "authentication", "invalid api key",
        "expired token", "token expired",
    )
    permanent_markers: tuple[str, ...] = (
        "forbidden", "model not found", "unknown model", "invalid model",
        "context length", "context window", "too long", "max tokens", "billing",
    )
    rate_limit_markers: tuple[str, ...] = ("rate limit", "rate-limit", "too many requests")
    transient_markers: tuple[str, ...] = (
        "timeout", "timed out", "connection reset", "connection refused",
        "broken pipe", "network", "service unavailable", "internal server error",
        "temporarily unavailable", "overloaded",
    )
    auth_codes: tuple[str | int, ...] = (401, "401")
    permanent_codes: tuple[str | int, ...] = (403, 404, "403", "404")
    rate_limit_codes: tuple[str | int, ...] = (429, "429")
    transient_codes: tuple[str | int, ...] = (500, 502, 503, 504, "500", "502", "503", "504")
    empty_is_transient: bool = True

    def classify(
        self,
        exit_code: int | None,
        stderr: str = "",
        stdout: str = "",
        *,
        code: str | int | None = None,
        detail: FailureDetail | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> HarnessError:
        """Classify both streams before bounding the diagnostics stored on the error."""
        from petri.harness.retry import parse_reset_hint

        text = f"{stderr}\n{stdout}".lower()
        detail = detail or FailureDetail(
            exit_code=exit_code, stderr_tail=stderr, stdout_tail=stdout,
        )
        message = f"Inference backend failed (exit code {exit_code})."

        def matches(markers: tuple[str, ...]) -> bool:
            return any(marker.lower() in text for marker in markers)

        if code in self.auth_codes or matches(self.auth_markers):
            return AuthExpiredError(detail=detail)
        if code in self.permanent_codes or matches(self.permanent_markers):
            return PermanentHarnessError(message, detail=detail)
        reset = parse_reset_hint(f"{stderr}\n{stdout}", clock=clock)
        if code in self.rate_limit_codes or matches(self.rate_limit_markers) or reset is not None:
            return RateLimitedError(message, retry_after_seconds=reset, detail=detail)
        if code in self.transient_codes or matches(self.transient_markers):
            return TransientHarnessError(message, detail=detail)
        if not text.strip() and exit_code not in (None, 0) and self.empty_is_transient:
            return TransientHarnessError(message, detail=detail)
        return PermanentHarnessError(message, detail=detail)
