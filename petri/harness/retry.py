"""Retry timing shared by inference backends; no transport-specific behavior."""

from __future__ import annotations

import asyncio
import math
import random
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TypeVar
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from petri.harness.errors import RateLimitedError, TransientHarnessError

T = TypeVar("T")


@dataclass(frozen=True)
class RetryPolicy:
    """Additional attempts and bounded exponential delay, measured in seconds."""

    max_retries: int = 2
    base: float = 1.5
    cap: float = 60.0
    jitter: float = 0.5

    def __post_init__(self) -> None:
        if type(self.max_retries) is not int or self.max_retries < 0:
            raise ValueError("max_retries must be a nonnegative integer")
        for name in ("base", "cap", "jitter"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")


def retry_delay(
    policy: RetryPolicy,
    retry_number: int,
    *,
    jitter_fraction: float = 0.0,
    error: TransientHarnessError | None = None,
) -> float | None:
    """Return a delay for a 1-based retry, or None when retrying is disallowed.

    The caller supplies a sample in [0, 1] to keep this function deterministic.
    Reset hints are minimum waits; hints beyond the cap surface immediately.
    """
    if type(retry_number) is not int or retry_number < 1:
        raise ValueError("retry_number must be a positive integer")
    if not math.isfinite(jitter_fraction) or not 0 <= jitter_fraction <= 1:
        raise ValueError("jitter_fraction must be between zero and one")
    if retry_number > policy.max_retries:
        return None
    hint = error.retry_after_seconds if isinstance(error, RateLimitedError) else None
    if hint is not None and hint > policy.cap:
        return None
    try:
        exponential = math.ldexp(policy.base, retry_number - 1)
    except OverflowError:
        exponential = policy.cap
    delay = min(policy.cap, exponential + policy.jitter * jitter_fraction)
    return max(delay, hint or 0.0)


async def retry(
    operation: Callable[[], Awaitable[T]],
    policy: RetryPolicy | None = None,
    *,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    random_sample: Callable[[], float] = random.random,
) -> T:
    """Retry typed transient failures; preserve permanent errors and cancellation."""
    policy = policy if policy is not None else RetryPolicy()
    retry_number = 1
    while True:
        try:
            return await operation()
        except TransientHarnessError as error:
            delay = retry_delay(
                policy, retry_number, jitter_fraction=random_sample(), error=error,
            )
            if delay is None:
                raise
            await sleep(delay)
            retry_number += 1


def parse_reset_hint(
    text: str, *, clock: Callable[[], datetime] | None = None,
) -> float | None:
    """Read relative waits or the next named-zone clock reset from backend text.

    Clock hints without a zone use UTC. The injected clock must return an aware
    datetime. Invalid or unsupported hints return None.
    """
    relative = re.search(
        r"try again in\s+(\d+(?:\.\d+)?)\s*(seconds?|minutes?|hours?)\b", text, re.IGNORECASE,
    )
    if relative:
        scale = {"second": 1, "minute": 60, "hour": 3600}[relative[2].lower().rstrip("s")]
        result = float(relative[1]) * scale
        return result if math.isfinite(result) else None
    match = re.search(
        r"\bresets?\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b(?:\s*\(([^)]+)\))?", text, re.IGNORECASE,
    )
    if not match:
        return None
    hour, minute = int(match[1]), int(match[2] or 0)
    if not 1 <= hour <= 12 or minute > 59:
        return None
    try:
        zone = ZoneInfo(match[4]) if match[4] else UTC
    except (ZoneInfoNotFoundError, ValueError):
        return None
    now = clock() if clock else datetime.now(UTC)
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("clock must return a timezone-aware datetime")
    local = now.astimezone(zone)
    hour = hour % 12 + (12 if match[3].lower() == "pm" else 0)
    target = local.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target.timestamp() <= now.timestamp():
        target += timedelta(days=1)
    return target.timestamp() - now.timestamp()
