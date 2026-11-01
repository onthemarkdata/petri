"""Deterministic retry tests use an injected clock, random sample, and sleeper."""

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from petri.harness.errors import PermanentHarnessError, RateLimitedError, TransientHarnessError
from petri.harness.retry import RetryPolicy, parse_reset_hint, retry, retry_delay


def test_exponential_growth_jitter_and_cap():
    policy = RetryPolicy(max_retries=10, base=1.5, cap=10, jitter=0.5)
    assert [retry_delay(policy, n) for n in range(1, 6)] == [1.5, 3, 6, 10, 10]
    for sample in (0, 0.5, 1):
        assert 3 <= retry_delay(policy, 2, jitter_fraction=sample) <= 3.5
        assert retry_delay(policy, 5, jitter_fraction=sample) == 10
    assert retry_delay(policy, 11) is None


def test_reset_hint_is_minimum_delay():
    policy = RetryPolicy(cap=60)
    assert retry_delay(policy, 1, error=RateLimitedError("wait", retry_after_seconds=40)) == 40
    assert retry_delay(policy, 1, error=RateLimitedError("wait", retry_after_seconds=60)) == 60
    assert retry_delay(policy, 1, error=RateLimitedError("wait", retry_after_seconds=61)) is None
    assert retry_delay(policy, 1, error=RateLimitedError("wait")) == 1.5


@pytest.mark.asyncio
async def test_zero_retries_performs_exactly_one_attempt():
    error = TransientHarnessError("failed")
    operation = AsyncMock(side_effect=error)
    sleep = AsyncMock()
    with pytest.raises(TransientHarnessError) as raised:
        await retry(operation, RetryPolicy(max_retries=0), sleep=sleep)
    assert raised.value is error
    assert operation.await_count == 1
    sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_long_reset_surfaces_without_retry():
    error = RateLimitedError("wait", retry_after_seconds=3600)
    operation = AsyncMock(side_effect=error)
    sleep = AsyncMock()
    with pytest.raises(RateLimitedError) as raised:
        await retry(operation, RetryPolicy(cap=60), sleep=sleep)
    assert raised.value is error
    assert operation.await_count == 1
    sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_transient_failure_retries_then_returns_result():
    operation = AsyncMock(side_effect=[TransientHarnessError("failed"), "ok"])
    sleep = AsyncMock()
    assert await retry(operation, sleep=sleep, random_sample=lambda: 1) == "ok"
    assert operation.await_count == 2
    sleep.assert_awaited_once_with(2)


@pytest.mark.asyncio
async def test_exhausted_retries_keep_last_failure():
    error = TransientHarnessError("failed")
    operation = AsyncMock(side_effect=error)
    sleep = AsyncMock()
    with pytest.raises(TransientHarnessError) as raised:
        await retry(operation, sleep=sleep, random_sample=lambda: 0)
    assert raised.value is error
    assert operation.await_count == 3
    assert [call.args[0] for call in sleep.await_args_list] == [1.5, 3]


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [PermanentHarnessError("bad input"), asyncio.CancelledError()])
async def test_permanent_errors_and_cancellation_propagate(error):
    operation = AsyncMock(side_effect=error)
    sleep = AsyncMock()
    with pytest.raises(type(error)):
        await retry(operation, sleep=sleep)
    assert operation.await_count == 1
    sleep.assert_not_awaited()


@pytest.mark.parametrize("text,expected", [
    ("try again in 5 minutes", 300), ("Try again in 1 minute", 60),
    ("try again in 2.5 seconds", 2.5), ("try again in 2 hours", 7200),
    ("no hint", None), ("resets 25pm (UTC)", None),
    ("resets 9:80pm (UTC)", None), ("resets 9pm (Not/A_Zone)", None),
])
def test_relative_and_invalid_hints(text, expected):
    assert parse_reset_hint(text) == expected


@pytest.mark.parametrize("now,expected", [
    (datetime(2026, 1, 1, 20, tzinfo=UTC), 9 * 3600),
    (datetime(2026, 1, 2, 6, tzinfo=UTC), 23 * 3600),
    (datetime(2026, 1, 2, 5, tzinfo=UTC), 24 * 3600),
    # The reset crosses the spring DST transition: 10pm to next 9pm takes 22 hours.
    (datetime(2026, 3, 8, 6, tzinfo=UTC), 22 * 3600),
])
def test_named_zone_clock_hints(now, expected):
    assert parse_reset_hint("resets 9pm (America/Los_Angeles)", clock=lambda: now) == expected


def test_naive_clock_rejected():
    with pytest.raises(ValueError, match="timezone-aware"):
        parse_reset_hint("resets 9pm", clock=lambda: datetime(2026, 1, 1, tzinfo=UTC).replace(tzinfo=None))


@pytest.mark.parametrize("kwargs", [
    {"max_retries": -1}, {"max_retries": 1.5}, {"base": -1},
    {"cap": float("inf")}, {"jitter": float("nan")},
])
def test_invalid_policies(kwargs):
    with pytest.raises(ValueError):
        RetryPolicy(**kwargs)


@pytest.mark.parametrize("number,sample", [(0, 0), (1.5, 0), (1, -0.1), (1, 1.1)])
def test_invalid_delay_inputs(number, sample):
    with pytest.raises(ValueError):
        retry_delay(RetryPolicy(), number, jitter_fraction=sample)


def test_large_retry_number_still_respects_cap():
    assert retry_delay(RetryPolicy(max_retries=2000), 2000) == 60
