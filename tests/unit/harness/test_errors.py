"""Failure classification and event diagnostics stay independent of transports."""

import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from petri.harness.errors import (
    AuthExpiredError,
    FailureClassifier,
    FailureDetail,
    PermanentHarnessError,
    RateLimitedError,
    TransientHarnessError,
)
from petri.models import Event


def test_detail_round_trip_in_event():
    detail = FailureDetail(
        exit_code=1, stderr_tail="failed", stdout_tail="reset",
        last_stream_event={"type": "error", "code": 429}, partial_output="partial",
    )
    assert FailureDetail.model_validate_json(detail.model_dump_json()) == detail
    event = Event(
        id="failure-1", cell_id="cell-1", timestamp="2026-01-01T00:00:00Z",
        type="verdict_issued", agent="test", iteration=1,
        data={"verdict": "EXECUTION_ERROR", "failure": detail.model_dump(mode="json")},
    )
    assert json.loads(event.model_dump_json())["data"]["failure"]["exit_code"] == 1


def test_detail_bounds_diagnostics_and_rejects_non_json_values():
    detail = FailureDetail(stdout_tail="x" * 10000, last_stream_event={"text": "y" * 20000})
    assert detail.stdout_tail == "x" * 4096
    assert detail.last_stream_event["truncated"] is True
    assert FailureDetail.model_validate_json(detail.model_dump_json()) == detail
    assert len(detail.model_dump_json()) < 9000
    with pytest.raises(ValidationError):
        FailureDetail(last_stream_event=object())
    with pytest.raises(ValidationError):
        FailureDetail(exit_code="1")


def test_stdout_reset_hint_with_empty_stderr():
    error = FailureClassifier().classify(
        1, "", "resets 9pm (America/Los_Angeles)",
        clock=lambda: datetime(2026, 1, 1, 20, tzinfo=UTC),
    )
    assert isinstance(error, RateLimitedError)
    assert error.retry_after_seconds == 9 * 3600
    assert error.detail.stdout_tail == "resets 9pm (America/Los_Angeles)"


@pytest.mark.parametrize("stderr,stdout,expected", [
    ("authentication expired", "rate limit", AuthExpiredError),
    ("rate limit", "model not found", PermanentHarnessError),
    ("model overloaded", "", TransientHarnessError),
    ("", "try again in 5 minutes", RateLimitedError),
    ("", "network timeout", TransientHarnessError),
    ("unrecognized failure", "", PermanentHarnessError),
    ("", "", TransientHarnessError),
])
def test_classification(stderr, stdout, expected):
    assert isinstance(FailureClassifier().classify(1, stderr, stdout), expected)


def test_backend_markers_and_structured_codes():
    classifier = FailureClassifier(
        permanent_markers=("bad input",), transient_codes=("busy",),
        rate_limit_codes=("quota",), auth_codes=("expired",),
    )
    assert isinstance(classifier.classify(1, code="busy"), TransientHarnessError)
    assert isinstance(classifier.classify(0, code="quota"), RateLimitedError)
    assert isinstance(classifier.classify(1, "bad input", code="busy"), PermanentHarnessError)
    assert isinstance(classifier.classify(1, "rate limit", code="expired"), AuthExpiredError)
    assert isinstance(FailureClassifier(empty_is_transient=False).classify(1), PermanentHarnessError)
    assert "Sign in" in str(classifier.classify(1, code="expired"))


@pytest.mark.parametrize("hint", [-1, float("inf"), float("nan")])
def test_invalid_retry_hint_rejected(hint):
    with pytest.raises(ValueError):
        RateLimitedError("limited", retry_after_seconds=hint)
