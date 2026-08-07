"""Offline pydantic-ai test harness utilities (v2 migration, M1-harness.4).

Factories and shared fixtures for ``TestModel`` / ``FunctionModel`` pre-loaded
with Petri-shaped canned outputs, typed against the result models in
``petri/models.py``: ``AssessmentResult``, ``DecompositionResult``,
``ClarifyingQuestion``, ``EvidenceMatch``.

These are tiers 1-2 of the v2 testing pattern:

- tier 1 — ``TestModel`` for smoke/shape tests (procedural, schema-valid
  output; pin exact values with ``custom_output_args``);
- tier 2 — ``FunctionModel`` for scripted multi-turn behaviour, e.g. a
  schema-invalid payload followed by a valid one to exercise pydantic-ai's
  validation-retry path deterministically.

The fixtures defined here are registered globally by an import at the bottom
of ``tests/conftest.py``. Tests under ``tests/unit/harness/`` additionally get
an autouse offline guard (see ``tests/unit/harness/conftest.py``).

Note on output shapes (pydantic-ai 2.x): for a list output type such as
``output_type=list[ClarifyingQuestion]`` the internal output tool wraps the
list in a ``response`` key. ``TestModel.custom_output_args`` applies that
wrapping itself, so pass the bare list; ``FunctionModel`` tool-call args must
match the tool schema exactly, so pass ``{"response": [...]}``.
"""

from collections.abc import Sequence
from typing import Any

import pytest
from pydantic_ai import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

# ── Canned payloads ──────────────────────────────────────────────────────
# Cell/colony IDs follow the canonical test scheme from tests/conftest.py
# (dish "test-dish", colony "colony"). Defined literally here to keep this
# module import-independent of conftest.

CANNED_ASSESSMENT_ARGS: dict[str, Any] = {
    "agent": "skeptic",
    "verdict": "PASS",
    "summary": "Claim is supported by primary evidence.",
    "arguments": "Two independent primary sources agree.",
    "evidence": "Peer-reviewed replication published in 2025.",
    "confidence": "HIGH",
    "sources_cited": [
        {
            "url": "https://example.org/primary-study",
            "title": "Primary study",
            "hierarchy_level": 1,
            "finding": "Direct experimental support.",
            "supports_or_contradicts": "supports",
            "confidence": "HIGH",
            "pub_date": "2025-06-01",
            "pub_year": "2025",
        }
    ],
}

# Schema-INVALID on purpose: AssessmentResult requires ``agent`` and
# ``verdict``. Feeding this to the output tool triggers a validation retry.
INVALID_ASSESSMENT_ARGS: dict[str, Any] = {
    "summary": "schema-invalid payload: required fields agent/verdict missing",
}

CANNED_DECOMPOSITION_ARGS: dict[str, Any] = {
    "colony_name": "colony",
    "center_claim": "Central thesis",
    "cells": [
        {
            "id": "test-dish-colony-000-000",
            "colony_id": "test-dish-colony",
            "claim_text": "Central thesis",
            "level": 0,
            "dependencies": [
                "test-dish-colony-001-001",
                "test-dish-colony-001-002",
            ],
        },
        {
            "id": "test-dish-colony-001-001",
            "colony_id": "test-dish-colony",
            "claim_text": "First premise",
            "level": 1,
        },
        {
            "id": "test-dish-colony-001-002",
            "colony_id": "test-dish-colony",
            "claim_text": "Second premise",
            "level": 1,
        },
    ],
    "edges": [
        {
            "from_cell": "test-dish-colony-000-000",
            "to_cell": "test-dish-colony-001-001",
        },
        {
            "from_cell": "test-dish-colony-000-000",
            "to_cell": "test-dish-colony-001-002",
        },
    ],
}

CANNED_CLARIFYING_QUESTIONS_ARGS: list[dict[str, Any]] = [
    {
        "question": "What time frame does the claim cover?",
        "options": ["2020-2025", "All time"],
    },
    {"question": "Which region is in scope?", "options": []},
]

CANNED_EVIDENCE_MATCHES_ARGS: list[dict[str, Any]] = [
    {
        "cell_id": "test-dish-colony-001-001",
        "relevance": 0.9,
        "reason": "Directly supports the first premise.",
    },
    {
        "cell_id": "test-dish-colony-001-002",
        "relevance": 0.4,
        "reason": "Tangentially related to the second premise.",
    },
]


# ── Tier 1: TestModel factories ──────────────────────────────────────────


def make_assessment_test_model() -> TestModel:
    """TestModel pinned to a canned payload valid as ``AssessmentResult``."""
    return TestModel(custom_output_args=CANNED_ASSESSMENT_ARGS)


def make_decomposition_test_model() -> TestModel:
    """TestModel pinned to a canned payload valid as ``DecompositionResult``."""
    return TestModel(custom_output_args=CANNED_DECOMPOSITION_ARGS)


def make_clarifying_questions_test_model() -> TestModel:
    """TestModel pinned to a canned payload valid as ``list[ClarifyingQuestion]``."""
    return TestModel(custom_output_args=CANNED_CLARIFYING_QUESTIONS_ARGS)


def make_evidence_matches_test_model() -> TestModel:
    """TestModel pinned to a canned payload valid as ``list[EvidenceMatch]``."""
    return TestModel(custom_output_args=CANNED_EVIDENCE_MATCHES_ARGS)


# ── Tier 2: FunctionModel scripted-sequence helper ───────────────────────


def make_scripted_output_model(
    turns: Sequence[dict[str, Any]],
    model_name: str = "scripted-output-model",
) -> FunctionModel:
    """Build a FunctionModel that replies to call *i* with ``turns[i]``.

    Each turn is passed verbatim as the args of a ``ToolCallPart`` naming the
    agent's output tool, so a schema-INVALID turn triggers pydantic-ai's
    validation-retry path (a ``RetryPromptPart`` request followed by a fresh
    model call) deterministically — script an invalid payload first and a
    valid one second to test exactly one retry.

    Calls past the end of the script repeat the final turn, so an all-invalid
    script exercises retry-budget exhaustion (``UnexpectedModelBehavior``).

    The turn index is derived from the message history (stateless), so one
    model instance can be reused across independent runs. For list output
    types pass the tool-schema shape, e.g. ``{"response": [...]}``.
    """
    if not turns:
        raise ValueError("turns must contain at least one payload")
    script: list[dict[str, Any]] = list(turns)

    def scripted(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        # Count replies, since the initial request can span several messages.
        replies = sum(isinstance(message, ModelResponse) for message in messages)
        index = min(replies, len(script) - 1)
        tool_name = info.output_tools[0].name
        return ModelResponse(parts=[ToolCallPart(tool_name, script[index])])

    return FunctionModel(scripted, model_name=model_name)


def make_retrying_assessment_model() -> FunctionModel:
    """Bad-then-good ``AssessmentResult`` script: one validation retry, then valid."""
    return make_scripted_output_model(
        [INVALID_ASSESSMENT_ARGS, CANNED_ASSESSMENT_ARGS],
        model_name="retrying-assessment-model",
    )


# ── Shared fixtures (registered via import in tests/conftest.py) ─────────


@pytest.fixture
def assessment_test_model() -> TestModel:
    """TestModel emitting the canned ``AssessmentResult`` payload."""
    return make_assessment_test_model()


@pytest.fixture
def decomposition_test_model() -> TestModel:
    """TestModel emitting the canned ``DecompositionResult`` payload."""
    return make_decomposition_test_model()


@pytest.fixture
def clarifying_questions_test_model() -> TestModel:
    """TestModel emitting the canned ``list[ClarifyingQuestion]`` payload."""
    return make_clarifying_questions_test_model()


@pytest.fixture
def evidence_matches_test_model() -> TestModel:
    """TestModel emitting the canned ``list[EvidenceMatch]`` payload."""
    return make_evidence_matches_test_model()


@pytest.fixture
def scripted_assessment_model() -> FunctionModel:
    """FunctionModel scripting an invalid ``AssessmentResult`` payload, then a valid one."""
    return make_retrying_assessment_model()
