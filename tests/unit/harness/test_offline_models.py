"""Offline TestModel/FunctionModel tests for the v2 harness fixtures.

Covers the M1-harness.4 acceptance criteria:
- scripted FunctionModel (schema-invalid payload, then valid) drives an
  ``Agent(output_type=AssessmentResult)`` through exactly one validation
  retry to a validated final output;
- plain ``TestModel`` produces a validated ``DecompositionResult`` with no
  custom scripting;
- the autouse offline guard (tests/unit/harness/conftest.py) trips loudly on
  network and real-CLI attempts.
"""

import socket
import subprocess

import pytest
from pydantic_ai import Agent, RetryPromptPart
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.models.test import TestModel

from petri.models import (
    AssessmentResult,
    ClarifyingQuestion,
    DecompositionResult,
    EvidenceMatch,
)
from tests.harness_utils import (
    CANNED_ASSESSMENT_ARGS,
    CANNED_CLARIFYING_QUESTIONS_ARGS,
    CANNED_DECOMPOSITION_ARGS,
    CANNED_EVIDENCE_MATCHES_ARGS,
    INVALID_ASSESSMENT_ARGS,
    make_scripted_output_model,
)


class TestCannedPayloadShapes:
    """The canned payloads really are typed against the petri result models."""

    def test_canned_assessment_validates(self):
        result = AssessmentResult.model_validate(CANNED_ASSESSMENT_ARGS)
        assert result.agent == "skeptic"
        assert result.sources_cited[0].url.startswith("https://")

    def test_invalid_assessment_payload_is_invalid(self):
        with pytest.raises(ValueError):
            AssessmentResult.model_validate(INVALID_ASSESSMENT_ARGS)

    def test_canned_decomposition_validates(self):
        result = DecompositionResult.model_validate(CANNED_DECOMPOSITION_ARGS)
        assert len(result.cells) == 3
        assert len(result.edges) == 2


class TestScriptedFunctionModel:
    """Tier 2: FunctionModel scripting the validation-retry path."""

    def test_bad_then_good_assessment_retries_once(self, scripted_assessment_model):
        """Bad payload then good: one retry, then a validated AssessmentResult."""
        agent = Agent(scripted_assessment_model, output_type=AssessmentResult, retries=2)
        result = agent.run_sync("assess the cell")

        assert isinstance(result.output, AssessmentResult)
        assert result.output.agent == CANNED_ASSESSMENT_ARGS["agent"]
        assert result.output.verdict == CANNED_ASSESSMENT_ARGS["verdict"]
        assert [s.url for s in result.output.sources_cited] == ["https://example.org/primary-study"]

        retry_parts = [
            part
            for message in result.all_messages()
            for part in message.parts
            if isinstance(part, RetryPromptPart)
        ]
        assert len(retry_parts) == 1, "expected exactly one validation retry"
        assert result.usage.requests == 2, "expected the bad call plus the good call"

    def test_all_invalid_script_exhausts_retry_budget(self):
        """An always-invalid script surfaces retry-budget exhaustion loudly."""
        model = make_scripted_output_model([INVALID_ASSESSMENT_ARGS])
        agent = Agent(model, output_type=AssessmentResult, retries=1)
        with pytest.raises(UnexpectedModelBehavior, match="Exceeded maximum output retries"):
            agent.run_sync("assess the cell")


class TestTestModelSmoke:
    """Tier 1: TestModel smoke/shape tests, no scripting required."""

    def test_plain_test_model_produces_validated_decomposition(self):
        """A bare TestModel yields a schema-valid DecompositionResult."""
        agent = Agent(TestModel(), output_type=DecompositionResult)
        result = agent.run_sync("decompose the claim")
        assert isinstance(result.output, DecompositionResult)

    def test_assessment_test_model_fixture(self, assessment_test_model):
        agent = Agent(assessment_test_model, output_type=AssessmentResult)
        result = agent.run_sync("assess the cell")
        assert result.output == AssessmentResult.model_validate(CANNED_ASSESSMENT_ARGS)

    def test_decomposition_test_model_fixture(self, decomposition_test_model):
        agent = Agent(decomposition_test_model, output_type=DecompositionResult)
        result = agent.run_sync("decompose the claim")
        assert result.output.colony_name == "colony"
        assert [cell.id for cell in result.output.cells] == [
            "test-dish-colony-000-000",
            "test-dish-colony-001-001",
            "test-dish-colony-001-002",
        ]
        assert all(cell.level >= 0 for cell in result.output.cells)

    def test_clarifying_questions_test_model_fixture(self, clarifying_questions_test_model):
        agent = Agent(clarifying_questions_test_model, output_type=list[ClarifyingQuestion])
        result = agent.run_sync("clarify the claim")
        assert [q.question for q in result.output] == [
            item["question"] for item in CANNED_CLARIFYING_QUESTIONS_ARGS
        ]

    def test_evidence_matches_test_model_fixture(self, evidence_matches_test_model):
        agent = Agent(evidence_matches_test_model, output_type=list[EvidenceMatch])
        result = agent.run_sync("match the evidence")
        assert [m.cell_id for m in result.output] == [
            item["cell_id"] for item in CANNED_EVIDENCE_MATCHES_ARGS
        ]
        assert result.output[0].relevance == pytest.approx(0.9)


class TestOfflineGuard:
    """The autouse guard trips loudly on network and real-CLI attempts."""

    def test_create_connection_is_blocked(self):
        with pytest.raises(RuntimeError, match="Offline guard"):
            socket.create_connection(("example.com", 443))

    def test_raw_socket_connect_is_blocked(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            with pytest.raises(RuntimeError, match="Offline guard"):
                sock.connect(("127.0.0.1", 9))
        finally:
            sock.close()

    def test_real_cli_subprocess_is_blocked(self):
        with pytest.raises(RuntimeError, match="Offline guard"):
            subprocess.run(["claude", "--version"], capture_output=True, check=False)
