"""The synchronous bridge validates outputs on one owned async loop."""

import asyncio
import threading

import pytest
from pydantic_ai import ModelResponse, ToolCallPart
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.models.function import FunctionModel

from petri.harness.legacy_bridge import HarnessInferenceProvider
from petri.models import ClarifyingQuestion, DecompositionResult
from petri.reasoning import prompts
from tests.harness_utils import make_scripted_output_model


def test_substance_contract_and_progress():
    payload = {"is_substantive": True, "reason": "Specific claim", "suggested_rewrite": ""}
    progress = []
    with HarnessInferenceProvider(make_scripted_output_model([payload])) as provider:
        assert provider.assess_claim_substance("claim", progress.append) == payload
    assert progress == ["substance: started\n", "substance: completed\n"]


def test_questions_are_typed():
    with HarnessInferenceProvider(make_scripted_output_model([
        {"response": [{"question": "Which region?", "options": ["North", "South"]}]},
    ])) as provider:
        questions = provider.generate_clarifying_questions("claim")
    assert isinstance(questions[0], ClarifyingQuestion)


def test_decomposition_assigns_placeholder_ids():
    with HarnessInferenceProvider(make_scripted_output_model([
        {"nodes": [{"claim_text": "premise"}]},
    ])) as provider:
        result = provider.decompose_claim("claim", [])
    assert isinstance(result, DecompositionResult)
    assert result.cells[0].id == "pending-premises-001-001"
    assert result.cells[0].claim_text == "premise"
    assert result.center_claim == "claim"


def test_why_retries_invalid_output_then_returns_children():
    with HarnessInferenceProvider(make_scripted_output_model([
        {"is_atomic": False, "sub_premises": []},
        {"is_atomic": False, "sub_premises": [{"claim_text": "cause", "is_atomic": True}]},
    ])) as provider:
        result = provider.decompose_why("claim", 1, 1)
    assert result == [{"claim_text": "cause", "is_atomic": True}]


def test_atomic_is_empty_but_invalid_output_raises():
    with HarnessInferenceProvider(make_scripted_output_model([
        {"is_atomic": True, "sub_premises": [], "reason": "A measurement"},
    ])) as provider:
        assert provider.decompose_why("claim", 1, 1) == []
    with HarnessInferenceProvider(make_scripted_output_model([{}]), retries=1) as provider:
        with pytest.raises(UnexpectedModelBehavior):
            provider.decompose_why("claim", 1, 1)


@pytest.mark.parametrize(("method", "args"), [
    ("assess_cell", ("cell", "claim", {}, "skeptic")),
    ("summarize_evidence", ("cell", "claim", "evidence", 1)),
    ("match_evidence", ("evidence", [])),
])
def test_followup_methods_are_explicit(method, args):
    with HarnessInferenceProvider(make_scripted_output_model([{}])) as provider:
        with pytest.raises(NotImplementedError, match="#27"):
            getattr(provider, method)(*args)


def test_bridge_prompts_omit_legacy_suffixes():
    recorded = []
    payloads = iter([
        {"is_substantive": True, "reason": "specific"},
        [{"question": "Scope?"}],
        {"nodes": [{"claim_text": "premise"}]},
        {"is_atomic": True, "sub_premises": []},
    ])

    def reply(messages, info):
        recorded.extend(part.content for message in messages for part in message.parts
                        if getattr(part, "part_kind", "") == "user-prompt")
        payload = next(payloads)
        if isinstance(payload, list):
            payload = {"response": payload}
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, payload)])

    with HarnessInferenceProvider(FunctionModel(reply)) as provider:
        provider.assess_claim_substance("claim")
        provider.generate_clarifying_questions("claim")
        provider.decompose_claim("claim", [])
        provider.decompose_why("claim", 1, 1)
    for prompt, suffix in zip(recorded, [
        prompts.assess_claim_substance_output_suffix(),
        prompts.generate_clarifying_questions_output_suffix(),
        prompts.decompose_claim_output_suffix(5),
        prompts.decompose_why_output_suffix(5),
    ], strict=True):
        assert suffix not in prompt


def test_worker_reuses_loop_across_caller_threads_and_closes():
    loops = []

    async def reply(messages, info):
        loops.append(asyncio.get_running_loop())
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {"is_substantive": True, "reason": "specific"})])

    provider = HarnessInferenceProvider(FunctionModel(reply))
    provider.assess_claim_substance("first")
    thread = threading.Thread(target=provider.assess_claim_substance, args=("second",))
    thread.start()
    thread.join()
    worker = provider._worker.thread
    provider.close()
    provider.close()
    assert len(loops) == 2 and loops[0] is loops[1]
    assert not worker.is_alive()
    assert loops[0].is_closed()
    with pytest.raises(RuntimeError, match="closed"):
        provider.assess_claim_substance("third")


@pytest.mark.asyncio
async def test_sync_call_inside_existing_event_loop():
    with HarnessInferenceProvider(make_scripted_output_model([
        {"is_substantive": False, "reason": "placeholder"},
    ])) as provider:
        assert not provider.assess_claim_substance("test")["is_substantive"]


def test_pi_process_is_reaped_on_exception(fake_pi):
    from petri.harness.pi.model import PiModel
    from petri.harness.pi.transport import PiRpcTransport

    fake_pi.scenario({"kind": "response", "text": "invalid JSON"})
    model = PiModel("fake", transport=PiRpcTransport(binary=str(fake_pi.path)))
    with pytest.raises(UnexpectedModelBehavior), HarnessInferenceProvider(model, retries=0) as provider:
        provider.assess_claim_substance("claim")
    assert model.transport.process.returncode is not None
    assert not provider._worker.thread.is_alive()
