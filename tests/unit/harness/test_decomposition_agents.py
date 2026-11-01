"""Typed decomposition retries, streaming, and SQLite persistence stay offline."""

import json
from datetime import datetime

import pytest
from pydantic import ValidationError
from pydantic_ai import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel

from petri.harness.legacy_bridge import HarnessInferenceProvider
from petri.reasoning.decomposer import decompose_claim
from petri.reasoning.decomposition_agents import (
    DecompositionError, DecompositionInput, five_whys, initial_decomposer,
    initial_instructions, run_decomposition_agent, why_instructions,
)
from petri.reasoning.decomposition_schemas import WhyExpansion
from petri.reasoning.decomposition_settings import DecompositionSettings
from petri.storage import database


def script(payloads):
    calls = []

    async def response(messages, info):
        calls.append(messages)
        payload = payloads[min(len(calls) - 1, len(payloads) - 1)]
        return ModelResponse(parts=[TextPart(json.dumps(payload))])

    return FunctionModel(response), calls


def test_atomicity_requires_explicit_boolean_and_reason():
    for invalid in ({"sub_premises": []}, {"sub_premises": [], "is_atomic": True},
                    {"sub_premises": [], "is_atomic": True, "bedrock_reason": " "}):
        with pytest.raises(ValidationError):
            WhyExpansion.model_validate(invalid)
    atomic = WhyExpansion(sub_premises=[], is_atomic=True, bedrock_reason="A definition")
    assert atomic.is_atomic


@pytest.mark.asyncio
async def test_invalid_then_valid_retries():
    model, calls = script([{}, {"nodes": [{"claim_text": "A falsifiable premise"}]}])
    result = await run_decomposition_agent(initial_decomposer, DecompositionInput("claim"), model=model)
    assert result.nodes[0].claim_text == "A falsifiable premise"
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_exhaustion_raises_without_an_atomic_result():
    model, calls = script([{}])
    with pytest.raises(DecompositionError, match="five_whys failed"):
        await run_decomposition_agent(five_whys, DecompositionInput("claim"), model=model)
    assert len(calls) == 3


def test_named_agents_preserve_the_initial_step_wording():
    assert initial_decomposer.name != five_whys.name
    for instructions in (initial_instructions(DecompositionInput("claim")), why_instructions(DecompositionInput("claim"))):
        for step in ("STEP 1 — BRAINSTORM", "STEP 2 — PRIORITIZE", "STEP 3 — SELECT", "STEP 4 — EMIT JSON"):
            assert step in instructions


@pytest.mark.asyncio
async def test_stream_handler_delivers_text_chunks():
    async def stream(messages, info):
        yield '{"nodes": ['
        yield '{"claim_text": "A streamed premise"}'
        yield ']}'

    chunks = []
    result = await run_decomposition_agent(initial_decomposer, DecompositionInput("claim"),
                                           model=FunctionModel(stream_function=stream),
                                           on_progress=chunks.append, stream=True)
    assert result.nodes[0].claim_text == "A streamed premise"
    assert '{"nodes": [' in "".join(chunks)
    assert chunks[-1] == "initial_decomposer: completed\n"


def test_atomic_stop_and_created_at_round_trip(tmp_path):
    model, calls = script([
        {"nodes": [{"claim_text": "A meter is a defined unit", "relation": "supports"}]},
        {"is_atomic": True, "sub_premises": [], "bedrock_reason": "A unit definition"},
    ])
    path = tmp_path / "petri.sqlite"
    settings = DecompositionSettings(depth=3, per_layer=5, model="test")
    with HarnessInferenceProvider(model) as runner:
        result = decompose_claim("claim", [], "dish", "colony", None, agent_runner=runner,
                                 db_path=path, settings=settings)
    assert len(calls) == 2
    assert len(result.cells) == 2
    rows = database.list_cells(path)
    assert len(rows) == 2 and len(database.read_edges(path)) == 1
    for row in rows:
        assert datetime.fromisoformat(row["created_at"]).utcoffset().total_seconds() == 0
    assert rows[1]["is_atomic"] == 1
    assert rows[1]["bedrock_reason"] == "A unit definition"
    assert rows[1]["relation"] == "supports"


@pytest.mark.parametrize("created_at", [None, ""])
def test_database_rejects_missing_creation_time(tmp_path, created_at):
    import sqlite3

    with pytest.raises(sqlite3.IntegrityError):
        database.put_cell(tmp_path / "petri.sqlite", {"id": "dish-colony-001-001", "colony": "dish-colony",
                          "level": 1, "seq": 1, "claim_text": "claim", "created_at": created_at})
