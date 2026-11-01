"""Real sequential agent hand-offs with one shared debate budget."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from pydantic_ai import Agent, capture_run_messages
from pydantic_ai.exceptions import UnexpectedModelBehavior, UsageLimitExceeded
from pydantic_ai.messages import ModelMessage
from pydantic_ai.usage import RunUsage, UsageLimits

from petri.agents.config import UsageLimitConfig
from petri.agents.contract import AgentName, DebateProtocol
from petri.agents.outputs import AgentAssessment
from petri.harness.errors import HarnessError

if TYPE_CHECKING:
    from petri.models import Event
    from petri.storage.database import SQLiteEventLog


class DebateStatus(StrEnum):
    COMPLETE = "complete"
    LIMIT_EXCEEDED = "limit_exceeded"
    EXECUTION_ERROR = "execution_error"


@dataclass(frozen=True)
class DebateExchange:
    speaker: AgentName
    round: int
    assessment: AgentAssessment


@dataclass
class DebateResult:
    protocol: DebateProtocol
    status: DebateStatus
    exchanges: list[DebateExchange] = field(default_factory=list)
    message_history: list[ModelMessage] = field(default_factory=list)
    usage: RunUsage = field(default_factory=RunUsage)
    detail: str = ""


async def run_debate(
    protocol: DebateProtocol,
    agents: Mapping[AgentName, Agent[Any, AgentAssessment]],
    claim: str,
    limits: UsageLimitConfig,
    *,
    message_history: Sequence[ModelMessage] = (),
) -> DebateResult:
    """Retain actual messages between turns and return partial progress on failure."""
    if any(name not in agents for name in protocol.pairing):
        raise ValueError("Both debate participants must be present in the built roster.")
    result = DebateResult(
        protocol=protocol, status=DebateStatus.COMPLETE, message_history=list(message_history)
    )
    budget = UsageLimits(
        request_limit=limits.request_limit,
        total_tokens_limit=limits.total_tokens_limit,
        tool_calls_limit=limits.tool_calls_limit,
    )
    turns = []
    for number in range(1, protocol.rounds + 1):
        if number == protocol.rounds and protocol.final_word is not None:
            turns.append((number, protocol.final_word))
        else:
            turns.extend((number, name) for name in protocol.pairing)
    for number, name in turns:
        prompt = (
            f"Debate claim: {claim}\nPurpose: {protocol.purpose}\n"
            f"Turn {len(result.exchanges) + 1}: respond as {name.value} to the prior "
            "participant, using the actual conversation history. Preserve source attribution."
        )
        with capture_run_messages() as attempted:
            try:
                response = await agents[name].run(
                    prompt,
                    message_history=result.message_history,
                    usage=result.usage,
                    usage_limits=budget,
                )
            except (UsageLimitExceeded, UnexpectedModelBehavior, HarnessError) as error:
                result.status = (
                    DebateStatus.LIMIT_EXCEEDED
                    if isinstance(error, UsageLimitExceeded)
                    else DebateStatus.EXECUTION_ERROR
                )
                result.detail = str(error)
                if attempted:
                    result.message_history = list(attempted)
                return result
        result.message_history = response.all_messages()
        result.exchanges.append(DebateExchange(name, number, response.output))
    return result


def log_debate(
    target: SQLiteEventLog,
    cell_id: str,
    iteration: int,
    result: DebateResult,
    *,
    round_number: int = 0,
) -> Event:
    """Append through the domain-event seam, preserving the standard event envelope."""
    from petri.storage.event_log import append_event

    summary = json.dumps(
        {
            "status": result.status.value,
            "detail": result.detail,
            "purpose": result.protocol.purpose,
            "exchanges": [
                {
                    "speaker": item.speaker.value,
                    "round": item.round,
                    "assessment": item.assessment.model_dump(mode="json"),
                }
                for item in result.exchanges
            ],
            "usage": {
                "requests": result.usage.requests,
                "input_tokens": result.usage.input_tokens,
                "output_tokens": result.usage.output_tokens,
            },
        }
    )
    return append_event(
        target,
        cell_id,
        "debate_mediated",
        "cell_lead",
        iteration,
        {
            "from_agent": result.protocol.pairing[0].value,
            "to_agent": result.protocol.pairing[1].value,
            "exchange_summary": summary,
        },
        round_number=round_number,
        node="debate:" + ":".join(name.value for name in result.protocol.pairing),
    )
