"""Typed domain attributes derived from explicit identity and measured usage."""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import Literal

from opentelemetry.trace import Span
from pydantic_ai.usage import RunUsage

from petri.agents.contract import AGENT_SPECS, AgentName, BlockingMode
from petri.telemetry.config import ModelPrice
from petri.telemetry.setup import get_tracer

Attributes = dict[str, str | int | float | bool]
CommandName = Literal["seed", "grow", "scan", "feed", "re-decompose"]


class AttributeKey(StrEnum):
    COMMAND = "petri.command"
    DISH_ID = "petri.dish_id"
    COLONY_ID = "petri.colony_id"
    CELL_ID = "petri.cell_id"
    LEVEL = "petri.level"
    SEQ = "petri.seq"
    PHASE = "petri.phase"
    AGENT_ROLE = "petri.agent_role"
    AGENT_BLOCKING = "petri.agent_blocking"
    VERDICT = "petri.verdict"
    ITERATION = "petri.iteration"
    WEAKEST_LINK = "petri.weakest_link"
    TOKENS_INPUT = "petri.tokens_input"
    TOKENS_OUTPUT = "petri.tokens_output"
    COST_USD = "petri.cost_usd"
    OUTCOME = "petri.outcome"


@dataclass(frozen=True)
class CellSpanContext:
    dish_id: str
    colony_id: str
    level: int
    seq: int

    def __post_init__(self) -> None:
        if not self.dish_id or not self.colony_id.startswith(self.dish_id + "-"):
            raise ValueError("colony_id must include its explicit dish_id prefix.")
        if self.colony_id == self.dish_id + "-":
            raise ValueError("colony_id must include a colony slug.")
        if (
            type(self.level) is not int
            or type(self.seq) is not int
            or min(self.level, self.seq) < 0
        ):
            raise ValueError("Cell level and sequence must be nonnegative integers.")

    @property
    def cell_id(self) -> str:
        from petri.models import build_cell_key

        colony_slug = self.colony_id[len(self.dish_id) + 1 :]
        return build_cell_key(self.dish_id, colony_slug, self.level, self.seq)


def cell_attributes(context: CellSpanContext) -> Attributes:
    return {
        AttributeKey.DISH_ID: context.dish_id,
        AttributeKey.COLONY_ID: context.colony_id,
        AttributeKey.CELL_ID: context.cell_id,
        AttributeKey.LEVEL: context.level,
        AttributeKey.SEQ: context.seq,
    }


def agent_attributes(agent: AgentName) -> Attributes:
    spec = AGENT_SPECS[agent]
    return {
        AttributeKey.AGENT_ROLE: agent.value,
        AttributeKey.AGENT_BLOCKING: spec.blocking_mode != BlockingMode.NON_BLOCKING,
    }


def verdict_attributes(
    verdict: str,
    *,
    usage: RunUsage | None = None,
    model: str | None = None,
    prices: Mapping[str, ModelPrice] | None = None,
    outcome: str | None = None,
) -> Attributes:
    attributes: Attributes = {AttributeKey.VERDICT: verdict}
    if outcome is not None:
        attributes[AttributeKey.OUTCOME] = outcome
    measured = usage is not None and (
        usage.unmeasured_requests == 0 or usage.input_tokens > 0 or usage.output_tokens > 0
    )
    if measured and usage is not None:
        attributes[AttributeKey.TOKENS_INPUT] = usage.input_tokens
        attributes[AttributeKey.TOKENS_OUTPUT] = usage.output_tokens
        if model and prices and model in prices and usage.unmeasured_requests == 0:
            price = prices[model]
            cost = (
                Decimal(str(price.input_per_million)) * usage.input_tokens
                + Decimal(str(price.output_per_million)) * usage.output_tokens
            ) / Decimal(1_000_000)
            attributes[AttributeKey.COST_USD] = float(cost)
    return attributes


def set_cell_attributes(span: Span, context: CellSpanContext) -> Attributes:
    attributes = cell_attributes(context)
    span.set_attributes(attributes)
    return attributes


def set_agent_attributes(span: Span, agent: AgentName) -> Attributes:
    attributes = agent_attributes(agent)
    span.set_attributes(attributes)
    return attributes


def set_verdict_attributes(
    span: Span,
    verdict: str,
    *,
    usage: RunUsage | None = None,
    model: str | None = None,
    prices: Mapping[str, ModelPrice] | None = None,
    outcome: str | None = None,
) -> Attributes:
    attributes = verdict_attributes(
        verdict, usage=usage, model=model, prices=prices, outcome=outcome
    )
    span.set_attributes(attributes)
    return attributes


@contextmanager
def phase_span(
    phase: str,
    context: CellSpanContext,
    *,
    iteration: int,
    agent: AgentName | None = None,
    weakest_link: str | None = None,
) -> Iterator[Span]:
    attributes = cell_attributes(context)
    attributes.update({AttributeKey.PHASE: phase, AttributeKey.ITERATION: iteration})
    if agent is not None:
        attributes.update(agent_attributes(agent))
    if weakest_link is not None:
        attributes[AttributeKey.WEAKEST_LINK] = weakest_link
    with get_tracer("petri.phases").start_as_current_span(
        f"petri.phase.{phase}", attributes=attributes
    ) as span:
        yield span


@contextmanager
def root_command_span(command: CommandName, dish_id: str) -> Iterator[Span]:
    attributes: Attributes = {AttributeKey.COMMAND: command, AttributeKey.DISH_ID: dish_id}
    with get_tracer("petri.cli").start_as_current_span(
        f"petri.{command}", attributes=attributes
    ) as span:
        yield span
