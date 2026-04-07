"""Pydantic models for the Petri data layer.

Shared by the event log manager and queue manager. Enforces type safety on all
event payloads and queue entries at write time -- invalid data is rejected
before it reaches the append-only JSONL files.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Optional, Protocol, runtime_checkable

import logging

from pydantic import AliasChoices, BaseModel, Field, field_validator

from petri.config import MAX_CONCURRENT, MAX_ITERATIONS


# ── Enums ────────────────────────────────────────────────────────────────


class NodeStatus(str, Enum):
    NEW = "NEW"
    RESEARCH = "RESEARCH"
    RED_TEAM = "RED_TEAM"
    EVALUATE = "EVALUATE"
    VALIDATED = "VALIDATED"
    DISPROVEN = "DISPROVEN"
    NEEDS_EXPERIMENT = "NEEDS_EXPERIMENT"
    DEFER_OPEN = "DEFER_OPEN"
    DEFER_CLOSED = "DEFER_CLOSED"
    STALLED = "STALLED"


class QueueState(str, Enum):
    queued = "queued"
    socratic_active = "socratic_active"
    research_active = "research_active"
    critique_active = "critique_active"
    mediating = "mediating"
    converged = "converged"
    stalled = "stalled"
    needs_human = "needs_human"
    red_team_active = "red_team_active"
    evaluating = "evaluating"
    done = "done"
    deferred_open = "deferred_open"
    deferred_closed = "deferred_closed"
    sync_conflict = "sync_conflict"


class EventType(str, Enum):
    search_executed = "search_executed"
    source_reviewed = "source_reviewed"
    freshness_checked = "freshness_checked"
    verdict_issued = "verdict_issued"
    evidence_appended = "evidence_appended"
    debate_mediated = "debate_mediated"
    convergence_checked = "convergence_checked"
    node_reopened = "node_reopened"
    propagation_triggered = "propagation_triggered"
    decomposition_created = "decomposition_created"
    decomposition_audit = "decomposition_audit"


class HierarchyLevel(int, Enum):
    direct_measurement = 1
    authoritative_docs = 2
    derived_calculation = 3
    expert_consensus = 4
    single_expert = 5
    community_report = 6


class SupportsOrContradicts(str, Enum):
    supports = "supports"
    contradicts = "contradicts"


class Confidence(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


# ── Event Data Payloads ──────────────────────────────────────────────────


class SearchExecutedData(BaseModel):
    query: str
    sources_found: int = Field(ge=0)


class SourceReviewedData(BaseModel):
    url: str
    title: str = ""
    pub_date: str = ""
    hierarchy_level: Optional[HierarchyLevel] = None
    finding: str = ""
    supports_or_contradicts: Optional[SupportsOrContradicts] = None
    confidence: Optional[Confidence] = None


class FreshnessCheckedData(BaseModel):
    source_url: str
    original_date: str = ""
    verdict: str = ""
    notes: str = ""


class VerdictIssuedData(BaseModel):
    verdict: str
    summary: str = ""
    arguments: str = ""
    evidence: str = ""
    confidence: str = ""
    sources_cited: list[dict] = Field(default_factory=list)


class EvidenceAppendedData(BaseModel):
    summary: str


class DebateMediatedData(BaseModel):
    from_agent: str
    to_agent: str
    exchange_summary: str = ""


class ConvergenceCheckedData(BaseModel):
    converged: bool
    blocking_verdicts: Optional[dict] = None
    weakest_link: Optional[str] = None
    focused_directive: Optional[str] = None


class NodeReopenedData(BaseModel):
    trigger: str
    prior_status: str


class PropagationTriggeredData(BaseModel):
    reopened_node_id: str
    flagged_dependents: list[str]


class DecompositionCreatedData(BaseModel):
    parent_node_id: str
    child_node_ids: list[str]


class DecompositionAuditData(BaseModel):
    """Logged when convergence failure triggers re-examination of the decomposition."""

    iteration: int = 0
    suggestion: str = ""
    should_restructure: bool = False


# ── Event Data Dispatch ──────────────────────────────────────────────────


EVENT_DATA_MODELS: dict[str, type[BaseModel]] = {
    "search_executed": SearchExecutedData,
    "source_reviewed": SourceReviewedData,
    "freshness_checked": FreshnessCheckedData,
    "verdict_issued": VerdictIssuedData,
    "evidence_appended": EvidenceAppendedData,
    "debate_mediated": DebateMediatedData,
    "convergence_checked": ConvergenceCheckedData,
    "node_reopened": NodeReopenedData,
    "propagation_triggered": PropagationTriggeredData,
    "decomposition_created": DecompositionCreatedData,
    "decomposition_audit": DecompositionAuditData,
}


def validate_event_data(event_type: str, data: dict) -> dict:
    """Validate event data against its Pydantic model. Returns validated dict."""
    model_cls = EVENT_DATA_MODELS.get(event_type)
    if model_cls is None:
        raise ValueError(f"Unknown event type: {event_type}")
    validated = model_cls.model_validate(data)
    return validated.model_dump(exclude_none=True)


# ── Core Entity Models ───────────────────────────────────────────────────


class Event(BaseModel):
    """An immutable record of an agent action."""

    id: str
    node_id: str
    timestamp: str
    type: EventType
    agent: str
    iteration: int = Field(ge=0)
    data: dict


class Verdict(BaseModel):
    """A verdict event -- convenience wrapper for querying."""

    node_id: str
    agent: str
    iteration: int
    verdict: str
    summary: str = ""


class Node(BaseModel):
    """A unit of logic -- claim, premise, or assumption."""

    id: str  # composite key: {dish}-{colony}-{level}-{seq}
    colony_id: str  # composite: {dish}-{colony}
    claim_text: str
    level: int = Field(ge=0)
    status: NodeStatus = NodeStatus.NEW
    dependencies: list[str] = Field(default_factory=list)  # composite node keys
    dependents: list[str] = Field(default_factory=list)  # composite node keys
    created_at: str = ""  # ISO 8601 UTC


class Edge(BaseModel):
    """A directed dependency between nodes."""

    from_node: str  # composite node key (the node that depends)
    to_node: str  # composite node key (the dependency)
    edge_type: str = "intra_colony"  # "intra_colony" or "cross_colony"


class Colony(BaseModel):
    """A connected DAG of nodes rooted at a colony center."""

    id: str  # composite: {dish}-{colony}
    dish: str
    center_claim: str
    center_node_id: str
    clarifications: list[dict] = Field(default_factory=list)  # list of {question, answer}
    node_paths: dict[str, str] = Field(default_factory=dict)  # node_id -> relative dir path
    created_at: str = ""


class PetriDish(BaseModel):
    """An independent research unit. Multiple dishes can coexist on a filesystem."""

    id: str  # dish slug
    path: str  # filesystem path to .petri/ root
    colonies: list[Colony] = Field(default_factory=list)
    config: dict = Field(default_factory=dict)  # parsed petri.yaml
    created_at: str = ""


class QueueEntry(BaseModel):
    """A node's position in the validation workflow."""

    node_id: str  # composite node key
    queue_state: QueueState = QueueState.queued
    iteration: int = Field(default=0, ge=0)
    max_iterations: int = Field(default=MAX_ITERATIONS, ge=1)
    cycle_start_iteration: int = Field(default=0, ge=0)
    weakest_link: Optional[str] = None
    focused_directive: Optional[str] = None
    entered_at: str = ""
    last_activity: str = ""


class AgentRole(BaseModel):
    """An agent's role definition from petri.yaml or defaults."""

    name: str
    display_name: str
    phase: Optional[int] = None  # 1 or 2, None for leads/post-convergence
    blocking: str = "false"  # "true", "false", or "conditional"
    is_lead: bool = False
    scope: Optional[str] = None  # "decomposition", "pipeline", "red_team", or None
    verdicts_pass: list[str] = Field(default_factory=list)
    verdicts_block: list[str] = Field(default_factory=list)
    redirect_on_block: Optional[str] = None
    instruction: str = ""  # Agent-specific prompt instruction from config


class Debate(BaseModel):
    """A structured debate pairing from petri.yaml or defaults."""

    pair: tuple[str, str]
    rounds: float
    purpose: str


class PetriConfig(BaseModel):
    """Top-level configuration parsed from petri.yaml."""

    name: str = ""
    model: dict = Field(default_factory=dict)  # {name, provider}
    harness: str = "claude-code"
    max_iterations: int = MAX_ITERATIONS
    max_concurrent: int = MAX_CONCURRENT
    agents: dict[str, AgentRole] = Field(default_factory=dict)
    debates: list[Debate] = Field(default_factory=list)
    source_hierarchy: dict = Field(default_factory=dict)


# ── Result Models (typed returns for provider and processor) ─────────────


class SourceCitation(BaseModel):
    """A source cited by an agent during assessment.

    Every source should have a valid URL. The ``url`` field accepts
    legacy data via the ``url_or_name`` alias.
    """

    url: str = Field(
        default="",
        validation_alias=AliasChoices("url", "url_or_name"),
    )
    title: str = ""
    hierarchy_level: int = 6  # default to weakest (community report)
    finding: str = ""
    supports_or_contradicts: str = "supports"
    confidence: Optional[str] = None
    pub_date: str = ""
    pub_year: str = ""

    @field_validator("url")
    @classmethod
    def _warn_missing_url(cls, url_value: str) -> str:
        if url_value and not url_value.startswith(("http://", "https://")):
            logging.getLogger("petri.models").warning(
                "Source without valid URL: %s", url_value[:80]
            )
        return url_value


class AssessmentResult(BaseModel):
    """Result from an agent's assess_node call."""

    agent: str
    verdict: str
    summary: str = ""
    arguments: str = ""
    evidence: str = ""
    confidence: str = ""
    sources_cited: list[SourceCitation] = Field(default_factory=list)


class ConvergenceOutcome(BaseModel):
    """Result from a convergence check."""

    outcome: str  # "converged", "iterate", "circuit_breaker", "short_circuit"
    type: Optional[str] = None  # for short_circuit: "needs_experiment", "defer_open"
    weakest_link: Optional[str] = None


class EvaluationResult(BaseModel):
    """Result from the evidence evaluation phase."""

    verdict: str  # EVIDENCE_CONFIRMS, EVIDENCE_REFUTES, EVIDENCE_INCONCLUSIVE
    final_status: str  # VALIDATED, DISPROVEN, DEFER_OPEN


class ProcessNodeResult(BaseModel):
    """Result from processing a single node through the pipeline."""

    node_id: str
    final_state: str
    iterations: int = 0
    events_logged: int = 0
    final_iteration: int = 0
    error: Optional[str] = None


class QueueProcessingResult(BaseModel):
    """Aggregated result from processing the queue."""

    processed: int = 0
    succeeded: int = 0
    failed: int = 0
    stalled: int = 0
    results: list[ProcessNodeResult] = Field(default_factory=list)
    dry_run: bool = False
    would_process: list[str] = Field(default_factory=list)


class IngestionResult(BaseModel):
    """Result from content ingestion (URL, file, PDF, or text)."""

    source_type: str  # "url", "file", "pdf", "text"
    source: str
    title: str = ""
    content: str = ""
    metadata: dict = Field(default_factory=dict)


class DebateExchange(BaseModel):
    """A single exchange in a debate."""

    speaker: str
    content: str
    round: float


class DebateResult(BaseModel):
    """Result from mediating a debate between two agents."""

    pair: tuple[str, str]
    rounds: float
    purpose: str
    exchanges: list[DebateExchange] = Field(default_factory=list)
    summary: str = ""


class ConvergenceCheckResult(BaseModel):
    """Result from check_convergence."""

    converged: bool
    blocking_results: dict[str, dict] = Field(default_factory=dict)
    non_blocking_results: dict[str, dict] = Field(default_factory=dict)
    missing_blocking: list[str] = Field(default_factory=list)
    weakest_link: Optional[str] = None


class ShortCircuitCondition(BaseModel):
    """A short-circuit condition detected during convergence."""

    type: str  # "needs_experiment", "defer_open"
    agent: str
    verdict: str


class DecompositionResult(BaseModel):
    """Result from decomposing a claim into a colony DAG."""

    nodes: list["Node"] = Field(default_factory=list)
    edges: list["Edge"] = Field(default_factory=list)
    colony_name: str = ""
    center_claim: str = ""


class ClarifyingQuestion(BaseModel):
    """A question to clarify a claim before decomposition."""

    question: str
    options: list[str] = Field(default_factory=list)
    answer: Optional[str] = None


class EvidenceMatch(BaseModel):
    """A match between new evidence and an existing node."""

    node_id: str
    relevance: float = 0.0
    reason: str = ""


# ── Composite Key Utilities ──────────────────────────────────────────────


def build_node_key(dish: str, colony: str, level: int, seq: int) -> str:
    """Build a composite node key: {dish}-{colony}-{level:03d}-{seq:03d}"""
    return f"{dish}-{colony}-{level:03d}-{seq:03d}"


def build_event_key(node_key: str, hex_id: str) -> str:
    """Build a composite event key: {node_key}-{8hex}"""
    return f"{node_key}-{hex_id}"


def parse_key(key: str, dish_id: str | None = None) -> dict:
    """Parse a composite key from right to left.

    Returns dict with available segments:
    - Always: 'raw' (original key)
    - If event key (5+ segments ending in 8-hex): 'event_hex', 'seq', 'level', 'colony_prefix'
    - If node key (ends in two 3-digit groups): 'seq', 'level', 'colony_prefix'
    - If dish_id provided: 'dish', 'colony' (split from colony_prefix)
    """
    parts = key.split("-")
    result: dict[str, Any] = {"raw": key}

    # Try to parse from the right
    # Check if last segment is an 8-char hex (event key)
    if len(parts) >= 5 and re.match(r"^[0-9a-f]{8}$", parts[-1]):
        result["event_hex"] = parts[-1]
        parts = parts[:-1]

    # Check if last two segments are 3-digit numbers (level + seq)
    if (
        len(parts) >= 3
        and re.match(r"^\d{3}$", parts[-1])
        and re.match(r"^\d{3}$", parts[-2])
    ):
        result["seq"] = int(parts[-1])
        result["level"] = int(parts[-2])
        result["colony_prefix"] = "-".join(parts[:-2])

        if dish_id:
            # Split colony_prefix into dish and colony
            prefix = result["colony_prefix"]
            if prefix.startswith(dish_id + "-"):
                result["dish"] = dish_id
                result["colony"] = prefix[len(dish_id) + 1 :]
            else:
                result["dish"] = dish_id
                result["colony"] = prefix

    return result


def parent_key(key: str) -> str:
    """Strip the last segment from a composite key."""
    parts = key.rsplit("-", 1)
    if len(parts) > 1:
        return parts[0]
    return key


def claim_to_slug(claim_text: str, max_len: int = 40) -> str:
    """Convert a claim string to a short kebab-case slug for directory names.

    Drops common stop words, keeps significant words, truncates to *max_len*.
    """
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "as", "into", "through", "and",
        "but", "or", "not", "that", "this", "it", "its", "there", "their",
    }
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", "", claim_text.lower())
    words = [w for w in cleaned.split() if w and w not in stop_words]
    if not words:
        words = [w for w in cleaned.split() if w]
    if not words:
        return "claim"

    parts: list[str] = []
    length = 0
    for w in words:
        needed = len(w) + (1 if parts else 0)
        if length + needed > max_len:
            break
        parts.append(w)
        length += needed
    return "-".join(parts) or "claim"


def validate_slug(slug: str) -> bool:
    """Validate a dish or colony slug: kebab-case, contains at least one letter."""
    if not slug:
        return False
    if not re.match(r"^[a-z0-9]+(-[a-z0-9]+)*$", slug):
        return False
    # Must contain at least one letter
    return bool(re.search(r"[a-z]", slug))


# ── InferenceProvider Protocol ───────────────────────────────────────────


@runtime_checkable
class InferenceProvider(Protocol):
    """Abstract interface for LLM inference. Injected by the harness or CLI."""

    def generate_clarifying_questions(
        self, claim: str, max_questions: int = 5
    ) -> list[ClarifyingQuestion]:
        """Generate clarifying questions for a claim."""
        ...

    def decompose_claim(
        self, claim: str, clarifications: list[ClarifyingQuestion]
    ) -> DecompositionResult:
        """Decompose a claim into nodes and edges."""
        ...

    def assess_node(
        self, node_id: str, claim_text: str, context: dict, agent_role: str
    ) -> AssessmentResult:
        """Run an agent role assessment on a node."""
        ...

    def match_evidence(
        self, content: str, nodes: list[Node]
    ) -> list[EvidenceMatch]:
        """Match new evidence to existing nodes.
        """
        ...


