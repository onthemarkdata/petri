"""Canonical agent structure, independent of editable YAML content."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, model_validator


class AgentName(StrEnum):
    DECOMPOSITION_LEAD = "decomposition_lead"
    CELL_LEAD = "cell_lead"
    RED_TEAM_LEAD = "red_team_lead"
    INVESTIGATOR = "investigator"
    FRESHNESS_CHECKER = "freshness_checker"
    DEPENDENCY_AUDITOR = "dependency_auditor"
    SKEPTIC = "skeptic"
    CHAMPION = "champion"
    PRAGMATIST = "pragmatist"
    SIMPLIFIER = "simplifier"
    TRIAGE = "triage"
    IMPACT_ASSESSOR = "impact_assessor"
    EVIDENCE_EVALUATOR = "evidence_evaluator"


class BlockingMode(StrEnum):
    BLOCKING = "blocking"
    NON_BLOCKING = "non_blocking"
    CONDITIONAL = "conditional"


class Phase(StrEnum):
    SOCRATIC = "socratic"
    RESEARCH = "research"
    CRITIQUE = "critique"
    POST_CONVERGENCE = "post_convergence"
    LEAD = "lead"


class AgentSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: AgentName
    display_name: str
    phase: Phase
    blocking_mode: BlockingMode
    is_lead: bool = False
    scope: str | None = None
    redirect_on_block: str | None = None
    verdicts_pass: tuple[str, ...] = ()
    verdicts_block: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_blocking(self):
        if self.is_lead and self.blocking_mode != BlockingMode.NON_BLOCKING:
            raise ValueError("Lead agents cannot block convergence.")
        if self.blocking_mode != BlockingMode.NON_BLOCKING and not self.verdicts_block:
            raise ValueError("Blocking and conditional agents require verdicts_block.")
        if self.blocking_mode == BlockingMode.CONDITIONAL and not self.redirect_on_block:
            raise ValueError("Conditional agents require redirect_on_block.")
        if set(self.verdicts_pass) & set(self.verdicts_block):
            raise ValueError("Pass and block verdicts must not overlap.")
        # Advisory block vocabulary stays representable; #31 owns the factory ruling.
        return self


class DecompositionLeadVerdict(StrEnum):
    DECOMPOSITION_COMPLETE = "DECOMPOSITION_COMPLETE"
    DECOMPOSITION_NEEDS_REFINEMENT = "DECOMPOSITION_NEEDS_REFINEMENT"


class CellLeadVerdict(StrEnum):
    PIPELINE_COMPLETE = "PIPELINE_COMPLETE"
    PIPELINE_STALLED = "PIPELINE_STALLED"


class RedTeamLeadVerdict(StrEnum):
    STRONG_DISPROVAL_CASE = "STRONG_DISPROVAL_CASE"
    WEAK_DISPROVAL_CASE = "WEAK_DISPROVAL_CASE"
    NO_CONTRADICTING_EVIDENCE = "NO_CONTRADICTING_EVIDENCE"


class InvestigatorVerdict(StrEnum):
    EVIDENCE_SUFFICIENT = "EVIDENCE_SUFFICIENT"
    EVIDENCE_CONTRADICTS = "EVIDENCE_CONTRADICTS"
    NEEDS_MORE_EVIDENCE = "NEEDS_MORE_EVIDENCE"
    CANNOT_DETERMINE = "CANNOT_DETERMINE"


class FreshnessCheckerVerdict(StrEnum):
    EVIDENCE_CURRENT = "EVIDENCE_CURRENT"
    STALE_BUT_HOLDS = "STALE_BUT_HOLDS"
    MATERIALLY_STALE = "MATERIALLY_STALE"


class DependencyAuditorVerdict(StrEnum):
    DEPENDENCIES_CLEAN = "DEPENDENCIES_CLEAN"
    UNVALIDATED_DEPS = "UNVALIDATED_DEPS"
    CIRCULAR_REASONING = "CIRCULAR_REASONING"


class SkepticVerdict(StrEnum):
    ARGUMENT_WITHSTANDS_CRITIQUE = "ARGUMENT_WITHSTANDS_CRITIQUE"
    CRITICAL_FLAW_FOUND = "CRITICAL_FLAW_FOUND"
    UNADDRESSED_COUNTERARGUMENT = "UNADDRESSED_COUNTERARGUMENT"


class ChampionVerdict(StrEnum):
    STRONG_CASE = "STRONG_CASE"
    DEFENSIBLE_WITH_CAVEATS = "DEFENSIBLE_WITH_CAVEATS"
    CANNOT_DEFEND = "CANNOT_DEFEND"


class PragmatistVerdict(StrEnum):
    PRODUCTION_READY = "PRODUCTION_READY"
    DIRECTIONALLY_CORRECT = "DIRECTIONALLY_CORRECT"
    WOULD_NOT_SHIP_THIS = "WOULD_NOT_SHIP_THIS"


class SimplifierVerdict(StrEnum):
    APPROPRIATELY_SCOPED = "APPROPRIATELY_SCOPED"
    COULD_BE_SIMPLER = "COULD_BE_SIMPLER"
    OVERCOMPLICATED = "OVERCOMPLICATED"


class TriageVerdict(StrEnum):
    HIGH_VALUE = "HIGH_VALUE"
    MODERATE_VALUE = "MODERATE_VALUE"
    LOW_VALUE_DEFER = "LOW_VALUE_DEFER"


class ImpactAssessorVerdict(StrEnum):
    CRITICAL_PATH = "CRITICAL_PATH"
    SUPPORTING_NOT_BLOCKING = "SUPPORTING_NOT_BLOCKING"
    ISOLATED_LOW_IMPACT = "ISOLATED_LOW_IMPACT"


class EvidenceEvaluatorVerdict(StrEnum):
    EVIDENCE_CONFIRMS = "EVIDENCE_CONFIRMS"
    EVIDENCE_REFUTES = "EVIDENCE_REFUTES"
    EVIDENCE_INCONCLUSIVE = "EVIDENCE_INCONCLUSIVE"


VERDICT_ENUMS: dict[AgentName, type[StrEnum]] = {
    AgentName.DECOMPOSITION_LEAD: DecompositionLeadVerdict,
    AgentName.CELL_LEAD: CellLeadVerdict,
    AgentName.RED_TEAM_LEAD: RedTeamLeadVerdict,
    AgentName.INVESTIGATOR: InvestigatorVerdict,
    AgentName.FRESHNESS_CHECKER: FreshnessCheckerVerdict,
    AgentName.DEPENDENCY_AUDITOR: DependencyAuditorVerdict,
    AgentName.SKEPTIC: SkepticVerdict,
    AgentName.CHAMPION: ChampionVerdict,
    AgentName.PRAGMATIST: PragmatistVerdict,
    AgentName.SIMPLIFIER: SimplifierVerdict,
    AgentName.TRIAGE: TriageVerdict,
    AgentName.IMPACT_ASSESSOR: ImpactAssessorVerdict,
    AgentName.EVIDENCE_EVALUATOR: EvidenceEvaluatorVerdict,
}


AGENT_SPECS: dict[AgentName, AgentSpec] = {
    AgentName.DECOMPOSITION_LEAD: AgentSpec(
        name=AgentName.DECOMPOSITION_LEAD,
        display_name="Decomposition Lead",
        phase=Phase.LEAD,
        blocking_mode=BlockingMode.NON_BLOCKING,
        is_lead=True,
        scope="decomposition",
        redirect_on_block=None,
        verdicts_pass=("DECOMPOSITION_COMPLETE", "DECOMPOSITION_NEEDS_REFINEMENT"),
        verdicts_block=(),
    ),
    AgentName.CELL_LEAD: AgentSpec(
        name=AgentName.CELL_LEAD,
        display_name="Cell Lead",
        phase=Phase.LEAD,
        blocking_mode=BlockingMode.NON_BLOCKING,
        is_lead=True,
        scope="pipeline",
        redirect_on_block=None,
        verdicts_pass=("PIPELINE_COMPLETE", "PIPELINE_STALLED"),
        verdicts_block=(),
    ),
    AgentName.RED_TEAM_LEAD: AgentSpec(
        name=AgentName.RED_TEAM_LEAD,
        display_name="Red Team Lead",
        phase=Phase.LEAD,
        blocking_mode=BlockingMode.NON_BLOCKING,
        is_lead=True,
        scope="red_team",
        redirect_on_block=None,
        verdicts_pass=("STRONG_DISPROVAL_CASE", "WEAK_DISPROVAL_CASE", "NO_CONTRADICTING_EVIDENCE"),
        verdicts_block=(),
    ),
    AgentName.INVESTIGATOR: AgentSpec(
        name=AgentName.INVESTIGATOR,
        display_name="Investigator",
        phase=Phase.RESEARCH,
        blocking_mode=BlockingMode.BLOCKING,
        is_lead=False,
        scope=None,
        redirect_on_block=None,
        verdicts_pass=("EVIDENCE_SUFFICIENT", "EVIDENCE_CONTRADICTS"),
        verdicts_block=("NEEDS_MORE_EVIDENCE", "CANNOT_DETERMINE"),
    ),
    AgentName.FRESHNESS_CHECKER: AgentSpec(
        name=AgentName.FRESHNESS_CHECKER,
        display_name="Freshness Checker",
        phase=Phase.RESEARCH,
        blocking_mode=BlockingMode.BLOCKING,
        is_lead=False,
        scope=None,
        redirect_on_block=None,
        verdicts_pass=("EVIDENCE_CURRENT", "STALE_BUT_HOLDS"),
        verdicts_block=("MATERIALLY_STALE",),
    ),
    AgentName.DEPENDENCY_AUDITOR: AgentSpec(
        name=AgentName.DEPENDENCY_AUDITOR,
        display_name="Dependency Auditor",
        phase=Phase.RESEARCH,
        blocking_mode=BlockingMode.BLOCKING,
        is_lead=False,
        scope=None,
        redirect_on_block=None,
        verdicts_pass=("DEPENDENCIES_CLEAN",),
        verdicts_block=("UNVALIDATED_DEPS", "CIRCULAR_REASONING"),
    ),
    AgentName.SKEPTIC: AgentSpec(
        name=AgentName.SKEPTIC,
        display_name="Skeptic",
        phase=Phase.CRITIQUE,
        blocking_mode=BlockingMode.BLOCKING,
        is_lead=False,
        scope=None,
        redirect_on_block=None,
        verdicts_pass=("ARGUMENT_WITHSTANDS_CRITIQUE",),
        verdicts_block=("CRITICAL_FLAW_FOUND", "UNADDRESSED_COUNTERARGUMENT"),
    ),
    AgentName.CHAMPION: AgentSpec(
        name=AgentName.CHAMPION,
        display_name="Champion",
        phase=Phase.CRITIQUE,
        blocking_mode=BlockingMode.BLOCKING,
        is_lead=False,
        scope=None,
        redirect_on_block=None,
        verdicts_pass=("STRONG_CASE", "DEFENSIBLE_WITH_CAVEATS"),
        verdicts_block=("CANNOT_DEFEND",),
    ),
    AgentName.PRAGMATIST: AgentSpec(
        name=AgentName.PRAGMATIST,
        display_name="Pragmatist",
        phase=Phase.CRITIQUE,
        blocking_mode=BlockingMode.BLOCKING,
        is_lead=False,
        scope=None,
        redirect_on_block=None,
        verdicts_pass=("PRODUCTION_READY", "DIRECTIONALLY_CORRECT"),
        verdicts_block=("WOULD_NOT_SHIP_THIS",),
    ),
    AgentName.SIMPLIFIER: AgentSpec(
        name=AgentName.SIMPLIFIER,
        display_name="Simplifier",
        phase=Phase.CRITIQUE,
        blocking_mode=BlockingMode.NON_BLOCKING,
        is_lead=False,
        scope=None,
        redirect_on_block=None,
        verdicts_pass=("APPROPRIATELY_SCOPED", "COULD_BE_SIMPLER"),
        verdicts_block=("OVERCOMPLICATED",),
    ),
    AgentName.TRIAGE: AgentSpec(
        name=AgentName.TRIAGE,
        display_name="Triage",
        phase=Phase.CRITIQUE,
        blocking_mode=BlockingMode.CONDITIONAL,
        is_lead=False,
        scope=None,
        redirect_on_block="DEFER_OPEN",
        verdicts_pass=("HIGH_VALUE", "MODERATE_VALUE"),
        verdicts_block=("LOW_VALUE_DEFER",),
    ),
    AgentName.IMPACT_ASSESSOR: AgentSpec(
        name=AgentName.IMPACT_ASSESSOR,
        display_name="Impact Assessor",
        phase=Phase.CRITIQUE,
        blocking_mode=BlockingMode.NON_BLOCKING,
        is_lead=False,
        scope=None,
        redirect_on_block=None,
        verdicts_pass=("CRITICAL_PATH", "SUPPORTING_NOT_BLOCKING", "ISOLATED_LOW_IMPACT"),
        verdicts_block=(),
    ),
    AgentName.EVIDENCE_EVALUATOR: AgentSpec(
        name=AgentName.EVIDENCE_EVALUATOR,
        display_name="Evidence Evaluator",
        phase=Phase.POST_CONVERGENCE,
        blocking_mode=BlockingMode.NON_BLOCKING,
        is_lead=False,
        scope=None,
        redirect_on_block=None,
        verdicts_pass=("EVIDENCE_CONFIRMS", "EVIDENCE_REFUTES", "EVIDENCE_INCONCLUSIVE"),
        verdicts_block=(),
    ),
}
