"""Offline phase callables shared by pipeline and lifecycle conformance tests."""

from dataclasses import dataclass, field

from petri.execution.pipeline import (
    ConvergenceDecision,
    EvaluationVerdict,
    PhaseResult,
    PipelineDeps,
    PipelineInput,
    RedTeamResult,
    SyncConflict,
)


@dataclass
class StubPhases:
    verdict: EvaluationVerdict = EvaluationVerdict.VALIDATED
    decisions: list[ConvergenceDecision] = field(default_factory=list)
    park_phase: str | None = None
    conflict: str | None = None
    red_team_defer: str | None = None
    calls: list[tuple[str, PipelineInput]] = field(default_factory=list)

    async def sync_check(self, value):
        self.calls.append(("sync_check", value))
        return SyncConflict(reason=self.conflict) if self.conflict else None

    async def phase(self, name, value):
        self.calls.append((name, value))
        return PhaseResult(
            needs_human_reason="Human review requested" if self.park_phase == name else None
        )

    async def socratic(self, value):
        return await self.phase("socratic", value)

    async def research(self, value):
        return await self.phase("research", value)

    async def critique(self, value):
        return await self.phase("critique", value)

    async def convergence(self, value):
        self.calls.append(("convergence", value))
        return self.decisions.pop(0) if self.decisions else ConvergenceDecision(outcome="converged")

    async def red_team(self, value):
        self.calls.append(("red_team", value))
        return RedTeamResult(defer_open_reason=self.red_team_defer)

    async def evaluate(self, value):
        self.calls.append(("evaluate", value))
        return self.verdict

    def dependencies(self):
        return PipelineDeps(
            self.socratic,
            self.research,
            self.critique,
            self.convergence,
            self.red_team,
            self.evaluate,
            self.sync_check,
        )
