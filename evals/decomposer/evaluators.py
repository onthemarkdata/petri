"""Seven deterministic quality checks; no judge model or network is required."""

from collections import Counter, defaultdict
from dataclasses import dataclass

from pydantic_evals.evaluators import Evaluator, EvaluatorContext

from petri.reasoning.decomposition_agents import DUPLICATE_THRESHOLD, parent_overlap
from evals.decomposer.types import EvalBatch, EvalInput

Context = EvaluatorContext[EvalInput, EvalBatch, None]


def layer_counts(result) -> dict[int, int]:
    return dict(Counter(cell.level for cell in result.cells if cell.level > 0))


class NoQuotaFill(Evaluator[EvalInput, EvalBatch, None]):
    def evaluate(self, ctx: Context) -> bool:
        for result in ctx.output.results:
            counts = layer_counts(result)
            if any(count == ctx.inputs.layer_ceiling and counts.get(level + 1) == count
                   for level, count in counts.items()):
                return False
        return True


@dataclass
class OrganicVariation(Evaluator[EvalInput, EvalBatch, None]):
    lower: int = 4
    upper: int = 13

    def evaluate(self, ctx: Context) -> bool:
        if ctx.inputs.kind != "organic":
            return True
        if not ctx.output.results:
            return False
        counts = [count for result in ctx.output.results for count in layer_counts(result).values()]
        if not counts or sum(self.lower <= count <= self.upper for count in counts) / len(counts) < 0.5:
            return False
        if len(counts) > 1 and len(set(counts)) == 1:
            return False
        roots = [layer_counts(result).get(1, 0) for result in ctx.output.results]
        return len(roots) < 2 or len(set(roots)) > 1


@dataclass
class CounterargumentPresence(Evaluator[EvalInput, EvalBatch, None]):
    threshold: float = 1.0

    def evaluate(self, ctx: Context) -> dict[str, bool | float]:
        cells = {cell.id: cell for result in ctx.output.results for cell in result.cells}
        children = defaultdict(set)
        for result in ctx.output.results:
            for edge in result.edges:
                children[edge.from_cell].add(edge.to_cell)
        expanded = [ids for ids in children.values() if len(ids) >= 2]
        coverage = (sum(any(identifier in cells and cells[identifier].relation in ("limits", "rebuts")
                            for identifier in ids) for ids in expanded) / len(expanded) if expanded else 1.0)
        return {"CounterargumentPresence": coverage >= self.threshold,
                "counterargument_coverage": coverage}


class BedrockStop(Evaluator[EvalInput, EvalBatch, None]):
    def evaluate(self, ctx: Context) -> bool:
        if ctx.inputs.kind != "axiom":
            return True
        cells = [cell for result in ctx.output.results for cell in result.cells]
        stopped = any(cell.is_atomic and cell.level <= 2 and cell.bedrock_reason
                      and cell.claim_text in ctx.inputs.axiom_texts for cell in cells)
        trivial_rate = sum(cell.id in ctx.output.trivial_cell_ids and not cell.is_atomic
                           for cell in cells) / max(1, len(cells))
        return stopped and trivial_rate < 0.25


class CrossColonyReferenceRate(Evaluator[EvalInput, EvalBatch, None]):
    def evaluate(self, ctx: Context) -> bool:
        if ctx.inputs.kind != "reference":
            return True
        cells = {cell.id: cell for result in ctx.output.results for cell in result.cells}
        return any(edge.edge_type == "cross_colony" and edge.to_cell in ctx.output.expected_reference_ids
                   and edge.to_cell in cells and cells[edge.to_cell].level > 0
                   for result in ctx.output.results for edge in result.edges)


class DepthOrganic(Evaluator[EvalInput, EvalBatch, None]):
    def evaluate(self, ctx: Context) -> bool:
        depths = [max((cell.level for cell in result.cells), default=0) for result in ctx.output.results]
        return len(depths) < 2 or len(set(depths)) > 1


class ParentChildDistinctness(Evaluator[EvalInput, EvalBatch, None]):
    def evaluate(self, ctx: Context) -> bool:
        cells = {cell.id: cell for result in ctx.output.results for cell in result.cells}
        return all(parent_overlap(cells[edge.from_cell].claim_text, cells[edge.to_cell].claim_text)
                   < DUPLICATE_THRESHOLD for result in ctx.output.results for edge in result.edges
                   if edge.from_cell in cells and edge.to_cell in cells)


QUALITY_EVALUATORS = (NoQuotaFill, OrganicVariation, CounterargumentPresence, BedrockStop,
                      CrossColonyReferenceRate, DepthOrganic, ParentChildDistinctness)
