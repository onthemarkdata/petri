"""Offline quality regressions use FunctionModel and public synthetic fixtures."""

import json
import os
from collections import Counter
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from evals.decomposer.archive import ArchiveGraph, load_archive, retrieval_report
from evals.decomposer.dataset import (
    assert_neutral_instructions, build_dataset, live_task, load_inputs, negative_fixture, scripted_task,
)
from evals.decomposer.evaluators import (
    BedrockStop, CounterargumentPresence, CrossColonyReferenceRate, DepthOrganic, NoQuotaFill,
    OrganicVariation, ParentChildDistinctness,
)
from evals.decomposer.types import EvalBatch, EvalInput
from petri.models import Cell, DecompositionResult, Edge


def shape(counts, *, name="sample"):
    root = Cell(id=f"eval-{name}-000-000", colony_id=f"eval-{name}", level=0, claim_text="Root claim")
    cells = [root]
    edges = []
    parents = [root]
    for level, count in enumerate(counts, 1):
        children = [Cell(id=f"eval-{name}-{level:03d}-{index:03d}", colony_id=f"eval-{name}", level=level,
                         claim_text=f"Constraint {name}L{level}N{index}",
                         relation="rebuts" if index == count else "supports") for index in range(1, count + 1)]
        edges.extend(Edge(from_cell=parents[min(index, len(parents) - 1)].id, to_cell=child.id)
                     for index, child in enumerate(children))
        cells.extend(children)
        parents = children
    return DecompositionResult(cells=cells, edges=edges, center_claim=root.claim_text, colony_name=name)


def context(results, *, kind="organic", ceiling=13, **kwargs):
    return SimpleNamespace(inputs=EvalInput(name="test", kind=kind, claims=["claim"], layer_ceiling=ceiling,
                                           axiom_texts=kwargs.pop("axiom_texts", []), layer_sizes=[[1]]),
                           output=EvalBatch(results=results, **kwargs))


def test_offline_dataset_runs_real_agents_without_network_or_subprocess(monkeypatch):
    import socket
    import subprocess

    def forbidden(*args, **kwargs):
        raise AssertionError("Offline eval attempted network or a subprocess")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    report = build_dataset(include_negatives=False).evaluate_sync(scripted_task, max_concurrency=1, progress=False)
    assert not report.failures
    assert len(report.cases) == 3
    for case in report.cases:
        assert case.assertions and all(item.value for item in case.assertions.values())


def test_no_quota_fill_distinguishes_v1_from_organic_shapes():
    assert not NoQuotaFill().evaluate(context([shape([13, 13, 13])]))
    assert NoQuotaFill().evaluate(context([shape([4, 7, 5])]))


def test_l1_and_depth_variation_are_checked_across_claims():
    uniform = context([shape([4, 7], name="first"), shape([4, 6], name="second")])
    varied = context([shape([4, 7], name="first"), shape([7, 5, 4], name="second")])
    assert not OrganicVariation().evaluate(uniform)
    assert OrganicVariation().evaluate(varied)
    assert not DepthOrganic().evaluate(uniform)
    assert DepthOrganic().evaluate(varied)


def test_restatement_child_fails_distinctness():
    result = shape([1])
    result.cells[1].claim_text = "ROOT claim!"
    assert not ParentChildDistinctness().evaluate(context([result]))


def test_counterargument_threshold_is_a_gate_and_a_fraction():
    result = shape([3])
    assert CounterargumentPresence().evaluate(context([result])) == {
        "CounterargumentPresence": True, "counterargument_coverage": 1.0,
    }
    result.cells[-1].relation = "supports"
    assert CounterargumentPresence().evaluate(context([result])) == {
        "CounterargumentPresence": False, "counterargument_coverage": 0.0,
    }


def test_axiom_stop_requires_the_expected_axiom_and_reason():
    inputs = next(item for item in load_inputs() if item.kind == "axiom")
    batch = scripted_task(inputs)
    ctx = SimpleNamespace(inputs=inputs, output=batch)
    assert BedrockStop().evaluate(ctx)
    batch.results[0].cells[1].is_atomic = False
    batch.trivial_cell_ids.add(batch.results[0].cells[1].id)
    assert not BedrockStop().evaluate(ctx)


def test_reference_targets_must_be_specific_existing_non_root_cells():
    inputs = next(item for item in load_inputs() if item.kind == "reference")
    batch = scripted_task(inputs)
    ctx = SimpleNamespace(inputs=inputs, output=batch)
    assert CrossColonyReferenceRate().evaluate(ctx)
    reference = next(edge for result in batch.results for edge in result.edges if edge.edge_type == "cross_colony")
    reference.to_cell = batch.results[0].cells[0].id
    batch.expected_reference_ids = {reference.to_cell}
    assert not CrossColonyReferenceRate().evaluate(ctx)


@pytest.mark.parametrize("inputs", [item for item in load_inputs() if item.kind == "negative"], ids=lambda item:item.name)
def test_negative_fixtures_score_poorly_on_counterargument_coverage(inputs):
    batch = negative_fixture(inputs)
    result = CounterargumentPresence().evaluate(SimpleNamespace(inputs=inputs, output=batch))
    assert result["CounterargumentPresence"] is False
    assert result["counterargument_coverage"] == 0


def test_quota_prompt_regression_is_rejected(monkeypatch):
    from petri.reasoning import decomposition_agents

    assert_neutral_instructions()
    original = decomposition_agents.initial_instructions
    monkeypatch.setattr(decomposition_agents, "initial_instructions",
                        lambda request: original(request) + "Pick the TOP 13 premises.")
    with pytest.raises(AssertionError, match="Numeric quota phrasing"):
        assert_neutral_instructions()
    assert not NoQuotaFill().evaluate(context([shape([13, 13])]))


def public_graph():
    return {"claims": [{"id": f"n{i}", "layer": i, "title": text, "status": "fixture"}
                       for i, text in enumerate(["Solar electricity generation", "Solar electricity output", "Wind generation"])],
            "edges": [["n0", "n1"], ["n1", "n2"]]}


def test_archive_loader_reports_real_search_precision_and_recall(tmp_path):
    path = tmp_path / "synthetic.json"
    path.write_text(json.dumps(public_graph()))
    with pytest.raises(ValueError, match="differ from expected"):
        load_archive(path)
    graph = load_archive(path, expected_counts=(3, 2, 2))
    metrics = retrieval_report(graph, tmp_path / "search.sqlite", top_k=2)
    assert metrics["exact_title_recall"] == 1.0
    assert metrics["linked_query_count"] == 2
    assert 0 <= metrics["precision"] <= 1 and 0 <= metrics["recall"] <= 1


def test_archive_loader_rejects_cycles_and_unknown_endpoints():
    graph = public_graph()
    graph["edges"].append(["n2", "n0"])
    with pytest.raises(ValidationError, match="acyclic"):
        ArchiveGraph.model_validate(graph)
    graph["edges"] = [["n0", "absent"]]
    with pytest.raises(ValidationError, match="unknown claim"):
        ArchiveGraph.model_validate(graph)


@pytest.mark.private_archive
def test_opt_in_private_archive_bundle(tmp_path):
    source = os.environ.get("PETRI_PRIVATE_ARCHIVE")
    if not source:
        pytest.skip("Set PETRI_PRIVATE_ARCHIVE to a local private fixture bundle.")
    bundle = Path(source)
    graph = load_archive(bundle / "claims-graph.json")
    assert graph.counts == (88, 128, 60)
    layers = json.loads((bundle / "layer-distribution.json").read_text())["per_layer"]
    assert {int(key): value for key, value in layers.items()} == dict(Counter(claim.layer for claim in graph.claims))
    paired = json.loads((bundle / "paired-waves.json").read_text())["pairs"]
    assert len(paired) == 7
    for pair in paired:
        assert pair["before"]["center_claim"] == pair["after"]["center_claim"]
        for wave in ("before", "after"):
            assert pair[wave]["cells"]
            for row in pair[wave]["cells"]:
                Cell.model_validate(row)
    for prefix in ("08-04a", "08-04b", "08-07a", "08-07b", "08-09a", "08-09b"):
        files = list((bundle / "principle-measurement").glob(prefix + "*.md"))
        assert len(files) == 1 and files[0].read_text().strip()
    print({"nodes": 88, "edges": 128, "cross_layer_edges": 60, "paired_colonies": 7,
           "principle_measurement_files": 6, **retrieval_report(graph, tmp_path / "archive-search.sqlite")})


@pytest.mark.integration
@pytest.mark.live
def test_opt_in_live_dataset_report():
    model = os.environ.get("PETRI_EVAL_MODEL")
    if os.environ.get("PETRI_LIVE_EVALS") != "1" or not model:
        pytest.skip("Set PETRI_LIVE_EVALS=1 and PETRI_EVAL_MODEL for an explicit live report.")
    report = build_dataset(include_negatives=False).evaluate_sync(partial(live_task, model_name=model),
                                                                 max_concurrency=1, progress=False)
    report.print(include_input=False, include_output=False)
    assert not report.failures
    reference = next(case for case in report.cases if case.name == "shared-generation-premise")
    assert reference.assertions["CrossColonyReferenceRate"].value
