#!/usr/bin/env python
"""THROWAWAY SCRATCH driver: run Petri end-to-end (seed -> grow) on the `pi`
harness, printing every pi request/response. Lives OUTSIDE the repo tree.

Injects a PiInferenceProvider at the real resolve_provider seam, seeds a claim
(decompose), grows the colony (process_queue), prints a summary.

Run:
    PI_BIN=/opt/node22/bin/pi uv run python <this> --claim "..."
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path

# scratch provider lives beside this file (outside the repo); make it importable.
sys.path.insert(0, str(Path(__file__).resolve().parent))

DEFAULT_CLAIM = "Regular expressions cannot parse arbitrary HTML."
_HR = "=" * 78
_CALL = {"n": 0}


def _short(s, limit=4000):
    s = s or ""
    return s if len(s) <= limit else s[:limit] + f"\n... [truncated {len(s)-limit} chars]"


def _logger():
    def _log(prompt, response_text, pi_result, error):
        _CALL["n"] += 1
        n = _CALL["n"]
        print(f"\n{_HR}\n[pi call #{n}] REQUEST\n{_HR}")
        print(_short(prompt, 1200))
        if error is not None:
            print(f"\n[pi call #{n}] ERROR channel={getattr(error,'channel','?')}: {error}")
        else:
            print(f"\n{_HR}\n[pi call #{n}] RESPONSE\n{_HR}")
            print(_short(response_text))
            u = getattr(pi_result, "usage", None)
            if u:
                print(f"[pi call #{n}] usage: {u}")
    return _log


def run(claim, workdir):
    import pi_provider_scratch as pp
    from petri.cli import _bootstrap
    from petri.cli.init import create_petri_dish
    from petri.graph.colony import ColonyGraph, serialize_colony, deserialize_colony
    from petri.models import Cell, Colony, build_cell_key

    petri_dir = workdir / ".petri"
    dish_id = "scratch"
    create_petri_dish(petri_dir, dish_name=dish_id, model_name="anthropic/claude-sonnet-4-6")
    print(f"Dish at {petri_dir}")
    print(f"pi provider={pp.DEFAULT_PI_PROVIDER} model={pp.DEFAULT_PI_MODEL} bin={pp.DEFAULT_PI_BIN}")

    provider = pp.PiInferenceProvider(on_transport=_logger())
    _bootstrap.resolve_provider = lambda _d: provider  # type: ignore
    provider = _bootstrap.resolve_provider(petri_dir)
    print("Injected PiInferenceProvider at resolve_provider seam")

    colony_name = "demo"
    colony_id = f"{dish_id}-{colony_name}"
    center_id = build_cell_key(dish_id, colony_name, 0, 0)
    colony_path = petri_dir / "petri-dishes" / colony_name
    center = Cell(id=center_id, colony_id=colony_id, claim_text=claim, level=0)
    graph = ColonyGraph(colony_id=colony_id)
    graph.add_cell(center)
    colony = Colony(id=colony_id, dish=dish_id, center_claim=claim, center_cell_id=center_id,
                    clarifications=[], created_at=datetime.now(timezone.utc).isoformat())
    serialize_colony(graph, colony, colony_path)

    print(f"\n{_HR}\nSEED: decompose\n{_HR}\n{claim}")
    decomposition = None
    try:
        from petri.reasoning.decomposer import decompose_claim
        decomposition = decompose_claim(claim=claim, clarifications=[], dish_id=dish_id,
                                        colony_name=colony_name, provider=provider, center=center)
        for cell in decomposition.cells:
            if cell.id != center_id:
                graph.add_cell(cell)
        for edge in decomposition.edges:
            graph.add_edge(edge)
        serialize_colony(graph, colony, colony_path)
        print(f"\nSeed -> {len(decomposition.cells)} cells, {len(decomposition.edges)} edges.")
    except Exception as exc:
        print(f"\n[SEED FAILED] {type(exc).__name__}: {exc}")
        traceback.print_exc()

    print(f"\n{_HR}\nGROW: validation pipeline\n{_HR}")
    grow_result = None
    try:
        from petri.engine.processor import process_queue
        grow_result = process_queue(petri_dir=petri_dir, provider=provider,
                                    max_concurrent=int(os.environ.get("PI_CONCURRENCY", "2")),
                                    all_cells=True, dry_run=False)
        print(f"\nGrow -> {grow_result}")
    except Exception as exc:
        print(f"\n[GROW FAILED] {type(exc).__name__}: {exc}")
        traceback.print_exc()

    print(f"\n{_HR}\nSUMMARY\n{_HR}")
    print(f"Total pi calls : {_CALL['n']}")
    print(f"Cells created  : {len(decomposition.cells) if decomposition else 1}")
    try:
        g, _ = deserialize_colony(colony_path, dish_id)
        print("Cell states:")
        for cell in g.get_all_cells():
            print(f"  - {cell.id} [{getattr(cell.status,'value',cell.status)}] :: {cell.claim_text[:70]}")
    except Exception as exc:
        print(f"  (could not read cells: {exc})")
    if grow_result is not None:
        for attr in ("processed", "succeeded", "failed", "stalled"):
            if hasattr(grow_result, attr):
                print(f"grow.{attr} = {getattr(grow_result, attr)}")
    print(f"Accumulated pi usage : {getattr(provider,'total_usage',{}) or '(none)'}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--claim", default=DEFAULT_CLAIM)
    ap.add_argument("--workdir", default=None)
    a = ap.parse_args()
    if a.workdir:
        wd = Path(a.workdir).resolve()
        wd.mkdir(parents=True, exist_ok=True)
        return run(a.claim, wd)
    with tempfile.TemporaryDirectory(prefix="petri-pi-e2e-") as tmp:
        return run(a.claim, Path(tmp))


if __name__ == "__main__":
    raise SystemExit(main())
