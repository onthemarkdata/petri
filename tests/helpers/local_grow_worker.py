"""Explicit local-Gemma grow probe with real research and preserved artifacts."""

import argparse
import json
import os
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import yaml
from typer.testing import CliRunner

from petri.cli import app
from petri.cli.init import create_petri_dish
from petri.graph.colony import ColonyGraph
from petri.models import Cell, Colony
from petri.storage import database
from petri.storage.colonies import reserve_colony, save_colony
from petri.storage.event_log import append_event
from petri.storage.paths import dish_db_path

LOCAL_ENDPOINT = "http://127.0.0.1:12434/engines/llama.cpp/v1"
GEMMA = "huggingface.co/google/gemma-4-e2b-it-qat-q4_0-gguf:Q4_0"
CASES = {
    "isolated": (
        "In SQLite, two separate connections opened with the exact filename ':memory:' "
        "create independent databases; a table created through one is not visible through the other.",
        "VALIDATED",
    ),
    "shared": (
        "In SQLite, two separate connections opened with the exact filename ':memory:' "
        "share one database; a table created through one is immediately visible through the other.",
        "DISPROVEN",
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--harness", required=True, choices=["provider", "pi"])
    parser.add_argument("--case", required=True, choices=CASES)
    parser.add_argument("--output-mode", choices=["tool", "prompted"],
                        help="Choose final-result delivery without changing host retrieval tools.")
    parser.add_argument("--expanded-budget", action="store_true",
                        help="Use a bounded acceptance budget for the full roster and source retrieval.")
    parser.add_argument("--max-tokens", type=int, default=4096,
                        help="Bound each native provider response; Pi uses its provider's explicit cap.")
    parser.add_argument(
        "--seed",
        action="store_true",
        help="Generate and approve the colony through the actual seed command.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the preserved dish without replacing its configuration or cells.",
    )
    parser.add_argument(
        "--requeue",
        metavar="REQUEST_ID",
        help="Start an explicit new validation round in a preserved dish.",
    )
    args = parser.parse_args()
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive.")
    if os.environ.get("PETRI_RUN_LOCAL_GROW") != "1":
        parser.error(
            "Set PETRI_RUN_LOCAL_GROW=1 to enable real local inference and public source retrieval."
        )
    os.environ["OPENAI_BASE_URL"] = LOCAL_ENDPOINT
    os.environ["OPENAI_API_KEY"] = "local-no-auth"
    if args.harness == "pi":
        agent_dir = os.environ.get("PI_CODING_AGENT_DIR")
        if not agent_dir:
            parser.error(
                "Provide PI_CODING_AGENT_DIR with an explicitly local petri-local provider."
            )
        provider = (
            json.loads((Path(agent_dir) / "models.json").read_text())
            .get("providers", {})
            .get("petri-local", {})
        )
        if provider.get("baseUrl", "").rstrip("/") != LOCAL_ENDPOINT:
            parser.error(
                "The petri-local Pi provider must point at the local Docker Model Runner endpoint."
            )
        selected = next((model for model in provider.get("models", []) if model.get("id") == GEMMA), {})
        if selected.get("maxTokens") != args.max_tokens:
            parser.error("The Pi provider's maxTokens must match --max-tokens.")
    root = args.directory.resolve()
    if root.exists() and any(root.iterdir()) and not args.resume:
        parser.error("Use an empty output directory; existing artifacts are preserved.")
    root.mkdir(parents=True, exist_ok=True)
    os.chdir(root)
    dish = root / ".petri"
    if not args.resume:
        create_petri_dish(dish, dish_name="grow-live")
        config = yaml.safe_load(
            (Path(__file__).parents[2] / "petri/defaults/petri.yaml").read_text()
        )
        config.update(
            name="grow-live",
            execution={"backend": "dbos", "worker_concurrency": 1},
            research={"public_web": True},
            max_decomposition_depth=1,
            level_one_cap=2,
        )
        config["model"] = f"openai-chat:{GEMMA}"
        if args.harness == "provider":
            config["model_settings"] = {"max_tokens": args.max_tokens}
        if args.expanded_budget:
            config["limits"] = {
                "run": {"request_limit": 120, "total_tokens_limit": 600000, "tool_calls_limit": 100},
                "debate": {"request_limit": 24, "total_tokens_limit": 100000, "tool_calls_limit": 20},
            }
        config["harness"] = (
            {"backend": "provider", "model": f"openai-chat:{GEMMA}"}
            if args.harness == "provider"
            else {"backend": "pi", "model": GEMMA, "provider": "petri-local", "tools": []}
        )
        if args.output_mode is not None:
            config["harness"]["output_mode"] = args.output_mode
        (dish / "defaults/petri.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    claim, expected = CASES[args.case]
    path = dish_db_path(dish, "grow-live")
    colony_id = f"grow-live-{args.case}"
    runner = CliRunner()
    seed_exit = None
    if args.resume:
        if not database.get_cell(path, f"{colony_id}-000-000"):
            parser.error("The preserved dish does not contain the requested case.")
    elif args.seed:
        seeded = runner.invoke(app, ["seed", claim, "--colony", args.case, "--no-questions"])
        (root / "seed.log").write_text(seeded.output)
        seed_exit = seeded.exit_code
    else:
        center = Cell(
            id=f"{colony_id}-000-000",
            colony_id=colony_id,
            claim_text=claim,
            level=0,
            created_at=datetime.now(UTC).isoformat(),
        )
        colony = Colony(
            id=colony_id, dish="grow-live", center_claim=claim, center_cell_id=center.id
        )
        reserve_colony(path, colony)
        graph = ColonyGraph(colony_id=colony_id)
        graph.add_cell(center)
        save_colony(path, graph, colony)
        append_event(
            database.SQLiteEventLog(path),
            center.id,
            "colony_approved",
            "human",
            0,
            {"fixture_kind": "public_single_cell_known_outcome"},
        )
    if args.requeue:
        from petri.execution import resolve_backend
        from petri.execution.models import ReopenRequest

        backend = resolve_backend(dish)
        backend.reopen_cell(
            ReopenRequest(
                f"{colony_id}-000-000",
                args.requeue,
                datetime.now(UTC).date().isoformat(),
                trigger="Human revalidation after recorded format failure",
                max_iterations=yaml.safe_load((dish / "defaults/petri.yaml").read_text())[
                    "max_iterations"
                ],
            ),
            petri_dir=dish,
        )
    grown = None
    if seed_exit in (None, 0):
        grown = runner.invoke(app, ["grow", "--colony", args.case, "--max-concurrent", "1"])
        (root / "grow.log").write_text(grown.output)
    events = database.read_events(path) if path.exists() else []
    rows = database.list_cells(path) if path.exists() else []
    sources = [event for event in events if event["type"] == "source_reviewed"
               and event["data"].get("retrieval_method") == "public-web"]
    searches = [event for event in events if event["type"] == "search_executed"]
    failures = [event for event in events if event["type"] == "agent_execution_failed"]
    last_reopen = max(
        (index for index, event in enumerate(events) if event["type"] == "cell_reopened"),
        default=-1,
    )
    current_failures = [
        event for event in events[last_reopen + 1 :] if event["type"] == "agent_execution_failed"
    ]
    verdicts = [
        {"cell_id": event["cell_id"], "agent": event["agent"], **event["data"]}
        for event in events
        if event["type"] == "verdict_issued"
    ]
    center = next((row for row in rows if row["level"] == 0), None)
    config = yaml.safe_load((dish / "defaults/petri.yaml").read_text())
    with database.connect(path) as connection:
        usage = [dict(row) for row in connection.execute("SELECT * FROM usage ORDER BY span_id")]
    report = {
        "fixture_kind": "model_generated_seed_approve_grow"
        if (root / "seed.log").exists()
        else "public_single_cell_known_outcome",
        "resumed": args.resume,
        "harness": args.harness,
        "model": GEMMA,
        "configured_limits": config["limits"],
        "response_max_tokens": config.get("model_settings", {}).get("max_tokens")
        if args.harness == "provider" else selected.get("maxTokens"),
        "usage": usage,
        "case": args.case,
        "expected_status": expected,
        "seed_exit": seed_exit,
        "grow_exit": grown.exit_code if grown else None,
        "grow_exception": repr(grown.exception) if grown and grown.exception else None,
        "center_status": center["status"] if center else None,
        "cells": [{"id": row["id"], "status": row["status"]} for row in rows],
        "event_counts": dict(Counter(event["type"] for event in events)),
        "searches": searches,
        "sources": sources,
        "verdicts": verdicts,
        "failures": failures,
        "current_round_failure_count": len(current_failures),
    }
    report["passed"] = bool(
        center and center["status"] == expected and sources and searches and not current_failures
    )
    (root / "result.json").write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key not in {"sources", "verdicts", "failures", "searches"}
            },
            indent=2,
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
