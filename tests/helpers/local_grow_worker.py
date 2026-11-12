"""Opt-in local grow diagnostics with preserved configuration and command outcomes."""

import argparse
import hashlib
import json
import os
import sys
import traceback
from collections import Counter
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import yaml
from typer.testing import CliRunner

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from petri.cli import app
from petri.cli.init import create_petri_dish
from petri.graph.colony import ColonyGraph
from petri.models import Cell, Colony
from petri.storage import database
from petri.storage.colonies import reserve_colony, save_colony
from petri.storage.event_log import append_event, derive_event_id
from petri.storage.paths import dish_db_path
from scripts.e2e.local_endpoint import (
    configuration_hash,
    connection_arguments,
    endpoint_from_args,
    environment_values,
    save_identity,
    source_sha,
)

CASES = {
    "isolated": (
        ("In SQLite, two separate connections opened with the exact filename ':memory:' "
         "create independent databases; a table created through one is not visible through the other."),
        "VALIDATED",
    ),
    "shared": (
        ("In SQLite, two separate connections opened with the exact filename ':memory:' "
         "share one database; a table created through one is immediately visible through the other."),
        "DISPROVEN",
    ),
}


def now():
    return datetime.now(UTC).isoformat()


def make_config(args, endpoint):
    config = yaml.safe_load((Path(__file__).parents[2] / "petri/defaults/petri.yaml").read_text())
    config.update(
        name="grow-live", execution={"backend": "dbos", "worker_concurrency": 1},
        research={"public_web": True}, max_decomposition_depth=1, level_one_cap=2,
        model=f"openai-chat:{endpoint.model}",
    )
    if args.harness == "provider":
        config["model_settings"] = {"max_tokens": endpoint.max_tokens}
    if args.expanded_budget:
        config["limits"] = {
            "run": {"request_limit": 120, "total_tokens_limit": 600000, "tool_calls_limit": 100},
            "debate": {"request_limit": 24, "total_tokens_limit": 100000, "tool_calls_limit": 20},
        }
    config["harness"] = (
        {"backend": "provider", "model": f"openai-chat:{endpoint.model}", "provider": None, "tools": []}
        if args.harness == "provider"
        else {"backend": "pi", "model": endpoint.model, "provider": "petri-local", "tools": []}
    )
    if args.output_mode is not None:
        config["harness"]["output_mode"] = args.output_mode
    return config


def diagnostic_summary(events, rows, cell_id, round_number, config, *, seed_ok, grow_exit):
    """Require exact current-round retrieval identities and retain the full history."""
    from petri.execution.phases import SOCRATIC_STEPS

    selected = [event for event in events if event["cell_id"] == cell_id]
    boundary = max((index for index, event in enumerate(selected)
                    if event["type"] == "cell_reopened"), default=-1)
    current = selected[boundary + 1:]
    nodes = ["research", "critique", "red_team", "evaluate", "evaluation",
             "socratic:verification", *(f"socratic:{step}" for step, _ in SOCRATIC_STEPS),
             *("debate:" + ":".join(debate["pair"]) for debate in config.get("debates", []))]

    def retrieval_matches(event):
        data = event["data"]
        if event["type"] == "source_reviewed":
            if not (data.get("retrieval_method") == "public-web" and data.get("url")
                    and data.get("retrieved_at") and data.get("content_sha256") and data.get("finding")):
                return False
            identity = data["url"] + ":" + data["content_sha256"]
        else:
            if not data.get("sources_found", 0) > 0 or not data.get("source_urls"):
                return False
            identity = hashlib.sha256(json.dumps(
                [data["query"], data["source_urls"]], sort_keys=True
            ).encode()).hexdigest()
        return any(event["id"] == database.stable_event_id(
            cell_id, "public-web", round_number, node, event["agent"], event["iteration"],
            event["type"], identity,
        ) for node in nodes)

    sources = [event for event in current if event["type"] == "source_reviewed"
               and retrieval_matches(event)]
    searches = [event for event in current if event["type"] == "search_executed"
                and retrieval_matches(event)]
    failures = [event for event in selected if event["type"] == "agent_execution_failed"]
    # Unknown failure coordinates after the selected cell's reopen remain blocking.
    current_failures = [event for event in current if event["type"] == "agent_execution_failed"]
    for event in failures:
        if event not in current_failures and any(event["id"] == derive_event_id(
            cell_id, round_number=round_number, node=node, agent=event["agent"],
            iteration=event["iteration"], event_type=event["type"],
        ) for node in nodes):
            current_failures.append(event)
    center = next((row for row in rows if row["id"] == cell_id), None)
    return {
        "cell_id": cell_id, "round_number": round_number, "workflow_id": f"{cell_id}:r{round_number}",
        "center_status": center["status"] if center else None,
        "sources": sources, "searches": searches, "failures": failures,
        "current_round_failure_count": len(current_failures),
        "selected_cell_events": selected,
        "verdicts": [event for event in selected if event["type"] == "verdict_issued"],
        "command_success": bool(seed_ok and grow_exit == 0),
    }


def diagnostic_passed(summary, expected):
    return bool(summary["command_success"] and summary["center_status"] == expected
                and summary["sources"] and summary["searches"]
                and not summary["current_round_failure_count"])


def validate_pi_config(path, endpoint):
    document = json.loads(path.read_text())
    if document != endpoint.pi_document():
        raise ValueError("The isolated Pi configuration differs from the requested endpoint/model/limits/auth.")
    return document


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--harness", required=True, choices=["provider", "pi"])
    parser.add_argument("--case", required=True, choices=CASES)
    parser.add_argument("--output-mode", choices=["tool", "prompted"],
                        help="Choose final-result delivery without changing host retrieval tools.")
    parser.add_argument("--expanded-budget", action="store_true",
                        help="Use a bounded budget for the full roster and source retrieval.")
    connection_arguments(parser)
    parser.add_argument("--seed", action="store_true",
                        help="Generate and approve the colony through the actual seed command.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume only a dish with the same saved run identity.")
    parser.add_argument("--requeue", metavar="REQUEST_ID",
                        help="Start an explicit new validation round in a preserved dish.")
    args = parser.parse_args()
    if os.environ.get("PETRI_RUN_LOCAL_GROW") != "1":
        parser.error("Set PETRI_RUN_LOCAL_GROW=1 to enable real local inference and public retrieval.")
    if args.requeue and not args.resume:
        parser.error("--requeue requires --resume.")
    try:
        endpoint = endpoint_from_args(args)
        config = make_config(args, endpoint)
        root = args.directory.resolve()
        identity_path = root / "run-identity.json"
        fixture = "model_generated_seed_approve_grow" if args.seed else "public_single_cell_known_outcome"
        if args.resume:
            saved = json.loads(identity_path.read_text())
            if not args.seed:
                fixture = saved["fixture_kind"]
        elif root.exists() and any(root.iterdir()):
            parser.error("Use an empty output directory; existing artifacts are preserved.")
        identity = {
            **endpoint.identity(), "harness": args.harness, "case": args.case,
            "claim": CASES[args.case][0], "expected_status": CASES[args.case][1],
            "fixture_kind": fixture, "source_sha": source_sha(),
            "configuration_hash": configuration_hash({"petri": config, "endpoint": endpoint.identity(),
                                                      "pi": endpoint.pi_document() if args.harness == "pi" else None}),
        }
        root.mkdir(parents=True, exist_ok=True)
        dish = root / ".petri"
        if args.resume:
            if yaml.safe_load((dish / "defaults/petri.yaml").read_text()) != config:
                raise ValueError("Preserved Petri configuration differs from the requested configuration.")
            if args.harness == "pi":
                validate_pi_config(root / "pi-agent/models.json", endpoint)
        save_identity(identity_path, identity, resume=args.resume)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    if not args.resume:
        create_petri_dish(dish, dish_name="grow-live")
        (dish / "defaults/petri.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        if args.harness == "pi":
            endpoint.pi_config(root / "pi-agent")
    attempt = root / "attempts" / (datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:8])
    attempt.mkdir(parents=True)
    command_file = root / "commands.jsonl"
    prior_commands = [json.loads(line) for line in command_file.read_text().splitlines()] if command_file.exists() else []
    commands = []
    runner = CliRunner()
    started = now()
    claim, expected = CASES[args.case]
    colony_id = f"grow-live-{args.case}"
    cell_id = f"{colony_id}-000-000"
    path = dish_db_path(dish, "grow-live")

    def invoke(command):
        began = now()
        result = runner.invoke(app, command)
        name = command[0]
        (attempt / f"{name}.log").write_text(endpoint.redact(result.output))
        detail = "".join(traceback.format_exception(result.exception)) if result.exception else None
        if detail:
            (attempt / f"{name}-exception.log").write_text(endpoint.redact(detail))
        record = {"command": command, "started_at_utc": began, "finished_at_utc": now(),
                  "exit_code": result.exit_code, "exception": endpoint.redact(detail) if detail else None,
                  "attempt": attempt.name}
        with command_file.open("a") as stream:
            stream.write(endpoint.diagnostic_json(record) + "\n")
        commands.append(record)
        return record

    seed_ok = fixture == "public_single_cell_known_outcome"
    grow_exit = None
    error = None
    old_cwd = Path.cwd()
    try:
        os.chdir(root)
        if args.resume:
            if not path.exists() or not database.get_cell(path, cell_id):
                raise ValueError("The preserved dish does not contain the requested root cell.")
            if not seed_ok:
                seeds = [record for record in prior_commands if record["command"][0] == "seed"]
                seed_ok = bool(seeds and seeds[-1]["exit_code"] == 0 and not seeds[-1]["exception"])
        # Clients reach a task-owned relay. The exact upstream URL remains in the saved identity
        # and Pi configuration; no upstream credential is given to the SDK or Pi process.
        with endpoint.guarded_route() as (base_url, token):
            env = {"OPENAI_BASE_URL": base_url, "OPENAI_API_KEY": token,
                   "PI_TELEMETRY": "0", "PI_OFFLINE": "1", "PETRI_LOCAL_RELAY_KEY": token}
            if args.harness == "pi":
                runtime_agent = attempt / "pi-agent"
                document = endpoint.pi_document(base_url)
                provider = document["providers"]["petri-local"]
                provider.update(apiKey="PETRI_LOCAL_RELAY_KEY", authHeader=True)
                provider.pop("headers", None)
                runtime_agent.mkdir()
                (runtime_agent / "models.json").write_text(json.dumps(document, indent=2))
                env["PI_CODING_AGENT_DIR"] = str(runtime_agent)
            with environment_values(env):
                if not args.resume and args.seed:
                    seed = invoke(["seed", claim, "--colony", args.case, "--no-questions"])
                    seed_ok = seed["exit_code"] == 0 and not seed["exception"]
                elif not args.resume:
                    center = Cell(id=cell_id, colony_id=colony_id, claim_text=claim, level=0, created_at=now())
                    colony = Colony(id=colony_id, dish="grow-live", center_claim=claim, center_cell_id=cell_id)
                    reserve_colony(path, colony)
                    graph = ColonyGraph(colony_id=colony_id)
                    graph.add_cell(center)
                    save_colony(path, graph, colony)
                    append_event(database.SQLiteEventLog(path), cell_id, "colony_approved", "human", 0,
                                 {"fixture_kind": fixture})
                if args.requeue:
                    from petri.execution import resolve_backend
                    from petri.execution.models import ReopenRequest

                    resolve_backend(dish).reopen_cell(ReopenRequest(
                        cell_id, args.requeue, datetime.now(UTC).date().isoformat(),
                        trigger="Human revalidation after recorded format failure",
                        max_iterations=config["max_iterations"],
                    ), petri_dir=dish)
                if seed_ok:
                    grown = invoke(["grow", "--colony", args.case, "--max-concurrent", "1"])
                    grow_exit = grown["exit_code"] if not grown["exception"] else 1
    except Exception as failure:  # noqa: BLE001 - Preserve diagnostics for any stage failure.
        error = endpoint.redact("".join(traceback.format_exception(failure)))
        (attempt / "worker-exception.log").write_text(error)
    finally:
        os.chdir(old_cwd)
    events = database.read_events(path) if path.exists() else []
    rows = database.list_cells(path) if path.exists() else []
    round_number = 1
    usage = []
    if path.exists():
        from petri.execution.scheduler import current_round

        with closing(database.connect(path)) as connection:
            tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if {"execution_round_requests", "execution_admissions"} <= tables:
                round_number = current_round(connection, cell_id)
            if "usage" in tables:
                usage = [dict(row) for row in connection.execute("SELECT * FROM usage ORDER BY span_id")]
    summary = diagnostic_summary(events, rows, cell_id, round_number, config,
                                 seed_ok=seed_ok, grow_exit=grow_exit)
    report = {
        "scope": "limited grow diagnostic; complete lifecycle acceptance is not evaluated",
        "complete_cycle_passed": False, "fixture_kind": fixture, "resumed": args.resume,
        "identity": identity, "configured_limits": config["limits"],
        "started_at_utc": started, "finished_at_utc": now(), "commands": commands,
        "expected_status": expected, "error": error, **summary,
        "dish_usage_diagnostics": usage, "dish_event_counts_diagnostics": dict(Counter(event["type"] for event in events)),
        "cells": [{"id": row["id"], "status": row["status"]} for row in rows],
    }
    report["passed"] = not error and diagnostic_passed(summary, expected)
    rendered = endpoint.diagnostic_json(report, indent=2) + "\n"
    (attempt / "result.json").write_text(rendered)
    # Each attempt is retained; this pointer never replaces an earlier report.
    print(json.dumps({"result_path": str(attempt / "result.json"), "passed": report["passed"],
                      "complete_cycle_passed": False, "center_status": summary["center_status"]}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
