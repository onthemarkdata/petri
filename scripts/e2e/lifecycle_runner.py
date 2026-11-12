"""Opt-in, resumable Petri lifecycle acceptance; never enabled by offline tests."""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

import yaml
from filelock import FileLock, Timeout

from scripts.e2e.lifecycle_checks import (
    AcceptanceError,
    canonical,
    check_feed_plan,
    check_response_readback,
    continuity,
    cross_references,
    digest,
    domain,
    feed_report,
    fetched_source,
    logical_rows,
    require,
    process_identity,
    same_process,
    research_report,
    workflow_snapshot,
)

REPO = Path(__file__).resolve().parents[2]
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
STAGES = (
    "prepare", "seed_recovery", "seed_isolated", "approval_isolated", "seed_continuity",
    "seed_shared", "approval_shared", "reference", "grow_recovery", "grow",
    "grow_continuity", "pre_feed", "feed", "post_feed", "repeat_feed", "storage",
    "dashboard_api", "browser", "trace_readback",
)
EXTERNAL = {"dashboard_api", "browser", "trace_readback"}
BROWSER_CHECKS = {"graph", "cells", "usage", "trace_waterfall", "seed_controls", "run_controls",
                  "read_only_during_cli", "no_console_errors"}
TRACE_CHECKS = {"workflow_step_agent_model_tool_ancestry", "request_response_readback",
                "source_verdict_readback", "round_usage_readback"}


def now():
    return datetime.now(UTC).isoformat()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def load(path):
    return json.loads(path.read_text())


def artifact(root, path):
    path = path.resolve()
    require(path.is_relative_to(root.resolve()) and path.is_file() and path.stat().st_size > 0,
            "Evidence must be a nonempty artifact inside the run directory.")
    return {"path": str(path.relative_to(root)), "sha256": digest(path.read_bytes())}


class Manifest:
    def __init__(self, root, *, resume=False):
        self.root = root
        self.path = root / "manifest.json"
        if resume:
            self.data = load(self.path)
            for stage in self.data["stages"].values():
                if stage["status"] == "running":
                    stage["status"] = "interrupted"
                    stage["attempts"][-1]["status"] = "interrupted"
        else:
            self.data = {"schema_version": 2, "retained_inputs": {}, "created_at": now(), "live_acceptance": False,
                         "stages": {}}
            for cycle in range(1, 4):
                for stage in STAGES:
                    self.data["stages"][f"cycle-{cycle}/{stage}"] = {
                        "status": "pending" if stage in EXTERNAL else "missing", "attempts": []}
            self.data["stages"]["deterministic_recovery"] = {"status": "pending", "attempts": []}
        self.save()

    def save(self):
        self.data["updated_at"] = now()
        self.data["live_acceptance"] = self.accepted()
        write_json(self.path, self.data)

    def accepted(self):
        expected = {f"cycle-{cycle}/{stage}" for cycle in range(1, 4) for stage in STAGES}
        expected.add("deterministic_recovery")
        if not expected <= self.data["stages"].keys():
            return False
        for key in expected:
            stage = self.data["stages"][key]
            if stage["status"] != "passed" or not stage.get("attempts"):
                return False
            files = stage["attempts"][-1].get("artifacts", [])
            if not files:
                return False
            for record in files:
                path = (self.root / record["path"]).resolve()
                if (not path.is_relative_to(self.root.resolve()) or not path.is_file()
                        or digest(path.read_bytes()) != record["sha256"]):
                    return False
        return True

    def retain(self, path, folder):
        key = str(path.relative_to(self.root))
        if key in self.data["retained_inputs"]:
            self.validate_retained()
            return
        saved = folder / ("retained-" + path.name)
        shutil.copyfile(path, saved)
        self.data["retained_inputs"][key] = artifact(self.root, saved)
        self.save()

    def validate_retained(self):
        require(self.data.get("schema_version") == 2, "Resume requires the reviewed v2 manifest; preserve the older run.")
        for key, record in self.data["retained_inputs"].items():
            require(artifact(self.root, self.root / record["path"]) == record,
                    "Retained sidecar evidence changed: " + key)
            current = self.root / key
            if not current.exists() and current.name == "source-evidence.txt":
                current = current.with_name(".source-evidence-retained.txt")
            require(current.is_file() and digest(current.read_bytes()) == record["sha256"],
                    "Live sidecar differs from retained attempt evidence: " + key)

    def run(self, name, action):
        stage = self.data["stages"][name]
        if stage["status"] == "passed":
            records = stage["attempts"][-1]["artifacts"]
            for item in records:
                require(artifact(self.root, self.root / item["path"]) == item,
                        "A previously passed stage artifact changed.")
            return load(self.root / records[0]["path"])
        folder = self.root / "attempts" / name / str(len(stage["attempts"]) + 1)
        folder.mkdir(parents=True)
        attempt = {"status": "running", "started_at": now(), "directory": str(folder.relative_to(self.root)), "artifacts": []}
        stage["attempts"].append(attempt)
        stage["status"] = "running"
        self.save()
        try:
            result = action(folder)
            write_json(folder / "result.json", result)
            attempt["artifacts"] = [artifact(self.root, folder / "result.json")]
            # Logs, snapshots, and reports must remain available for readback.
            attempt["artifacts"] += [artifact(self.root, path) for path in sorted(folder.rglob("*"))
                                     if path.is_file() and path.name != "result.json"
                                     and path.stat().st_size > 0]
            attempt["status"] = stage["status"] = "passed"
            return result
        except BaseException as error:
            attempt["status"] = stage["status"] = (
                "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed")
            attempt["error"] = str(error)
            raise
        finally:
            attempt["finished_at"] = now()
            self.save()


def child_worker(ticket):
    """Wait for a durable parent authorization before executing the actual Petri CLI."""
    planned = load(ticket)
    identity = process_identity(os.getpid())
    write_json(ticket.with_suffix(".child.json"), identity)
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        current = load(ticket)
        if not same_process(planned["parent"], process_identity(planned["parent"]["pid"])):
            return 125
        require(current["launch_id"] == planned["launch_id"], "The child launch ticket changed.")
        if current.get("authorized"):
            require(same_process(identity, current.get("child")), "Child authorization has another birth identity.")
            os.execv(sys.executable, planned["command"])
        time.sleep(0.05)
    return 125


def reconcile_children(root):
    """Fail on surviving owned children; a reused PID is never a signal target."""
    outcomes = []
    for ticket in sorted((root / "attempts").rglob("*.process.json")):
        record = load(ticket)
        child = record.get("child")
        ack = ticket.with_suffix(".child.json")
        if child is None and ack.exists():
            child = load(ack)
        current = process_identity(child["pid"]) if child else None
        if same_process(child, current) and current["state"] != "Z":
            raise AcceptanceError(
                f"Earlier task-owned child PID {child['pid']} (start {child['start_ticks']}) remains active. "
                "Wait for it to exit or have Root inspect and stop that exact process before resume; no new command was started.")
        if not child and record.get("authorized"):
            raise AcceptanceError("An authorized child has no retained birth identity; Root must inspect the launch ticket.")
        outcomes.append({"ticket": str(ticket.relative_to(root)),
                         "state": "pid_reused" if current else "exited_or_not_authorized"})
    write_json(root / "child-reconciliation.json", {"observed_at": now(), "children": outcomes})
    for number in range(1, 4):
        petri = root / f"cycle-{number}" / ".petri"
        if petri.is_dir():
            try:
                with FileLock(str(petri / ".execution-owner.lock"), timeout=0):
                    pass
            except Timeout:
                raise AcceptanceError("A cycle still has an execution owner; Root must reconcile it before resume.") from None


class Commands:
    def __init__(self, endpoint, timeout, transport_key="", *, preflight=None):
        self.endpoint, self.timeout, self.transport_key = endpoint, timeout, transport_key
        self.preflight = preflight

    def launch(self, record, ticket, raw):
        # A wrapper cannot enter Petri until both its birth identity and authorization are durable.
        process = None
        try:
            process = subprocess.Popen(
                [sys.executable, "-m", "scripts.e2e.lifecycle_runner", "_child", str(ticket)],
                cwd=record["cwd"], stdout=raw, stderr=raw, stdin=subprocess.DEVNULL)
            record["pid"] = process.pid
            deadline = time.monotonic() + 10
            ack = ticket.with_suffix(".child.json")
            while not ack.exists() and process.poll() is None:
                require(time.monotonic() < deadline, "Child launch acknowledgement timed out.")
                time.sleep(0.05)
            if ack.exists():
                child = load(ack)
                require(child["pid"] == process.pid and same_process(child, process_identity(process.pid)),
                        "Child acknowledgement has another process birth identity.")
                record.update(child=child, authorized=True)
                write_json(ticket, record)
            return process
        except BaseException:
            if process is not None:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=15)
                record["exit_code"] = process.returncode
                record["launch_aborted"] = True
                write_json(ticket, record)
            raise

    def run(self, cwd, arguments, folder, label, *, checkpoint=None):
        """Own one acknowledged child and persist each interruption boundary."""
        if self.preflight:
            self.preflight()
        command = [sys.executable, "-m", "petri", *arguments]
        raw = folder / (label + ".raw")
        ticket = folder / (label + ".process.json")
        started = time.monotonic()
        record = {"command": command, "cwd": str(cwd.resolve()), "started_at": now(), "pid": None,
                  "parent": process_identity(os.getpid()), "launch_id": uuid4().hex, "authorized": False,
                  "timeout_seconds": self.timeout, "checkpoint": None}
        write_json(ticket, record)
        try:
            with raw.open("wb") as stream:
                raw.chmod(0o600)
                process = self.launch(record, ticket, stream)
                try:
                    while process.poll() is None:
                        if checkpoint:
                            observation = checkpoint()
                            if observation:
                                record["checkpoint"] = observation
                                write_json(ticket, record)
                                require(process.poll() is None, "Petri exited before the observed crash boundary.")
                                record["termination_requested"] = True
                                write_json(ticket, record)
                                process.kill()
                                process.wait(timeout=15)
                                break
                        if time.monotonic() - started > self.timeout:
                            record["timed_out"] = True
                            raise AcceptanceError("Petri exceeded the declared command time bound.")
                        time.sleep(0.5)
                finally:
                    if process.poll() is None:
                        record["cleanup_termination"] = True
                        process.kill()
                        process.wait(timeout=15)
                    record["exit_code"] = process.returncode
                    write_json(ticket, record)
        finally:
            record["elapsed_seconds"] = time.monotonic() - started
            if raw.exists():
                text = self.endpoint.redact(raw.read_text(errors="replace"))
                if self.transport_key:
                    text = text.replace(self.transport_key, "[REDACTED]")
                (folder / (label + ".log")).write_text(text)
                raw.unlink()
            write_json(ticket, record)
            write_json(folder / (label + ".json"), record)
        if checkpoint:
            require(record["checkpoint"] and record.get("termination_requested")
                    and record["exit_code"] == -signal.SIGKILL,
                    "No task-owned process was terminated after an active durable checkpoint.")
        else:
            require(record["exit_code"] == 0, f"Petri command failed: {arguments[0]}.")
        return record


def otlp_loopback_url(value):
    try:
        parsed = urlsplit(value)
        host = parsed.hostname
        local = host == "localhost" or ipaddress.ip_address(host).is_loopback
        port = parsed.port
        require(local and parsed.scheme in {"http", "https"} and not parsed.username
                and not parsed.password and not parsed.query and not parsed.fragment
                and port != 0 and not any(character.isspace() for character in value),
                "Use a loopback OTLP URL without credentials, query, or fragment.")
    except (TypeError, ValueError):
        raise AcceptanceError("Use a loopback OTLP URL without credentials, query, or fragment.") from None
    return value


def configuration(endpoint, args, dish):
    config = yaml.safe_load((REPO / "petri/defaults/petri.yaml").read_text())
    config.update(name=dish, model=f"openai-chat:{endpoint.model}",
                  execution={"backend": "dbos", "worker_concurrency": 1},
                  research={"public_web": True}, max_decomposition_depth=2,
                  level_one_cap=2, per_parent_cap=2, max_nodes_per_layer=4)
    config["harness"] = {"backend": "provider", "model": config["model"], "provider": None,
                         "tools": [], "output_mode": args.output_mode}
    compatibility = profile_selection(args)["openai_chat_supports_multiple_system_messages"]
    if compatibility is not None:
        config["harness"]["openai_chat_supports_multiple_system_messages"] = compatibility
    config["model_settings"] = {"max_tokens": endpoint.max_tokens}
    config["limits"] = {
        "run": {"request_limit": args.request_limit, "total_tokens_limit": args.token_limit,
                "tool_calls_limit": args.tool_limit},
        "debate": {"request_limit": 24, "total_tokens_limit": 100000, "tool_calls_limit": 20},
    }
    capture = getattr(args, "capture_content", False)
    otlp = getattr(args, "otlp_traces_endpoint", None)
    if capture or otlp:
        telemetry = config.setdefault("telemetry", {})
        telemetry["enabled"] = True
        if capture:
            telemetry["capture_content"] = True
            telemetry["redact_env_vars"] = sorted(set(telemetry.get("redact_env_vars", [])) | {"OPENAI_API_KEY"})
        if otlp:
            telemetry.setdefault("otlp", {})["traces_endpoint"] = otlp_loopback_url(otlp)
    return config


def profile_selection(args):
    choice = getattr(args, "openai_chat_system_messages", "default")
    require(choice in {"default", "single", "multiple"}, "Unknown OpenAI Chat system-message selection.")
    require(args.output_mode in {"tool", "prompted"}, "Unknown structured output mode.")
    return {"output_mode": args.output_mode,
            "openai_chat_supports_multiple_system_messages": {
                "default": None, "single": False, "multiple": True}[choice]}


def validate_provider(config):
    """Use the public resolver; explicit profile support requires its schema contract."""
    from petri.agents.config import HarnessContent
    from petri.harness.resolve import resolve_model

    harness = config["harness"]
    require(harness["backend"] == "provider" and
            (harness.get("model") or config["model"]).startswith("openai-chat:"),
            "Lifecycle inference requires the explicit provider/openai-chat route.")
    key = "openai_chat_supports_multiple_system_messages"
    require(key not in harness or key in HarnessContent.model_fields,
            "Integrate the reviewed OpenAI Chat profile resolver before selecting single or multiple system messages.")
    resolve_model(config)


def run_identity(endpoint_module, endpoint, configs, args):
    identity = {"source_sha": endpoint_module.source_sha(), "endpoint": endpoint.identity(),
            "configuration_hash": endpoint_module.configuration_hash(configs), "configurations": configs,
            "cases": json.loads(json.dumps(CASES)), "timeout_seconds": args.timeout, "concurrency": 1,
            "coverage": "model_generated_lifecycle", "provider_selection": profile_selection(args),
            "diagnostics": {"capture_content": args.capture_content,
                            "otlp_traces_endpoint": args.otlp_traces_endpoint,
                            "redact_env_vars": ["OPENAI_API_KEY"] if args.capture_content else []}}
    # YAML source levels use integer keys; the saved identity must compare in JSON form.
    return json.loads(json.dumps(identity))


class Cycle:
    def __init__(self, manifest, number, commands, config):
        self.manifest, self.commands, self.config = manifest, commands, config
        self.label = f"cycle-{number}"
        self.root = manifest.root / self.label
        self.petri = self.root / ".petri"
        self.dish = config["name"]
        self.db = self.petri / "petri-dishes" / self.dish / "petri.sqlite"

    def stage(self, name, action):
        return self.manifest.run(self.label + "/" + name, action)

    def snapshot(self):
        return {"domain": domain(self.db), "workflows": workflow_snapshot(self.petri)}

    def validate_state(self):
        self.manifest.validate_retained()
        path = self.petri / "defaults/petri.yaml"
        if self.root.exists():
            require(path.is_file(), "An existing cycle is missing its live configuration.")
            require(yaml.safe_load(path.read_text()) == self.config,
                    "Live cycle configuration differs from the saved run identity: " + self.label)
            # Use Petri's effective loader as well as the exact saved YAML comparison.
            from petri.config import load_dish_config
            load_dish_config(self.petri)
        for name in ("seed-observed.json", "seed-checkpoint.json", "grow-observed.json",
                     "grow-checkpoint.json", "feed-before.json", "source-evidence.txt"):
            path = self.root / name
            if path.exists():
                require(str(path.relative_to(self.manifest.root)) in self.manifest.data["retained_inputs"],
                        "An unretained lifecycle sidecar needs review before resume: " + name)

    def prepare(self, folder):
        from petri.cli.init import create_petri_dish

        self.root.mkdir(exist_ok=True)
        path = self.petri / "defaults/petri.yaml"
        if not path.exists():
            create_petri_dish(self.petri, dish_name=self.dish)
            path.write_text(yaml.safe_dump(self.config, sort_keys=False))
        require(yaml.safe_load(path.read_text()) == self.config, "Preserved configuration differs.")
        write_json(folder / "configuration.json", self.config)
        return {"dish_id": self.dish, "configuration": str(path)}

    def recover(self, folder, kind):
        saved = self.root / (kind + "-checkpoint.json")
        self.manifest.validate_retained()
        if saved.exists():
            return load(saved)
        previous = sorted((self.manifest.root / "attempts" / self.label / (kind + "_recovery")).rglob(
            kind + "-interrupt.process.json"))
        if previous:
            terminated = load(previous[-1])
            require(terminated.get("exit_code") == -signal.SIGKILL
                    and terminated.get("termination_requested") and terminated.get("checkpoint"),
                    "The previous recovery experiment has no confirmed crash outcome. Root must inspect its ticket; do not start a new experiment on old checkpoints.")
            reference = terminated["checkpoint"]
            observed = self.manifest.root / reference["path"]
            require(artifact(self.manifest.root, observed) == reference, "The retained crash observation changed.")
        else:
            target = "petri.decompose_seed_step" if kind == "seed" else "petri.cell_phase"

            def checkpoint():
                observation = self.snapshot()
                active = [(key, row, step) for key, row in observation["workflows"].items()
                          if row["status"] == "PENDING"
                          and (key == self.dish + "-isolated" if kind == "seed" else
                               key.startswith(self.dish + "-") and row["name"] == "petri.validate_cell")
                          for step in row["steps"] if step["function_name"] == target and not step["error"]]
                if not active:
                    return None
                workflow, row, step = active[0]
                observation["crash_observation"] = {
                    "workflow_id": workflow, "status": row["status"], "checkpoint": step,
                    "observed_at": now()}
                observed = self.root / (kind + "-observed.json")
                write_json(observed, observation)
                self.manifest.retain(observed, folder)
                return artifact(self.manifest.root, observed)

            arguments = (["seed", CASES["isolated"][0], "--colony", "isolated", "--no-questions"]
                         if kind == "seed" else ["grow", "--max-concurrent", "1"])
            if kind == "seed" and any(row["id"] == self.dish + "-isolated"
                                      for row in domain(self.db)["colonies"]):
                arguments = ["seed", "--resume", "--colony", "isolated", "--no-questions"]
            terminated = self.commands.run(self.root, arguments, folder, kind + "-interrupt", checkpoint=checkpoint)
            observed = self.root / (kind + "-observed.json")
        observation = load(observed)
        active = observation.get("crash_observation", {})
        require(active.get("status") == "PENDING" and active.get("checkpoint"),
                "Recovery requires an observed active workflow and its committed checkpoint.")
        observation["termination"] = terminated
        write_json(saved, observation)
        self.manifest.retain(saved, folder)
        write_json(folder / "checkpoint.json", observation)
        return observation

    def seed(self, folder, case):
        colony = self.dish + "-" + case
        rows = domain(self.db)["colonies"]
        arguments = (["seed", "--resume", "--colony", case, "--no-questions"]
                     if any(row["id"] == colony for row in rows)
                     else ["seed", CASES[case][0], "--colony", case, "--no-questions"])
        result = self.commands.run(self.root, arguments, folder, "seed")
        write_json(folder / "snapshot.json", self.snapshot())
        return result

    def approval(self, folder, case):
        snapshot = domain(self.db)
        center = self.dish + "-" + case + "-000-000"
        colony = next((row for row in snapshot["colonies"] if row["id"] == self.dish + "-" + case), None)
        require(colony and colony["center_cell_id"] == center and colony["center_claim"] == CASES[case][0],
                "The seeded colony changed its declared root claim or identity.")
        events = [event for event in snapshot["events"] if event["cell_id"] == center]
        approved = [event for event in events if event["event_type"] == "colony_approved"]
        generated = [event for event in events if event["event_type"] == "decomposition_completed"]
        require(approved and generated, "Model decomposition and actual approval events are required.")
        result = {"dish_id": self.dish, "colony_id": colony["id"], "center_cell_id": center,
                  "approval_mode": "CLI --no-questions auto-accept", "approval": approved,
                  "decomposition": generated, "expected_pre_feed": CASES[case][1],
                  "expected_post_feed": CASES[case][1]}
        write_json(folder / "approval.json", result)
        return result

    def verify_continuity(self, folder, kind):
        before = load(self.root / (kind + "-checkpoint.json"))
        after = self.snapshot()
        count = continuity(before["workflows"], after["workflows"])
        later = {event["id"]: event for event in after["domain"]["events"]}
        require(all(later.get(event["id"]) == event for event in before["domain"]["events"]),
                "A committed event changed or disappeared after recovery.")
        write_json(folder / "after.json", after)
        return {"committed_steps_preserved": count,
                "committed_events_preserved": len(before["domain"]["events"]),
                "external_call_repetition": "An uncommitted external call may repeat."}

    def reference(self, folder):
        snapshot = domain(self.db)
        references = cross_references(snapshot, self.dish)
        require(references, "The second colony did not reuse a non-root foundation cell.")
        require(not list(self.petri.rglob("edges.json")) and not list(self.petri.rglob("edges.jsonl")),
                "Unexpected dish-level edge side files exist.")
        write_json(folder / "graph.json", snapshot)
        return {"dish_id": self.dish, "references": references}

    def grow(self, folder):
        result = self.commands.run(self.root, ["grow", "--max-concurrent", "1"], folder, "grow")
        write_json(folder / "snapshot.json", self.snapshot())
        return result

    def research(self, folder, *, post=False):
        snapshot = self.snapshot()
        expectations = ({self.dish + "-" + case + "-000-000": expected
                         for case, (_, expected) in CASES.items()})
        if post:
            expectations.update(load(self.root / "feed-before.json")["targets"])
        reports = []
        for cell_id, expected in expectations.items():
            number = 2 if post and cell_id in load(self.root / "feed-before.json")["targets"] else 1
            reports.append(research_report(snapshot["domain"], snapshot["workflows"], cell_id,
                                           number, expected, self.config))
        write_json(folder / "snapshot.json", snapshot)
        write_json(folder / "research.json", reports)
        require(all(report["passed"] for report in reports), "At least one required cell-round failed research acceptance.")
        return {"dish_id": self.dish, "rounds": reports}

    def feed(self, folder):
        context = self.root / "feed-before.json"
        source = self.root / "source-evidence.txt"
        if not context.exists():
            before = self.snapshot()
            targets = check_feed_plan(before["domain"], self.dish + "-isolated")
            require(targets and self.dish + "-isolated-000-000" in targets,
                    "The foundation has no validated root eligible for feed.")
            candidates = [event for event in before["domain"]["events"]
                          if event["cell_id"] in targets and event["event_type"] == "source_reviewed"
                          and event["data"].get("retrieval_method") == "public-web"
                          and fetched_source(event["data"])]
            require(candidates, "No real fetched source is available to feed.")
            # Feed source text actually returned by Petri retrieval, with its provenance.
            evidence = "\n\n".join(canonical(event["data"]) for event in candidates)
            source.write_text(evidence)
            write_json(context, {**before, "targets": targets, "evidence": evidence,
                                 "source_events": candidates})
            self.manifest.retain(context, folder)
            self.manifest.retain(source, folder)
        self.manifest.validate_retained()
        before = load(context)
        check_feed_plan(self.snapshot()["domain"], self.dish + "-isolated", before["targets"])
        require(source.read_text() == before["evidence"], "The preserved feed source changed.")
        self.commands.run(self.root, ["feed", str(source), "--colony", "isolated", "--auto-reopen"],
                          folder, "feed")
        after = self.snapshot()
        write_json(folder / "after.json", after)
        result = feed_report(before["domain"], after["domain"], before["workflows"],
                             after["workflows"], before["targets"], before["evidence"])
        for fork in result["forks"]:
            path = (self.db.parent / fork["evidence_path"]).resolve()
            require(path.is_relative_to(self.db.parent.resolve()) and path.is_file(),
                    "Retained fed evidence file is missing or outside the dish.")
            require(digest(load(path)["content"].encode()) == fork["content_sha256"],
                    "Retained fed evidence content differs from the committed digest.")
        write_json(folder / "feed-before.json", before)
        return result

    def repeat_feed(self, folder):
        self.manifest.validate_retained()
        plan = load(self.root / "feed-before.json")
        before = self.snapshot()
        check_feed_plan(before["domain"], self.dish + "-isolated", plan["targets"])
        self.commands.run(self.root, ["feed", str(self.root / "source-evidence.txt"), "--colony",
                                     "isolated", "--auto-reopen"], folder, "repeat-feed")
        after = self.snapshot()
        kinds = {"evidence_fed", "cell_reopened", "propagation_triggered"}
        require([event for event in before["domain"]["events"] if event["event_type"] in kinds]
                == [event for event in after["domain"]["events"] if event["event_type"] in kinds],
                "Repeated evidence created another fork or propagation effect.")
        old = {key for key, value in before["workflows"].items() if value["name"] == "petri.validate_cell"}
        new = {key for key, value in after["workflows"].items() if value["name"] == "petri.validate_cell"}
        require(old == new, "Repeated evidence created another validation workflow.")
        write_json(folder / "before.json", before)
        write_json(folder / "after.json", after)
        return {"workflow_ids": sorted(new), "evidence_identity_deduplicated": True}

    def storage(self, folder):
        from petri.storage.backup import list_snapshots

        def exported(root):
            return {str(path.relative_to(root)): digest(path.read_bytes())
                    for path in sorted(root.rglob("*")) if path.is_file()}

        exports = self.db.parent / "exports"
        self.commands.run(self.root, ["export"], folder, "export-1")
        first = exported(exports)
        self.commands.run(self.root, ["export"], folder, "export-2")
        require(first and first == exported(exports), "Repeated exports differ.")
        original = logical_rows(self.db)
        self.commands.run(self.root, ["backup", "--with-export"], folder, "backup")
        self.commands.run(self.root, ["backup", "--list"], folder, "backup-list")
        snapshots = list_snapshots(self.petri, self.dish)
        require(snapshots, "The CLI did not create a snapshot.")
        snapshot = snapshots[0]
        require(snapshot.name in (folder / "backup-list.log").read_text(), "Snapshot missing from backup --list.")
        restored = folder / "isolated-restore"
        backup = restored / ".petri/backups" / snapshot.name
        backup.parent.mkdir(parents=True)
        shutil.copytree(snapshot.path, backup)
        # A marker exercises the production warning; this is not restored DBOS control state.
        (restored / ".petri/dbos.sqlite").touch()
        self.commands.run(restored, ["backup", "--restore", snapshot.name], folder, "restore")
        require("WARNING: DBOS" in (folder / "restore.log").read_text()
                and "desynchronize" in (folder / "restore.log").read_text(),
                "The domain/DBOS desynchronization warning was lost.")
        target = restored / ".petri/petri-dishes" / self.dish / "petri.sqlite"
        require(original == logical_rows(target), "Restored logical rows differ.")
        # Overwrite must fail without --force. Keep the failed command as expected evidence.
        try:
            self.commands.run(restored, ["backup", "--restore", snapshot.name], folder, "overwrite-guard")
        except AcceptanceError:
            record = load(folder / "overwrite-guard.json")
            require(record.get("exit_code", 0) != 0 and "--force" in
                    (folder / "overwrite-guard.log").read_text(), "Restore failed for an unexpected reason.")
        else:
            raise AcceptanceError("Restore overwrote an existing destination without --force.")
        require(original == logical_rows(target), "The overwrite guard changed destination rows.")
        require(not (target.parent / "evidence").exists(), "Restore unexpectedly copied source evidence files.")
        source = self.root / "source-evidence.txt"
        held = self.root / ".source-evidence-retained.txt"
        if held.exists() and not source.exists():
            held.replace(source)
        require(source.is_file() and not held.exists(), "The preserved feed source is unavailable.")
        source.replace(held)
        try:
            self.commands.run(restored, ["export"], folder, "restored-export")
        finally:
            held.replace(source)
        restored_exports = target.parent / "exports"
        require(first == exported(restored_exports), "Restored exports differ from the source exports.")
        fed = [event for event in domain(target)["events"] if event["event_type"] == "evidence_fed"]
        require(fed, "The restored database has no fed evidence.")
        for event in fed:
            markdown = (restored_exports / "cells" / event["cell_id"] / "cell.md").read_text()
            require(event["data"]["content"].splitlines()[0] in markdown,
                    "Exported restored evidence is missing without its original source file.")
        require(original == logical_rows(self.db), "Storage checks changed the source dish.")
        write_json(folder / "export-hashes.json", first)
        return {"dish_id": self.dish, "snapshot": snapshot.name, "integrity": "ok",
                "logical_rows_equal": True, "overwrite_protected": True,
                "dbos_warning": "Domain restore can desynchronize DBOS checkpoints; not complete DBOS recovery."}

    def execute(self):
        self.validate_state()
        self.stage("prepare", self.prepare)
        self.stage("seed_recovery", lambda folder: self.recover(folder, "seed"))
        self.stage("seed_isolated", lambda folder: self.seed(folder, "isolated"))
        self.stage("approval_isolated", lambda folder: self.approval(folder, "isolated"))
        self.stage("seed_continuity", lambda folder: self.verify_continuity(folder, "seed"))
        self.stage("seed_shared", lambda folder: self.seed(folder, "shared"))
        self.stage("approval_shared", lambda folder: self.approval(folder, "shared"))
        self.stage("reference", self.reference)
        self.stage("grow_recovery", lambda folder: self.recover(folder, "grow"))
        self.stage("grow", self.grow)
        self.stage("grow_continuity", lambda folder: self.verify_continuity(folder, "grow"))
        self.stage("pre_feed", self.research)
        self.stage("feed", self.feed)
        self.stage("post_feed", lambda folder: self.research(folder, post=True))
        self.stage("repeat_feed", self.repeat_feed)
        self.stage("storage", self.storage)


def external_evidence(manifest, name, evidence_path):
    """Validate Root's scoped observations and retain their referenced artifacts."""
    evidence = load(evidence_path)
    require(evidence.get("observer") and evidence.get("observed_at"), "Name the observer and capture time.")
    if name == "deterministic_recovery":
        require(evidence.get("source_sha") == load(manifest.root / "identity.json")["source_sha"],
                "Deterministic recovery evidence belongs to another source revision.")
        required = {"seed_exact_counts", "grow_exact_counts", "feed_exact_counts", "stop_semantics"}
    else:
        cycle, kind = name.split("/")
        require(kind in EXTERNAL, "Only Root-operated stages accept evidence attachments.")
        dish = "lifecycle-" + cycle
        require(evidence.get("dish_id") == dish, "Evidence belongs to another dish.")
        db = manifest.root / cycle / ".petri/petri-dishes" / dish / "petri.sqlite"
        snapshot = domain(db)
        roots = {dish + "-" + case + "-000-000": status for case, (_, status) in CASES.items()}
        require(all(any(row["id"] == cell and row["status"] == status for row in snapshot["cells"])
                    for cell, status in roots.items()), "Browser/readback evidence needs successful known-outcome dishes.")
        require(evidence.get("root_statuses") == roots, "Evidence omits or changes required root statuses.")
        required = BROWSER_CHECKS if kind == "browser" else TRACE_CHECKS
        if kind == "dashboard_api":
            required = {"graph_payload", "cell_payloads", "usage_payloads", "trace_payloads",
                        "sse_committed_events", "read_only_during_cli"}
        if kind == "trace_readback":
            spans = {row["span_id"]: row for row in snapshot["spans"]}
            require(evidence.get("span_ids") and all(identity in spans for identity in evidence["span_ids"]),
                    "Stored trace readback cites missing span identities.")
            workflows = workflow_snapshot(manifest.root / cycle / ".petri")
            require(evidence.get("workflow_ids") and all(key in workflows for key in evidence["workflow_ids"]),
                    "Stored trace readback cites missing workflow identities.")
            expected = {cell + ":r1" for cell in roots}
            context = manifest.root / cycle / "feed-before.json"
            require(context.is_file(), "Trace readback needs the declared feed target set.")
            expected |= {cell + ":r2" for cell in load(context)["targets"]}
            require(set(evidence["workflow_ids"]) == expected, "Trace readback omits a required cell-round.")
            selected = [json.loads(spans[key]["attributes"]) for key in evidence["span_ids"]]
            observed = {f"{attrs['petri.cell_id']}:r{attrs['petri.round']}" for attrs in selected
                        if "petri.cell_id" in attrs and "petri.round" in attrs}
            require(expected <= observed, "Trace readback omits required workflow span identities.")
            for record in evidence.get("artifacts", []):
                require(artifact(manifest.root, manifest.root / record["path"]) == record,
                        "A response readback attachment changed.")
            manifest.validate_retained()
            check_response_readback(snapshot, expected, evidence, manifest.root)
    require(set(evidence.get("checks", {})) == required and
            all(value is True for value in evidence["checks"].values()), "Required observations are incomplete.")
    files = evidence.get("artifacts", [])
    require(files, "Evidence attachments are missing.")
    for record in files:
        require(artifact(manifest.root, manifest.root / record["path"]) == record,
                "An evidence attachment is missing or changed.")
    def attach(folder):
        write_json(folder / "observation.json", evidence)
        # Retain independent copies so the stage cannot pass after a capture disappears.
        for index, record in enumerate(files):
            shutil.copyfile(manifest.root / record["path"], folder / f"capture-{index}.artifact")
        return evidence
    return manifest.run(name, attach)


def main(argv=None):
    values = argv if argv is not None else sys.argv[1:]
    if values[:1] == ["_child"]:
        return child_worker(Path(values[1]))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["run", "status", "attach", "capture-dashboard"])
    parser.add_argument("directory", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cycle", type=int, choices=[1, 2, 3])
    parser.add_argument("--dashboard-url")
    parser.add_argument("--cli-pid", type=int)
    parser.add_argument("--capture-timeout", type=float, default=120)
    parser.add_argument("--retry-interval", type=float, default=1)
    parser.add_argument("--final-status-timeout", type=float, default=14400)
    parser.add_argument("--stage")
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--timeout", type=int, default=14400)
    parser.add_argument("--request-limit", type=int, default=120)
    parser.add_argument("--token-limit", type=int, default=600000)
    parser.add_argument("--tool-limit", type=int, default=100)
    parser.add_argument("--capture-content", action="store_true",
                        help="Opt in to stored model/tool content; requires telemetry integration.")
    parser.add_argument("--otlp-traces-endpoint", type=otlp_loopback_url,
                        help="Explicit loopback OTLP HTTP/protobuf trace URL.")
    parser.add_argument("--output-mode", choices=["tool", "prompted"], default="tool")
    parser.add_argument("--openai-chat-system-messages", choices=["default", "single", "multiple"],
                        default="default", help="Explicit OpenAI Chat profile; requires the reviewed resolver.")
    # Deliverable A is deliberately optional for status, attachments, and offline tests.
    endpoint_module = None
    if (argv or sys.argv[1:])[:1] == ["run"]:
        try:
            from scripts.e2e import local_endpoint as endpoint_module
        except ImportError:
            parser.error("Integrate Deliverable A's scripts.e2e.local_endpoint before live execution.")
        endpoint_module.connection_arguments(parser)
    args = parser.parse_args(argv)
    root = args.directory.resolve()
    if args.action == "capture-dashboard":
        from scripts.e2e.lifecycle_dashboard import capture_dashboard
        require(args.cycle and args.dashboard_url and args.cli_pid,
                "Dashboard capture requires --cycle, --dashboard-url, and --cli-pid.")
        print(capture_dashboard(root, args.cycle, args.dashboard_url, args.cli_pid,
                                capture_timeout=args.capture_timeout, retry_interval=args.retry_interval,
                                final_status_timeout=args.final_status_timeout))
        return 0
    if args.action == "run":
        require(os.environ.get("PETRI_RUN_LIFECYCLE") == "1",
                "Set PETRI_RUN_LIFECYCLE=1 only after Root validates the assigned endpoint and device headroom.")
        require(min(args.timeout, args.request_limit, args.token_limit, args.tool_limit) > 0,
                "Request, token, tool, and command time bounds must be positive.")
        require(args.resume or not root.exists() or not any(root.iterdir()), "Use a new empty artifact directory.")
        root.mkdir(parents=True, exist_ok=True)
    require(root.is_dir(), "The run directory is missing.")
    with FileLock(str(root / ".lifecycle-owner.lock"), timeout=0):
        if args.action == "run":
            endpoint = endpoint_module.endpoint_from_args(args)
            configs = [configuration(endpoint, args, f"lifecycle-cycle-{number}") for number in range(1, 4)]
            identity = run_identity(endpoint_module, endpoint, configs, args)
            endpoint_module.save_identity(root / "identity.json", identity, resume=args.resume)
            manifest = Manifest(root, resume=args.resume)
            cycles = [Cycle(manifest, number, None, config) for number, config in enumerate(configs, 1)]
            def preflight():
                for cycle in cycles:
                    cycle.validate_state()
                reconcile_children(root)
            preflight()
            with endpoint.guarded_route() as (transport, key):
                environment = {"OPENAI_BASE_URL": transport, "OPENAI_API_KEY": key,
                               "PYTHONPATH": str(REPO), "PYDANTIC_AI_NO_BANNER": "1"}
                with endpoint_module.environment_values(environment):
                    # Model construction validates the public route using the temporary relay environment.
                    for config in configs:
                        validate_provider(config)
                    commands = Commands(endpoint, args.timeout, key, preflight=preflight)
                    for cycle in cycles:
                        cycle.commands = commands
                        cycle.execute()
        else:
            manifest = Manifest(root, resume=True)
            if args.action == "attach":
                require(args.stage and args.evidence, "Attach requires --stage and --evidence.")
                external_evidence(manifest, args.stage, args.evidence)
        manifest.save()
        print(json.dumps({"manifest": str(manifest.path), "live_acceptance": manifest.accepted(),
                          "stages": {name: stage["status"] for name, stage in manifest.data["stages"].items()}}, indent=2))
        return 0 if manifest.accepted() else 1


if __name__ == "__main__":
    def terminate(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, terminate)
    try:
        raise SystemExit(main())
    except (AcceptanceError, OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1)
