"""Synthetic lifecycle readback and mocked commands; no inference, HTTP, or Petri children."""

import copy
import json
import signal
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import httpx
import pytest
import yaml

from petri.storage import database
from petri.storage.event_log import derive_event_id
from scripts.e2e import lifecycle_checks as checks
from scripts.e2e import lifecycle_dashboard as dashboard
from scripts.e2e import lifecycle_runner as runner

DISH = "lifecycle-cycle-1"
ISOLATED = DISH + "-isolated-000-000"
SHARED = DISH + "-shared-000-000"
LEAF = DISH + "-isolated-001-001"


@pytest.fixture
def config():
    return {"debates": [{"pair": ["skeptic", "champion"]}]}


def event(cell, kind, payload, *, number=1, agent="investigator", iteration=1):
    identity = None
    if kind == "search_executed":
        identity = checks.digest(json.dumps([payload["query"], payload["source_urls"]], sort_keys=True).encode())
    elif kind == "source_reviewed":
        identity = payload["url"] + ":" + payload["content_sha256"]
    key = (database.stable_event_id(cell, "public-web", number, "research", agent, iteration, kind, identity)
           if identity else derive_event_id(cell, round_number=number, node="evaluation",
                                           agent=agent, iteration=iteration, event_type=kind))
    return {"id": key, "cell_id": cell, "event_type": kind, "agent": agent,
            "iteration": iteration, "data": payload, "created_at": "2026-11-11T12:00:00Z"}


def add_research(cell, *, number=1, expected="VALIDATED"):
    url = "https://www.sqlite.org/inmemorydb.html"
    direction = "supports" if expected == "VALIDATED" else "contradicts"
    citation = {"url": url, "hierarchy_level": 2, "finding": "Separate connections have private databases.",
                "supports_or_contradicts": direction}
    return [event(cell, "search_executed", {"query": "SQLite memory databases", "sources_found": 1,
                                           "source_urls": [url]}, number=number),
            event(cell, "source_reviewed", {"url": url, "finding": citation["finding"],
                                            "retrieval_method": "public-web", "content_sha256": "a" * 64,
                                            "retrieved_at": "2026-11-11T12:00:00Z"}, number=number),
            event(cell, "verdict_issued", {"verdict": "EVIDENCE_CONFIRMS" if expected == "VALIDATED"
                                          else "EVIDENCE_REFUTES", "sources_cited": [citation]},
                  number=number, agent="evidence_evaluator")]


def steps():
    return [{"function_id": number, "function_name": "petri.cell_phase", "phase": phase,
             "output_sha256": str(number), "error": None}
            for number, phase in enumerate(["sync_check", "socratic", "research", "evaluation"], 1)]


@pytest.fixture
def synthetic(tmp_path):
    path = tmp_path / "petri.sqlite"
    for index, (cell, colony, level, status) in enumerate([
        (SHARED, DISH + "-shared", 0, "DISPROVEN"),
        (ISOLATED, DISH + "-isolated", 0, "VALIDATED"),
        (LEAF, DISH + "-isolated", 1, "VALIDATED"),
    ]):
        database.put_cell(path, {"id": cell, "colony": colony, "level": level, "seq": index,
                                 "claim_text": "A synthetic scoped claim", "status": status,
                                 "created_at": "2026-11-11"})
    for parent in (ISOLATED, SHARED):
        database.add_edge(path, from_cell=parent, to_cell=LEAF,
                          edge_type="cross_colony" if parent == SHARED else "intra_colony",
                          created_at="2026-11-11")
    events = add_research(ISOLATED) + add_research(SHARED, expected="DISPROVEN") + add_research(LEAF)
    with database.connect(path) as connection:
        for row in events:
            connection.execute("INSERT INTO events VALUES (?,?,?,?,?,?,?)", (
                row["id"], row["cell_id"], row["event_type"], row["agent"], row["iteration"],
                json.dumps(row["data"]), row["created_at"]))
        for index, cell in enumerate([ISOLATED, SHARED, LEAF]):
            parent = f"workflow-{index}"
            child = f"usage-{index}"
            attrs = {"petri.cell_id": cell, "petri.round": 1}
            for span_id, ancestor, attributes in [(parent, None, attrs), (child, parent, {})]:
                connection.execute("INSERT INTO spans VALUES (?,?,?,?,?,?,?,?,?,?,?)", (
                    span_id, "trace", ancestor, "span", "INTERNAL", 1, 2, "OK", None,
                    json.dumps(attributes), "{}"))
            connection.execute("INSERT INTO usage VALUES (?,?,?,?,?,?,?,?)", (
                child, "trace", cell, "research", "investigator", 100, 20, None))
    workflows = {cell + ":r1": {"name": "petri.validate_cell", "status": "SUCCESS", "steps": steps()}
                 for cell in (ISOLATED, SHARED, LEAF)}
    return path, checks.domain(path), workflows


def test_exact_root_selection_does_not_use_first_root(synthetic, config):
    _, snapshot, workflows = synthetic
    assert snapshot["cells"][0]["id"] == SHARED
    assert checks.research_report(snapshot, workflows, ISOLATED, 1, "VALIDATED", config)["passed"]
    assert checks.research_report(snapshot, workflows, SHARED, 1, "DISPROVEN", config)["passed"]
    assert not checks.research_report(snapshot, workflows, SHARED, 1, "VALIDATED", config)["passed"]
    assert not checks.research_report(snapshot, workflows, "missing-root", 1, "VALIDATED", config)["passed"]


def test_one_success_does_not_hide_other_colony_failure(synthetic, config, tmp_path):
    _, snapshot, workflows = synthetic
    snapshot["cells"][0]["status"] = "STALLED"
    manifest = runner.Manifest(tmp_path / "run")
    cycle = runner.Cycle(manifest, 1, None, {"name": DISH, **config})
    cycle.snapshot = lambda: {"domain": snapshot, "workflows": workflows}
    with pytest.raises(checks.AcceptanceError, match="At least one"):
        cycle.stage("pre_feed", cycle.research)
    assert manifest.data["stages"]["cycle-1/pre_feed"]["status"] == "failed"
    reports = json.loads(next((manifest.root / "attempts").rglob("research.json")).read_text())
    assert [report["passed"] for report in reports] == [True, False]
    assert not manifest.accepted()


@pytest.mark.parametrize("bad_status", ["STALLED", "NEW", "RESEARCH", "NEEDS_HUMAN", "DEFER_OPEN"])
def test_only_declared_terminal_outcome_counts(synthetic, config, bad_status):
    _, snapshot, workflows = synthetic
    next(row for row in snapshot["cells"] if row["id"] == ISOLATED)["status"] = bad_status
    assert not checks.research_report(snapshot, workflows, ISOLATED, 1, "VALIDATED", config)["passed"]


def test_old_round_sources_and_usage_cannot_satisfy_new_round(synthetic, config):
    _, snapshot, workflows = synthetic
    workflows[ISOLATED + ":r2"] = {"name": "petri.validate_cell", "status": "SUCCESS", "steps": steps()}
    snapshot["events"] += [add_research(ISOLATED, number=2)[-1]]
    report = checks.research_report(snapshot, workflows, ISOLATED, 2, "VALIDATED", config)
    assert not report["passed"] and not report["sources"] and not report["searches"] and not report["usage"]


@pytest.mark.parametrize("field,value", [("retrieved_at", ""), ("content_sha256", ""), ("finding", "")])
def test_incomplete_fetch_provenance_rejected(synthetic, config, field, value):
    _, snapshot, workflows = synthetic
    for row in snapshot["events"]:
        if row["cell_id"] == ISOLATED and row["event_type"] == "source_reviewed":
            row["data"][field] = value
    assert not checks.research_report(snapshot, workflows, ISOLATED, 1, "VALIDATED", config)["passed"]


def test_zero_hits_and_failed_fetch_do_not_count(synthetic, config):
    _, snapshot, workflows = synthetic
    snapshot["events"] = [row for row in snapshot["events"]
                          if row["cell_id"] != ISOLATED or row["event_type"] == "verdict_issued"]
    snapshot["events"] += [event(ISOLATED, "search_executed", {"query": "query", "sources_found": 0,
                                                            "source_urls": []})]
    report = checks.research_report(snapshot, workflows, ISOLATED, 1, "VALIDATED", config)
    assert not report["passed"] and not report["searches"] and not report["sources"]


def test_expected_verdict_needs_fetched_supporting_citation(synthetic, config):
    _, snapshot, workflows = synthetic
    for row in snapshot["events"]:
        if row["cell_id"] == ISOLATED and row["event_type"] == "verdict_issued":
            row["data"]["sources_cited"][0]["supports_or_contradicts"] = "contradicts"
    assert not checks.research_report(snapshot, workflows, ISOLATED, 1, "VALIDATED", config)["passed"]


def test_failure_is_scoped_by_hashed_round_identity(synthetic, config):
    _, snapshot, workflows = synthetic
    row = event(ISOLATED, "agent_execution_failed", {"error": "offline failure"}, number=1)
    row["id"] = derive_event_id(ISOLATED, round_number=1, node="research", agent=row["agent"],
                                iteration=1, event_type="agent_execution_failed")
    snapshot["events"].append(row)
    assert not checks.research_report(snapshot, workflows, ISOLATED, 1, "VALIDATED", config)["passed"]
    assert not checks.event_in_round(row, 2, config)


def test_reference_must_target_nonroot_foundation(synthetic):
    _, snapshot, _ = synthetic
    assert checks.cross_references(snapshot, DISH)
    snapshot["edges"][1]["to_cell"] = ISOLATED
    assert not checks.cross_references(snapshot, DISH)


def test_continuity_requires_exact_committed_step_outputs(synthetic):
    _, _, before = synthetic
    after = copy.deepcopy(before)
    after[ISOLATED + ":r1"]["steps"].append({"function_id": 5})
    assert checks.continuity(before, after) == 12
    after[ISOLATED + ":r1"]["steps"][0]["output_sha256"] = "changed"
    with pytest.raises(checks.AcceptanceError, match="changed or disappeared"):
        checks.continuity(before, after)
    with pytest.raises(checks.AcceptanceError, match="No durable"):
        checks.continuity({}, {})


@pytest.fixture
def fed(synthetic):
    _, before, workflows = synthetic
    after = copy.deepcopy(before)
    later = copy.deepcopy(workflows)
    targets = {ISOLATED: "VALIDATED", LEAF: "VALIDATED"}
    evidence = "Synthetic evidence retained in the database."
    for cell in targets:
        later[cell + ":r2"] = {"name": "petri.validate_cell", "status": "SUCCESS", "steps": steps()}
        after["events"].append({"id": "fed-" + cell, "cell_id": cell, "event_type": "evidence_fed",
                               "data": {"content": evidence, "content_sha256": checks.digest(evidence.encode()),
                                        "original_workflow_id": cell + ":r1", "forked_workflow_id": cell + ":r2",
                                        "generation": 2}})
    for cell in targets:
        data = next(row["data"] for row in after["events"]
                    if row["event_type"] == "evidence_fed" and row["cell_id"] == cell)
        after["events"].append({"id": "reopen-" + cell, "cell_id": cell,
                               "event_type": "cell_reopened", "data": data.copy()})
    for cell in (ISOLATED, SHARED):
        after["events"].append({"id": "flag-" + cell, "cell_id": cell,
                               "event_type": "propagation_triggered", "data": {"reopened_cell_id": LEAF}})
    return before, after, workflows, later, targets, evidence


def test_feed_checks_copy_research_evidence_and_external_dependents(fed):
    report = checks.feed_report(*fed)
    assert report["targets"] == {ISOLATED: "VALIDATED", LEAF: "VALIDATED"}
    assert SHARED in report["flagged_dependents"]


@pytest.mark.parametrize("damage", ["copy", "research", "flag", "enqueue", "evidence", "duplicate", "status"])
def test_feed_rejects_missing_or_duplicate_effects(fed, damage):
    before, after, old, new, targets, evidence = fed
    if damage == "copy":
        new[ISOLATED + ":r2"]["steps"][1]["output_sha256"] = "different"
    elif damage == "research":
        new[ISOLATED + ":r2"]["steps"] = new[ISOLATED + ":r2"]["steps"][:2]
    elif damage == "flag":
        after["events"] = [row for row in after["events"] if row["event_type"] != "propagation_triggered"]
    elif damage == "enqueue":
        new[SHARED + ":r2"] = {"name": "petri.validate_cell"}
    elif damage == "evidence":
        evidence = "Changed source"
    elif damage == "duplicate":
        after["events"].append({**next(row for row in after["events"]
                                     if row["event_type"] == "evidence_fed"), "id": "duplicate"})
    else:
        next(row for row in after["cells"] if row["id"] == SHARED)["status"] = "NEW"
    with pytest.raises(checks.AcceptanceError):
        checks.feed_report(before, after, old, new, targets, evidence)


class FakeProcess:
    def __init__(self, command, **kwargs):
        self.pid = 12345
        self.returncode = 1
        kwargs["stdout"].write(b"Synthetic command failure with secret.\n")

    def poll(self):
        return self.returncode


def test_command_failure_preserves_exit_and_redacted_log(tmp_path, monkeypatch):
    monkeypatch.setattr(runner.subprocess, "Popen", FakeProcess)
    endpoint = SimpleNamespace(redact=lambda text: text.replace("secret", "[REDACTED]"))
    command = runner.Commands(endpoint, 10)
    with pytest.raises(checks.AcceptanceError, match="command failed"):
        command.run(tmp_path, ["grow"], tmp_path, "grow")
    assert runner.load(tmp_path / "grow.json")["exit_code"] == 1
    assert "[REDACTED]" in (tmp_path / "grow.log").read_text()
    assert not (tmp_path / "grow.raw").exists()


def test_recovery_requires_checkpoint_and_task_owned_kill(tmp_path, monkeypatch):
    class Owned(FakeProcess):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.returncode = None
        def kill(self):
            self.returncode = -signal.SIGKILL
        def wait(self, **kwargs):
            return self.returncode
    # This test exercises the command loop; launch acknowledgement has separate tests.
    monkeypatch.setattr(runner.Commands, "launch", lambda self, record, ticket, raw:
                        Owned([], stdout=raw))
    command = runner.Commands(SimpleNamespace(redact=lambda text: text), 10)
    record = command.run(tmp_path, ["grow"], tmp_path, "interrupted", checkpoint=lambda: {"steps": 1})
    assert record["checkpoint"] == {"steps": 1} and record["exit_code"] == -signal.SIGKILL
    monkeypatch.setattr(runner.Commands, "launch", lambda self, record, ticket, raw:
                        FakeProcess([], stdout=raw))
    with pytest.raises(checks.AcceptanceError, match="No task-owned"):
        command.run(tmp_path, ["grow"], tmp_path, "no-checkpoint", checkpoint=lambda: None)


def test_interrupted_stage_retains_diagnostics_and_resume_history(tmp_path):
    manifest = runner.Manifest(tmp_path)
    def interrupted(folder):
        (folder / "partial.log").write_text("Committed work remains available.")
        raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        manifest.run("cycle-1/grow", interrupted)
    resumed = runner.Manifest(tmp_path, resume=True)
    assert resumed.data["stages"]["cycle-1/grow"]["status"] == "interrupted"
    resumed.run("cycle-1/grow", lambda folder: {"command_exit": 0})
    stage = resumed.data["stages"]["cycle-1/grow"]
    assert len(stage["attempts"]) == 2 and stage["attempts"][0]["status"] == "interrupted"
    assert next((tmp_path / "attempts").rglob("partial.log")).is_file()


def test_manifest_aggregate_requires_every_stage_and_artifact(tmp_path):
    manifest = runner.Manifest(tmp_path)
    assert manifest.data["stages"]["cycle-1/browser"]["status"] == "pending"
    for name in manifest.data["stages"]:
        manifest.run(name, lambda folder: {"synthetic": True})
    assert manifest.accepted()
    for state in ("missing", "pending", "skipped", "failed", "interrupted", "running"):
        manifest.data["stages"]["cycle-3/post_feed"]["status"] = state
        assert not manifest.accepted()
    manifest.data["stages"]["cycle-3/post_feed"]["status"] = "passed"
    stage = manifest.data["stages"]["cycle-2/browser"]
    (tmp_path / stage["attempts"][-1]["artifacts"][0]["path"]).unlink()
    assert not manifest.accepted()


def test_missing_external_readback_is_rejected(tmp_path):
    manifest = runner.Manifest(tmp_path)
    runner.write_json(tmp_path / "identity.json", {"source_sha": "synthetic"})
    evidence = tmp_path / "evidence.json"
    runner.write_json(evidence, {"observer": "Root", "observed_at": "2026-11-11",
                                "source_sha": "synthetic", "checks": {"seed_exact_counts": True}})
    with pytest.raises(checks.AcceptanceError, match="incomplete"):
        runner.external_evidence(manifest, "deterministic_recovery", evidence)
    assert not manifest.accepted()


def test_domain_readback_does_not_create_missing_database(tmp_path):
    path = tmp_path / "missing.sqlite"
    with pytest.raises(checks.AcceptanceError, match="missing"):
        checks.domain(path)
    assert not path.exists()


def test_configuration_keeps_context_distinct_from_output_cap():
    endpoint = SimpleNamespace(model="synthetic-loopback", max_tokens=4096, context_size=262144)
    args = SimpleNamespace(request_limit=120, token_limit=600000, tool_limit=100, output_mode="tool")
    config = runner.configuration(endpoint, args, DISH)
    assert config["model_settings"]["max_tokens"] == 4096
    assert config["execution"]["worker_concurrency"] == 1
    assert len(config["agents"]) == 13
    assert config["source_hierarchy"]["minimum_terminal_level"] == 4


def test_diagnostic_capture_config_is_explicit_and_redacts_relay_key():
    endpoint = SimpleNamespace(model="synthetic", max_tokens=4096, context_size=262144)
    args = SimpleNamespace(request_limit=120, token_limit=600000, tool_limit=100, output_mode="tool",
                           capture_content=True, otlp_traces_endpoint="http://127.0.0.1:14318/v1/traces")
    config = runner.configuration(endpoint, args, DISH)
    assert config["telemetry"]["capture_content"] is True
    assert "OPENAI_API_KEY" in config["telemetry"]["redact_env_vars"]
    assert config["telemetry"]["otlp"]["traces_endpoint"] == args.otlp_traces_endpoint
    assert config["model_settings"]["max_tokens"] == 4096
    assert config["execution"]["worker_concurrency"] == 1
    args.capture_content = False
    args.otlp_traces_endpoint = None
    assert "capture_content" not in runner.configuration(endpoint, args, DISH).get("telemetry", {})


@pytest.mark.parametrize("url", ["https://example.com/v1/traces", "http://127.0.0.1@host/v1/traces",
                                 "http://user:password@127.0.0.1/v1/traces", "http://127.0.0.1:0/v1/traces",
                                 "http://127.0.0.1/v1/traces?key=secret", "file:///tmp/traces"])
def test_otlp_route_must_be_loopback_without_credentials(url):
    with pytest.raises(checks.AcceptanceError, match="loopback OTLP"):
        runner.otlp_loopback_url(url)


@pytest.mark.parametrize("url", ["http://localhost:4318/v1/traces", "http://127.0.0.1:14318/v1/traces",
                                 "https://[::1]:4318/custom/traces"])
def test_loopback_otlp_route_preserves_path(url):
    assert runner.otlp_loopback_url(url) == url


def test_mixed_feed_outcomes_fail_before_command(synthetic, tmp_path):
    _, snapshot, workflows = synthetic
    snapshot["cells"] += [{"id": "disproven-child", "colony": DISH + "-isolated", "status": "DISPROVEN"},
                          {"id": "deferred-child", "colony": DISH + "-isolated", "status": "DEFER_OPEN"}]
    assert checks.feed_eligible(snapshot, DISH + "-isolated") == {
        ISOLATED: "VALIDATED", LEAF: "VALIDATED", "disproven-child": "DISPROVEN",
        "deferred-child": "DEFER_OPEN"}
    manifest = runner.Manifest(tmp_path)
    calls = []
    cycle = runner.Cycle(manifest, 1, SimpleNamespace(run=lambda *a, **k: calls.append(a)), {"name": DISH})
    cycle.root.mkdir()
    cycle.snapshot = lambda: {"domain": snapshot, "workflows": workflows}
    with pytest.raises(checks.AcceptanceError, match="unsupported foundation outcomes"):
        cycle.stage("feed", cycle.feed)
    assert not calls and not (cycle.root / "feed-before.json").exists()


def test_partial_feed_retains_original_new_and_completed_targets(synthetic, tmp_path):
    _, snapshot, _ = synthetic
    targets = checks.check_feed_plan(snapshot, DISH + "-isolated")
    snapshot["cells"][1]["status"] = "NEW"
    snapshot["events"].append({"cell_id": ISOLATED, "event_type": "evidence_fed"})
    assert checks.check_feed_plan(snapshot, DISH + "-isolated", targets) == targets
    snapshot["cells"][2]["status"] = "DISPROVEN"
    with pytest.raises(checks.AcceptanceError, match="unsupported outcome"):
        checks.check_feed_plan(snapshot, DISH + "-isolated", targets)
    snapshot["cells"][2]["status"] = "NEW"
    with pytest.raises(checks.AcceptanceError, match="original target set"):
        checks.check_feed_plan(snapshot, DISH + "-isolated", targets)


def test_post_feed_research_covers_every_retained_target(synthetic, tmp_path, monkeypatch):
    _, snapshot, workflows = synthetic
    manifest = runner.Manifest(tmp_path)
    cycle = runner.Cycle(manifest, 1, None, {"name": DISH})
    targets = {ISOLATED: "VALIDATED", LEAF: "VALIDATED"}
    runner.write_json(cycle.root / "feed-before.json", {"targets": targets})
    cycle.snapshot = lambda: {"domain": snapshot, "workflows": workflows}
    observed = []
    def research(snapshot, workflows, cell, number, expected, config):
        observed.append((cell, number, expected))
        return {"passed": True}
    monkeypatch.setattr(runner, "research_report", research)
    cycle.stage("post_feed", lambda folder: cycle.research(folder, post=True))
    assert set(observed) == {(ISOLATED, 2, "VALIDATED"), (SHARED, 1, "DISPROVEN"), (LEAF, 2, "VALIDATED")}


def test_partial_feed_retry_uses_retained_before_snapshot(fed, tmp_path, monkeypatch):
    before, after, old, new, targets, evidence = fed
    manifest = runner.Manifest(tmp_path)
    calls = []
    cycle = runner.Cycle(manifest, 1, SimpleNamespace(run=lambda *a, **k: calls.append(a)), {"name": DISH})
    context = cycle.root / "feed-before.json"
    source = cycle.root / "source-evidence.txt"
    original = {"domain": before, "workflows": old, "targets": targets, "evidence": evidence}
    runner.write_json(context, original)
    source.write_text(evidence)
    retained = tmp_path / "retained"
    retained.mkdir()
    manifest.retain(context, retained)
    manifest.retain(source, retained)
    partial = copy.deepcopy(before)
    partial["cells"][1]["status"] = "NEW"
    partial["events"] += [row for row in after["events"] if row["event_type"] == "evidence_fed" and row["cell_id"] == ISOLATED]
    snapshots = [{"domain": partial, "workflows": old}, {"domain": after, "workflows": new}]
    cycle.snapshot = lambda: snapshots.pop(0)
    def report(saved, final, previous_workflows, final_workflows, plan, text):
        assert saved == before and previous_workflows == old
        assert final == after and final_workflows == new and plan == targets and text == evidence
        return {"forks": []}
    monkeypatch.setattr(runner, "feed_report", report)
    cycle.stage("feed", cycle.feed)
    assert len(calls) == 1 and calls[0][1][0] == "feed"
    assert runner.load(context) == original


@pytest.mark.parametrize("name", ["seed-observed.json", "grow-checkpoint.json", "feed-before.json", "source-evidence.txt"])
def test_retained_sidecar_mutation_rejected(tmp_path, name):
    manifest = runner.Manifest(tmp_path)
    path = tmp_path / "cycle-1" / name
    runner.write_json(path, {"original": True})
    folder = tmp_path / "retained"
    folder.mkdir()
    manifest.retain(path, folder)
    path.write_text("changed")
    with pytest.raises(checks.AcceptanceError, match="Live sidecar differs"):
        runner.Manifest(tmp_path, resume=True).validate_retained()


def test_retained_evidence_and_unretained_sidecars_fail_closed(tmp_path):
    manifest = runner.Manifest(tmp_path)
    cycle = runner.Cycle(manifest, 1, None, {"name": DISH})
    cycle.root.mkdir()
    runner.write_json(cycle.root / "feed-before.json", {})
    with pytest.raises(checks.AcceptanceError, match="missing its live configuration"):
        cycle.validate_state()
    folder = tmp_path / "retained"
    folder.mkdir()
    manifest.retain(cycle.root / "feed-before.json", folder)
    (folder / "retained-feed-before.json").write_text("changed")
    with pytest.raises(checks.AcceptanceError, match="Retained sidecar evidence changed"):
        manifest.validate_retained()


def birth(pid=12345, start="10", *, cwd="/synthetic/cycle", state="S"):
    return {"pid": pid, "start_ticks": start, "boot_id": "synthetic-boot", "state": state,
            "cwd": cwd, "argv": ["python", "-m", "petri", "grow"]}


@pytest.mark.parametrize("boundary", ["before_ack", "after_ack", "after_authorization"])
def test_reconcile_surviving_child_blocks_before_new_command(tmp_path, monkeypatch, boundary):
    ticket = tmp_path / "attempts/cycle-1/grow/1/grow.process.json"
    record = {"authorized": boundary == "after_authorization"}
    child = birth()
    if boundary == "after_authorization":
        record["child"] = child
    runner.write_json(ticket, record)
    if boundary != "before_ack":
        runner.write_json(ticket.with_suffix(".child.json"), child)
    monkeypatch.setattr(runner, "process_identity", lambda pid: child)
    if boundary == "before_ack":
        # The unacknowledged wrapper cannot enter Petri, even if its parent dies here.
        runner.write_json(ticket, {"parent": birth(1), "launch_id": "launch", "command": [], "authorized": False})
        monkeypatch.setattr(runner.os, "getpid", lambda: child["pid"])
        monkeypatch.setattr(runner, "process_identity", lambda pid: child if pid == child["pid"] else None)
        assert runner.child_worker(ticket) == 125
    else:
        with pytest.raises(checks.AcceptanceError, match="remains active"):
            runner.reconcile_children(tmp_path)


def test_reconcile_pid_reuse_never_signals_and_checks_execution_lock(tmp_path, monkeypatch):
    ticket = tmp_path / "attempts/cycle-1/grow/1/grow.process.json"
    runner.write_json(ticket, {"authorized": True, "child": birth()})
    monkeypatch.setattr(runner, "process_identity", lambda pid: birth(start="11"))
    monkeypatch.setattr(runner.os, "kill", lambda *a: pytest.fail("A reused PID must never be signalled."))
    runner.reconcile_children(tmp_path)
    assert runner.load(tmp_path / "child-reconciliation.json")["children"][0]["state"] == "pid_reused"
    petri = tmp_path / "cycle-1/.petri"
    petri.mkdir(parents=True)
    with runner.FileLock(str(petri / ".execution-owner.lock")):
        with pytest.raises(checks.AcceptanceError, match="execution owner"):
            runner.reconcile_children(tmp_path)


def test_launch_authorizes_acknowledged_birth_and_cleans_up_on_parent_termination(tmp_path, monkeypatch):
    child = birth(cwd=str(tmp_path))
    class Acknowledged(FakeProcess):
        def __init__(self, command, **kwargs):
            super().__init__(command, **kwargs)
            self.returncode = None
            runner.write_json(tmp_path / "grow.process.child.json", child)
        def kill(self):
            self.returncode = -signal.SIGKILL
        def wait(self, **kwargs):
            return self.returncode
    monkeypatch.setattr(runner.subprocess, "Popen", Acknowledged)
    monkeypatch.setattr(runner, "process_identity", lambda pid: child if pid == child["pid"] else birth(pid))
    def interrupted_checkpoint():
        record = runner.load(tmp_path / "grow.process.json")
        assert record["authorized"] and record["child"] == child
        raise SystemExit(143)
    with pytest.raises(SystemExit):
        runner.Commands(SimpleNamespace(redact=lambda text: text), 10).run(
            tmp_path, ["grow"], tmp_path, "grow", checkpoint=interrupted_checkpoint)
    record = runner.load(tmp_path / "grow.process.json")
    assert record["cleanup_termination"] and record["exit_code"] == -signal.SIGKILL


def test_parent_termination_before_durable_authorization_aborts_wrapper(tmp_path, monkeypatch):
    child = birth(cwd=str(tmp_path))
    processes = []
    class Wrapper(FakeProcess):
        def __init__(self, command, **kwargs):
            super().__init__(command, **kwargs)
            self.returncode = None
            processes.append(self)
            runner.write_json(tmp_path / "grow.process.child.json", child)
        def kill(self):
            self.returncode = -signal.SIGKILL
        def wait(self, **kwargs):
            return self.returncode
    monkeypatch.setattr(runner.subprocess, "Popen", Wrapper)
    monkeypatch.setattr(runner, "process_identity", lambda pid: child if pid == child["pid"] else birth(pid))
    write = runner.write_json
    interrupted = []
    def terminate_before_save(path, value):
        if path.name == "grow.process.json" and value.get("authorized") and not interrupted:
            interrupted.append(True)
            raise SystemExit(143)
        write(path, value)
    monkeypatch.setattr(runner, "write_json", terminate_before_save)
    with pytest.raises(SystemExit):
        runner.Commands(SimpleNamespace(redact=lambda text: text), 10).run(tmp_path, ["grow"], tmp_path, "grow")
    assert processes[0].returncode == -signal.SIGKILL
    assert runner.load(tmp_path / "grow.process.json")["launch_aborted"]


@pytest.mark.parametrize("status", ["SUCCESS", "PENDING"])
def test_recovery_only_observes_active_workflow(synthetic, tmp_path, status):
    _, snapshot, workflows = synthetic
    manifest = runner.Manifest(tmp_path)
    workflows[ISOLATED + ":r1"]["status"] = status
    cycle = runner.Cycle(manifest, 1, None, {"name": DISH})
    cycle.root.mkdir()
    cycle.snapshot = lambda: {"domain": snapshot, "workflows": workflows}
    def run(cwd, args, folder, label, checkpoint):
        observation = checkpoint()
        if status == "SUCCESS":
            assert observation is None
            raise checks.AcceptanceError("No active target workflow.")
        assert runner.load(manifest.root / observation["path"])["crash_observation"]["status"] == "PENDING"
        return {"checkpoint": observation, "termination_requested": True, "exit_code": -signal.SIGKILL}
    cycle.commands = SimpleNamespace(run=run)
    if status == "SUCCESS":
        with pytest.raises(checks.AcceptanceError, match="No active"):
            cycle.stage("grow_recovery", lambda folder: cycle.recover(folder, "grow"))
        assert not (cycle.root / "grow-observed.json").exists()
    else:
        result = cycle.stage("grow_recovery", lambda folder: cycle.recover(folder, "grow"))
        assert result["crash_observation"]["workflow_id"] == ISOLATED + ":r1"


@pytest.mark.parametrize("boundary", ["observed", "kill_requested", "killed", "completed"])
def test_recovery_retry_reuses_confirmed_kill_or_fails_closed(tmp_path, boundary):
    manifest = runner.Manifest(tmp_path)
    cycle = runner.Cycle(manifest, 1, SimpleNamespace(run=lambda *a, **k: pytest.fail("No second experiment.")), {"name": DISH})
    folder = tmp_path / "attempts/cycle-1/grow_recovery/1"
    folder.mkdir(parents=True)
    observed = cycle.root / "grow-observed.json"
    runner.write_json(observed, {"crash_observation": {"workflow_id": ISOLATED + ":r1", "status": "PENDING", "checkpoint": steps()[0]}})
    manifest.retain(observed, folder)
    record = {"checkpoint": runner.artifact(tmp_path, observed)}
    if boundary != "observed":
        record["termination_requested"] = True
    if boundary in {"killed", "completed"}:
        record["exit_code"] = -signal.SIGKILL if boundary == "killed" else 0
    runner.write_json(folder / "grow-interrupt.process.json", record)
    manifest.data["stages"]["cycle-1/grow_recovery"].update(
        status="interrupted", attempts=[{"status": "interrupted", "artifacts": []}])
    manifest.save()
    if boundary == "killed":
        result = cycle.stage("grow_recovery", lambda folder: cycle.recover(folder, "grow"))
        assert result["termination"] == record
    else:
        with pytest.raises(checks.AcceptanceError, match="no confirmed crash outcome"):
            cycle.stage("grow_recovery", lambda folder: cycle.recover(folder, "grow"))


@pytest.fixture
def response_readback(synthetic, tmp_path):
    _, snapshot, workflows = synthetic
    expected = {ISOLATED + ":r1", SHARED + ":r1", ISOLATED + ":r2", LEAF + ":r2"}
    evidence = {"workflow_ids": sorted(expected), "span_ids": [], "response_ids": [], "response_readbacks": [], "artifacts": []}
    for index, workflow in enumerate(sorted(expected)):
        cell, number = workflow.rsplit(":r", 1)
        parent, model, response_id = f"round-{index}", f"model-{index}", f"response:{index}/verbatim"
        for span, ancestor, attrs in [(parent, None, {"petri.cell_id": cell, "petri.round": int(number)}),
                                     (model, parent, {"gen_ai.operation.name": "chat", "gen_ai.response.id": response_id})]:
            snapshot["spans"].append({"span_id": span, "parent_span_id": ancestor, "trace_id": f"trace-{index}", "attributes": json.dumps(attrs)})
            evidence["span_ids"].append(span)
        mapping = {"response_id": response_id, "span_id": model, "trace_id": f"trace-{index}", "workflow_id": workflow,
                   "artifact_path": f"response-{index}.json"}
        path = tmp_path / mapping["artifact_path"]
        runner.write_json(path, {**mapping, "remote_record": {"content": "Synthetic stored response."}})
        evidence["artifacts"].append(runner.artifact(tmp_path, path))
        evidence["response_ids"].append(response_id)
        evidence["response_readbacks"].append(mapping)
        workflows[workflow] = {"name": "petri.validate_cell", "status": "SUCCESS", "steps": steps()}
    return snapshot, workflows, expected, evidence, tmp_path


@pytest.mark.parametrize("damage", [None, "unknown", "other_round", "missing_round", "missing_response", "artifact", "trace"])
def test_response_readback_joins_exact_stored_coordinates(response_readback, damage):
    snapshot, _, expected, evidence, root = response_readback
    if damage == "unknown":
        evidence["response_readbacks"][0]["response_id"] = "not-a-stored-response"
    elif damage == "other_round":
        evidence["response_readbacks"][0]["workflow_id"] = LEAF + ":r1"
    elif damage == "missing_round":
        evidence["response_readbacks"].pop()
    elif damage == "missing_response":
        snapshot["spans"].append({"span_id": "extra-model", "parent_span_id": "round-0", "trace_id": "trace-0",
                                  "attributes": json.dumps({"gen_ai.operation.name": "chat", "gen_ai.response.id": "extra"})})
    elif damage == "artifact":
        runner.write_json(root / evidence["response_readbacks"][0]["artifact_path"], {"remote_record": "No IDs."})
    elif damage == "trace":
        snapshot["spans"][-1]["trace_id"] = "another-trace"
    if damage:
        with pytest.raises(checks.AcceptanceError):
            checks.check_response_readback(snapshot, expected, evidence, root)
    else:
        checks.check_response_readback(snapshot, expected, evidence, root)


def test_external_trace_requires_exact_feed_target_rounds(response_readback, monkeypatch):
    snapshot, workflows, _, evidence, root = response_readback
    manifest = runner.Manifest(root)
    evidence.update(observer="Root", observed_at="2026-11-11T12:00:00Z", dish_id=DISH,
                    root_statuses={ISOLATED: "VALIDATED", SHARED: "DISPROVEN"},
                    checks=dict.fromkeys(runner.TRACE_CHECKS, True))
    context = root / "cycle-1/feed-before.json"
    runner.write_json(context, {"targets": {ISOLATED: "VALIDATED", LEAF: "VALIDATED"}})
    retained = root / "retained"
    retained.mkdir()
    manifest.retain(context, retained)
    monkeypatch.setattr(runner, "domain", lambda path: snapshot)
    monkeypatch.setattr(runner, "workflow_snapshot", lambda path: workflows)
    path = root / "trace-evidence.json"
    incomplete = {**evidence, "workflow_ids": [key for key in evidence["workflow_ids"] if key != LEAF + ":r2"]}
    runner.write_json(path, incomplete)
    with pytest.raises(checks.AcceptanceError, match="omits a required cell-round"):
        runner.external_evidence(manifest, "cycle-1/trace_readback", path)
    runner.write_json(path, evidence)
    runner.external_evidence(manifest, "cycle-1/trace_readback", path)
    assert manifest.data["stages"]["cycle-1/trace_readback"]["status"] == "passed"


@pytest.fixture
def endpoint_seam(tmp_path, monkeypatch):
    endpoint = SimpleNamespace(model="synthetic-model", max_tokens=4096, identity=lambda: {"model": "synthetic-model"},
                               guarded_route=lambda: nullcontext(("http://127.0.0.1:1", "synthetic-key")))
    def save_identity(path, identity, resume=False):
        if resume:
            checks.require(runner.load(path) == identity, "Run identity differs; preserve the earlier run.")
        else:
            runner.write_json(path, identity)
    module = SimpleNamespace(connection_arguments=lambda parser: None, endpoint_from_args=lambda args: endpoint,
                             source_sha=lambda: "synthetic-sha", configuration_hash=lambda value: checks.digest(checks.canonical(value).encode()),
                             save_identity=save_identity, environment_values=lambda mapping: nullcontext(),
                             provider_validator=runner.validate_provider)
    monkeypatch.setitem(sys.modules, "scripts.e2e.local_endpoint", module)
    import scripts.e2e
    monkeypatch.setattr(scripts.e2e, "local_endpoint", module, raising=False)
    monkeypatch.setenv("PETRI_RUN_LIFECYCLE", "1")
    monkeypatch.setattr(runner, "subprocess", SimpleNamespace(
        Popen=lambda *a, **k: pytest.fail("No inference subprocess is authorized by this test.")))
    monkeypatch.setattr(runner.Cycle, "execute", lambda self: self.stage("prepare", self.prepare))
    monkeypatch.setattr(runner, "validate_provider", lambda config: None)
    return tmp_path / "run", module, endpoint


@pytest.mark.parametrize("choice,value", [("default", None), ("single", False), ("multiple", True)])
def test_profile_mapping_and_identity(endpoint_seam, monkeypatch, choice, value):
    if value is not None:
        # The separate resolver/schema patch is deliberately absent in B's worktree.
        import petri.config
        monkeypatch.setattr(petri.config, "load_dish_config", lambda path: None)
    root, module, _ = endpoint_seam
    assert runner.main(["run", str(root), "--openai-chat-system-messages", choice]) == 1
    identity = runner.load(root / "identity.json")
    assert identity["provider_selection"] == {"output_mode": "tool", "openai_chat_supports_multiple_system_messages": value}
    for config in identity["configurations"]:
        if value is None:
            assert "openai_chat_supports_multiple_system_messages" not in config["harness"]
        else:
            assert config["harness"]["openai_chat_supports_multiple_system_messages"] is value
        assert config["harness"]["output_mode"] == "tool"
    assert identity["configuration_hash"] == module.configuration_hash(identity["configurations"])
    assert runner.main(["run", str(root), "--resume", "--openai-chat-system-messages", choice]) == 1


@pytest.mark.parametrize("change", ["output", "profile", "model_yaml", "concurrency_yaml", "diagnostic_yaml", "profile_yaml"])
def test_resume_rejects_changed_identity_or_effective_yaml_before_child(endpoint_seam, change):
    root, _, _ = endpoint_seam
    args = ["run", str(root)]
    assert runner.main(args) == 1
    config_path = root / "cycle-1/.petri/defaults/petri.yaml"
    if change == "output":
        args += ["--output-mode", "prompted"]
    elif change == "profile":
        args += ["--openai-chat-system-messages", "single"]
    else:
        config = yaml.safe_load(config_path.read_text())
        if change == "model_yaml":
            config["model"] = "openai-chat:another-model"
        elif change == "concurrency_yaml":
            config["execution"]["worker_concurrency"] = 2
        elif change == "diagnostic_yaml":
            config.setdefault("telemetry", {})["capture_content"] = True
        else:
            config["harness"]["openai_chat_supports_multiple_system_messages"] = False
        config_path.write_text(yaml.safe_dump(config))
    with pytest.raises(checks.AcceptanceError, match="identity"):
        runner.main([*args, "--resume"])


@pytest.mark.parametrize("changed_arguments", [["--output-mode", "prompted"],
                                               ["--openai-chat-system-messages", "single"]])
def test_endpoint_seam_replaces_preimported_real_endpoint(request, monkeypatch, changed_arguments):
    real_endpoint = pytest.importorskip("scripts.e2e.local_endpoint")
    import scripts.e2e
    assert scripts.e2e.local_endpoint is real_endpoint
    root, module, _ = request.getfixturevalue("endpoint_seam")
    from scripts.e2e import local_endpoint
    assert local_endpoint is module and local_endpoint is not real_endpoint
    assert sys.modules["scripts.e2e.local_endpoint"] is module
    saved = []
    save_identity = module.save_identity
    def observed_save(path, identity, resume=False):
        saved.append(resume)
        return save_identity(path, identity, resume=resume)
    monkeypatch.setattr(module, "save_identity", observed_save)
    assert runner.main(["run", str(root)]) == 1
    with pytest.raises(checks.AcceptanceError, match="Run identity differs"):
        runner.main(["run", str(root), "--resume", *changed_arguments])
    assert saved == [False, True]


def test_output_and_profile_make_distinct_config_hashes(endpoint_seam):
    _, module, endpoint = endpoint_seam
    identities = []
    for mode, choice in [("tool", "default"), ("prompted", "default"), ("prompted", "single"), ("prompted", "multiple")]:
        args = SimpleNamespace(output_mode=mode, openai_chat_system_messages=choice, request_limit=120,
                               token_limit=600000, tool_limit=100, timeout=14400, capture_content=False,
                               otlp_traces_endpoint=None)
        config = runner.configuration(endpoint, args, DISH)
        identities.append(runner.run_identity(module, endpoint, [config], args))
    assert len({item["configuration_hash"] for item in identities}) == 4
    assert len({checks.canonical(item["provider_selection"]) for item in identities}) == 4


def test_explicit_profile_requires_resolver_contract_and_uses_public_resolver(monkeypatch):
    import petri.harness.resolve as resolver
    from petri.agents.config import HarnessContent
    config = {"model": "openai-chat:synthetic", "harness": {"backend": "provider", "model": "openai-chat:synthetic"}}
    calls = []
    monkeypatch.setattr(resolver, "resolve_model", lambda config: calls.append(copy.deepcopy(config)))
    runner.validate_provider(config)
    assert calls == [config]
    config["harness"]["openai_chat_supports_multiple_system_messages"] = False
    monkeypatch.setattr(HarnessContent, "model_fields", {})
    with pytest.raises(checks.AcceptanceError, match="Integrate the reviewed"):
        runner.validate_provider(config)
    monkeypatch.setattr(HarnessContent, "model_fields", {"openai_chat_supports_multiple_system_messages": object()})
    def reject(config):
        raise ValueError("Synthetic resolver rejection.")
    monkeypatch.setattr(resolver, "resolve_model", reject)
    with pytest.raises(ValueError, match="resolver rejection"):
        runner.validate_provider(config)
    config["harness"]["model"] = "openai:synthetic"
    with pytest.raises(checks.AcceptanceError, match="provider/openai-chat"):
        runner.validate_provider(config)


def test_public_resolver_rejection_prevents_inference_subprocess(endpoint_seam, monkeypatch):
    import petri.harness.resolve as resolver
    root, module, _ = endpoint_seam
    monkeypatch.setattr(runner, "validate_provider", module.provider_validator)
    def reject(config):
        raise ValueError("Synthetic unsupported resolver configuration.")
    monkeypatch.setattr(resolver, "resolve_model", reject)
    with pytest.raises(ValueError, match="unsupported resolver configuration"):
        runner.main(["run", str(root)])
    assert not (root / "cycle-1").exists()


class Clock:
    def __init__(self):
        self.value = 0
    def monotonic(self):
        return self.value
    def sleep(self, seconds):
        self.value += seconds


@pytest.fixture
def dashboard_seam(synthetic, tmp_path, monkeypatch):
    _, snapshot, _ = synthetic
    clock = Clock()
    monkeypatch.setattr(dashboard, "time", clock)
    identity = birth(cwd=str(tmp_path / "cycle-1"))
    monkeypatch.setattr(dashboard, "process_identity", lambda pid: identity)
    snapshots = []
    def read_domain(path):
        return copy.deepcopy(snapshots.pop(0) if len(snapshots) > 1 else snapshots[0])
    snapshots.append(snapshot)
    monkeypatch.setattr(dashboard, "domain", read_domain)
    requests = []
    def respond(request):
        route = request.url.path
        requests.append(str(request.url))
        if route == "/api/dishes":
            payload = [{"id": DISH}]
        elif route == "/api/cells":
            payload = [{"cell_id": cell, "dish_id": DISH} for cell in (ISOLATED, SHARED)]
        elif route == "/api/edges":
            payload = [{"edge_type": "cross_colony"}]
        elif route == "/api/queue":
            payload = {"owner": "cli"}
        elif route.startswith("/api/cell/"):
            payload = {"dish_id": DISH}
        elif route == "/api/costs":
            payload = {"cells": [{"input_tokens": 100}]}
        elif route == "/api/traces":
            payload = [{"trace_id": "synthetic-trace"}]
        elif route.startswith("/api/traces/"):
            payload = [{"span_id": "synthetic-span"}]
        elif route == "/api/stream":
            row = next(row for row in snapshot["events"] if row["event_type"] == "source_reviewed")
            return httpx.Response(200, headers={"content-type": "text/event-stream"},
                                  text="data: " + json.dumps({"id": row["id"], "cell_id": row["cell_id"]}) + "\n\n")
        else:
            pytest.fail("Unexpected dashboard route: " + route)
        return httpx.Response(200, json=payload)
    client = httpx.Client
    transport = httpx.MockTransport(respond)
    monkeypatch.setattr(dashboard.httpx, "Client", lambda **kwargs: client(transport=transport, **kwargs))
    monkeypatch.setattr(dashboard, "check_owner", lambda petri, saved: identity)
    return tmp_path, snapshot, snapshots, identity, requests, clock, client


def test_dashboard_retries_ownership_gaps_then_confirms_terminal_roots(dashboard_seam, monkeypatch):
    root, snapshot, snapshots, identity, requests, clock, _ = dashboard_seam
    reopened = copy.deepcopy(snapshot)
    reopened["cells"][1]["status"] = "RESEARCH"
    snapshots[:] = [reopened, reopened, snapshot]
    calls = []
    def owner(petri, saved):
        calls.append(saved)
        if len(calls) == 1:
            raise checks.AcceptanceError("Between per-cell feed owners.")
        return identity
    monkeypatch.setattr(dashboard, "check_owner", owner)
    path = dashboard.capture_dashboard(root, 1, "http://127.0.0.1:1234", identity["pid"], capture_timeout=5,
                                       retry_interval=1, final_status_timeout=5)
    evidence = runner.load(path)
    assert evidence["root_statuses"] == {ISOLATED: "VALIDATED", SHARED: "DISPROVEN"}
    assert runner.load(path.parent / "ownership.json")["root_statuses_during_ownership"][ISOLATED] == "RESEARCH"
    assert [row["status"] for row in runner.load(path.parent / "attempts.json")] == ["failed", "passed"]
    assert len(calls) == 3 and clock.value == 2
    assert any(url.endswith("/api/stream") for url in requests)
    assert all(runner.artifact(root, root / record["path"]) == record for record in evidence["artifacts"])


@pytest.mark.parametrize("failure", ["owner_after", "wrong_dish", "sse", "final_roots"])
def test_dashboard_failure_never_writes_pass_evidence(dashboard_seam, monkeypatch, failure):
    root, snapshot, snapshots, identity, _, _, client = dashboard_seam
    if failure == "owner_after":
        calls = []
        def owner(petri, saved):
            calls.append(saved)
            if len(calls) % 2 == 0:
                raise checks.AcceptanceError("Ownership ended during reads.")
            return identity
        monkeypatch.setattr(dashboard, "check_owner", owner)
    elif failure in {"wrong_dish", "sse"}:
        # Replace only the test transport; no sockets are opened.
        transport = httpx.MockTransport(lambda request: httpx.Response(200, json=[{"id": "wrong-dish"}]))
        if failure == "sse":
            original = dashboard.capture_payloads
            def empty_stream(*args):
                raise httpx.ReadTimeout("Synthetic SSE timeout.")
            monkeypatch.setattr(dashboard, "capture_payloads", empty_stream)
            assert original
        else:
            monkeypatch.setattr(dashboard.httpx, "Client", lambda **kwargs: client(transport=transport, **kwargs))
    else:
        snapshot["cells"][1]["status"] = "NEW"
        snapshots[:] = [snapshot]
    with pytest.raises(checks.AcceptanceError, match="window|roots"):
        dashboard.capture_dashboard(root, 1, "http://localhost:1234", identity["pid"], capture_timeout=2,
                                    retry_interval=1, final_status_timeout=2)
    assert not list((root / "captures").rglob("evidence.json"))
    if failure == "final_roots":
        assert list((root / "captures").rglob("ownership.json"))


def test_dashboard_owner_check_rejects_pid_reuse_before_lock(tmp_path, monkeypatch):
    saved = birth(cwd=str(tmp_path))
    monkeypatch.setattr(dashboard, "process_identity", lambda pid: birth(start="11", cwd=str(tmp_path)))
    with pytest.raises(checks.AcceptanceError, match="PID was reused"):
        dashboard.check_owner(tmp_path / ".petri", saved)
