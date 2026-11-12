"""Read-only acceptance checks for opt-in lifecycle artifacts."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from datetime import datetime

from petri.query.connection import has_table, read_connection
from petri.storage.database import stable_event_id
from petri.storage.event_log import derive_event_id


class AcceptanceError(ValueError):
    """A required observation is missing or differs from the declared outcome."""


def require(condition, message):
    if not condition:
        raise AcceptanceError(message)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def canonical(value):
    if dataclasses.is_dataclass(value):
        value = dataclasses.asdict(value)
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))


def domain(path):
    """Read one committed domain snapshot, including failed attempts and telemetry."""
    with read_connection(path) as connection:
        require(connection is not None, "The domain database is missing.")
        require(connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok",
                "SQLite integrity_check failed.")
        result = {}
        for table in ("cells", "colonies", "edges", "events", "spans", "usage",
                      "execution_round_requests"):
            result[table] = ([dict(row) for row in connection.execute(f'SELECT * FROM "{table}"')]
                             if has_table(connection, table) else [])
            if table == "events":
                for row in result[table]:
                    row["data"] = json.loads(row.pop("payload"))
        return result


def logical_rows(path):
    """Compare every domain table, including evidence, without relying on file bytes."""
    with read_connection(path) as connection:
        require(connection is not None, "The comparison database is missing.")
        require(connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok",
                "SQLite integrity_check failed.")
        tables = [row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )]
        return {table: sorted(canonical(list(row)) for row in connection.execute(
            'SELECT * FROM "' + table.replace('"', '""') + '"')) for table in tables}


def workflow_snapshot(petri_dir):
    """Read checkpoints through the pinned DBOS client; never launch a worker."""
    path = petri_dir / "dbos.sqlite"
    if not path.is_file():
        return {}
    # Avoid creating DBOS client tables while a seed is still initializing SQLite.
    with read_connection(path) as connection:
        if not has_table(connection, "operation_outputs"):
            return {}
    from dbos import DBOSClient

    client = DBOSClient(system_database_url=f"sqlite:///{path}")
    try:
        result = {}
        for run in client.list_workflows(load_input=False, load_output=False):
            result[run.workflow_id] = {
                "name": run.name, "status": run.status,
                "steps": [{
                    "function_id": step["function_id"],
                    "function_name": step["function_name"],
                    "phase": getattr(step.get("output"), "phase", None),
                    "output_sha256": digest(canonical(step.get("output")).encode()),
                    "error": str(step["error"]) if step.get("error") else None,
                } for step in client.list_workflow_steps(run.workflow_id)],
            }
        return result
    finally:
        client.destroy()


def nodes(config):
    return ["research", "critique", "red_team", "evaluation", "decomposition_audit",
            "socratic:clarify", "socratic:challenge_assumptions",
            "socratic:identify_evidence_needed", "socratic:verification"] + [
                "debate:" + ":".join(item["pair"]) for item in config["debates"]]


def event_in_round(event, number, config):
    """Reconstruct hashed execution coordinates; timestamps do not identify rounds."""
    data = event["data"]
    kind = event["event_type"]
    for node in nodes(config):
        if event["id"] == derive_event_id(
            event["cell_id"], round_number=number, node=node, agent=event["agent"],
            iteration=event["iteration"], event_type=kind,
        ):
            return True
        identity = None
        if kind == "search_executed" and "source_urls" in data:
            identity = digest(json.dumps([data["query"], data["source_urls"]], sort_keys=True).encode())
        if kind == "source_reviewed" and data.get("retrieval_method") == "public-web":
            identity = data.get("url", "") + ":" + data.get("content_sha256", "")
        if identity is not None and event["id"] == stable_event_id(
            event["cell_id"], "public-web", number, node, event["agent"],
            event["iteration"], kind, identity,
        ):
            return True
    return False


def scoped_usage(snapshot, cell_id, number):
    """Follow parent spans to the explicit cell-round workflow span."""
    spans = {row["span_id"]: {**row, "attrs": json.loads(row["attributes"])}
             for row in snapshot["spans"]}
    result = []
    for row in snapshot["usage"]:
        if row["cell_id"] != cell_id:
            continue
        span = spans.get(row["span_id"])
        seen = set()
        while span and span["span_id"] not in seen:
            seen.add(span["span_id"])
            attrs = span["attrs"]
            if attrs.get("petri.cell_id") == cell_id and attrs.get("petri.round") == number:
                result.append({**row, "round": number, "workflow_id": f"{cell_id}:r{number}"})
                break
            span = spans.get(span["parent_span_id"])
    return result


def fetched_source(data):
    try:
        timestamp = datetime.fromisoformat(data.get("retrieved_at", ""))
    except (ValueError, TypeError):
        return False
    return bool(timestamp.tzinfo and data.get("url", "").startswith(("https://", "http://"))
                and re.fullmatch("[a-f0-9]{64}", data.get("content_sha256", ""))
                and data.get("finding") and "[Failed to fetch" not in data["finding"])


def research_report(snapshot, workflows, cell_id, number, expected, config):
    """Require a successful exact root/round, retrieval, and supporting evaluation."""
    cell = next((row for row in snapshot["cells"] if row["id"] == cell_id), None)
    workflow_id = f"{cell_id}:r{number}"
    events = [event for event in snapshot["events"] if event["cell_id"] == cell_id
              and event_in_round(event, number, config)]
    searches = [event for event in events if event["event_type"] == "search_executed"
                and event["data"].get("sources_found", 0) > 0]
    sources = [event for event in events if event["event_type"] == "source_reviewed"
               and event["data"].get("retrieval_method") == "public-web"]
    failures = [event for event in events if event["event_type"] == "agent_execution_failed"]
    direction = "supports" if expected == "VALIDATED" else "contradicts"
    verdict = "EVIDENCE_CONFIRMS" if expected == "VALIDATED" else "EVIDENCE_REFUTES"
    evaluations = [event for event in events if event["event_type"] == "verdict_issued"
                   and event["agent"] == "evidence_evaluator"
                   and event["data"].get("verdict") == verdict]
    valid_sources = {event["data"]["url"] for event in sources
                     if fetched_source(event["data"])}
    supported = [citation for event in evaluations for citation in event["data"].get("sources_cited", [])
                 if citation.get("url") in valid_sources
                 and citation.get("supports_or_contradicts") == direction
                 and 1 <= citation.get("hierarchy_level", 6) <= 4 and citation.get("finding")]
    usage = scoped_usage(snapshot, cell_id, number)
    problems = []
    for valid, message in (
        (cell is not None and cell["status"] == expected, "Expected terminal status is absent."),
        (workflows.get(workflow_id, {}).get("status") == "SUCCESS", "Workflow did not succeed."),
        (bool(searches), "No search with nonzero hits in this round."),
        (bool(valid_sources), "No successful fetched source with full provenance in this round."),
        (bool(supported), "No fetched eligible citation supports the expected evaluation."),
        (not failures, "This round contains an execution failure."),
        (bool(usage), "No usage attributed through the exact cell-round workflow."),
    ):
        if not valid:
            problems.append(message)
    return {"cell_id": cell_id, "colony_id": cell["colony"] if cell else None,
            "round": number, "workflow_id": workflow_id, "expected_status": expected,
            "actual_status": cell["status"] if cell else None, "searches": searches,
            "sources": sources, "evaluations": evaluations, "failures": failures,
            "usage": usage, "problems": problems, "passed": not problems}


def cross_references(snapshot, dish):
    cells = {row["id"]: row for row in snapshot["cells"]}
    return [edge for edge in snapshot["edges"] if edge["edge_type"] == "cross_colony"
            and edge["from_cell"] in cells and edge["to_cell"] in cells
            and cells[edge["from_cell"]]["colony"] == dish + "-shared"
            and cells[edge["to_cell"]]["colony"] == dish + "-isolated"
            and cells[edge["to_cell"]]["level"] > 0]


def continuity(before, after):
    """Committed outputs must survive; no claim is made about uncommitted calls."""
    count = 0
    for identity, run in before.items():
        later = {step["function_id"]: step for step in after.get(identity, {}).get("steps", [])}
        for step in run["steps"]:
            require(later.get(step["function_id"]) == step,
                    f"Committed checkpoint changed or disappeared: {identity}/{step['function_id']}.")
            count += 1
    require(count > 0, "No durable checkpoint was observed before termination.")
    return count


def feed_report(before, after, old_workflows, new_workflows, targets, evidence):
    rows = {row["id"]: row for row in after["cells"]}
    previous = {row["id"]: row for row in before["cells"]}
    old_events = {row["id"]: row for row in before["events"]}
    events = [row for row in after["events"] if row["id"] not in old_events]
    fed = [row for row in events if row["event_type"] == "evidence_fed"]
    require({row["cell_id"] for row in fed} == set(targets) and len(fed) == len(targets),
            "Feed did not create exactly one evidence event per declared target.")
    forks = []
    reopened = [row for row in events if row["event_type"] == "cell_reopened"]
    require({row["cell_id"] for row in reopened} == set(targets) and len(reopened) == len(targets),
            "Feed did not create exactly one reopen event per declared target.")
    dependents = set()
    for event in fed:
        data = event["data"]
        cell_id = event["cell_id"]
        require(data["content_sha256"] == digest(evidence.encode()) and data["content"] == evidence,
                "Fed evidence differs from the supplied source.")
        original, fork = data["original_workflow_id"], data["forked_workflow_id"]
        context = next(row["data"] for row in reopened if row["cell_id"] == cell_id)
        require(all(context.get(key) == data[key] for key in (
            "original_workflow_id", "forked_workflow_id", "generation")),
            "The reopen event names another fork or round.")
        require(original == cell_id + ":r1" and fork == cell_id + ":r2" and data["generation"] == 2,
                "Feed did not fork the declared first round into round two.")
        old_steps = old_workflows.get(original, {}).get("steps", [])
        new_steps = new_workflows.get(fork, {}).get("steps", [])
        research = [step["function_id"] for step in old_steps if step["phase"] == "research"]
        require(research, "Original research checkpoint is missing.")
        boundary = min(research)
        copied = [step for step in old_steps if step["function_id"] < boundary]
        require(any(step["phase"] == "socratic" for step in copied), "No Socratic work was copied.")
        require(copied == [step for step in new_steps if step["function_id"] < boundary],
                "Forked Socratic checkpoints differ from the original.")
        require(any(step["phase"] == "research" for step in new_steps), "New research steps are missing.")
        require(rows[cell_id]["status"] == targets[cell_id], "Fed cell has the wrong post-feed status.")
        frontier = {cell_id}
        reached = set()
        while frontier:
            frontier = {edge["from_cell"] for edge in before["edges"] if edge["to_cell"] in frontier} - reached
            reached |= frontier
        dependents |= reached
        flags = {row["cell_id"] for row in events if row["event_type"] == "propagation_triggered"
                 and row["data"].get("reopened_cell_id") == cell_id}
        require(reached <= flags, "Transitive dependents were not flagged.")
        forks.append({"cell_id": cell_id, **data, "start_step": boundary})
    new_validations = {key for key, value in new_workflows.items()
                       if key not in old_workflows and value["name"] == "petri.validate_cell"}
    require(new_validations == {row["forked_workflow_id"] for row in forks},
            "Feed enqueued an undeclared validation workflow.")
    for cell_id in dependents - targets.keys():
        require(rows[cell_id]["status"] == previous[cell_id]["status"], "A dependent was reopened.")
    require(dependents - targets.keys(), "Feed did not exercise an external dependent.")
    return {"forks": forks, "flagged_dependents": sorted(dependents), "targets": targets}


def feed_eligible(snapshot, colony):
    """Mirror the public CLI's terminal and previously fed NEW eligibility."""
    fed = {event["cell_id"] for event in snapshot["events"] if event["event_type"] == "evidence_fed"}
    return {row["id"]: row["status"] for row in snapshot["cells"] if row["colony"] == colony
            and (row["status"] in {"VALIDATED", "DISPROVEN", "DEFER_OPEN"}
                 or row["status"] == "NEW" and row["id"] in fed)}


def check_feed_plan(snapshot, colony, targets=None):
    eligible = feed_eligible(snapshot, colony)
    if targets is None:
        unsupported = {key: status for key, status in eligible.items() if status != "VALIDATED"}
        require(not unsupported, "Feed preflight rejected unsupported foundation outcomes: "
                + canonical(unsupported) + ". Preserve this dish for review; no feed was started.")
        require(eligible, "The foundation has no eligible validated cells.")
        return eligible
    require(targets and all(status == "VALIDATED" for status in targets.values()),
            "The original feed plan contains unsupported expected outcomes.")
    require(set(eligible) == set(targets),
            "CLI feed eligibility differs from the original target set; preserve the dish for review.")
    require(all(status in {"VALIDATED", "NEW"} for status in eligible.values()),
            "A retained feed target has an unsupported outcome; no resumed feed was started.")
    return targets


def span_workflow(span, spans):
    """Resolve a stored span to its explicit workflow round without crossing traces."""
    seen = set()
    trace_id = span["trace_id"]
    while span and span["span_id"] not in seen:
        seen.add(span["span_id"])
        if span["trace_id"] != trace_id:
            return None
        attrs = json.loads(span["attributes"])
        if "petri.cell_id" in attrs and type(attrs.get("petri.round")) is int:
            return f"{attrs['petri.cell_id']}:r{attrs['petri.round']}"
        span = spans.get(span["parent_span_id"])
    return None


def check_response_readback(snapshot, expected, evidence, root):
    """Join remote readback IDs to local model spans in every required cell-round."""
    spans = {row["span_id"]: row for row in snapshot["spans"]}
    selected = set(evidence["span_ids"])
    model_responses = {}
    for span_id, span in spans.items():
        attrs = json.loads(span["attributes"])
        response_id = attrs.get("gen_ai.response.id")
        workflow = span_workflow(span, spans)
        if (attrs.get("gen_ai.operation.name") == "chat" and response_id
                and workflow in expected):
            model_responses[(response_id, span_id, workflow)] = span["trace_id"]
    mappings = evidence.get("response_readbacks", [])
    require(mappings, "Stored response readback mappings are missing.")
    joined = set()
    records = {item["path"]: item for item in evidence.get("artifacts", [])}
    for item in mappings:
        key = (item.get("response_id"), item.get("span_id"), item.get("workflow_id"))
        require(key in model_responses and item.get("trace_id") == model_responses[key],
                "Response readback does not match a local model span in the required round.")
        path = item.get("artifact_path")
        require(path in records, "Response readback does not name a retained attachment.")
        captured = json.loads((root / path).read_text())
        require(isinstance(captured, dict) and all(captured.get(field) == item.get(field)
                for field in ("response_id", "span_id", "trace_id", "workflow_id")),
                "Remote readback artifact does not name the matching local response coordinates.")
        require(captured.get("remote_record"), "Remote response content readback is missing.")
        joined.add(key)
    require({key[2] for key in joined} == expected, "Model response readback omits a required cell-round.")
    require(joined == set(model_responses) and all(key[1] in selected for key in joined),
            "Model response readback omits stored responses in a required cell-round.")
    require(set(evidence.get("response_ids", [])) == {key[0] for key in joined},
            "Final response IDs differ from the joined readback mappings.")


def process_identity(pid):
    """Read Linux process birth coordinates; never inspect environment values."""
    from pathlib import Path
    try:
        proc = Path("/proc") / str(pid)
        fields = (proc / "stat").read_text().rsplit(")", 1)[1].split()
        return {"pid": pid, "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                "start_ticks": fields[19], "state": fields[0],
                "cwd": str((proc / "cwd").resolve(strict=True)),
                "argv": [part.decode() for part in (proc / "cmdline").read_bytes().split(b"\0") if part]}
    except (FileNotFoundError, ProcessLookupError):
        return None


def same_process(saved, current):
    return bool(saved and current and all(saved.get(key) == current.get(key)
                                         for key in ("pid", "boot_id", "start_ticks")))
