# /// script
# requires-python = ">=3.11"
# dependencies = ["dbos==3.4.0"]
# ///
"""Run isolated SQLite durability experiments without model calls or production code."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import signal
import sqlite3
import statistics
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

from dbos import DBOS, SetEnqueueOptions, SetWorkflowID
from dbos import error as dboserror

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests/fixtures/dbos-sqlite"
HERE: Path


def audit(kind, key, value=None):
    with sqlite3.connect(HERE / "audit.sqlite", timeout=30) as conn:
        conn.execute(
            "INSERT INTO audit VALUES (?,?,?,?,?)",
            (time.time(), os.getpid(), kind, str(key), json.dumps(value)),
        )


def rows(kind):
    with sqlite3.connect(HERE / "audit.sqlite") as conn:
        return conn.execute(
            "SELECT t,pid,key,value FROM audit WHERE kind=? ORDER BY t", (kind,)
        ).fetchall()


def initialize(path, executor="one", caps=(8, 4), limiter=None, polling=0.05):
    global HERE
    HERE = Path(path)
    HERE.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(HERE / "audit.sqlite") as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS audit(t REAL,pid INTEGER,kind TEXT,key TEXT,value TEXT)"
        )
        conn.execute("CREATE TABLE IF NOT EXISTS events(id TEXT PRIMARY KEY,payload TEXT)")
    DBOS(
        config={
            "name": "petri-sqlite-spike",
            "application_version": "spike-1",
            "system_database_url": f"sqlite:///{HERE / 'system.sqlite'}",
            "executor_id": executor,
            "use_listen_notify": False,
            "enable_otlp": False,
            "log_level": "WARNING",
            "max_executor_threads": 64,
        }
    )
    DBOS.launch()
    DBOS.register_queue(
        "work",
        global_concurrency=caps[0],
        worker_concurrency=caps[1],
        limiter=limiter,
        polling_interval_sec=polling,
    )


@DBOS.step()
def paid_step(key, block=False):
    audit("call", key)
    if block and not (HERE / "release").exists():
        (HERE / "in-step").write_text(key)
        while not (HERE / "release").exists():
            time.sleep(0.02)
    audit("call-completed", key)
    return f"result:{key}"


@DBOS.step()
def domain_event(key, payload):
    with sqlite3.connect(HERE / "audit.sqlite", timeout=30) as conn:
        conn.execute("INSERT OR IGNORE INTO events VALUES (?,?)", (key, json.dumps(payload)))
    return key


@DBOS.workflow()
def crash_flow():
    first = paid_step("completed")
    second = paid_step("uncheckpointed", True)
    return [first, second]


@DBOS.workflow()
def queued(key, submitted, hold=0.2, events=0):
    return queue_step(key, submitted, hold, events)


@DBOS.step()
def queue_step(key, submitted, hold, events):
    audit("start", key, {"submitted": submitted})
    if events:
        wait_for(lambda: (HERE / "work-release").exists())
    event_latencies = []
    for index in range(events):
        started = time.monotonic()
        with sqlite3.connect(HERE / "audit.sqlite", timeout=30) as conn:
            conn.execute("INSERT OR IGNORE INTO events VALUES (?,?)", (f"{key}:{index}", "event"))
        event_latencies.append(time.monotonic() - started)
    audit("event-latencies", key, event_latencies)
    time.sleep(hold)
    audit("end", key)
    return key


@DBOS.workflow()
def fork_flow():
    return [paid_step("zero"), paid_step("one"), paid_step("two")]


@DBOS.workflow()
def cancel_flow():
    paid_step("cancel-active", True)
    paid_step("cancel-next")


@DBOS.workflow()
def research_flow(key, sources):
    for source in sources:
        value = paid_step(f"{key}:{source}")
        domain_event(f"{key}:{source}", value)
    domain_event(f"{key}:validated", "PASS")
    return "PASS"


@DBOS.step()
def payload_step(size):
    return b"x" * size


@DBOS.workflow()
def payload_flow(size):
    return len(payload_step(size))


def wait_for(predicate, timeout=90):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise TimeoutError("Experiment did not reach its expected state")
        time.sleep(0.02)


def child(mode, path, *extra):
    with open(Path(path) / f"{mode}.log", "w") as log:
        return subprocess.Popen(
            [sys.executable, __file__, mode, str(path), *map(str, extra)],
            stdout=log,
            stderr=subprocess.STDOUT,
        )


def finish(process, timeout=120):
    try:
        code = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        raise
    if code:
        raise RuntimeError(f"Experiment process exited with {code}")


def crash_experiment(path):
    process = child("crash", path)
    try:
        wait_for(lambda: (path / "in-step").exists())
        process.kill()  # SIGKILL: completed checkpoint survives, active external work may repeat.
        assert process.wait() == -signal.SIGKILL
        (path / "release").touch()
        started = time.monotonic()
        recovery = child("recover", path)
        try:
            wait_for(lambda: (path / "recovered").exists())
            elapsed = time.monotonic() - started
        finally:
            finish(recovery)
        with sqlite3.connect(path / "audit.sqlite") as conn:
            counts = dict(
                conn.execute("SELECT key,count(*) FROM audit WHERE kind='call' GROUP BY key")
            )
        assert counts == {"completed": 1, "uncheckpointed": 2}, counts
        return {
            "pass": True,
            "relaunch_to_result_seconds": elapsed,
            "external_call_counts": counts,
            "external_exactly_once": False,
        }
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def percentile(values, p):
    return sorted(values)[min(len(values) - 1, int((len(values) - 1) * p))]


def queue_experiment(
    path, entries, events, limiter=False, single=False, global_cap=None, fast=False
):
    caps = (entries, entries // 2) if events else (8, 4)
    if global_cap is not None:
        caps = (global_cap, 6)
    rate = {"limit": 2, "period": 1.0} if limiter else None
    initialize(path, "one", caps, rate)
    peer = None if single else child("peer", path, int(limiter), *caps)
    try:
        if peer:
            wait_for(lambda: (path / "peer-ready").exists())
        handles = []
        for i in range(entries):
            handles.append(
                DBOS.enqueue_workflow(
                    "work",
                    queued,
                    str(i),
                    time.time(),
                    0.01 if fast else (1.0 if global_cap else 0.5),
                    events,
                )
            )
        if events:
            wait_for(lambda: len(rows("start")) == entries)
            (path / "work-release").touch()
        for handle in handles:
            wait_for(lambda h=handle: h.get_status().status in ("SUCCESS", "ERROR"))
            handle.get_result()
        active = 0
        by_pid = defaultdict(int)
        maxima = defaultdict(int)
        maximum = 0
        timeline = [(t, pid, 1) for t, pid, _, _ in rows("start")]
        timeline += [(t, pid, -1) for t, pid, _, _ in rows("end")]
        for _, pid, delta in sorted(timeline):
            active += delta
            by_pid[pid] += delta
            maximum = max(maximum, active)
            maxima[pid] = max(maxima[pid], by_pid[pid])
        starts = rows("start")
        latencies = [t - json.loads(value)["submitted"] for t, _, _, value in starts]
        cap_pass = maximum <= caps[0] and all(v <= caps[1] for v in maxima.values())
        participation_pass = len(maxima) == (1 if single else 2)
        with sqlite3.connect(HERE / "audit.sqlite") as conn:
            recorded = conn.execute("SELECT count(*) FROM events").fetchone()[0]
        assert recorded == entries * events
        event_times = [v for _, _, _, values in rows("event-latencies") for v in json.loads(values)]
        result = {
            "pass": cap_pass and participation_pass,
            "participation_pass": participation_pass,
            "caps_pass": cap_pass,
            "workers": len(maxima),
            "entries": entries,
            "hold_seconds": 0.01 if fast else (1.0 if global_cap else 0.5),
            "events_per_cell": events,
            "events_recorded": recorded,
            "event_write_p50_seconds": statistics.median(event_times) if event_times else None,
            "event_write_p95_seconds": percentile(event_times, 0.95) if event_times else None,
            "global_cap": caps[0],
            "worker_cap": caps[1],
            "global_peak": maximum,
            "worker_peaks": sorted(maxima.values()),
            "latency_p50_seconds": statistics.median(latencies),
            "latency_p95_seconds": percentile(latencies, 0.95),
            "start_times_relative_seconds": [round(t - starts[0][0], 4) for t, _, _, _ in starts],
        }
        if limiter:
            # DBOS documents a rolling-window limiter. Allow 100 ms scheduling tolerance.
            times = [t for t, _, _, _ in starts]
            gaps = [times[i + 2] - times[i] for i in range(len(times) - 2)]
            result["two_start_gap_min_seconds"] = min(gaps)
            result["limiter_pass"] = min(gaps) >= 0.90
            result["pass"] = result["pass"] and result["limiter_pass"]
        return result
    finally:
        if peer:
            (path / "peer-stop").touch()
            finish(peer)
        DBOS.destroy(destroy_registry=False, workflow_completion_timeout_sec=10)


def semantics_experiment(path):
    initialize(path)
    try:
        with SetWorkflowID("identity"):
            one = DBOS.start_workflow(fork_flow)
        one.get_result()
        with SetWorkflowID("identity"):
            two = DBOS.start_workflow(fork_flow)
        assert one.get_result() == two.get_result()
        assert len(rows("call")) == 3
        steps = DBOS.list_workflow_steps("identity")
        assert [s["function_id"] for s in steps] == [1, 2, 3], steps
        fork = DBOS.fork_workflow("identity", start_step=3)
        assert fork.get_result() == one.get_result()
        assert Counter(row[2] for row in rows("call")) == {"zero": 1, "one": 1, "two": 2}
        raw_fork = DBOS.fork_workflow("identity", start_step=2)
        assert raw_fork.get_result() == one.get_result()
        assert Counter(row[2] for row in rows("call")) == {"zero": 1, "one": 2, "two": 3}
        with SetEnqueueOptions(deduplication_id="same-cell"):
            first = DBOS.enqueue_workflow("work", queued, "dedup", time.time(), 2.0)
        deduplicated = False
        try:
            with SetEnqueueOptions(deduplication_id="same-cell"):
                DBOS.enqueue_workflow("work", queued, "dedup", time.time(), 2.0)
        except dboserror.DBOSQueueDeduplicatedError:
            deduplicated = True
        assert deduplicated
        first.get_result()
        handle = DBOS.start_workflow(cancel_flow)
        wait_for(lambda: (path / "in-step").exists())
        DBOS.cancel_workflow(handle.workflow_id)
        (path / "release").touch()
        wait_for(lambda: handle.get_status().status == "CANCELLED")
        DBOS.destroy(workflow_completion_timeout_sec=10)
        assert "cancel-next" not in [row[2] for row in rows("call")]
        assert "cancel-active" in [row[2] for row in rows("call-completed")]
        return {
            "pass": True,
            "duplicate_id_calls": 3,
            "fork_step_ids": [s["function_id"] for s in steps],
            "fork_copied_steps": [0, 1],
            "fork_reran_steps": [2],
            "logical_step_2_dbos_id": 3,
            "raw_start_step_2_reruns_logical_steps": [1, 2],
            "queue_deduplicated": True,
            "cancel_status": "CANCELLED",
            "active_step_completed": True,
            "next_step_started": False,
        }
    finally:
        DBOS.destroy()


def replay_experiment(path):
    initialize(path)
    try:
        fixtures = json.loads((FIXTURES / "double-spend.json").read_text())
        results = []
        for fixture in fixtures:
            key = fixture["name"]
            baseline_calls = 0
            for count in fixture["attempt_source_counts"]:
                for source in fixture["sources"][:count]:
                    audit("naive-call", f"{key}:{source}")
                    baseline_calls += 1
            for _ in fixture["attempt_source_counts"]:
                with SetWorkflowID(key):
                    handle = DBOS.start_workflow(research_flow, key, fixture["sources"])
                assert handle.get_result() == "PASS"
            count = sum(row[2].startswith(key + ":") for row in rows("call"))
            assert count == len(fixture["sources"])
            # Replay a domain write independently: recovery across the two DBs must also deduplicate.
            event_key = f"{key}:{fixture['sources'][0]}"
            with sqlite3.connect(path / "audit.sqlite") as conn:
                conn.execute("INSERT OR IGNORE INTO events VALUES (?,?)", (event_key, '"replay"'))
            results.append(
                {
                    "name": key,
                    "uncheckpointed_calls": baseline_calls,
                    "checkpointed_calls": count,
                }
            )
        exported = []
        with sqlite3.connect(path / "audit.sqlite") as conn:
            for key, payload in conn.execute("SELECT id,payload FROM events ORDER BY id"):
                exported.append({"id": key, "payload": json.loads(payload)})
        expected = load_golden()
        assert exported == expected, "Golden exported events differ"
        return {"pass": True, "replays": results, "golden_events": len(exported)}
    finally:
        DBOS.destroy(workflow_completion_timeout_sec=10)


def load_golden():
    """Load portable, sorted events from the deterministic fake-research grow fixture."""
    events = [
        json.loads(line) for line in (FIXTURES / "golden-grow.jsonl").read_text().splitlines()
    ]
    keys = [event["id"] for event in events]
    assert keys == sorted(set(keys))
    return events


def size_experiment(path):
    initialize(path)
    try:
        sizes = [1024, 1024 * 1024, 16 * 1024 * 1024]
        timings = []
        for size in sizes:
            started = time.monotonic()
            assert DBOS.start_workflow(payload_flow, size).get_result() == size
            timings.append(time.monotonic() - started)
        with sqlite3.connect(path / "system.sqlite") as conn:
            ceiling = conn.getlimit(sqlite3.SQLITE_LIMIT_LENGTH)
        with sqlite3.connect(path / "system.sqlite") as conn:
            before = (
                conn.execute("PRAGMA page_count").fetchone()[0]
                * conn.execute("PRAGMA page_size").fetchone()[0]
            )
        started = time.monotonic()
        handles = [
            DBOS.enqueue_workflow("work", queued, f"cell-{i}", time.time(), 0, 0)
            for i in range(1000)
        ]
        for handle in handles:
            handle.get_result()
        elapsed = time.monotonic() - started
        DBOS.destroy(workflow_completion_timeout_sec=10)
        with sqlite3.connect(path / "system.sqlite") as conn:
            after = (
                conn.execute("PRAGMA page_count").fetchone()[0]
                * conn.execute("PRAGMA page_size").fetchone()[0]
            )
        return {
            "pass": True,
            "payload_bytes": sizes,
            "roundtrip_seconds": timings,
            "sqlite_length_limit_bytes": ceiling,
            "largest_payload_tested_bytes": sizes[-1],
            "cells": 1000,
            "cell_seconds": elapsed,
            "database_growth_bytes": after - before,
            "universal_pickle_ceiling_established": False,
        }
    finally:
        DBOS.destroy()


def run_all(output):
    results = {
        "dbos": importlib.metadata.version("dbos"),
        "python": platform.python_version(),
        "platform": platform.system() + " " + platform.machine(),
        "sqlite": sqlite3.sqlite_version,
        "polling_seconds": 0.05,
        "use_listen_notify": False,
    }
    with tempfile.TemporaryDirectory(prefix="petri-dbos-spike-") as temp:
        root = Path(temp)
        for scenario in [
            "crash",
            "semantics",
            "replay",
            "size",
            "queue",
            "rate",
            "rate-single",
            "rate-fast",
            "caps-global",
            "volume-10-10",
            "volume-10-500",
            "volume-50-10",
            "volume-50-500",
        ]:
            path = root / scenario
            path.mkdir()
            print(f"Running {scenario}", flush=True)
            if scenario == "crash":
                results[scenario] = crash_experiment(path)
            else:
                process = child("scenario", path, scenario)
                try:
                    finish(process, 300)
                except Exception:
                    print((path / "scenario.log").read_text(), file=sys.stderr)
                    raise
                results[scenario] = json.loads((path / "result.json").read_text())
            logs = "\n".join(p.read_text() for p in path.glob("*.log"))
            results[scenario]["lock_error_log_lines"] = [
                line
                for line in logs.splitlines()
                if "database is locked" in line or "SQLITE_BUSY" in line
            ]
            if results[scenario]["lock_error_log_lines"]:
                results[scenario]["pass"] = False
            output.write_text(json.dumps(results, indent=2) + "\n")
    failures = [
        name
        for name, result in results.items()
        if isinstance(result, dict) and not result.get("pass", True)
    ]
    print(f"Experiment results saved to {output}; failed criteria: {failures}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["run", "crash", "recover", "peer", "scenario"])
    parser.add_argument("path", type=Path)
    parser.add_argument("extra", nargs="*")
    args = parser.parse_args()
    if args.mode == "run":
        run_all(args.path)
    elif args.mode in ("crash", "recover"):
        initialize(args.path)
        if args.mode == "crash":
            with SetWorkflowID("crash"):
                handle = DBOS.start_workflow(crash_flow)
        else:
            handle = DBOS.retrieve_workflow("crash")
        assert handle.get_result() == ["result:completed", "result:uncheckpointed"]
        (args.path / "recovered").touch()
        DBOS.destroy(workflow_completion_timeout_sec=10)
    elif args.mode == "peer":
        initialize(
            args.path,
            "two",
            caps=(int(args.extra[1]), int(args.extra[2])),
            limiter={"limit": 2, "period": 1.0} if int(args.extra[0]) else None,
        )
        (args.path / "peer-ready").touch()
        wait_for(lambda: (args.path / "peer-stop").exists(), 290)
        DBOS.destroy(workflow_completion_timeout_sec=10)
    else:
        scenario = args.extra[0]
        if scenario == "semantics":
            result = semantics_experiment(args.path)
        elif scenario == "replay":
            result = replay_experiment(args.path)
        elif scenario == "size":
            result = size_experiment(args.path)
        else:
            entries, events = (
                map(int, scenario.split("-")[1:])
                if scenario.startswith("volume")
                else (64 if scenario == "rate-fast" else 16, 0)
            )
            result = queue_experiment(
                args.path,
                entries,
                events,
                scenario.startswith("rate"),
                single=scenario == "rate-single",
                global_cap=8 if scenario == "caps-global" else None,
                fast=scenario == "rate-fast",
            )
        (args.path / "result.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
