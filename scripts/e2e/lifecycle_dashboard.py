"""Root-operated, read-only dashboard capture against the same lifecycle dish."""

from __future__ import annotations

import json
import math
import time
from urllib.parse import urlsplit
from uuid import uuid4

import httpx
from filelock import FileLock, Timeout

from scripts.e2e.lifecycle_checks import AcceptanceError, domain, process_identity, require, same_process


def check_owner(petri, identity):
    current = process_identity(identity["pid"])
    require(same_process(identity, current) and current["state"] != "Z",
            "The supplied CLI process exited or its PID was reused.")
    require(current["cwd"] == str(petri.parent.resolve()) and "petri" in current["argv"]
            and any(value in current["argv"] for value in ("grow", "feed", "seed")),
            "The supplied PID is not a Petri CLI command in this cycle directory.")
    lock = FileLock(str(petri / ".execution-owner.lock"), timeout=0)
    try:
        lock.acquire()
    except Timeout:
        return current
    else:
        lock.release()
        raise AcceptanceError("The CLI does not hold the execution owner lock.")


def capture_dashboard(root, cycle, base_url, cli_pid, *, capture_timeout=120, retry_interval=1,
                      final_status_timeout=14400):
    """Retry CLI-owned reads, then confirm final root outcomes in a separate snapshot."""
    from scripts.e2e.lifecycle_runner import CASES, artifact, now, write_json

    require(all(math.isfinite(value) and value > 0 for value in
                (capture_timeout, retry_interval, final_status_timeout)),
            "Dashboard capture and retry bounds must be finite positive seconds.")
    parsed = urlsplit(base_url)
    require(parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost", "::1"}
            and parsed.path in {"", "/"} and not parsed.username and not parsed.password
            and not parsed.query and not parsed.fragment, "Use a loopback dashboard HTTP origin.")
    directory = root / f"cycle-{cycle}"
    petri = directory / ".petri"
    dish = f"lifecycle-cycle-{cycle}"
    database = petri / "petri-dishes" / dish / "petri.sqlite"
    roots = {dish + "-" + case + "-000-000": status for case, (_, status) in CASES.items()}
    identity = process_identity(cli_pid)
    require(identity and identity["cwd"] == str(directory.resolve()),
            "The CLI PID is missing or belongs to another working directory.")
    output = root / "captures" / f"cycle-{cycle}" / uuid4().hex
    output.mkdir(parents=True)
    deadline = time.monotonic() + capture_timeout
    attempts = []
    while True:
        attempt = output / str(len(attempts) + 1)
        attempt.mkdir()
        try:
            owner_before = check_owner(petri, identity)
            snapshot = domain(database)
            files, inserted = capture_payloads(root, attempt, base_url, dish, roots, snapshot, deadline)
            owner_after = check_owner(petri, identity)
        except (AcceptanceError, httpx.HTTPError, OSError, ValueError) as error:
            attempts.append({"status": "failed", "observed_at": now(), "error": str(error)})
            write_json(attempt / "observation.json", attempts[-1])
            write_json(output / "attempts.json", attempts)
            current = process_identity(cli_pid)
            require(same_process(identity, current) and current["state"] != "Z",
                    "CLI exited before a complete owned capture; retain failed attempts and retry during another task-owned command.")
            remaining = deadline - time.monotonic()
            require(remaining > 0, "Dashboard ownership capture window expired; failed attempts are retained.")
            time.sleep(min(retry_interval, remaining))
        else:
            attempts.append({"status": "passed", "observed_at": now(), "directory": str(attempt.relative_to(root))})
            write_json(output / "attempts.json", attempts)
            write_json(output / "ownership.json", {
                "before": owner_before, "after": owner_after, "observed_at": now(),
                "root_statuses_during_ownership": {row["id"]: row["status"] for row in snapshot["cells"]
                                                   if row["id"] in roots}})
            files += [artifact(root, output / "attempts.json"), artifact(root, output / "ownership.json")]
            break
    # Feed may reopen a root during the owned observation. Confirm outcomes afterward.
    deadline = time.monotonic() + final_status_timeout
    while True:
        final = domain(database)
        statuses = {row["id"]: row["status"] for row in final["cells"] if row["id"] in roots}
        write_json(output / "final-roots.json", {"observed_at": now(), "root_statuses": statuses})
        if statuses == roots:
            break
        remaining = deadline - time.monotonic()
        require(remaining > 0, "Owned capture retained, but final known-outcome roots did not pass within the confirmation window.")
        time.sleep(min(retry_interval, remaining))
    files.append(artifact(root, output / "final-roots.json"))
    evidence = {"observer": "Root-operated capture", "observed_at": now(), "dish_id": dish,
                "root_statuses": statuses, "cli_pid": cli_pid, "cli_identity": identity,
                "sse_event": inserted, "artifacts": files,
                "checks": {key: True for key in ("graph_payload", "cell_payloads", "usage_payloads",
                                                  "trace_payloads", "sse_committed_events", "read_only_during_cli")}}
    write_json(output / "evidence.json", evidence)
    return output / "evidence.json"


def capture_payloads(root, output, base_url, dish, roots, snapshot, deadline):
    from scripts.e2e.lifecycle_runner import artifact, write_json

    def remaining():
        seconds = deadline - time.monotonic()
        require(seconds > 0, "Dashboard capture window expired during requests.")
        return min(10, seconds)

    files = []
    responses = {}
    with httpx.Client(base_url=base_url.rstrip("/"), timeout=remaining(), trust_env=False,
                      follow_redirects=False) as client:
        routes = ["/api/dishes", "/api/cells", "/api/edges", "/api/queue"]
        for cell in roots:
            routes += ["/api/cell/" + cell, "/api/costs?cell_id=" + cell,
                       "/api/traces?cell_id=" + cell]
        for index, route in enumerate(routes):
            response = client.get(route, timeout=remaining())
            require(response.status_code == 200, "Dashboard read failed.")
            payload = response.json()
            responses[route] = payload
            path = output / f"api-{index}.json"
            write_json(path, {"route": route, "status": response.status_code, "body": payload})
            files.append(artifact(root, path))
        require({row["id"] for row in responses["/api/dishes"]} == {dish},
                "Dashboard is serving another dish.")
        require(set(roots) <= {row["cell_id"] for row in responses["/api/cells"]
                              if row["dish_id"] == dish}, "Dashboard cells omit a declared root.")
        require(any(edge["edge_type"] == "cross_colony" for edge in responses["/api/edges"]),
                "Dashboard graph omits cross-colony reuse.")
        for index, cell in enumerate(roots):
            require(responses["/api/cell/" + cell]["dish_id"] == dish, "Cell detail belongs to another dish.")
            costs = responses["/api/costs?cell_id=" + cell]
            require(costs.get("cells"), "Dashboard usage has no cell readback.")
            traces = responses["/api/traces?cell_id=" + cell]
            require(traces, "Dashboard trace listing is empty.")
            trace = client.get("/api/traces/" + traces[0]["trace_id"], timeout=remaining())
            require(trace.status_code == 200 and trace.json(), "Dashboard waterfall spans are missing.")
            path = output / f"trace-{index}.json"
            write_json(path, trace.json())
            files.append(artifact(root, path))
        known = {row["id"]: row for row in snapshot["events"]}
        lines = []
        inserted = None
        with client.stream("GET", "/api/stream", timeout=remaining()) as response:
            require(response.status_code == 200 and "text/event-stream" in
                    response.headers.get("content-type", ""), "Dashboard SSE did not open.")
            for line in response.iter_lines():
                lines.append(line)
                require(time.monotonic() < deadline and len(lines) < 20000,
                        "No committed research event was observed on SSE within the capture bound.")
                if line.startswith("data: "):
                    payload = json.loads(line[6:])
                    row = known.get(payload.get("id"))
                    if row and row["event_type"] in {"source_reviewed", "verdict_issued"}:
                        require(payload.get("cell_id") == row["cell_id"], "SSE event has the wrong cell identity.")
                        inserted = payload
                        break
        require(inserted, "SSE contained no matching committed research event.")
        path = output / "stream.sse"
        path.write_text("\n".join(lines) + "\n")
        files.append(artifact(root, path))
    return files, inserted
