#!/usr/bin/env python3
"""Create the Petri v2 migration backlog on GitHub.

Reads docs/v2/issues/backlog.json and creates, in order:
  labels -> milestones -> 7 epic issues -> 81 issues -> sub-issue links ->
  dependency links -> epic issue tables.

GitHub itself is the only state store: every phase queries before writing
(matching on exact titles), so the script is idempotent — re-running after a
crash resumes where it left off and a full re-run reports "0 created".

Requires: gh CLI (authenticated with repo scope). Stdlib only.

Usage:
  python scripts/create_v2_issues.py --dry-run     # print the plan, write nothing
  python scripts/create_v2_issues.py               # run all phases
  python scripts/create_v2_issues.py --through 5   # stop after phase 5
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = "onthemarkdata/petri"
BLOB = f"https://github.com/{REPO}/blob/main"
DEPS_API_VERSION = "2026-03-10"
BACKLOG = Path(__file__).resolve().parent.parent / "docs" / "v2" / "issues" / "backlog.json"

EPIC_NAMES = {
    "M1-harness": "Harness & inference layer",
    "M2-agents": "Agents on Pydantic AI",
    "M3-decomposer": "Agentic decomposer + evals",
    "M4-dbos": "Durable execution",
    "M5-otel": "OpenTelemetry",
    "M6-storage": "Storage & dashboard reads",
    "M7-lifecycle": "Iterative lifecycle",
}
EPIC_AREA = {
    "M1-harness": "harness",
    "M2-agents": "agents",
    "M3-decomposer": "decomposer",
    "M4-dbos": "durable-execution",
    "M5-otel": "observability",
    "M6-storage": "storage",
    "M7-lifecycle": "lifecycle",
}

# name -> (color, description). GitHub defaults (`documentation`, `good first issue`,
# `bug`, `enhancement`) are reused, never re-created or modified.
LABEL_DEFS = {
    "migration-v2": ("5319E7", "Part of the v2 migration backlog"),
    "epic": ("3E4B9E", "Milestone tracking issue with sub-issues"),
    "harness": ("1D76DB", "Inference harness layer (pi, Claude Code, providers)"),
    "agents": ("1D76DB", "Agent roster, debates, convergence"),
    "decomposer": ("1D76DB", "Claim decomposition and evals"),
    "durable-execution": ("1D76DB", "ExecutionBackend seam, DBOS, queues"),
    "observability": ("1D76DB", "OpenTelemetry, spans, cost accounting"),
    "storage": ("1D76DB", "Domain store, query layer, export, backup"),
    "dashboard": ("1D76DB", "petri launch web UI and REST/SSE API"),
    "cli": ("1D76DB", "Typer CLI surface"),
    "lifecycle": ("1D76DB", "Re-decomposition, feed, scan, analyst"),
    "spike": ("D4C5F9", "Timeboxed investigation; deliverable is a document"),
    "rfc": ("C5DEF5", "Design-first; discussion expected before implementation"),
    "breaking-change": ("B60205", "Observable behavior change for existing users"),
    "housekeeping": ("FEF2C0", "Repo hygiene and cleanup"),
    "size:S": ("C2E0C6", "Rough effort: hours"),
    "size:M": ("FBCA04", "Rough effort: a day or two"),
    "size:L": ("E99695", "Rough effort: several days"),
}
LABEL_MAP = {"docs": "documentation", "good-first-issue": "good first issue"}

FIELD_REF_RE = re.compile(r"(?<![\w&`])#(\d{1,2})\b")
HEADING_RE = re.compile(r"^\*\*(Context|Scope|Out of scope|Touched files)\.\*\*\s*", re.M)
MILESTONE_SLUG_RE = re.compile(r"^(M\d+-[a-z-]+)")
ISSUES_BLOCK_RE = re.compile(r"<!-- issues:begin -->.*<!-- issues:end -->", re.S)

DRY_RUN = False
_writes = {"labels": 0, "milestones": 0, "epics": 0, "issues": 0, "sub_links": 0, "dep_links": 0, "patches": 0}


def gh(args: list[str], payload: dict | None = None, ok_statuses: tuple = ()) -> dict | list | None:
    """Run `gh api` with retry on secondary rate limits. Returns parsed JSON."""
    cmd = ["gh", "api", *args]
    stdin = json.dumps(payload) if payload is not None else None
    if payload is not None:
        cmd += ["--input", "-"]
    delay = 60
    for attempt in range(5):
        proc = subprocess.run(cmd, input=stdin, capture_output=True, text=True)
        if proc.returncode == 0:
            return json.loads(proc.stdout) if proc.stdout.strip() else None
        err = proc.stderr + proc.stdout
        if "HTTP 403" in err or "HTTP 429" in err or "rate limit" in err.lower():
            print(f"    rate-limited; sleeping {delay}s (attempt {attempt + 1}/5)", flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 900)
            continue
        for status in ok_statuses:
            if f"HTTP {status}" in err:
                return None
        print(f"FATAL: gh api {' '.join(args)}\n{err}", file=sys.stderr)
        sys.exit(1)
    print("FATAL: exhausted retries", file=sys.stderr)
    sys.exit(1)


def paginate(path: str, headers: list[str] | None = None) -> list:
    args = []
    for h in headers or []:
        args += ["-H", h]
    args += [path, "--paginate"]
    out = gh(args)
    return out if isinstance(out, list) else []


def md_escape(text: str) -> str:
    """Escape angle brackets so GitHub's HTML sanitizer doesn't strip <tokens>."""
    return text.replace("<", "\\<")


def code_span_field_refs(text: str) -> str:
    """Bare #N (N in 2..15) refers to docs/field-reports.md entries, not issues in
    this repo — wrap in code spans so GitHub doesn't autolink them to v2 issues."""

    def sub(m: re.Match) -> str:
        n = int(m.group(1))
        return f"`#{n}`" if 2 <= n <= 15 else m.group(0)

    return FIELD_REF_RE.sub(sub, text)


def render_issue_body(issue: dict, slug: str, epic_num: int, title_to_num: dict[str, int]) -> str:
    body = HEADING_RE.sub(lambda m: f"### {m.group(1)}\n\n", issue["body"])
    body = code_span_field_refs(body)

    parts = [
        f"> Part of **{slug}** ([epic #{epic_num}](https://github.com/{REPO}/issues/{epic_num})) "
        f"· Size **{issue['size']}** "
        f"· [Migration plan]({BLOB}/docs/v2/MIGRATION_PLAN.md) "
        f"· [Architecture]({BLOB}/docs/ARCHITECTURE-V2.md)",
        "",
        body.strip(),
        "",
        "### Acceptance criteria",
        "",
    ]
    parts += [f"- [ ] {code_span_field_refs(ac)}" for ac in issue["acceptance_criteria"]]

    if issue["depends_on"]:
        parts += ["", "### Dependencies", ""]
        for dep in issue["depends_on"]:
            parts.append(f"- Blocked by #{title_to_num[dep]} ({md_escape(dep)})")

    if issue["field_issues_absorbed"]:
        closes = [f"`#{f[1:]}`" for f in issue["field_issues_absorbed"] if f.startswith("#")]
        relates = [f"`{f.split(':')[1]}`" for f in issue["field_issues_absorbed"] if f.startswith("relates:")]
        line = f"**Field reports** ([index]({BLOB}/docs/field-reports.md)):"
        if closes:
            line += f" closes {', '.join(closes)};"
        if relates:
            line += f" relates {', '.join(relates)}"
        parts += ["", line.rstrip(";")]

    parts += [
        "",
        "---",
        f"`#N` numbers above refer to entries in [docs/field-reports.md]({BLOB}/docs/field-reports.md), "
        "not issues in this repo.",
        f"New contributor? See [CONTRIBUTING.md]({BLOB}/CONTRIBUTING.md) — comment to claim this issue; "
        "questions welcome.",
    ]
    return "\n".join(parts)


def render_epic_body(m: dict, epic_deps: list[tuple[int, str]]) -> str:
    slug = m["milestone"]
    parts = [
        code_span_field_refs(m["goal"]),
        "",
        "### Shippable release",
        "",
        code_span_field_refs(m["shippable_release"]),
    ]
    if m.get("risks"):
        parts += ["", "### Milestone risks", ""]
        parts += [f"- {code_span_field_refs(r)}" for r in m["risks"]]
    if epic_deps:
        parts += ["", "### Depends on", ""]
        parts += [f"- Blocked by #{num} ({md_escape(title)})" for num, title in epic_deps]
    parts += [
        "",
        "### Issues",
        "",
        "<!-- issues:begin -->",
        "_Populated automatically once all sub-issues exist._",
        "<!-- issues:end -->",
        "",
        "### References",
        "",
        f"- [Migration plan]({BLOB}/docs/v2/MIGRATION_PLAN.md)",
        f"- [Architecture / decision record]({BLOB}/docs/ARCHITECTURE-V2.md)",
        f"- [Field reports]({BLOB}/docs/field-reports.md) (cited as `#N` in issue text)",
        f"- [Full issue backlog for this milestone]({BLOB}/docs/v2/issues/{slug}.md)",
    ]
    return "\n".join(parts)


def issue_labels(issue: dict) -> list[str]:
    labels = [LABEL_MAP.get(lbl, lbl) for lbl in issue["labels"]]
    labels.append(f"size:{issue['size']}")
    if issue["good_first_issue"] and "good first issue" not in labels:
        labels.append("good first issue")
    return sorted(set(labels))


def epic_title(slug: str) -> str:
    return f"[EPIC] {slug} — {EPIC_NAMES[slug]}"


# ---------------------------------------------------------------- phases

def phase0_load() -> list[dict]:
    data = json.loads(BACKLOG.read_text())
    assert len(data) == 7, f"expected 7 milestones, got {len(data)}"
    titles: list[str] = []
    seen_per_milestone: dict[str, set] = {}
    known_labels = set(LABEL_DEFS) | set(LABEL_MAP)
    for m in data:
        seen_per_milestone[m["milestone"]] = set()
        for issue in m["issues"]:
            titles.append(issue["title"])
            seen_per_milestone[m["milestone"]].add(issue["title"])
            assert issue["size"] in ("S", "M", "L"), issue["title"]
            for lbl in issue["labels"]:
                assert lbl in known_labels, f"unknown label {lbl!r} on {issue['title']!r}"
    assert len(titles) == 81, f"expected 81 issues, got {len(titles)}"
    assert len(set(titles)) == 81, "duplicate issue titles"
    # deps resolve to same-or-earlier position (topological order)
    position = {t: i for i, t in enumerate(titles)}
    for m in data:
        for issue in m["issues"]:
            for dep in issue["depends_on"]:
                assert dep in position, f"dangling dep {dep!r}"
                assert position[dep] < position[issue["title"]], f"forward dep: {issue['title']!r}"
    print("phase 0: backlog valid — 7 milestones, 81 issues, deps topologically ordered")
    return data


def phase1_labels() -> None:
    existing = {lbl["name"] for lbl in paginate(f"repos/{REPO}/labels?per_page=100")}
    for name, (color, desc) in LABEL_DEFS.items():
        if name in existing:
            continue
        print(f"  creating label {name!r}")
        if not DRY_RUN:
            gh(["-X", "POST", f"repos/{REPO}/labels"], {"name": name, "color": color, "description": desc})
            _writes["labels"] += 1
            time.sleep(1)
    print(f"phase 1: labels done ({_writes['labels']} created)")


def phase2_milestones(data: list[dict]) -> dict[str, int]:
    existing = {m["title"]: m["number"] for m in paginate(f"repos/{REPO}/milestones?state=all&per_page=100")}
    out = {}
    for m in data:
        slug = m["milestone"]
        if slug in existing:
            out[slug] = existing[slug]
            continue
        desc = f"{m['goal'].split('. ')[0]}. Ships: {m['shippable_release'].split('. ')[0]}."
        print(f"  creating milestone {slug}")
        if DRY_RUN:
            out[slug] = -1
            continue
        resp = gh(["-X", "POST", f"repos/{REPO}/milestones"], {"title": slug, "description": desc[:700]})
        out[slug] = resp["number"]
        _writes["milestones"] += 1
        time.sleep(1)
    print(f"phase 2: milestones done ({_writes['milestones']} created)")
    return out


def phase3_snapshot() -> dict[str, dict]:
    issues = paginate(f"repos/{REPO}/issues?state=all&per_page=100")
    out = {i["title"]: {"number": i["number"], "id": i["id"]} for i in issues if "pull_request" not in i}
    print(f"phase 3: snapshot — {len(out)} existing issues")
    return out


def phase4_epics(data, milestones, snapshot) -> None:
    epic_nums: dict[str, tuple[int, str]] = {}
    for m in data:
        slug = m["milestone"]
        title = epic_title(slug)
        if title in snapshot:
            epic_nums[slug] = (snapshot[title]["number"], title)
            continue
        deps = []
        for raw in m["depends_on_milestones"]:
            dep_slug = MILESTONE_SLUG_RE.match(raw).group(1)
            deps.append((epic_nums[dep_slug][0], epic_title(dep_slug)))
        print(f"  creating epic {title!r}")
        if DRY_RUN:
            snapshot[title] = {"number": -1, "id": -1}
            epic_nums[slug] = (-1, title)
            _writes["epics"] += 1
            continue
        resp = gh(["-X", "POST", f"repos/{REPO}/issues"], {
            "title": title,
            "body": render_epic_body(m, deps),
            "labels": ["epic", "migration-v2", EPIC_AREA[slug]],
            "milestone": milestones[slug],
        })
        snapshot[title] = {"number": resp["number"], "id": resp["id"]}
        epic_nums[slug] = (resp["number"], title)
        _writes["epics"] += 1
        time.sleep(2)
    print(f"phase 4: epics done ({_writes['epics']} created)")


def phase5_issues(data, milestones, snapshot) -> None:
    title_to_num = {t: v["number"] for t, v in snapshot.items()}
    for m in data:
        slug = m["milestone"]
        epic_num = snapshot[epic_title(slug)]["number"]
        for issue in m["issues"]:
            if issue["title"] in snapshot:
                continue
            print(f"  creating [{slug}] {issue['title'][:70]!r}")
            if DRY_RUN:
                snapshot[issue["title"]] = {"number": -1, "id": -1}
                title_to_num[issue["title"]] = -1
                _writes["issues"] += 1
                continue
            resp = gh(["-X", "POST", f"repos/{REPO}/issues"], {
                "title": issue["title"],
                "body": render_issue_body(issue, slug, epic_num, title_to_num),
                "labels": issue_labels(issue),
                "milestone": milestones[slug],
            })
            snapshot[issue["title"]] = {"number": resp["number"], "id": resp["id"]}
            title_to_num[issue["title"]] = resp["number"]
            _writes["issues"] += 1
            time.sleep(2)
    print(f"phase 5: issues done ({_writes['issues']} created)")


def phase6_sub_issues(data, snapshot) -> None:
    for m in data:
        epic = snapshot[epic_title(m["milestone"])]
        existing = set() if epic["number"] == -1 else {
            s["id"] for s in paginate(f"repos/{REPO}/issues/{epic['number']}/sub_issues?per_page=100")}
        for issue in m["issues"]:
            child = snapshot[issue["title"]]
            if child["id"] in existing:
                continue
            if not DRY_RUN:
                gh(["-X", "POST", f"repos/{REPO}/issues/{epic['number']}/sub_issues"],
                   {"sub_issue_id": child["id"]})
                time.sleep(1)
            _writes["sub_links"] += 1
    print(f"phase 6: sub-issue links done ({_writes['sub_links']} created)")


def phase7_dependencies(data, snapshot) -> None:
    header = f"X-GitHub-Api-Version: {DEPS_API_VERSION}"

    def link(blocked_title: str, blocker_title: str) -> None:
        blocked = snapshot[blocked_title]
        blocker = snapshot[blocker_title]
        existing = set() if blocked["number"] == -1 else {d["id"] for d in paginate(
            f"repos/{REPO}/issues/{blocked['number']}/dependencies/blocked_by?per_page=100", [header])}
        if blocker["id"] in existing:
            return
        if not DRY_RUN:
            gh(["-X", "POST", "-H", header,
                f"repos/{REPO}/issues/{blocked['number']}/dependencies/blocked_by"],
               {"issue_id": blocker["id"]})
            time.sleep(1)
        _writes["dep_links"] += 1

    for m in data:
        for raw in m["depends_on_milestones"]:
            dep_slug = MILESTONE_SLUG_RE.match(raw).group(1)
            link(epic_title(m["milestone"]), epic_title(dep_slug))
        for issue in m["issues"]:
            for dep in issue["depends_on"]:
                link(issue["title"], dep)
    print(f"phase 7: dependency links done ({_writes['dep_links']} created)")


def phase8_epic_tables(data, snapshot) -> None:
    for m in data:
        epic = snapshot[epic_title(m["milestone"])]
        if epic["number"] == -1:
            continue
        rows = ["| # | Issue | Size | Good first issue |", "|---|-------|------|------------------|"]
        for issue in m["issues"]:
            child = snapshot[issue["title"]]
            gfi = "✓" if issue["good_first_issue"] else ""
            rows.append(f"| #{child['number']} | {md_escape(issue['title'])} | {issue['size']} | {gfi} |")
        table = "<!-- issues:begin -->\n" + "\n".join(rows) + "\n<!-- issues:end -->"
        current = gh([f"repos/{REPO}/issues/{epic['number']}"])
        new_body = ISSUES_BLOCK_RE.sub(lambda _: table, current["body"])
        if new_body != current["body"] and not DRY_RUN:
            gh(["-X", "PATCH", f"repos/{REPO}/issues/{epic['number']}"], {"body": new_body})
            _writes["patches"] += 1
            time.sleep(1)
    print(f"phase 8: epic tables done ({_writes['patches']} patched)")


def report(data, snapshot) -> None:
    total = sum(len(m["issues"]) for m in data)
    created_all = all(i["title"] in snapshot for m in data for i in m["issues"])
    print("\n=== report ===")
    print(f"writes this run: {_writes}")
    print(f"all {total} issues + 7 epics present: {created_all and all(epic_title(m['milestone']) in snapshot for m in data)}")
    for m in data:
        et = epic_title(m["milestone"])
        n = snapshot.get(et, {}).get("number", "?")
        print(f"  {m['milestone']}: epic #{n}, {len(m['issues'])} issues")


def main() -> None:
    global DRY_RUN
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--through", type=int, default=8, choices=range(0, 9),
                    help="stop after this phase (default: 8, run everything)")
    args = ap.parse_args()
    DRY_RUN = args.dry_run
    if DRY_RUN:
        print("DRY RUN — no writes\n")

    data = phase0_load()
    if args.through == 0:
        return
    phase1_labels()
    if args.through == 1:
        return
    milestones = phase2_milestones(data)
    if args.through == 2:
        return
    snapshot = phase3_snapshot()
    if args.through == 3:
        return
    phase4_epics(data, milestones, snapshot)
    if args.through == 4:
        return
    phase5_issues(data, milestones, snapshot)
    if args.through == 5:
        return
    phase6_sub_issues(data, snapshot)
    if args.through == 6:
        return
    phase7_dependencies(data, snapshot)
    if args.through == 7:
        return
    phase8_epic_tables(data, snapshot)
    report(data, snapshot)


if __name__ == "__main__":
    main()
