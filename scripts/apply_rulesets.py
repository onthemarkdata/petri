#!/usr/bin/env python3
"""Bring the Petri GitHub branch-protection rulesets to their desired state.

Two rulesets are managed, keyed by their live numeric ids:

  14761004  "default"  -> targets ~DEFAULT_BRANCH (main)
  18881836  "dev"      -> targets refs/heads/dev

GitHub itself is the only state store. For each ruleset the script fetches the
live JSON, applies an id-specific mutation IN A DEEP COPY, strips the read-only
top-level keys, and compares against the (stripped) live state. If they already
match it reports "OK" and writes nothing; otherwise it prints a readable diff
and — unless --dry-run — PUTs the desired state back. Re-running is safe and
reports "already in desired state" once applied.

LOCKOUT SAFETY: before any PUT the script asserts that bypass_actors still
contains the admin RepositoryRole (actor_id 5). That bypass is what lets the
maintainer self-merge and force-push; if it is ever missing the script aborts
WITHOUT writing rather than risk locking the repo owner out.

The committed snapshots in .github/rulesets/{main,dev}.json are the reviewable
source of truth; `--export .github/rulesets` regenerates them byte-for-byte.

Requires: gh CLI (authenticated with repo admin scope). Stdlib only.

Usage:
  python scripts/apply_rulesets.py --dry-run                  # print diffs, write nothing
  python scripts/apply_rulesets.py                            # apply changes via gh api PUT
  python scripts/apply_rulesets.py --export .github/rulesets  # write desired snapshots, no PUT
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = "onthemarkdata/petri"
RULESETS = {14761004: "main", 18881836: "dev"}

# Top-level keys GitHub returns but rejects (or ignores) on PUT — always stripped
# before comparing or writing.
READONLY_KEYS = (
    "id",
    "node_id",
    "created_at",
    "updated_at",
    "source",
    "source_type",
    "current_user_can_bypass",
    "_links",
)

INTEGRATION_ID = 15368  # GitHub Actions app id for this repo's checks
STATUS_CHECK_CONTEXTS = (
    "test (3.11)",
    "test (3.12)",
    "test (3.13)",
    "test (3.14)",
    "lint",
    "build",
    "typecheck",
)

# Desired parameters for the rules the dev ruleset must contain. Order here is
# load-bearing: it is the order written to .github/rulesets/dev.json, so keep it
# in sync with that snapshot.
# Key order mirrors what GitHub echoes back on GET (see the live main ruleset),
# and required_reviewers / dismissal_restriction are included because GitHub adds
# them as defaults on any pull_request rule — omitting them makes a no-op run
# diff forever (stripped_live would carry two extra keys) and re-PUT every time.
DEV_PULL_REQUEST_PARAMS = {
    "required_approving_review_count": 0,
    "dismiss_stale_reviews_on_push": False,
    "required_reviewers": [],
    "require_code_owner_review": False,
    "dismissal_restriction": {"enabled": False, "allowed_actors": []},
    "require_last_push_approval": False,
    "required_review_thread_resolution": False,
    "allowed_merge_methods": ["merge", "squash", "rebase"],
}
DEV_STATUS_CHECKS_PARAMS = {
    "strict_required_status_checks_policy": True,
    "do_not_enforce_on_create": False,
    "required_status_checks": [
        {"context": context, "integration_id": INTEGRATION_ID}
        for context in STATUS_CHECK_CONTEXTS
    ],
}


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
        lowered = err.lower()
        # Only retry genuine rate limits. A bare HTTP 403 is usually an auth/scope
        # failure (not repo admin, missing token scope) — fail fast on those rather
        # than stall ~25 min through the backoff loop. GitHub's secondary-rate-limit
        # responses still carry a "rate limit"/"secondary rate" body, so they match.
        is_rate_limited = (
            "HTTP 429" in err
            or "rate limit" in lowered
            or "secondary rate" in lowered
            or "abuse detection" in lowered
        )
        if is_rate_limited:
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


# ---------------------------------------------------------------- mutations

def _upsert_rule(rules: list[dict], rule_type: str, parameters: dict) -> None:
    """Set `parameters` on the existing rule of this type, or append a new one.

    Idempotent: a second run finds the rule and rewrites identical parameters
    rather than adding a duplicate.
    """
    for rule in rules:
        if rule.get("type") == rule_type:
            rule["parameters"] = copy.deepcopy(parameters)
            return
    rules.append({"type": rule_type, "parameters": copy.deepcopy(parameters)})


def mutate_main(ruleset: dict) -> None:
    """Require one approving review with code-owner review and stale dismissal,
    and keep the required status checks (including typecheck) in sync.

    Touches the pull_request rule's parameters and the required_status_checks
    rule's context list; the checks' strict policy and every other rule
    (deletion, non_fast_forward, creation, update, required_linear_history) and
    bypass_actors are left exactly as-is.
    """
    saw_pull_request = False
    for rule in ruleset.get("rules", []):
        if rule.get("type") == "pull_request":
            params = rule.setdefault("parameters", {})
            params["required_approving_review_count"] = 1
            params["require_code_owner_review"] = True
            params["dismiss_stale_reviews_on_push"] = True
            saw_pull_request = True
        elif rule.get("type") == "required_status_checks":
            params = rule.setdefault("parameters", {})
            params["required_status_checks"] = [
                {"context": context, "integration_id": INTEGRATION_ID}
                for context in STATUS_CHECK_CONTEXTS
            ]
    if not saw_pull_request:
        print("FATAL: main ruleset has no pull_request rule to mutate", file=sys.stderr)
        sys.exit(1)


def mutate_dev(ruleset: dict) -> None:
    """Ensure the dev ruleset carries a pull_request and required_status_checks
    rule, keeping the existing deletion / non_fast_forward rules and
    bypass_actors untouched. Idempotent via _upsert_rule.
    """
    rules = ruleset.setdefault("rules", [])
    _upsert_rule(rules, "pull_request", DEV_PULL_REQUEST_PARAMS)
    _upsert_rule(rules, "required_status_checks", DEV_STATUS_CHECKS_PARAMS)


MUTATORS = {"main": mutate_main, "dev": mutate_dev}


# ---------------------------------------------------------------- helpers

def strip_readonly(ruleset: dict) -> dict:
    """Drop server-managed top-level keys, preserving remaining key order."""
    return {key: value for key, value in ruleset.items() if key not in READONLY_KEYS}


def has_admin_bypass(ruleset: dict) -> bool:
    """True if the admin RepositoryRole (actor_id 5) can still bypass the rules."""
    return any(
        actor.get("actor_id") == 5 and actor.get("actor_type") == "RepositoryRole"
        for actor in ruleset.get("bypass_actors", [])
    )


def summarize_changes(live: dict, desired: dict) -> list[str]:
    """Human-readable diff between stripped live and desired rulesets."""
    lines: list[str] = []
    for key in sorted(set(live) | set(desired)):
        if key == "rules":
            continue
        if live.get(key) != desired.get(key):
            lines.append(f"  {key}: {json.dumps(live.get(key))} -> {json.dumps(desired.get(key))}")

    live_rules = {rule.get("type"): rule for rule in live.get("rules", [])}
    desired_rules = {rule.get("type"): rule for rule in desired.get("rules", [])}
    for rule_type in sorted(set(live_rules) | set(desired_rules)):
        if rule_type not in live_rules:
            lines.append(f"  + rule '{rule_type}' added")
        elif rule_type not in desired_rules:
            lines.append(f"  - rule '{rule_type}' removed")
        else:
            live_params = live_rules[rule_type].get("parameters", {})
            desired_params = desired_rules[rule_type].get("parameters", {})
            for param in sorted(set(live_params) | set(desired_params)):
                if live_params.get(param) != desired_params.get(param):
                    lines.append(
                        f"  rule '{rule_type}'.{param}: "
                        f"{json.dumps(live_params.get(param))} -> {json.dumps(desired_params.get(param))}"
                    )
    return lines


def write_snapshot(path: Path, ruleset: dict) -> None:
    path.write_text(json.dumps(ruleset, indent=2) + "\n")


# ---------------------------------------------------------------- driver

def process_ruleset(ruleset_id: int, name: str, dry_run: bool, export_dir: Path | None) -> None:
    live = gh([f"repos/{REPO}/rulesets/{ruleset_id}"])
    if not isinstance(live, dict):
        print(f"FATAL: unexpected response for ruleset {ruleset_id}", file=sys.stderr)
        sys.exit(1)

    desired = copy.deepcopy(live)
    MUTATORS[name](desired)
    stripped_live = strip_readonly(live)
    stripped_desired = strip_readonly(desired)

    if export_dir is not None:
        out_path = export_dir / f"{name}.json"
        write_snapshot(out_path, stripped_desired)
        print(f"exported {name}: {out_path}")
        return

    if stripped_live == stripped_desired:
        print(f"OK {name}: already in desired state")
        return

    print(f"{name}: changes needed")
    for line in summarize_changes(stripped_live, stripped_desired):
        print(line)

    if dry_run:
        print(f"  (dry-run) not writing {name}")
        return

    # LOCKOUT SAFETY: never write a ruleset that would strip the admin bypass.
    if not has_admin_bypass(stripped_desired):
        print(
            f"ERROR: {name} bypass_actors is missing the admin RepositoryRole "
            "(actor_id 5); aborting WITHOUT writing to avoid locking out the maintainer.",
            file=sys.stderr,
        )
        sys.exit(1)

    gh(["-X", "PUT", f"repos/{REPO}/rulesets/{ruleset_id}"], payload=stripped_desired)
    print(f"  PUT {name}: applied")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="print diffs; never PUT")
    parser.add_argument(
        "--export",
        metavar="DIR",
        help="write each ruleset's desired end-state JSON to DIR/{name}.json; no PUT",
    )
    args = parser.parse_args()

    export_dir: Path | None = None
    if args.export:
        export_dir = Path(args.export)
        export_dir.mkdir(parents=True, exist_ok=True)
    elif args.dry_run:
        print("DRY RUN — no writes\n")

    for ruleset_id, name in RULESETS.items():
        process_ruleset(ruleset_id, name, args.dry_run, export_dir)


if __name__ == "__main__":
    main()
