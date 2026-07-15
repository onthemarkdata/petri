# Contributing to Petri

Thanks for your interest. Petri is a colony-based research orchestration framework, currently in
the thick of its **v2 migration**: re-platforming onto Pydantic AI, pydantic-graph, and DBOS while
keeping the zero-infrastructure, local-first design.

## Where to start

- **Vision:** [VISION.md](VISION.md). The durable, model-independent research runtime Petri is growing
  into, and the thinking behind it.
- **Roadmap:** [docs/v2/MIGRATION_PLAN.md](docs/v2/MIGRATION_PLAN.md). Seven milestones, each with a
  tracking epic issue and a set of scoped, independently landable issues.
- **Good first issues:** [filter the tracker](https://github.com/onthemarkdata/petri/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22).
  These are self-contained and need no prior context beyond the linked docs.
- **Architecture background:** [docs/ARCHITECTURE-V2.md](docs/ARCHITECTURE-V2.md) (the v2 decision
  record) and [docs/field-reports.md](docs/field-reports.md) (the real-world evidence behind the
  migration; issues cite these as `` `#N` `` field-report numbers).

## Branch model

| Branch | Purpose | PRs target it? |
|--------|---------|----------------|
| `dev`  | Integration branch; all v2 work lands here | **Yes, always target `dev`** |
| `main` | Released code only (what's on PyPI) | No, only release-promotion PRs from `dev` |

CI runs on every PR: tests on Python 3.11–3.14, ruff lint, and a package build. It also measures
coverage and runs an advisory mypy type-check.

## Development setup

```bash
git clone https://github.com/<you>/petri && cd petri
git checkout dev
uv sync --extra test        # creates .venv from the committed lockfile
uv run pytest tests/ -q     # 500+ tests, all offline; no API keys or claude CLI needed
uvx ruff check .            # lint (default rules)
uvx pre-commit install      # optional: runs ruff plus hygiene hooks on each commit
```

**Tests must stay offline.** The suite uses a dependency-injected `FakeProvider`
(see `tests/conftest.py`) instead of real LLM calls. If your change needs inference to test,
mock at the provider seam. Never call a network or a real CLI from tests.

## Claiming and working an issue

1. **Comment on the issue** ("I'd like to take this"), and the maintainer will assign it to you.
2. **Check its "Blocked by" relationships first.** If a blocking issue is open, start there
   instead, or pick something unblocked.
3. One issue per PR. The issue's **acceptance criteria checkboxes are the definition of done**, so
   your PR should let a reviewer tick every box.
4. Questions are welcome; comment on the issue anytime. Unclear scope is a bug in the issue, not
   in you.

## Label taxonomy

- `migration-v2`: part of the v2 migration backlog
- `epic`: a milestone tracking issue (has sub-issues)
- Area labels (`harness`, `agents`, `decomposer`, `durable-execution`, `observability`,
  `storage`, `dashboard`, `lifecycle`): which subsystem the issue touches
- `size:S` / `size:M` / `size:L`: rough effort estimate
- `spike`: timeboxed investigation; the deliverable is a written document, not code
- `rfc`: design-first; discussion expected before implementation
- `breaking-change`: observably changes behavior for existing users
- `good first issue`: newcomer-safe and self-contained

## Release process (maintainer)

1. When `dev` is ready to release, open a PR `dev` → `main` titled `Release X.Y.Z`; CI must be green.
2. Merge with **Rebase and merge** (`main` requires linear history).
3. Run `gh release create vX.Y.Z --target main --title "vX.Y.Z" --generate-notes`. The git tag sets
   the package version (via setuptools-scm), and the `release: published` event triggers the publish
   workflow, which ships to PyPI via trusted publishing.
4. Resync dev, since rebase-merge rewrites SHAs:
   ```bash
   git fetch origin && git checkout dev
   git reset --hard origin/main
   git push --force-with-lease origin dev   # relies on the maintainer admin bypass on the dev ruleset
   ```

Tip: a true fast-forward promotion of `dev` to `main` avoids the reset. Consider it if open
contributor PRs ever make the hard reset risky.

## Code of conduct

Be kind, be constructive, and assume good faith. Research tooling attracts people from many
backgrounds, so welcome them.
