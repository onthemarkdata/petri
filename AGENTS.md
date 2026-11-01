# Petri

Petri is an open-source research orchestration framework. Shared configuration
and contributor instructions must work without a maintainer's private tools,
accounts, or repositories. Keep personal agent settings and hooks local.

Read [CONTRIBUTING.md](CONTRIBUTING.md) for the development workflow and
[ARCHITECTURE.md](ARCHITECTURE.md) for the current runtime. The
[v2 plan](docs/v2/MIGRATION_PLAN.md) describes planned work, not shipped behavior.

## Workflow

- Use GitHub Issues as the work tracker. Read the issue's acceptance criteria,
  dependencies, and linked PRs before implementing it. Avoid duplicating open PRs.
- Keep changes focused and preserve unrelated work. Development PRs target `dev`;
  `main` receives release promotions. Maintainers review and merge.
- Follow the user's current scope and order. Inspecting an issue does not
  authorize claiming it, posting comments, or starting the next issue.

## Checks

```sh
uv sync --locked --extra test
uv run pytest tests/ -q
uvx ruff check .
git diff --check
```

Run checks appropriate to the change and report failures or skipped checks.
Tests must stay offline: use `FakeProvider` or the relevant inference seam.
Do not run real model calls, `petri seed`, or `petri grow` as routine validation.
Preserve existing research data and append-only event logs.

## Configuration and documentation

- Keep personal Codex configuration in the ignored `.codex/` directory and local
  agent instructions in `AGENTS.override.md`. See [Codex setup](docs/codex-setup.md).
- Never commit credentials, local session data, or private working notes. Example
  environment files must contain placeholders only.
- Write documentation and user-facing text for contributors. Preserve facts,
  links, conditions, identifiers, and code behavior when editing prose.
