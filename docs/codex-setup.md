# Codex setup

Codex is optional for contributing to Petri. Read [AGENTS.md](../AGENTS.md) for
shared instructions and [CONTRIBUTING.md](../CONTRIBUTING.md) for development setup.
No maintainer-specific plugin, account, or private repository is required.

## Personal configuration

Keep project-specific Codex settings in `.codex/config.toml`. The entire
`.codex/` directory is ignored, including local launchers, hooks, credentials,
and session data. Keep personal agent instructions in the ignored root
`AGENTS.override.md`; include a reminder to read the shared `AGENTS.md`.

User-level settings and hooks can remain in your existing Codex home. Avoid
registering the same hook at both user and project scope. Review any new or
changed hooks through `/hooks` in the Codex CLI before relying on them. See the
[official hook documentation](https://learn.chatgpt.com/docs/hooks).

Local setup files are not copied by Git into a fresh clone or linked worktree.
Configure each checkout as needed, or use user-level settings for tools you want
available across projects. Keep secrets out of shared configuration.

## Before committing

Check `git status --short --untracked-files=all` and review the staged diff.
`.env` and `.env.*` are ignored except `.env.example`, `.env.sample`, and
`.env.template`. Those examples must contain placeholders, never real credentials.

Ignore rules do not remove files already tracked by Git or erase history. If a
credential was previously committed, removing the file alone is not sufficient;
revoke or rotate the credential as well.
