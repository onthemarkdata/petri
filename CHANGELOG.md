# Changelog

## Unreleased

- **Breaking:** DBOS is the only execution backend. The v1 processor, grow loop,
  propagation engine, adaptive load balancer, and file-backed queue are removed.
  `execution.backend: v1` fails with a pointer to the last v1 release, v0.3.7.
- Existing v0.x dishes are not migrated or rewritten. Keep them for text search
  or run them with v0.3.7; initialize a fresh dish for v2.
- Fresh dishes use validated v2 settings and per-dish SQLite data. Configuration
  is loaded at call time, without global constants or a loader cache.
- The Unix `fcntl` queue dependency is removed. Importing Petri no longer requires
  Unix file locking; this change alone does not certify every harness on Windows.
- Generated adapters use `launch` and typed pipeline transitions. The obsolete
  `analyze` command and queue-update skill are removed. The old Gemma default was
  already replaced by the model named in the validated packaged configuration.
