# V1 retirement

DBOS is the only execution backend. v0.3.7 is the last release for running v1
dishes. This release does not migrate, rewrite, or remove old research data.
Initialize a fresh dish for v2; keep old exports and directories for text search.

The removed modules are the v1 processor, grow loop, adaptive load balancer,
file-backed queue, and propagation engine. Domain-only impact reporting and
propagation flags now live in `petri.lifecycle.propagation`. The legacy Claude
provider is replaced by the first-party Pydantic AI model and typed synchronous
seed bridge. Adapter exports no longer contain the queue-update skill or the
obsolete `analyze` command.

Tests that existed only to exercise deleted implementations were removed with
those implementations. Their behavioral replacements remain:

| Retired coverage | V2 coverage |
|---|---|
| Queue transitions and invalid edges | `test_pipeline_conformance.py` preserves the frozen transition oracle and checks every typed node or End disposition, including invalid edges |
| Processor phases, evidence, and sources | `test_research_phases.py`, `test_decision_phases.py`, and source/tool conformance tests exercise typed agents and committed SQLite evidence |
| Busy-wait grow loop and adaptive workers | `test_durable_scheduler.py` and `test_dbos_backend.py` check dependency gates, deduplication, concurrency, rate limits, and typed failures |
| Stop sentinels and processor callbacks | Cancellation tests and `test_grow_resume.py` cover graceful/force stop, public-client resume, backend polling, real SIGKILL, and Ctrl+C |
| Legacy provider transport and parsing | First-party Claude Code and pi model/transport tests cover typed output, retries, streaming, and resource cleanup |
| File-backed reopen and propagation | Durable round tests cover r+1 allocation; propagation and feed tests cover impact flags without dependent reopening |

`test_retire_v1.py` runs init, seed, grow, check, and stop with local fake inference
on a fresh dish. It verifies that no queue file appears and that a neighboring
legacy dish retains its bytes and modification times. It also checks imports for
retired modules and verifies that importing Petri does not require `fcntl`.

Configuration now returns a validated `AgentConfig` and is loaded afresh on each
call. Runtime overrides inherit packaged v2 values and are validated before use;
explicitly requested config files never fall back when missing. No import-time
configuration constants or loader cache remain.
