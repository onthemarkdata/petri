# Dish database transition

V2 seed stores cells, edges, colony metadata, and events in
`.petri/petri-dishes/<dish_id>/petri.sqlite`. Initialization creates this database
at the current schema version and creates an empty `exports/` directory.
`dish_dir` and `dish_db_path` resolve the location. `load_colony` assembles the
graph from rows and preserves outgoing cross-colony edges.

Seed reserves a colony name in a transaction before creating its rows. A second
run cannot overwrite it. Regeneration replaces only the current draft's rows;
its events remain append-only. An approved colony, or a draft referenced by
another colony, cannot be discarded. The checkpoint work adds interrupted-seed
resumption separately.

CLI check, graph, connect, stop events, propagation, and dashboard cell views now
use the domain store. Listing ignores legacy colony directories. Fixtures retain
legacy files only where tests still exercise the v1 engine or disposable index.
The temporary v1 execution adapter reads database rows for grow dry-run; actual
durable execution replaces the v1 file-based processor in the execution stack.
Dashboard event queries and launch indexing complete their transition in the
query-layer and index-retirement changes (#75–77).

Validation on 2026-11-01: the combined offline suite passed 1,172 tests; mypy
checked 81 modules and Ruff 0.15.21 passed. Two local tests then passed seed plus
grow dry-run against SQLite in 8.63 seconds, through pi and the direct Docker
Model Runner API with the locked Gemma Q4_0 model. These checks do not constitute
a full grow or crash-recovery run.
