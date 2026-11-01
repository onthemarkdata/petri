# Dependency intelligence

Petri computes graph metrics over the active `DishGraph`. Cross-colony references
count alongside references within a colony. The aim is to expose shared premises
and long dependency chains, building on [field report #11](field-reports.md#11--cross-colony-cell-search-highest-leverage)'s
finding that prose references to other colonies were missing from the graph.

An edge `A -> B` means A depends on B. The metrics use that direction:

| Metric | Formula |
|---|---|
| Importance | Number of distinct cells directly depending on the cell: its in-degree. |
| Cross-colony importance | Direct dependents whose colony differs from the target cell's colony. |
| Longest dependency chain | Longest incoming path to the cell plus longest outgoing path from it, measured in edges. |
| Transitive dependents | Number of distinct cells reachable by following incoming edges; shared paths count each cell once. |
| Fragility | Longest dependency-chain length plus transitive-dependent count. |
| Convergence point | A cell with direct dependents from at least two distinct colonies. |

Convergence points rank by the number of depending colonies, then importance,
then fragility, then level and cell ID. Importance and fragility rankings are also
available separately. These are topology measures, not probabilities that a claim
is true or reliable.

`petri.analysis.edge_intelligence` contains pure functions and the typed
`EdgeIntelligenceReport`. It does not load a model, run inference, schedule work,
or write a database. Reports serialize as JSON for downstream CLI, dashboard, and
scheduling consumers. Tombstoned edges and retired cells do not contribute.

## CLI and dashboard

`petri check --targets 5` prints the highest-ranked convergence points with their
importance and fragility. The default limit is five. `petri graph` marks those
points in text and DOT output and labels references across colonies. The dashboard
uses the same `/api/intelligence` report for its five validation targets;
`/api/edges` supplies typed edge records. Shared presentation settings define edge
labels and styles for the dashboard and DOT renderer.

### Fixture verification

Run these checks against a dish with two colonies whose cells share a dependency:

- [x] Compare `petri check --targets 5` with the first five IDs from
  `/api/intelligence`. The three-cell fixture returned the same target,
  `intelligence-alpha-001-001`, with importance 2 and fragility 3.
- [x] Inspect `petri graph --format dot`: the target has a double outline and
  convergence annotation; the cross-colony reference has a dashed blue edge.
- [x] Read `/api/edges` and validate every row as an `EdgeRecord`. The fixture
  contains one intra-colony edge and one cross-colony edge.
- [x] Verify both routes emit dish/outcome span attributes. Automated fixture
  checks also cover cell/colony filters and a schema-valid report.
- [ ] Open the dashboard and inspect the target panel and distinct edge styles
  in a browser. Browser rendering remains unverified in this environment because
  the browser tool could not create a tab.

The CLI and TestClient steps were executed on the local fixture. No inference or
live grow was needed. Browser appearance still requires the final visual check.
