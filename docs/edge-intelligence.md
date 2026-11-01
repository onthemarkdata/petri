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
