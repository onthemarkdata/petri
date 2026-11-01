# Fixture provenance

The committed cases are public synthetic regressions. They are not the private
research archive or a substitute for its ground truth.

- `hotdog-template.json` represents the five-generic-premise failure described in
  public issue #48. Its wording is synthetic, not a captured model response. It
  should fail counterargument coverage.
- `legacy-default-template.json` reproduces the five claim templates in
  `petri/decomposer.py` at public tag `v0.1.0`, instantiated with the synthetic
  root claim. It should fail counterargument coverage. These templates are
  historical behavior, not current decomposition output.
- The positive shapes in `claims.yaml` test code contracts through FunctionModel.
  Their counts are synthetic examples using the publicly documented 4–13 range;
  no measured distribution is inferred from them.

The actual 88-node, 128-edge graph, its 60 cross-layer edges, seven paired waves,
and the six principle/measurement documents remain private by the maintainer's decision. They are not included. Opt-in local
archive tests validate and search a separately supplied bundle; CI does not
substitute a synthetic graph and claim the archive checks passed. Public results
contain aggregate counts and retrieval scores only.


## Local validation record

On November 1, 2026, the opt-in private archive check verified 88 nodes, 128
edges, 60 cross-layer edges, an acyclic graph, seven same-claim colony pairs,
and six split documents. Lexical search over 81 linked-claim queries measured
precision 0.11227 and recall 0.33594; exact-title recall was 1.0. These are
retrieval measurements, not claims of semantic equivalence.

The explicitly enabled Dataset run against the exact local Gemma model failed
all three positive cases when output validation exhausted retries. It did not
produce a passing live reference case. The report preserved those failures;
scripted offline success is not presented as live model success.
