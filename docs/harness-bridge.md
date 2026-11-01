# Legacy provider bridge

`HarnessInferenceProvider` adapts Pydantic AI agents to the synchronous
`InferenceProvider` interface. Each instance owns one serialized worker loop, so
pi requests and subprocess cleanup stay on the loop that created them. Use the
provider as a context manager or call `close()` in a `finally` block. A finalizer
also closes abandoned instances. Progress callbacks run on the caller's thread
and report start, completion, or failure; token streaming is not implemented.

The live protocol differs from the frozen #26/#27 descriptions in three places:

- It has seven methods, including `summarize_evidence`.
- `assess_claim_substance` returns a dict with `is_substantive`, `reason`, and
  `suggested_rewrite`. `AssessmentResult` is a role verdict and cannot represent
  that result. The bridge validates a separate `SubstanceAssessment` and returns
  its dict to preserve existing callers.
- `decompose_claim` promises `DecompositionResult`, but the legacy implementation
  returns a `nodes` dict. The bridge validates a small premise schema and returns
  typed cells with temporary `pending-premises` identifiers. Those identifiers
  carry no dish identity; the decomposer must assign real cell identifiers before
  persistence. Models do not choose filesystem identifiers.

The bridge uses the shared prompt bodies without their legacy JSON suffixes.
Pydantic AI supplies the output schema and validation feedback. A malformed
Five Whys result consumes the configured output retries and then raises; only a
validated atomic result returns an empty list. Outputs over the requested limit
also trigger validation feedback rather than being silently truncated.
