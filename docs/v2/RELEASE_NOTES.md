# V2 migration release notes

The typed convergence engine resolves weakest-link ties in canonical `AgentName`
order. Editing YAML key order no longer changes which failing agent is retried
first. Missing blocking verdicts still outrank explicit failures, and the last
nonempty verdict per agent still wins. `EXECUTION_ERROR` counts as a failure;
it never becomes a passing or inconclusive research verdict.

V2 agent content requires `schema_version: 2`. Structural keys move to typed
code, and missing files or invalid content fail explicitly. The migration keeps
a separate legacy default file until the final CLI switch; existing v0.3.x dish
configuration is not automatically converted by the v2 loader.
