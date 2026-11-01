# RFC: GitHub review feedback as structured re-entry

**Status: design-first proposal for #95. No implementation is scheduled by this RFC.**

Petri already carries `weakest_link` and `focused_directive` through its
convergence and execution paths. A reviewer should be able to challenge a
particular cell through a pull request, retain the source of that challenge,
and ask Petri to revisit it without restarting unrelated research.

This proposal defines identity, intake, and human decisions. It does not add
code, register a GitHub App, create pull requests, configure webhooks, or change
the dashboard. A later milestone must authorize those changes.

## Cell-to-PR identity

| Option | Benefit | Cost | Recommendation |
| --- | --- | --- | --- |
| Branch and PR per cell | Review, history, and re-entry have one cell owner | Many small branches and PRs; shared evidence needs explicit links | Default for cells selected for GitHub review |
| PR per colony | Fewer PRs; reviewers see the argument together | Comments need cell attribution and a merge cannot identify which cells passed | Optional presentation bundle, not the primary feedback identity |
| Labels alone | Easy to discover and filter PRs | Mutable labels cannot bind a comment to a claim revision | Discovery only |

Recommend one PR per reviewed cell, with a branch such as
`petri/cell/<cell-id>`. Exported artifacts and an explicit PR-body metadata block
identify the dish, colony, stable cell ID, claim revision digest, and exported
commit SHA. Store the repository ID, PR node ID, and this mapping locally after
a human confirms it. Branch names and labels are hints, not identity. Reject
ambiguous mappings; a colony PR must name the target cell for every directive.

Do not automatically publish private research. Mapping an existing PR and
publishing an export are separate actions requiring their own authorization.

## Proposed event contract

A future implementation would add `github_feedback_received` to `EventType`
and register a typed `GitHubFeedbackReceivedData` in `EVENT_DATA_MODELS`. The
existing event writer remains the sole append path. This RFC makes no registry
change. Proposed payload:

```json
{
  "version": 1,
  "repository_id": 123,
  "repository": "example/research",
  "pull_request_node_id": "PR_example",
  "pull_request_number": 7,
  "cell_id": "dish-colony-001-001",
  "claim_revision": "sha256:example",
  "head_sha": "example-head-sha",
  "source_kind": "review",
  "source_id": "review-42",
  "source_updated_at": "2026-11-01T00:00:00Z",
  "source_url": "https://github.com/example/research/pull/7#pullrequestreview-42",
  "reviewer": "reviewer-login",
  "review_state": "CHANGES_REQUESTED",
  "comment_ids": [101],
  "source_text": "The claimed threshold needs a primary measurement.",
  "focused_directive": "Test the threshold against a primary measurement.",
  "disposition": "needs_human",
  "supersedes_event_id": null
}
```

The eventual schema must constrain source kind, review state, and disposition;
validate repository/PR/cell identity; and retain the exact source text alongside
any human-approved directive. Deduplicate by repository ID, source kind, source
ID, and source update version. Re-reading the same source appends nothing.
Edits, dismissals, and deletions append a superseding observation; they never
rewrite earlier events. Keep intake cursors separate from domain events, and
commit the cursor only after all fetched observations have been journaled.

GitHub exposes reviews as a state, body, author, and commit, with associated
review comments. Fetch both the review and its comments rather than treating
the summary as the complete review. [GitHub review API](https://docs.github.com/en/rest/pulls/reviews).

## Extracting a directive

Recommend deterministic extraction followed by human approval, with no model
call required for intake:

1. An allowlisted reviewer's submitted `CHANGES_REQUESTED` review creates a
   candidate containing its nonempty body and associated inline comments. Keep
   comment IDs, links, file positions, and the reviewed commit attached.
2. Ordinary discussion and `COMMENTED` reviews remain observations unless a
   human explicitly selects text for re-entry. Approval supplies no directive.
   Empty, deleted, or dismissed feedback cannot create a new directive.
3. Present the candidate for one cell. A human may select or summarize its
   actionable request; preserve the source and the approved wording. Comment
   contents are research input, not commands to execute or configuration edits.
4. If the claim digest or reviewed head differs from the current export, mark
   the candidate stale and require reconciliation before requeueing.
5. Combine compatible requests with source attribution. Conflicting reviewer
   requests remain separate, unresolved candidates until a human chooses a
   directive or records a reconciliation. Neither majority vote nor the latest
   timestamp silently settles a disagreement.

Accepted feedback flags the cell and affected dependents for attention. It does
not start inference. Explicit human approval of a directive uses the same
re-decomposition approval boundary as #87 and the existing focused-directive
execution path.

## Merge and convergence

**GitHub approval never substitutes for mechanical convergence.** A review or
merge must not write `DONE`, fabricate specialist verdicts, clear blockers, or
declare a dependency satisfied.

Recommend **flag, do not requeue** as the default for all incoming feedback,
including merge. A merge observation records the merged SHA and makes relevant
dependents visible for reconsideration. A human can explicitly request their
requeue after inspecting the current graph. The normal dependency and
convergence checks still decide whether they can run. A closed, unmerged PR
remains an observation with no unblock effect. Merge after conflicting or stale
feedback does not discard those unresolved candidates.

```mermaid
sequenceDiagram
    actor Reviewer
    participant GitHub
    participant Intake as Local intake
    participant Events as Domain event log
    actor Maintainer
    participant Queue as Execution backend
    Reviewer->>GitHub: Submit changes-requested review
    Maintainer->>Intake: Fetch mapped PR feedback
    Intake->>GitHub: Read review, comments, and current head
    GitHub-->>Intake: Versioned source records
    Intake->>Events: Append github_feedback_received once
    Intake-->>Maintainer: Flag cell; show attributed candidate
    Maintainer->>Queue: Explicitly approve directive and requeue
    Queue->>Queue: Apply dependency and convergence checks
    Reviewer->>GitHub: Approve; PR is later merged
    Intake->>Events: Record merge; flag affected dependents
    Maintainer->>Queue: Explicitly request dependent requeue
    Queue->>Queue: Recheck mechanical gates
```

## Authentication and delivery

| Choice | Recommendation and rationale |
| --- | --- |
| Existing `gh` login | Local default. Use `gh api` under the user's selected host/account; do not print or copy its token into Petri files. |
| Explicit token | Opt-in for unattended local runs. Read from a credential provider/environment, request only the repository access needed, and keep it outside dish configuration and exports. |
| GitHub App | Defer until a shared deployment needs installation-level identity and permissions; it adds registration, key storage, token renewal, and operational ownership. |

The CLI supports token environment variables, and App installation tokens carry
the installation's repository access and permissions. These are different
operational choices, not interchangeable credentials. [CLI environment](https://cli.github.com/manual/gh_help_environment),
[App installation authentication](https://docs.github.com/en/apps/creating-github-apps/authenticating-with-a-github-app/authenticating-as-a-github-app-installation).

Recommend an explicit local fetch command first, with optional bounded polling
only while Petri is running. A local tool should not require a permanent public
receiver. Fetch only mapped PRs, paginate completely, use authenticated
conditional requests, retain ETags, and honor rate-limit and polling headers.
Stop visibly on authentication or mapping errors; do not advance the cursor
past unrecorded feedback.

GitHub recommends webhooks for continuous integrations and efficient polling
when webhooks are unavailable. Webhooks require a receiving server; adopt them
only with a later hosted deployment, signature verification, durable delivery
deduplication, and missed-delivery reconciliation. Both delivery modes must feed
the same event intake. [REST best practices](https://docs.github.com/en/rest/using-the-rest-api/best-practices-for-using-the-rest-api),
[webhooks](https://docs.github.com/en/webhooks/about-webhooks).

## Proposed follow-on issues

These are planning candidates, not scheduled work or newly created issues.

| Work | Rough size | Completion evidence |
| --- | --- | --- |
| Cell/PR mapping and export metadata | S | Ambiguous or stale mappings rejected; private exports remain opt-in |
| Typed feedback event and idempotent local fetch | M | Pagination, edited/dismissed feedback, crash recovery, and duplicate intake tests |
| Candidate review and explicit focused-directive requeue | M | Conflicting reviewers and stale claims require human resolution; approval alone cannot change convergence |
| Merge observations and dependent flags | S | Merge flags dependents once; explicit requeue retains all mechanical gates |
| Optional App/webhook receiver | M | Separate deployment decision; signatures, delivery replay, permission boundaries, and reconciliation tested |
