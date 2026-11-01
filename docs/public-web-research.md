# Public-web research tools

Direct provider models can retrieve public sources through Petri's host tools.
Enable them explicitly in the dish's `petri.yaml`:

```yaml
research:
  public_web: true
```

The default is `false`. When enabled, research, critique, red-team, and evaluation
agents receive `web_search(query, limit=5)` and `web_fetch(url)`, including during
debates. Decomposition and cell leads do not receive these tools. The flag also works with CLI
harnesses that support Petri's function-tool bridge; their native CLI tool
allowlists remain separate. Constructing agents makes no network requests.

Search uses DuckDuckGo's unauthenticated HTML results and returns up to five URLs
and titles. Fetch retrieves HTML, plain text, or JSON from public HTTP/HTTPS
addresses on ports 80 or 443. It follows at most three redirects and validates
each destination. Each retrieval has a 15-second deadline and reads at most 256 KiB of response
data; returned text is capped at 12,000 characters. Scripts and styles are omitted from HTML text.
The source may still contain misleading claims or instructions: tool results are
marked as untrusted evidence, and agent instructions prohibit following source
instructions or inventing results when retrieval fails. An output validator rejects
citations whose final URLs were not returned by a successful fetch in that run.
The investigator must attempt `web_search` or `web_fetch` before returning an
assessment when public-web research is enabled. A failed retrieval counts as
an attempt, but supplies no evidence and may justify an inconclusive result.
Other roles may assess supplied evidence without a mandatory new search.

During a cell phase, successful searches append `search_executed` events with the
query, returned URLs, and count. Successful fetches append `source_reviewed`
events with the final URL, title, retrieval timestamp, a SHA-256 digest of the
bounded response, and an excerpt. This records retrieval provenance without
assigning an evidence-hierarchy level. The agent must still assess the source.
The same cell/round/node/agent/query or content identity reuses its existing event
on replay. Standalone runs without a cell context return sources without writing
cell events.

`PublicWebClient` accepts an HTTP transport and hostname resolver for offline
fixtures; `build_agents(..., research_client=...)` accepts a retrieval dependency.
Tests exercise the real agent tool loop, provenance, redirects, limits, private
address rejection, and a disabled configuration without network access. A manual
smoke check also searched for the Python sqlite3 documentation and retrieved
text from the [official Python page](https://docs.python.org/3/library/sqlite3.html).
No paid search service, new credential, or model call was used for that check.
