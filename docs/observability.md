# Observability

Petri records traces locally by default. `petri launch` shows a Traces panel,
span waterfall, and token and cost totals. Each cell's **View trace** button
opens its traces. Docker, Postgres, a collector, and a hosted account are never
required by Petri itself. The backend recipes below are optional external services.

Local spans and usage live beside domain data in
`.petri/petri-dishes/<dish_id>/petri.sqlite`. Dashboard reads use short-lived,
read-only SQLite connections. No external export is configured by default.

## Token and cost visibility

Usage rows contain measurements reported by the inference harness. Spans without
usage attributes create no usage row. Missing measurements and unconfigured prices
appear as **n/a**; a dry run does not invent token use. A known zero remains zero.
An aggregate cost remains unknown if any contributing usage row is unpriced.

Rates are optional and keyed by exact model ID in `petri.yaml`:

```yaml
prices:
  example-model-id:
    input_per_million: 1.0
    output_per_million: 2.0
```

These are example USD rates, not a price quote. Replace them with rates for your
model. Petri multiplies each reported token count by its configured rate and divides
by one million. Cache discounts and other billing rules are not modeled, so the
result is an estimate rather than a provider invoice.

## Configuration reference

Set `telemetry` in `.petri/defaults/petri.yaml`. This complete reference matches
`TelemetryConfig`; omitted fields take the values shown here.

<!-- telemetry-config-reference -->
```yaml
telemetry:
  enabled: true
  capture_content: false
  capture_limits: null
  redact_env_vars: []
  otlp:
    traces_endpoint: null
    logs_endpoint: null
    headers: {}
    protocol: http/protobuf
  local:
    enabled: true
    retention_days: 30
  sample_ratio: 1.0
  service_name: petri
```

- `enabled: false` disables tracing. `local.enabled: false` disables local storage
  while allowing an explicitly configured OTLP trace exporter.
- `traces_endpoint` is the full OTLP/HTTP trace URL, including `/v1/traces`.
  `logs_endpoint` is a reserved configuration field; Petri does not export logs.
- `headers` is a mapping of header names to values for external trace export.
- `protocol` accepts only `http/protobuf`. Port 4317 is conventionally gRPC and
  cannot be used as the HTTP exporter endpoint.
- `retention_days` must be a positive integer. On `petri launch`, local retention
  removes telemetry older than that age; domain events, cells, and edges are kept.
- `sample_ratio` ranges from 0 to 1. Root traces use that ratio; child spans follow
  their parent's sampling decision. The default records every root trace.
- `service_name` is a nonempty string. Resource attributes also include the installed
  `service.version` and explicit `petri.dish_id`.

Two environment variables override the corresponding YAML settings:

| Variable | Behavior |
|---|---|
| `OTEL_EXPORTER_OTLP_ENDPOINT` | Base URL; Petri appends `/v1/traces` and `/v1/logs` to the respective configuration fields. Supply the base, not the complete trace URL. |
| `OTEL_EXPORTER_OTLP_HEADERS` | Replaces the header mapping with comma-separated `key=value` pairs. Keys and values are URL-decoded; an empty value clears the mapping. |

For example, `OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4318` selects
`http://127.0.0.1:4318/v1/traces`. Signal-specific OpenTelemetry endpoint variables
are not Petri configuration overrides. Neither environment variable overrides
`telemetry.enabled: false`.

## Opt-in diagnostic content

`telemetry.capture_content: true` enables content and model-request parameter
capture in the existing Pydantic AI instrumentation. The default is `false`.
CLI and DBOS initialization use the current dish's validated setting; changing
the dish or configuration selects the corresponding provider and instrumentation.
`telemetry.enabled: false` still disables all tracing.

With capture enabled, native agent, model-request, and `execute_tool` spans can
retain prompts, instructions, returned model text, tool arguments and results,
and the model settings exposed by the pinned instrumentation. Model-request
usage uses `gen_ai.usage.*`; cumulative agent usage uses
`gen_ai.aggregated_usage.*`. Sum request measurements, not both sets. These native
attributes do not create extra Petri usage rows. With capture off, message shapes
and tool identities may remain, but private text is excluded by the instrumentation.
Binary content remains excluded.

Registered `web_search` and `web_fetch` tools already emit native tool spans.
Their results are captured when the tools finish, independently of a later model
request. Offline tests retain the local tool result after cancellation before it
is sent to either inference transport. They also verify workflow → DBOS step →
agent → tool ancestry, invalid arguments, HTTP failures, and inference retries
without duplicate tool execution. Invalid arguments produce an error tool span
without executing retrieval. Public-web HTTP failures return an error object, so
the native tool span remains `UNSET`; the existing client exposes the exception
class rather than the numeric HTTP status. The pinned native instrumentation also
leaves a cancelled model span `UNSET`, without final output or usage. Neither
status should be interpreted as proof of completion.

Capture describes what this path received, within its bounds. It cannot recover
hidden reasoning, omitted provider output, or Pi subprocess tool internals. Pi
host function tools are visible; its internal tool activity is not a separate
Petri tool-span tree. Pi rejects unsupported per-request model settings. Internal
provider retry attempts may share a model-request span; their intermediate errors
are not necessarily emitted. This is diagnostic span data, not a full transcript.

### Redaction before export

One processor sanitizes completed spans before either the SQLite or OTLP batch
processor receives them. It applies the same policy to attributes (including
nested and JSON-encoded values, including repeatedly encoded string scalars),
span names, exception and status text, events,
resource attributes, instrumentation scope, and link attributes. SQLite retains
sanitized events, with their timestamps, in `petri.telemetry.events` inside the
existing attributes column. OTLP retains native events. No domain-event rows are
rewritten.

The policy replaces recognized credential values with `[REDACTED]`: authorization
and proxy authorization, API keys, passwords, tokens, secrets, private keys,
credentials, cookies, HTTP URL user information, and Bearer/Basic authentication
material. It also handles these credential assignments in ordinary text and URL
query strings. Public evidence text, source URLs without credentials, IDs,
timestamps, usage, provider `gen_ai.response.id`, and tool-call correlation remain
available unless a capture
bound applies.

Export-header values are protected automatically. For secrets that may appear
without a recognizable field name, list only the environment variable **names**
to protect:

```yaml
telemetry:
  capture_content: true
  redact_env_vars:
    - MY_PROVIDER_API_KEY
```

Supply the values through your protected runtime environment, never public YAML.
Only those named variables are read; Petri does not collect the environment.
Embedding callers can pass `protected_secret_values` to `init_telemetry` instead.
Protected values and their URL-encoded/decoded forms are scrubbed. Provider reuse
adds newly supplied values to its redactor, retaining protection for older values
until shutdown. Secret values are not included in provider cache fingerprints or
redaction warnings. This policy cannot detect arbitrary unknown secrets or every
encoding. Capture remains an explicit choice for material authorized for tracing.

Span redaction does **not** scrub every artifact. Existing bounded `FailureDetail`
and `model_failure_detail` values, domain events (including retrieved evidence),
CLI output, and other logs have no general redactor in this change. They require
separate review before handling secret-bearing research. This change leaves
research inputs, source/verdict validation, and append-only events unchanged.

### Bounds and delivery limits

With `capture_limits: null` (the default), each string retains at most 65,536 characters plus a
`[TRUNCATED]` marker. Nested collections retain at most 128 entries per collection
and 32 levels of nesting. `petri.telemetry.bounded_values` reports values or
entries affected by these limits. Text cut inside a JSON attribute is no longer
valid JSON; the marker and count identify that omission. `petri.telemetry.redacted`
marks spans whose values were scrubbed. For credential assignments embedded in
escaped text, an internal quote can make the value's boundary ambiguous. The
redactor conservatively omits the remaining tail, including any following public
text, and adds `[TRUNCATED: escaped credential tail]` with a bounded-value count.
This does not affect the inference input.

For a finite, larger diagnostic envelope, explicitly enable content capture and
select a policy. An empty `capture_limits: {}` selects all ceilings below; each
field can be reduced to a strict positive integer. Booleans, strings, zero,
negative values, unknown fields and values above these ceilings are rejected.
A policy without `capture_content: true` is rejected before provider setup.

<!-- telemetry-capture-limits-reference -->
```yaml
telemetry:
  capture_content: true
  capture_limits:
    max_text_chars: 1048576
    max_collection_items: 512
    max_depth: 64
    max_nodes: 16384
    max_input_bytes: 8388608
    max_span_bytes: 4194304
```

| Field | Unit and scope |
|---|---|
| `max_text_chars` | Unicode characters per sanitized string, including serialized JSON framing; a plain-text truncation marker is additional |
| `max_collection_items` | Retained entries per collection |
| `max_depth` | Redactor recursion depth, starting at zero; decoded JSON adds a level |
| `max_nodes` | Visited containers, scalars and keys per span; decoded layers count again |
| `max_input_bytes` | Cumulative UTF-8 bytes inspected as strings or keys per span; decoded layers count again |
| `max_span_bytes` | UTF-8 bytes of the sanitized span envelope serialized as ASCII-escaped JSON, including identities, resource/scope, events, links, markers and framing |

All six limits apply together. The serialized envelope is a conservative JSON
accounting representation, not a sink schema, OTLP wire-size limit or RSS bound.
The processor checks the input budget before decoding JSON and rejects broad or
deep JSON before allocating its decoded tree. Aggregate overflow drops the whole
span before either owned sink and uses the existing type-only redaction warning
(`CaptureBudgetExceeded`). The warning is emitted once per processor, so it is
not an exact drop count. Missing spans still fail strict completeness checks.
Collection and recursion omissions retain `petri.telemetry.bounded_values`.
A structured value exceeding the text limit becomes a parseable JSON string
`"[TRUNCATED: text limit]"`; plain text retains the existing prefix and marker.
Neither form is a complete capture. Redaction happens before truncation; the
processor never exports an unsanitized prefix to avoid work.

A policy belongs to one provider. Changed policies select distinct providers;
identical dish/path/configuration reuses its provider and can extend runtime
secret protection. No policy changes inference inputs, model settings or usage.
These ceilings define a testable capture envelope, not completeness for every
model context. Capture truncation does not establish inference-input truncation.
Earlier failed captures remain failed; exact content and identity comparisons
are still required for new captures.

Providers with a policy use queues of 16 spans and batches of at most 4 at each
owned sink; providers without a policy keep their existing batching defaults.
The larger 4 MiB serialized-span envelope motivates this bounded queue size.
For two sinks, `2 × (16 + 4) × 4 MiB = 160 MiB` estimates serialized queued/batch
volume. It excludes Python objects, original histories, decoding/redaction
copies, protobuf/compression buffers and exporter/runtime overhead. It is not a
process-memory ceiling. Measure synthetic peak RSS and allocations with the
intended envelope and sink concurrency, then reserve host headroom before live
use. Smaller queues can drop bursts; they add no delivery guarantee. Existing
SDK limits and drop safeguards remain in force independently of this policy.

The existing OpenTelemetry SDK limits also apply (normally 128 span attributes,
128 events, 128 links, and 128 attributes per event or link). SDK count drops are
exposed through `petri.telemetry.dropped_attributes`, `dropped_events`,
`dropped_links`, per-event dropped-attribute counters, and
`dropped_link_attributes`, using the same prefix. If an explicit SDK attribute
length limit is configured, a retained value at that limit is conservatively
replaced by `[TRUNCATED: SDK length limit]` and counted as bounded: the processor
cannot tell whether the SDK cut a secret in half. Existing retrieval bounds
remain: public-web fetches keep at most 256 KiB of response bytes and 12,000
characters of extracted text, with the tool result's `truncated` flag. Earlier
provider or harness omissions cannot be reconstructed from spans.

Both exporters use in-memory batching. SQLite waits at most 100 ms for a lock,
drops a failed batch, and warns once per exporter. OTLP export is best effort;
there is no durable spool or outage-replay guarantee. Forced termination can lose
buffered spans, and sampling can omit entire traces. A redaction failure drops
the affected span and warns with only the exception type. A successful flush or
HTTP response does not prove backend storage; query the configured backend to
verify stored span IDs and payloads. This change does not establish request
propagation or a link between Petri workflow traces and gateway traces.

## Optional external backend: Jaeger

The [Jaeger all-in-one recipe](https://www.jaegertracing.io/docs/2.22/getting-started/)
provides a collector and trace UI with transient in-memory storage. Run it separately:

```sh
docker run --rm --name petri-jaeger \
  -p 127.0.0.1:16686:16686 \
  -p 127.0.0.1:4318:4318 \
  cr.jaegertracing.io/jaegertracing/jaeger:2.22.0
```

In another terminal, from an initialized project with seeded work:

```sh
OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4318 petri grow --dry-run
```

The trace endpoint is `http://127.0.0.1:4318/v1/traces`; this local recipe needs no
authentication header. Open `http://127.0.0.1:16686`, choose service **petri**, and
search for the command trace. Dry runs exercise command tracing without model
inference. An actual grow run also records instrumented agent activity.

Verified on November 1, 2026: an actual `petri grow --dry-run` selected one seeded
cell, exported its `petri.grow` span, and the matching trace ID was read back from
Jaeger 2.22.0 through `/api/v3/traces/<trace_id>`. The same trace was opened in the
Jaeger UI: its timeline showed one 64.3 ms `petri.grow` span with the command and
dish attributes. This verifies command export; the dry run contains no model
or workflow child spans.

## Optional external backend: SigNoz

Community verification wanted. Install a separate SigNoz instance using its
[installation guide](https://signoz.io/docs/install/), then configure Petri to use
the collector's HTTP port:

```sh
OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4318 petri grow --dry-run
```

For a collector on the same machine, the full trace endpoint is
`http://127.0.0.1:4318/v1/traces`. Self-hosted SigNoz needs no ingestion key by
default. A proxy may require its own authentication headers. See SigNoz's
[self-hosted ingestion reference](https://signoz.io/docs/ingestion/self-hosted/overview/)
for deployment addresses and authentication conditions.

Jaeger and SigNoz cannot both bind the same local port; run one recipe at a time
or change the host-side port and Petri's endpoint together.

## Optional external backend: Langfuse

Community verification wanted. With a separately installed local Langfuse instance,
use `http://127.0.0.1:3000/api/public/otel` as the base URL. Petri appends
`/v1/traces`, producing `http://127.0.0.1:3000/api/public/otel/v1/traces`.

Langfuse requires HTTP Basic authentication using the public key as the username
and secret key as the password. Replace the placeholder below with the Base64
encoding of `public-key:secret-key`, supplied through your credential environment:

```sh
export OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:3000/api/public/otel
export OTEL_EXPORTER_OTLP_HEADERS='Authorization=Basic <BASE64_PUBLIC_KEY_COLON_SECRET_KEY>,x-langfuse-ingestion-version=4'
petri grow --dry-run
```

The ingestion version header selects real-time ingestion for Langfuse v4. The
[Langfuse OpenTelemetry reference](https://langfuse.com/integrations/native/opentelemetry)
also lists hosted region endpoints. This recipe is external and optional;
Langfuse's deployment dependencies do not become Petri requirements.

## Domain span attributes

The following table is shared with [the architecture reference](ARCHITECTURE-V2.md).
Attributes are conditional; missing values are omitted rather than guessed.

| Attribute | Type | Present when |
|---|---|---|
| `petri.command` | string | Root command: `seed`, `grow`, `scan`, `feed`, or `re-decompose` |
| `petri.dish_id` | string | Command roots and cell/phase spans |
| `petri.colony_id` | string | Cell context is supplied; full composite colony ID |
| `petri.cell_id` | string | Cell context is supplied; `build_cell_key` output |
| `petri.level` | integer | Cell context is supplied |
| `petri.seq` | integer | Cell context is supplied |
| `petri.phase` | string | A phase span is open |
| `petri.agent_role` | string | A specialist is supplied; canonical `AgentName` |
| `petri.agent_blocking` | boolean | A specialist is supplied; true for blocking and conditional roles |
| `petri.verdict` | string | A validated verdict or mechanical convergence result is available |
| `petri.iteration` | integer | A phase span is open; dispatcher-owned iteration |
| `petri.weakest_link` | string | A weakest link is known |
| `petri.tokens_input` | integer | RunUsage includes measured token usage |
| `petri.tokens_output` | integer | RunUsage includes measured token usage |
| `petri.cost_usd` | number | All requests have measured usage and the exact model has configured prices |
| `petri.outcome` | string | A mechanical pipeline outcome is available |

## Troubleshooting

1. Check `telemetry.enabled`, `telemetry.local.enabled`, and the active project's
   `.petri/defaults/petri.yaml`. A local-enabled provider needs the CLI bootstrap's
   explicit dish path; importing `petri.telemetry` alone creates no files.
2. Check sampling. With `sample_ratio: 0`, unsampled roots produce no stored spans.
   Already sampled parents retain their sampling decision.
3. Check retention. `petri launch` prunes old telemetry using `retention_days`.
   Retention cannot recover spans that were never sampled or exported.
4. For OTLP, check that the configured HTTP endpoint is reachable and has the
   correct path. A base environment endpoint gets `/v1/traces` appended; the YAML
   `traces_endpoint` must already include it. Inspect collector logs for rejected
   authentication or payloads.
5. A local exporter waits at most 100 ms for a database lock. It drops a failed batch
   and logs one warning per exporter rather than blocking research. Check directory
   permissions and competing writers. Each batch uses its own short transaction;
   there is no transaction held between batches.
6. Normal shutdown flushes buffered spans. Forced termination can lose spans still
   in the processor's memory. Domain events are stored separately and do not depend
   on span export.
7. If tokens or costs show n/a, check that the harness reports usage and that an
   exact model-ID entry exists in `prices`. Model-generated usage claims do not
   count as measurements.

The trace API is read-only: `/api/traces`, `/api/traces/<trace_id>`, and `/api/costs`.
A missing or older dish with no telemetry returns an empty list or empty usage
rollup. An unknown trace ID returns 404. Only v2 per-dish databases are enumerated;
legacy flat JSONL dishes and backup/export directories are ignored.


## Live progress and reconnects

`/api/stream` tails committed domain events with an independent integer rowid
watermark for each dish. Its SSE event ID encodes the watermark map as JSON.
Reconnect with the last SSE ID in the `Last-Event-ID` header or `cursor` query
parameter. Timestamp ties do not affect replay order.

`agent_progress` is a transient, in-process channel. The stream adapter can fan
out text deltas to connected clients, and a slow client retains the latest 100
progress messages. Detached CLI processes remain visible through persisted domain
events, but their live text does not cross process boundaries. Production agent
stream-handler wiring and cross-process live text are separate work; no detached
CLI text coverage is claimed by this adapter.

## Analytics

The domain database includes four research-health views. They use domain event
history, so telemetry sampling does not affect them. Replace `DISH` in these
examples with a v2 dish directory.

- `blocking_verdict_patterns` counts non-passing verdicts from blocking or
  conditional agents by colony, level, and agent. Advisory agents are excluded.
  Counts include historical verdicts, not only each cell's latest verdict.

  ```sh
  sqlite3 .petri/petri-dishes/DISH/petri.sqlite 'SELECT * FROM blocking_verdict_patterns ORDER BY blocked_verdicts DESC;'
  ```

- `cell_velocity` reports each cell's first and latest event, first positive
  convergence check, and elapsed seconds to that check. Cells with no events are
  absent; unconverged cells have null convergence time.

  ```sh
  sqlite3 .petri/petri-dishes/DISH/petri.sqlite 'SELECT cell_id,seconds_to_convergence FROM cell_velocity;'
  ```

- `stalled_cells` lists unconverged cells with event history. The wrapper's default
  inactivity window is 24 hours; use an explicit cutoff in ad-hoc SQL.

  ```sh
  sqlite3 .petri/petri-dishes/DISH/petri.sqlite "SELECT * FROM stalled_cells WHERE julianday(last_event_at)<=julianday('now','-24 hours');"
  ```

- `colony_convergence_rates` divides cells with a positive convergence check by all
  cells in that colony. Colonies with no events are absent.

  ```sh
  sqlite3 .petri/petri-dishes/DISH/petri.sqlite 'SELECT * FROM colony_convergence_rates;'
  ```

Typed read-only wrappers live in `petri.query.analytics`. These are historical
signals; a past convergence check does not prove that a reopened cell is currently
validated.

Actual research adapters emit domain phase spans around each agent assessment,
debate, and mechanical convergence check. Assessment usage is the difference
between the measured run counters before and after that call; replayed cached
verdicts add no usage. This keeps cell totals from counting earlier calls again.
Failed calls retain measured usage and an error outcome. Debate usage comes from
its own run counter, and convergence records its iteration and mechanical outcome.
