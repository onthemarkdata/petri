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
Jaeger 2.22.0 through `/api/v3/traces/<trace_id>`. The Jaeger UI screenshot remains
pending; API verification does not complete that visual check.

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
