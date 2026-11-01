CREATE TABLE spans (
    span_id TEXT PRIMARY KEY NOT NULL,
    trace_id TEXT NOT NULL,
    parent_span_id TEXT,
    name TEXT NOT NULL,
    kind TEXT NOT NULL,
    start_time_ns INTEGER NOT NULL,
    end_time_ns INTEGER NOT NULL,
    status_code TEXT NOT NULL,
    status_message TEXT,
    attributes TEXT NOT NULL CHECK(json_valid(attributes)),
    resource TEXT NOT NULL CHECK(json_valid(resource))
);
CREATE INDEX spans_trace ON spans(trace_id, start_time_ns);
CREATE INDEX spans_start ON spans(start_time_ns);
CREATE INDEX spans_cell ON spans(json_extract(attributes, '$."petri.cell_id"'));
CREATE TABLE usage (
    span_id TEXT PRIMARY KEY NOT NULL REFERENCES spans(span_id) ON DELETE CASCADE,
    trace_id TEXT NOT NULL,
    cell_id TEXT,
    phase TEXT,
    agent_role TEXT,
    tokens_input INTEGER,
    tokens_output INTEGER,
    cost_usd REAL
);
CREATE INDEX usage_cell_phase ON usage(cell_id, phase);
