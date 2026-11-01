CREATE TABLE cells (
    id TEXT PRIMARY KEY NOT NULL,
    colony TEXT NOT NULL,
    level INTEGER NOT NULL CHECK(level >= 0),
    seq INTEGER NOT NULL CHECK(seq >= 0),
    claim_text TEXT NOT NULL,
    status TEXT NOT NULL,
    relation TEXT,
    is_atomic INTEGER NOT NULL DEFAULT 0 CHECK(is_atomic IN (0, 1)),
    bedrock_reason TEXT,
    created_at TEXT NOT NULL CHECK(created_at <> '')
);
CREATE INDEX cells_colony_level ON cells(colony, level, seq);
CREATE TABLE edges (
    from_cell TEXT NOT NULL REFERENCES cells(id),
    to_cell TEXT NOT NULL REFERENCES cells(id),
    edge_type TEXT NOT NULL,
    reason TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL CHECK(created_at <> ''),
    PRIMARY KEY(from_cell, to_cell, edge_type),
    CHECK(from_cell <> to_cell)
);
CREATE VIEW dependents AS
    SELECT to_cell AS cell_id, from_cell AS dependent_cell_id, edge_type, reason, created_at
    FROM edges;
CREATE TABLE events (
    id TEXT PRIMARY KEY NOT NULL,
    cell_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    agent TEXT NOT NULL,
    iteration INTEGER NOT NULL CHECK(iteration >= 0),
    payload TEXT NOT NULL CHECK(json_valid(payload)),
    created_at TEXT NOT NULL CHECK(created_at <> '')
);
CREATE INDEX events_cell ON events(cell_id);
CREATE INDEX events_type ON events(event_type);
CREATE TRIGGER events_no_update BEFORE UPDATE ON events
BEGIN
    SELECT RAISE(ABORT, 'events are append-only: UPDATE is forbidden');
END;
CREATE TRIGGER events_no_delete BEFORE DELETE ON events
BEGIN
    SELECT RAISE(ABORT, 'events are append-only: DELETE is forbidden');
END;
