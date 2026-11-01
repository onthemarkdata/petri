DROP VIEW dependents;
ALTER TABLE edges RENAME TO previous_edges;
CREATE TABLE edges (
    from_cell TEXT NOT NULL REFERENCES cells(id),
    to_cell TEXT NOT NULL REFERENCES cells(id),
    edge_type TEXT NOT NULL,
    reason TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL CHECK(created_at <> ''),
    PRIMARY KEY(from_cell,to_cell,edge_type,created_at),
    CHECK(from_cell <> to_cell)
);
INSERT INTO edges SELECT * FROM previous_edges ORDER BY rowid;
DROP TABLE previous_edges;
CREATE TABLE edge_tombstones (
    from_cell TEXT NOT NULL,
    to_cell TEXT NOT NULL,
    edge_type TEXT NOT NULL,
    edge_created_at TEXT NOT NULL,
    reason TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL CHECK(created_at <> ''),
    PRIMARY KEY(from_cell,to_cell,edge_type,edge_created_at)
);
CREATE VIEW active_edges AS
SELECT e.* FROM edges e WHERE NOT EXISTS (
    SELECT 1 FROM edge_tombstones t WHERE t.from_cell=e.from_cell AND t.to_cell=e.to_cell
    AND t.edge_type=e.edge_type AND t.edge_created_at=e.created_at
);
CREATE VIEW dependents AS
SELECT to_cell AS cell_id,from_cell AS dependent_cell_id,edge_type,reason,created_at FROM active_edges;
CREATE TRIGGER edge_tombstones_no_update BEFORE UPDATE ON edge_tombstones
BEGIN
    SELECT RAISE(ABORT,'edge tombstones are append-only');
END;
CREATE TRIGGER edge_tombstones_no_delete BEFORE DELETE ON edge_tombstones
BEGIN
    SELECT RAISE(ABORT,'edge tombstones are append-only');
END;
CREATE TABLE cell_tombstones (
    cell_id TEXT PRIMARY KEY NOT NULL,
    reason TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL CHECK(created_at <> '')
);
CREATE VIEW active_cells AS
SELECT c.* FROM cells c WHERE NOT EXISTS (SELECT 1 FROM cell_tombstones t WHERE t.cell_id=c.id);
CREATE TRIGGER cell_tombstones_no_update BEFORE UPDATE ON cell_tombstones
BEGIN
    SELECT RAISE(ABORT,'cell tombstones are append-only');
END;
CREATE TRIGGER cell_tombstones_no_delete BEFORE DELETE ON cell_tombstones
BEGIN
    SELECT RAISE(ABORT,'cell tombstones are append-only');
END;
