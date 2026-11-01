CREATE TABLE redecomposition_proposals (
    id TEXT PRIMARY KEY NOT NULL,
    parent_cell_id TEXT NOT NULL REFERENCES cells(id),
    generation INTEGER NOT NULL CHECK(generation > 0),
    state TEXT NOT NULL CHECK(state IN ('reserved','proposed','approved','rejected','halted')),
    capacity INTEGER NOT NULL CHECK(capacity >= 0),
    trigger_json TEXT NOT NULL CHECK(json_valid(trigger_json)),
    output_json TEXT CHECK(output_json IS NULL OR json_valid(output_json)),
    child_ids TEXT NOT NULL DEFAULT '[]' CHECK(json_valid(child_ids)),
    reason TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    UNIQUE(parent_cell_id,generation)
);
