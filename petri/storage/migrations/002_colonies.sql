CREATE TABLE colonies (
    id TEXT PRIMARY KEY NOT NULL,
    dish TEXT NOT NULL,
    center_cell_id TEXT NOT NULL,
    center_claim TEXT NOT NULL,
    clarifications TEXT NOT NULL DEFAULT '[]' CHECK(json_valid(clarifications)),
    created_at TEXT NOT NULL CHECK(created_at <> '')
);
