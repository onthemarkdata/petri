CREATE TABLE analyst_reports (
    id TEXT PRIMARY KEY NOT NULL,
    dish_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    report_json TEXT NOT NULL CHECK(json_valid(report_json))
);
CREATE INDEX analyst_reports_created ON analyst_reports(created_at);
CREATE TRIGGER analyst_reports_no_update BEFORE UPDATE ON analyst_reports
BEGIN SELECT RAISE(ABORT,'analyst reports are append-only'); END;
CREATE TRIGGER analyst_reports_no_delete BEFORE DELETE ON analyst_reports
BEGIN SELECT RAISE(ABORT,'analyst reports are append-only'); END;
