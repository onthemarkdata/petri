ALTER TABLE cells ADD COLUMN evidence_md TEXT NOT NULL DEFAULT '';
ALTER TABLE cells ADD COLUMN summary_md TEXT NOT NULL DEFAULT '';
UPDATE cells SET evidence_md=COALESCE((
    SELECT group_concat(summary, char(10)||char(10)) FROM (
        SELECT json_extract(payload,'$.summary') AS summary FROM events
        WHERE cell_id=cells.id AND event_type='evidence_appended'
        AND COALESCE(json_extract(payload,'$.summary'),'')<>'' ORDER BY rowid
    )
),''), summary_md=COALESCE((
    SELECT json_extract(payload,'$.summary') FROM events
    WHERE cell_id=cells.id AND event_type='verdict_issued' AND agent='cell_lead'
    ORDER BY rowid DESC LIMIT 1
),'');
CREATE TRIGGER cells_content_from_event AFTER INSERT ON events
BEGIN
    UPDATE cells SET evidence_md=CASE
        WHEN NEW.event_type='evidence_appended' AND COALESCE(json_extract(NEW.payload,'$.summary'),'')<>''
        THEN evidence_md || CASE WHEN evidence_md='' THEN '' ELSE char(10)||char(10) END || json_extract(NEW.payload,'$.summary')
        ELSE evidence_md END,
        summary_md=CASE WHEN NEW.event_type='verdict_issued' AND NEW.agent='cell_lead'
        THEN COALESCE(json_extract(NEW.payload,'$.summary'),'') ELSE summary_md END
    WHERE id=NEW.cell_id;
END;
CREATE TRIGGER cells_content_on_insert AFTER INSERT ON cells
BEGIN
    UPDATE cells SET evidence_md=COALESCE((
        SELECT group_concat(summary, char(10)||char(10)) FROM (
            SELECT json_extract(payload,'$.summary') AS summary FROM events
            WHERE cell_id=NEW.id AND event_type='evidence_appended'
            AND COALESCE(json_extract(payload,'$.summary'),'')<>'' ORDER BY rowid
        )
    ),''), summary_md=COALESCE((
        SELECT json_extract(payload,'$.summary') FROM events
        WHERE cell_id=NEW.id AND event_type='verdict_issued' AND agent='cell_lead'
        ORDER BY rowid DESC LIMIT 1
    ),'') WHERE id=NEW.id;
END;
