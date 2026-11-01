CREATE VIEW blocking_verdict_events AS
SELECT e.id,e.cell_id,c.colony,c.level,e.agent,e.iteration,e.created_at,
       json_extract(e.payload,'$.verdict') AS verdict
FROM events e JOIN cells c ON c.id=e.cell_id
WHERE e.event_type='verdict_issued' AND (
(e.agent='investigator' AND json_extract(e.payload,'$.verdict') NOT IN ('EVIDENCE_SUFFICIENT','EVIDENCE_CONTRADICTS'))
 OR (e.agent='freshness_checker' AND json_extract(e.payload,'$.verdict') NOT IN ('EVIDENCE_CURRENT','STALE_BUT_HOLDS'))
 OR (e.agent='dependency_auditor' AND json_extract(e.payload,'$.verdict') NOT IN ('DEPENDENCIES_CLEAN'))
 OR (e.agent='skeptic' AND json_extract(e.payload,'$.verdict') NOT IN ('ARGUMENT_WITHSTANDS_CRITIQUE'))
 OR (e.agent='champion' AND json_extract(e.payload,'$.verdict') NOT IN ('STRONG_CASE','DEFENSIBLE_WITH_CAVEATS'))
 OR (e.agent='pragmatist' AND json_extract(e.payload,'$.verdict') NOT IN ('PRODUCTION_READY','DIRECTIONALLY_CORRECT'))
 OR (e.agent='triage' AND json_extract(e.payload,'$.verdict') NOT IN ('HIGH_VALUE','MODERATE_VALUE'))
);
CREATE VIEW blocking_verdict_patterns AS
SELECT colony,level,agent,COUNT(*) AS blocked_verdicts,
       COUNT(DISTINCT cell_id) AS blocked_cells,
       json_group_array(DISTINCT cell_id) AS cell_ids,
       MAX(created_at) AS last_blocked_at
FROM blocking_verdict_events GROUP BY colony,level,agent;
CREATE VIEW cell_velocity AS
SELECT c.id AS cell_id,c.colony,c.level,MIN(e.created_at) AS first_event_at,
       MAX(e.created_at) AS last_event_at,
       MIN(CASE WHEN e.event_type='convergence_checked' AND json_extract(e.payload,'$.converged')=1
                THEN e.created_at END) AS converged_at,
       (julianday(MIN(CASE WHEN e.event_type='convergence_checked' AND json_extract(e.payload,'$.converged')=1
                          THEN e.created_at END))-julianday(MIN(e.created_at)))*86400.0 AS seconds_to_convergence
FROM cells c JOIN events e ON e.cell_id=c.id GROUP BY c.id;
CREATE VIEW stalled_cells AS
SELECT cell_id,colony,level,first_event_at,last_event_at FROM cell_velocity
WHERE converged_at IS NULL;
CREATE VIEW colony_convergence_rates AS
SELECT c.colony,COUNT(*) AS total_cells,
       SUM(CASE WHEN v.converged_at IS NOT NULL THEN 1 ELSE 0 END) AS converged_cells,
       1.0*SUM(CASE WHEN v.converged_at IS NOT NULL THEN 1 ELSE 0 END)/COUNT(*) AS convergence_rate
FROM cells c LEFT JOIN cell_velocity v ON v.cell_id=c.id
WHERE EXISTS (SELECT 1 FROM events e JOIN cells active ON active.id=e.cell_id WHERE active.colony=c.colony)
GROUP BY c.colony;
