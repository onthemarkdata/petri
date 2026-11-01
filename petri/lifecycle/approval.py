"""Persist approval and eligible child rounds in one domain transaction."""

from contextlib import closing
from pathlib import Path

from petri.execution.scheduler import prepare_rounds
from petri.lifecycle.config import LifecycleConfig
from petri.lifecycle.proposals import Proposal, _event, _proposal
from petri.storage import database


def list_proposals(db_path: Path) -> list[Proposal]:
    with closing(database.connect(db_path)) as conn:
        return [_proposal(row) for row in conn.execute(
            "SELECT * FROM redecomposition_proposals ORDER BY rowid"
        )]


def decide_proposal(db_path: Path, proposal_id: str, *, approve: bool, config: LifecycleConfig) -> tuple[Proposal, bool]:
    """Commit a decision once; a resumed scheduler can admit approved children."""
    prepare_rounds(db_path)
    with database.write_connection(db_path) as conn:
        row = conn.execute("SELECT * FROM redecomposition_proposals WHERE id=?", (proposal_id,)).fetchone()
        if row is None:
            raise KeyError(proposal_id)
        proposal = _proposal(row)
        if proposal.state != "proposed":
            return proposal, False
        if approve:
            accepted = conn.execute(
                "SELECT COUNT(*) FROM redecomposition_proposals WHERE parent_cell_id=? AND state='approved'",
                (proposal.parent_cell_id,),
            ).fetchone()[0]
            if accepted >= config.max_rounds_per_cell:
                raise ValueError("The cell has reached its re-decomposition round limit.")
            for child in proposal.child_ids:
                changed = conn.execute("UPDATE cells SET status='NEW' WHERE id=? AND status='DEFER_CLOSED'", (child,))
                if changed.rowcount != 1:
                    raise ValueError(f"Proposed cell is no longer deferred: {child}")
                conn.execute(
                    "INSERT INTO execution_round_requests VALUES (?,?,?,?,?)",
                    (child, proposal.id, proposal.generation, "DEFER_CLOSED", "redecomposition"),
                )
        state = "approved" if approve else "rejected"
        conn.execute("UPDATE redecomposition_proposals SET state=? WHERE id=?", (state, proposal.id))
        _event(conn, db_path, proposal, f"redecomposition_{state}", {
            "proposal_id": proposal.id, "generation": proposal.generation, "child_cell_ids": proposal.child_ids,
        })
        return proposal.model_copy(update={"state": state}), True
