"""Shared labels and colors for dependency and research-status presentation."""

STATUS_COLORS = {
    "NEW": "#5c5c6e",
    "RESEARCH": "#4a9eff",
    "RED_TEAM": "#ef4444",
    "EVALUATE": "#a855f7",
    "VALIDATED": "#22c55e",
    "DISPROVEN": "#dc2626",
    "NEEDS_EXPERIMENT": "#eab308",
    "DEFER_OPEN": "#78716c",
    "DEFER_CLOSED": "#44403c",
    "STALLED": "#f97316",
}

EVENT_COLORS = {
    "search_executed": "#4a9eff",
    "source_reviewed": "#a855f7",
    "freshness_checked": "#eab308",
    "verdict_issued": "#22c55e",
    "evidence_appended": "#06b6d4",
    "debate_mediated": "#f97316",
    "convergence_checked": "#ef4444",
    "cell_reopened": "#ec4899",
    "propagation_triggered": "#818cf8",
    "decomposition_created": "#8b5cf6",
    "decomposition_audit": "#a78bfa",
}

QUEUE_ACTIVE_STATES = [
    "socratic_active",
    "research_active",
    "critique_active",
    "mediating",
    "red_team_active",
    "evaluating",
]

EDGE_STYLES = {
    'intra_colony': {'label':'Within colony','color':'#00ff88','dash':'','dot_color':'green','dot_style':'solid'},
    'cross_colony': {'label':'Across colonies','color':'#ff3399','dash':'3 3','dot_color':'blue','dot_style':'dashed'},
}
