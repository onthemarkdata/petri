"""Build the Petri Lab frontend HTML from the template.

Loads ``petri/templates/frontend.html`` and substitutes config-derived
values (status colors, event colors, verdicts, version) using
``string.Template.safe_substitute`` -- the same pattern used by
``petri.adapters.generators`` for agent and rule generation.
"""

from __future__ import annotations

import json
from pathlib import Path
from string import Template

_TEMPLATES_DIR = Path(__file__).parent.parent / "templates"

# ── Color maps ───────────────────────────────────────────────────────────

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

PASS_VERDICTS = [
    "EVIDENCE_SUFFICIENT",
    "EVIDENCE_CONTRADICTS",
    "EVIDENCE_CURRENT",
    "STALE_BUT_HOLDS",
    "DEPENDENCIES_CLEAN",
    "ARGUMENT_WITHSTANDS_CRITIQUE",
    "STRONG_CASE",
    "DEFENSIBLE_WITH_CAVEATS",
    "PRODUCTION_READY",
    "DIRECTIONALLY_CORRECT",
    "APPROPRIATELY_SCOPED",
    "COULD_BE_SIMPLER",
    "HIGH_VALUE",
    "MODERATE_VALUE",
    "DECOMPOSITION_COMPLETE",
    "PIPELINE_COMPLETE",
    "EVIDENCE_CONFIRMS",
    "CRITICAL_PATH",
    "SUPPORTING_NOT_BLOCKING",
    "STRONG_DISPROVAL_CASE",
    "WEAK_DISPROVAL_CASE",
    "NO_CONTRADICTING_EVIDENCE",
]

STARTER_CLAIM = (
    "Open source AI models are catching up to frontier lab models"
)

# ── Builder ──────────────────────────────────────────────────────────────


def build_frontend_html(version: str = "0.3.0") -> str:
    """Load the frontend template and substitute config values.

    Parameters
    ----------
    version:
        Version string shown in the CRT onboarding boot sequence.

    Returns
    -------
    str
        Complete HTML document ready to serve.
    """
    template_path = _TEMPLATES_DIR / "frontend.html"
    tmpl = Template(template_path.read_text())

    return tmpl.safe_substitute(
        status_colors_json=json.dumps(STATUS_COLORS),
        event_colors_json=json.dumps(EVENT_COLORS),
        queue_active_states_json=json.dumps(QUEUE_ACTIVE_STATES),
        pass_verdicts_json=json.dumps(PASS_VERDICTS),
        petri_version=f"v{version}",
        starter_claim=STARTER_CLAIM,
    )
