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

from petri.agents.contract import AGENT_SPECS
from petri.presentation import EDGE_STYLES, EVENT_COLORS, QUEUE_ACTIVE_STATES, STATUS_COLORS

_TEMPLATES_DIR = Path(__file__).parent.parent / "templates"

# ── Color maps ───────────────────────────────────────────────────────────

PASS_VERDICTS = sorted({verdict for spec in AGENT_SPECS.values() for verdict in spec.verdicts_pass})

STARTER_CLAIM = (
    "Open source AI models are catching up to frontier lab models"
)

# ── Builder ──────────────────────────────────────────────────────────────


def build_frontend_html(version: str | None = None) -> str:
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
    if version is None:
        from petri import __version__
        version = __version__
    template_path = _TEMPLATES_DIR / "frontend.html"
    tmpl = Template(template_path.read_text())

    return tmpl.safe_substitute(
        edge_styles_json=json.dumps(EDGE_STYLES),
        status_colors_json=json.dumps(STATUS_COLORS),
        event_colors_json=json.dumps(EVENT_COLORS),
        queue_active_states_json=json.dumps(QUEUE_ACTIVE_STATES),
        pass_verdicts_json=json.dumps(PASS_VERDICTS),
        petri_version=f"v{version}",
        starter_claim=STARTER_CLAIM,
    )
