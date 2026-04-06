"""Config generator functions for harness adapters.

Uses only stdlib ``string.Template`` + Python string assembly — no Jinja2.
Templates live in ``petri/templates/`` as plain-text files.  Each public
function takes structured config and returns a rendered string.
"""

from __future__ import annotations

from pathlib import Path
from string import Template

from petri.models import AgentRole, EventType, NodeStatus, QueueState

_TEMPLATES_DIR = Path(__file__).parent.parent / "templates"


def _load_template(name: str) -> Template:
    """Load a plain-text template from the templates directory."""
    path = _TEMPLATES_DIR / name
    return Template(path.read_text())


# ── Rule Generators ─────────────────────────────────────────────────────


def generate_constitution(constitution_text: str) -> str:
    """Generate the constitution rule file content."""
    tmpl = _load_template("rule_constitution.txt")
    return tmpl.safe_substitute(constitution_content=constitution_text)


def generate_data_model_rule(
    agent_roles: dict[str, AgentRole],
) -> str:
    """Generate the data-model rule file content."""
    tmpl = _load_template("rule_data_model.txt")

    # Event types list
    event_types = "\n".join(f"- `{e.value}`" for e in EventType)

    # Verdict table
    lines = []
    for name, role in sorted(agent_roles.items()):
        pass_str = ", ".join(role.verdicts_pass) or "(none)"
        block_str = ", ".join(role.verdicts_block) or "(none)"
        blocking_label = role.blocking if role.blocking != "false" else "no"
        lines.append(
            f"| {role.display_name} (`{name}`) | {pass_str} | {block_str} | {blocking_label} |"
        )
    verdict_table = (
        "| Agent | Pass Verdicts | Block Verdicts | Blocking? |\n"
        "|-------|--------------|----------------|----------|\n"
        + "\n".join(lines)
    )

    # Queue states
    queue_states = "\n".join(f"- `{s.value}`" for s in QueueState)

    return tmpl.safe_substitute(
        event_types=event_types,
        verdict_table=verdict_table,
        queue_states=queue_states,
    )


def generate_feedback_loop_rule(
    agent_roles: dict[str, AgentRole],
    max_iterations: int = 3,
) -> str:
    """Generate the feedback-loop rule file content."""
    tmpl = _load_template("rule_feedback_loop.txt")

    blocking_lines = []
    for name, role in sorted(agent_roles.items()):
        if role.blocking in ("true", "conditional") and not role.is_lead:
            pass_str = ", ".join(role.verdicts_pass)
            blocking_lines.append(f"- **{role.display_name}** (`{name}`): pass = {pass_str}")

    return tmpl.safe_substitute(
        blocking_agents="\n".join(blocking_lines),
        max_iterations=str(max_iterations),
    )


def generate_evidence_format_rule(
    source_hierarchy: dict,
) -> str:
    """Generate the evidence-format rule file content."""
    tmpl = _load_template("rule_evidence_format.txt")

    levels = source_hierarchy.get("levels", {})
    hierarchy_lines = []
    for level_num in sorted(levels.keys()):
        info = levels[level_num]
        name = info.get("name", f"Level {level_num}")
        desc = info.get("description", "")
        eligible = "Yes" if info.get("terminal_eligible", False) else "No"
        hierarchy_lines.append(
            f"- **Level {level_num}: {name}** — {desc} (terminal eligible: {eligible})"
        )

    min_level = source_hierarchy.get("minimum_terminal_level", 4)

    return tmpl.safe_substitute(
        source_hierarchy="\n".join(hierarchy_lines),
        minimum_terminal_level=str(min_level),
    )


def generate_research_methodology_rule() -> str:
    """Generate the research-methodology rule file content."""
    tmpl = _load_template("rule_research_methodology.txt")
    return tmpl.safe_substitute()


# ── Agent Generator ─────────────────────────────────────────────────────


_AGENT_DESCRIPTIONS: dict[str, str] = {
    "decomposition_lead": (
        "You orchestrate the decomposition phase. You generate clarifying "
        "questions, decompose claims into logical sub-nodes, and manage the "
        "colony graph structure."
    ),
    "node_lead": (
        "You orchestrate the validation pipeline for each node. You mediate "
        "debates, write convergence checks, and manage the transition through "
        "pipeline phases."
    ),
    "red_team_lead": (
        "You orchestrate the Red Team phase. You build the strongest possible "
        "case against each node, independent of the original research."
    ),
    "investigator": (
        "You gather evidence for the claim under review. Search for primary "
        "sources, evaluate their credibility, and determine whether sufficient "
        "evidence exists."
    ),
    "freshness_checker": (
        "You verify that evidence is current. Check publication dates, look "
        "for superseding information, and flag materially stale sources."
    ),
    "dependency_auditor": (
        "You verify that all dependencies of the current node are properly "
        "validated. Check for unvalidated dependencies and circular reasoning."
    ),
    "skeptic": (
        "You challenge every claim with rigorous counterarguments. Find flaws "
        "in reasoning, identify unaddressed counterarguments, and stress-test "
        "the logic."
    ),
    "champion": (
        "You build the strongest possible defense of the claim. Marshal "
        "evidence, address counterarguments, and articulate why the claim holds."
    ),
    "pragmatist": (
        "You evaluate practical viability. Assess whether the claim is "
        "production-ready, directionally correct, or fundamentally unshippable."
    ),
    "simplifier": (
        "You evaluate scope and complexity. Determine whether the claim is "
        "appropriately scoped or overcomplicated."
    ),
    "triage": (
        "You assess value and priority. Determine whether this node is high "
        "value, moderate value, or should be deferred."
    ),
    "impact_assessor": (
        "You assess the impact of this node on the broader colony. Determine "
        "if it's on the critical path, supporting, or isolated."
    ),
    "evidence_evaluator": (
        "You are the neutral judge. After the Red Team has made its case, you "
        "weigh ALL evidence — for and against — and issue a terminal "
        "recommendation."
    ),
}


def generate_agent(role: AgentRole) -> str:
    """Generate an agent definition markdown file."""
    tmpl = _load_template("agent.txt")

    description = _AGENT_DESCRIPTIONS.get(role.name, f"Agent: {role.display_name}")

    if role.phase is not None:
        phase_desc = f"Phase {role.phase} ({'Research' if role.phase == 1 else 'Critique'})"
    elif role.is_lead:
        phase_desc = f"Lead agent — orchestrates the {role.scope or 'pipeline'} phase"
    else:
        phase_desc = "Post-convergence"

    pass_verdicts = "\n".join(f"- `{v}`" for v in role.verdicts_pass) or "- (none)"
    block_verdicts = "\n".join(f"- `{v}`" for v in role.verdicts_block) or "- (none)"

    constraints = []
    if role.is_lead:
        constraints.append("- You are a **lead agent** (non-blocking orchestrator).")
        constraints.append("- You do NOT count toward convergence.")
    elif role.blocking == "true":
        constraints.append("- You are a **blocking** agent — your verdict directly affects convergence.")
    elif role.blocking == "conditional":
        constraints.append("- You are a **conditionally blocking** agent.")
        if role.redirect_on_block:
            constraints.append(f"- On block verdict, redirect to: `{role.redirect_on_block}`")
    else:
        constraints.append("- You are **non-blocking** (advisory only).")

    constraints.append("- NEVER write directly to `.jsonl` files — use event log skills only.")

    constitution_instruction = ""
    if role.is_lead:
        constitution_instruction = (
            "\n## Constitution Re-Read\n\n"
            "**IMPORTANT**: You MUST re-read the constitution at the start of every "
            "iteration. Read the file `.claude/rules/constitution.md` before taking "
            "any action each iteration. This ensures governance principles are fresh "
            "and prevents drift."
        )

    return tmpl.safe_substitute(
        display_name=role.display_name,
        agent_description=description,
        phase_description=phase_desc,
        pass_verdicts=pass_verdicts,
        block_verdicts=block_verdicts,
        behavioral_constraints="\n".join(constraints),
        constitution_instruction=constitution_instruction,
    )


# ── Skill Generators ────────────────────────────────────────────────────


def generate_skill(name: str, petri_dir: str, config: dict | None = None) -> str:
    """Generate a skill definition file.

    Parameters
    ----------
    name:
        One of: event_log_write, event_log_read, queue_update,
        convergence_check, read_node.
    petri_dir:
        The ``.petri/`` directory path (for template substitution).
    config:
        Optional config dict with max_iterations, event_types, etc.
    """
    config = config or {}
    template_name = f"skill_{name}.txt"
    tmpl = _load_template(template_name)

    subs: dict[str, str] = {"petri_dir": petri_dir}

    if name == "event_log_write":
        event_types = "\n".join(f"- `{e.value}`" for e in EventType)
        subs["event_types"] = event_types
    elif name == "queue_update":
        from petri.queue import VALID_TRANSITIONS

        lines = []
        for state, targets in VALID_TRANSITIONS.items():
            targets_str = ", ".join(f"`{t}`" for t in targets) if targets else "(terminal)"
            lines.append(f"- `{state}` → {targets_str}")
        subs["state_transitions"] = "\n".join(lines)
    elif name == "convergence_check":
        subs["max_iterations"] = str(config.get("max_iterations", 3))

    return tmpl.safe_substitute(**subs)


# ── Command Generator ───────────────────────────────────────────────────


_COMMAND_DEFS: dict[str, dict[str, str]] = {
    "seed": {
        "description": "Seed a new colony from a claim",
        "help": "Decompose a thesis into a colony of logical sub-nodes.",
        "usage": "petri seed \"<claim>\" [--colony NAME] [--no-questions]",
        "implementation": (
            "from petri.cli import app\n"
            "# Runs: petri seed <claim>\n"
            "# See petri/cli.py seed command for full implementation"
        ),
    },
    "grow": {
        "description": "Grow nodes through adversarial validation",
        "help": "Enqueue nodes and process through the validation pipeline.",
        "usage": "petri grow [NODE_IDS...] [--colony NAME] [--all] [--dry-run]",
        "implementation": (
            "from petri.cli import app\n"
            "# Runs: petri grow\n"
            "# See petri/cli.py grow command for full implementation"
        ),
    },
    "check": {
        "description": "Show current state of the petri dish",
        "help": "Display nodes grouped by level with status and queue state.",
        "usage": "petri check [--colony NAME] [--node ID] [--json]",
        "implementation": (
            "from petri.cli import app\n"
            "# Runs: petri check\n"
            "# See petri/cli.py check command for full implementation"
        ),
    },
    "feed": {
        "description": "Provide new evidence to the colony",
        "help": "Ingest new evidence, match to nodes, re-open and propagate.",
        "usage": "petri feed <source> [--colony NAME] [--auto-reopen]",
        "implementation": (
            "from petri.cli import app\n"
            "# Runs: petri feed <source>\n"
            "# See petri/cli.py feed command for full implementation"
        ),
    },
    "analyze": {
        "description": "Analysis and visualization tools",
        "help": "Run dashboard, scanner, graph, or connect sub-tools.",
        "usage": "petri analyze [--dashboard] [--scan] [--graph] [--connect N1 N2]",
        "implementation": (
            "from petri.cli import app\n"
            "# Runs: petri analyze\n"
            "# See petri/cli.py analyze command for full implementation"
        ),
    },
    "stop": {
        "description": "Gracefully stop all running tasks",
        "help": "Signal active processor to finish current phase and stop.",
        "usage": "petri stop [--force]",
        "implementation": (
            "from petri.cli import app\n"
            "# Runs: petri stop\n"
            "# See petri/cli.py stop command for full implementation"
        ),
    },
}


def generate_command(name: str) -> str:
    """Generate a command definition file."""
    if name not in _COMMAND_DEFS:
        raise ValueError(f"Unknown command: {name}. Valid: {', '.join(_COMMAND_DEFS)}")

    tmpl = _load_template("command.txt")
    defn = _COMMAND_DEFS[name]

    return tmpl.safe_substitute(
        command_name=name,
        command_description=defn["description"],
        command_help=defn["help"],
        command_usage=defn["usage"],
        command_implementation=defn["implementation"],
    )


# ── YAML Generator ──────────────────────────────────────────────────────


def generate_petri_yaml(config: dict) -> str:
    """Generate petri.yaml content from a config dict.

    This is a simple YAML serializer using string assembly (no PyYAML
    dependency required for generation).
    """
    lines = ["# Petri Dish Configuration"]
    lines.append(f"name: {config.get('name', 'petri-dish')}")
    lines.append("model:")

    model = config.get("model", {})
    lines.append(f"  name: {model.get('name', 'gemma-3-4b-it')}")
    lines.append(f"  provider: {model.get('provider', 'local')}")

    lines.append(f"harness: {config.get('harness', 'claude-code')}")
    lines.append(f"max_iterations: {config.get('max_iterations', 3)}")
    lines.append(f"max_concurrent: {config.get('max_concurrent', 4)}")

    return "\n".join(lines) + "\n"
