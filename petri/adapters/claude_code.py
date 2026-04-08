"""Claude Code adapter — generates a ``.claude/`` directory.

Produces rules, agents (13 definitions with lead constitution re-read),
skills (event log, queue, convergence), commands, and settings.json.
All generated from ``petri.yaml`` via stdlib ``string.Template`` using
plain-text templates in ``petri/templates/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from petri.adapters.base import AbstractAdapter
from petri.adapters.generators import (
    generate_agent,
    generate_command,
    generate_constitution,
    generate_data_model_rule,
    generate_evidence_format_rule,
    generate_feedback_loop_rule,
    generate_petri_yaml,
    generate_research_methodology_rule,
    generate_skill,
)
from petri.analysis.convergence import load_agent_roles
from petri.models import AgentRole, PetriConfig

# Skills to generate
_SKILL_NAMES = [
    "event_log_write",
    "event_log_read",
    "queue_update",
    "convergence_check",
    "read_cell",
]

# Commands to generate. Note: "analyze" is the historical name of the
# combined analyze command, preserved here because the adapter generates
# command skeletons that may still predate the CLI split.
_COMMAND_NAMES = ["seed", "grow", "check", "feed", "analyze", "stop"]


class ClaudeCodeAdapter(AbstractAdapter):
    """Generates a ``.claude/`` directory for Claude Code harness."""

    def generate(self, output_dir: Path) -> list[Path]:
        """Generate the full ``.claude/`` directory structure."""
        created: list[Path] = []

        # Load agent roles
        petri_yaml_path = self.petri_dir / "defaults" / "petri.yaml"
        agent_roles = load_agent_roles(petri_yaml_path)

        # Load source hierarchy from consolidated petri.yaml
        full_config = _load_yaml(petri_yaml_path)
        source_hierarchy = full_config.get("source_hierarchy", {})

        # Load constitution
        constitution_path = self.petri_dir / "defaults" / "constitution.md"
        constitution_text = ""
        if constitution_path.exists():
            constitution_text = constitution_path.read_text()

        petri_dir_str = str(self.petri_dir)

        # Create directory structure
        for subdir in ("rules", "agents", "skills", "commands"):
            (output_dir / subdir).mkdir(parents=True, exist_ok=True)

        # ── Rules ────────────────────────────────────────────────────
        rules = {
            "constitution": generate_constitution(constitution_text),
            "data-model": generate_data_model_rule(agent_roles),
            "feedback-loop": generate_feedback_loop_rule(
                agent_roles, self.config.max_iterations
            ),
            "evidence-format": generate_evidence_format_rule(source_hierarchy),
            "research-methodology": generate_research_methodology_rule(),
        }

        for name, content in rules.items():
            path = output_dir / "rules" / f"{name}.md"
            path.write_text(content)
            created.append(path)

        # ── Agents ───────────────────────────────────────────────────
        for agent_name, role in agent_roles.items():
            content = generate_agent(role)
            path = output_dir / "agents" / f"{agent_name}.md"
            path.write_text(content)
            created.append(path)

        # ── Skills ───────────────────────────────────────────────────
        skill_config = {"max_iterations": self.config.max_iterations}
        for skill_name in _SKILL_NAMES:
            content = generate_skill(skill_name, petri_dir_str, skill_config)
            path = output_dir / "skills" / f"{skill_name}.md"
            path.write_text(content)
            created.append(path)

        # ── Commands ─────────────────────────────────────────────────
        for cmd_name in _COMMAND_NAMES:
            content = generate_command(cmd_name)
            path = output_dir / "commands" / f"{cmd_name}.md"
            path.write_text(content)
            created.append(path)

        # ── settings.json ────────────────────────────────────────────
        settings = self._generate_settings()
        settings_path = output_dir / "settings.json"
        settings_path.write_text(json.dumps(settings, indent=2) + "\n")
        created.append(settings_path)

        return created

    def validate(self, config_dir: Path) -> list[str]:
        """Check generated files for consistency."""
        from petri.analysis.scanner import scan

        issues = scan(self.petri_dir, config_dir)
        return [f"[{i.category}] {i.description}" for i in issues]

    def get_generated_files(self) -> list[str]:
        """Return relative paths of all files that ``generate`` will create."""
        files: list[str] = []

        # Rules
        for name in ("constitution", "data-model", "feedback-loop",
                      "evidence-format", "research-methodology"):
            files.append(f"rules/{name}.md")

        # Agents
        agent_roles = load_agent_roles(
            self.petri_dir / "defaults" / "petri.yaml"
        )
        for agent_name in agent_roles:
            files.append(f"agents/{agent_name}.md")

        # Skills
        for skill_name in _SKILL_NAMES:
            files.append(f"skills/{skill_name}.md")

        # Commands
        for cmd_name in _COMMAND_NAMES:
            files.append(f"commands/{cmd_name}.md")

        # Settings
        files.append("settings.json")

        return files

    def _generate_settings(self) -> dict:
        """Generate settings.json with JSONL write denial."""
        return {
            "permissions": {
                "deny": [
                    "**/*.jsonl",
                ],
            },
            "env": {
                "PETRI_HARNESS": "claude-code",
                "PETRI_DIR": str(self.petri_dir),
            },
        }


def _load_yaml(path: Path) -> dict:
    """Load a YAML file, returning empty dict if missing or invalid."""
    if not path.exists():
        return {}
    try:
        return yaml.safe_load(path.read_text()) or {}
    except Exception:
        return {}
