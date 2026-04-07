"""Centralized config loader for Petri.

Single source of truth — loads petri.yaml once, provides typed accessors.
All other modules import from here instead of hardcoding values.
"""

from __future__ import annotations

import functools
from pathlib import Path

import yaml

_DEFAULTS_PATH = Path(__file__).parent / "defaults" / "petri.yaml"


@functools.lru_cache(maxsize=1)
def load_config(config_path: Path | None = None) -> dict:
    """Load and cache the full petri.yaml config."""
    path = config_path if config_path and config_path.exists() else _DEFAULTS_PATH
    return yaml.safe_load(path.read_text()) or {}


def get_model_name(config: dict | None = None) -> str:
    cfg = config or load_config()
    model = cfg.get("model", {})
    if isinstance(model, dict):
        return model.get("name", "")
    return str(model) if model else ""


def get_max_iterations(config: dict | None = None) -> int:
    return (config or load_config()).get("max_iterations", 3)


def get_max_concurrent(config: dict | None = None) -> int:
    return (config or load_config()).get("max_concurrent", 4)


def get_max_decomposition_depth(config: dict | None = None) -> int:
    return (config or load_config()).get("max_decomposition_depth", 3)


def get_minimum_terminal_level(config: dict | None = None) -> int:
    cfg = config or load_config()
    hierarchy = cfg.get("source_hierarchy", {})
    return hierarchy.get("minimum_terminal_level", 4)


def get_research_agents(config: dict | None = None) -> list[str]:
    """Get agent names assigned to the research phase (phase 1)."""
    cfg = config or load_config()
    agents = cfg.get("agents", {})
    return [
        name for name, defn in agents.items()
        if defn.get("phase") == 1 and not defn.get("is_lead", False)
    ]


def get_critique_agents(config: dict | None = None) -> list[str]:
    """Get agent names assigned to the critique phase (phase 2)."""
    cfg = config or load_config()
    agents = cfg.get("agents", {})
    return [
        name for name, defn in agents.items()
        if defn.get("phase") == 2 and not defn.get("is_lead", False)
    ]


def get_agents_with_sources(config: dict | None = None) -> list[str]:
    """Get agent names that should produce sources_cited in their output.

    These are agents whose role involves citing evidence: investigator-type
    (phase 1), champion, red_team_lead, and evidence_evaluator.
    """
    cfg = config or load_config()
    agents = cfg.get("agents", {})
    source_agents = []
    for name, defn in agents.items():
        # Phase 1 agents gather evidence, plus champion/red_team/evaluator cite sources
        instruction = defn.get("instruction", "")
        if "sources_cited" in instruction:
            source_agents.append(name)
    return source_agents


def get_agent_instruction(agent_name: str, config: dict | None = None) -> str:
    cfg = config or load_config()
    agents = cfg.get("agents", {})
    defn = agents.get(agent_name, {})
    return defn.get("instruction", "")


def get_agent_verdicts(agent_name: str, config: dict | None = None) -> list[str]:
    """Get all valid verdicts for an agent (pass + block, pass first)."""
    cfg = config or load_config()
    agents = cfg.get("agents", {})
    defn = agents.get(agent_name, {})
    return list(defn.get("verdicts_pass", [])) + list(defn.get("verdicts_block", []))


def get_all_agent_verdicts(config: dict | None = None) -> dict[str, list[str]]:
    cfg = config or load_config()
    agents = cfg.get("agents", {})
    return {
        name: list(defn.get("verdicts_pass", [])) + list(defn.get("verdicts_block", []))
        for name, defn in agents.items()
        if defn.get("verdicts_pass") or defn.get("verdicts_block")
    }


def get_all_agent_instructions(config: dict | None = None) -> dict[str, str]:
    cfg = config or load_config()
    agents = cfg.get("agents", {})
    return {
        name: defn["instruction"]
        for name, defn in agents.items()
        if defn.get("instruction")
    }


def get_short_circuit_rules(config: dict | None = None) -> list[dict]:
    """Get short-circuit rules from config.

    Derives rules entirely from agent definitions in petri.yaml:
    - Agents with ``redirect_on_block`` set generate redirect rules
    - Agents with ``CANNOT_DETERMINE`` in verdicts_block (and no redirect)
      generate ``needs_experiment`` rules
    """
    cfg = config or load_config()
    agents = cfg.get("agents", {})
    rules = []

    for agent_name, agent_defn in agents.items():
        redirect = agent_defn.get("redirect_on_block")
        verdicts_block = agent_defn.get("verdicts_block", [])

        if redirect:
            # Agent has an explicit redirect destination
            for verdict in verdicts_block:
                rules.append({
                    "agent": agent_name,
                    "verdict": verdict,
                    "type": redirect.lower(),
                })
        elif "CANNOT_DETERMINE" in verdicts_block:
            # Agent can signal "cannot determine" — triggers experiment mode
            rules.append({
                "agent": agent_name,
                "verdict": "CANNOT_DETERMINE",
                "type": "needs_experiment",
            })

    return rules
