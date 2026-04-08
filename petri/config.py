"""Centralized config loader for Petri.

Single source of truth — loads petri.yaml once, provides typed accessors.
All other modules import from here instead of hardcoding values.
If the config file is missing or malformed, Petri refuses to run.
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
    if not path.exists():
        raise FileNotFoundError(f"Petri config not found: {path}")
    cfg = yaml.safe_load(path.read_text())
    if not cfg or not isinstance(cfg, dict):
        raise ValueError(f"Petri config is empty or malformed: {path}")
    return cfg


def load_dish_config(petri_dir: Path) -> dict:
    """Load ``petri.yaml`` from a specific dish directory.

    Returns an empty dict if the file does not exist — callers fall back
    to defaults in that case. ``pyyaml`` is a hard dependency (see
    ``pyproject.toml``) so no fallback parser is needed.
    """
    config_path = petri_dir / "defaults" / "petri.yaml"
    if not config_path.exists():
        return {}
    cfg = yaml.safe_load(config_path.read_text())
    return cfg or {}


def get_model_name(config: dict | None = None) -> str:
    cfg = config or load_config()
    model = cfg.get("model")
    if model is None:
        raise KeyError("Missing 'model' in petri.yaml")
    if isinstance(model, dict):
        name = model.get("name")
        if not name:
            raise KeyError("Missing 'model.name' in petri.yaml")
        return name
    if not model:
        raise KeyError("Empty 'model' in petri.yaml")
    return str(model)


def get_max_iterations(config: dict | None = None) -> int:
    cfg = config or load_config()
    value = cfg.get("max_iterations")
    if value is None:
        raise KeyError("Missing 'max_iterations' in petri.yaml")
    return int(value)


def get_max_concurrent(config: dict | None = None) -> int:
    cfg = config or load_config()
    value = cfg.get("max_concurrent")
    if value is None:
        raise KeyError("Missing 'max_concurrent' in petri.yaml")
    return int(value)


def get_max_decomposition_depth(config: dict | None = None) -> int:
    cfg = config or load_config()
    value = cfg.get("max_decomposition_depth")
    if value is None:
        raise KeyError("Missing 'max_decomposition_depth' in petri.yaml")
    return int(value)


def get_max_nodes_per_layer(config: dict | None = None) -> int:
    """Per-layer cap on nodes created during seed-time decomposition.

    The decomposer asks the LLM to brainstorm broadly, prioritise, then
    return the top N most important premises at each level. This bound
    keeps the seed minimal so growth happens later via feed/grow rather
    than producing 100+ nodes up front.
    """
    cfg = config or load_config()
    value = cfg.get("max_nodes_per_layer")
    if value is None:
        raise KeyError("Missing 'max_nodes_per_layer' in petri.yaml")
    return int(value)


def get_minimum_terminal_level(config: dict | None = None) -> int:
    cfg = config or load_config()
    hierarchy = cfg.get("source_hierarchy")
    if hierarchy is None:
        raise KeyError("Missing 'source_hierarchy' in petri.yaml")
    value = hierarchy.get("minimum_terminal_level")
    if value is None:
        raise KeyError("Missing 'source_hierarchy.minimum_terminal_level' in petri.yaml")
    return int(value)


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

    All non-lead agents must cite sources with URLs.
    """
    cfg = config or load_config()
    agents = cfg.get("agents", {})
    return [
        name for name, defn in agents.items()
        if not defn.get("is_lead", False)
    ]


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


# ── Global config values ──────────────────────────────────────────────────
# Loaded once from petri.yaml at import time. The actual values live only
# in ``defaults/petri.yaml`` — these are the single entry points for every
# module that needs configured defaults.
LLM_INFERENCE_MODEL: str = get_model_name()
MAX_ITERATIONS: int = get_max_iterations()
MAX_CONCURRENT: int = get_max_concurrent()
