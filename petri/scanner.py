"""Contradiction scanner for Petri configuration files.

Cross-references generated and default config files across 10 categories,
applies a 6-level authority hierarchy, and optionally auto-fixes lower-
authority files when conflicts are detected.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

from petri.convergence import load_agent_roles
from petri.models import AgentRole, EventType, NodeStatus, QueueState
from petri.queue import VALID_TRANSITIONS

# ── Authority Hierarchy ─────────────────────────────────────────────────
# Higher number = higher authority.  Constitution is the highest.

AUTHORITY_LEVELS = {
    "constitution": 6,
    "code": 5,
    "rules": 4,
    "skills": 3,
    "agents": 2,
    "overview": 1,
}


# ── Issue ────────────────────────────────────────────────────────────────


class ScanIssue:
    """A single inconsistency found by the scanner."""

    def __init__(
        self,
        category: str,
        description: str,
        file_path: str | None = None,
        authority: str = "overview",
        fix: str | None = None,
        fix_path: str | None = None,
        fix_old: str | None = None,
        fix_new: str | None = None,
    ) -> None:
        self.category = category
        self.description = description
        self.file_path = file_path
        self.authority = authority
        self.fix = fix
        self.fix_path = fix_path
        self.fix_old = fix_old
        self.fix_new = fix_new

    def __repr__(self) -> str:
        return f"ScanIssue({self.category!r}, {self.description!r})"


# ── Scanner ──────────────────────────────────────────────────────────────


def scan(
    petri_dir: Path,
    generated_dir: Path | None = None,
) -> list[ScanIssue]:
    """Run the contradiction scanner across all config files.

    Parameters
    ----------
    petri_dir:
        Path to the ``.petri/`` directory.
    generated_dir:
        Path to the generated harness config directory (e.g. ``.claude/``).
        If None, scans only defaults and petri.yaml.

    Returns a list of ``ScanIssue`` objects (empty = clean).
    """
    issues: list[ScanIssue] = []

    # Load canonical data from code
    code_agent_roles = load_agent_roles()
    code_event_types = {e.value for e in EventType}
    code_queue_states = {s.value for s in QueueState}
    code_node_statuses = {s.value for s in NodeStatus}
    code_transitions = VALID_TRANSITIONS

    # Load consolidated petri.yaml from defaults
    defaults_dir = petri_dir / "defaults"
    petri_yaml = _load_yaml(defaults_dir / "petri.yaml")
    # Scanner functions expect these as separate dicts — extract sections
    defaults_agents = {"agents": petri_yaml.get("agents", {})}
    defaults_source_hierarchy = petri_yaml.get("source_hierarchy", {})
    defaults_debates = {"debates": petri_yaml.get("debates", [])}

    # 1. Agent names consistency
    issues.extend(
        _check_agent_names(code_agent_roles, defaults_agents, generated_dir)
    )

    # 2. Verdict vocabulary consistency
    issues.extend(
        _check_verdict_vocabulary(code_agent_roles, defaults_agents, generated_dir)
    )

    # 3. State machine consistency
    issues.extend(
        _check_state_machine(code_queue_states, code_transitions, generated_dir)
    )

    # 4. Event types consistency
    issues.extend(
        _check_event_types(code_event_types, generated_dir)
    )

    # 5. Queue schema consistency
    issues.extend(
        _check_queue_schema(petri_dir)
    )

    # 6. Convergence logic consistency
    issues.extend(
        _check_convergence_logic(code_agent_roles, defaults_agents)
    )

    # 7. Branching / node status consistency
    issues.extend(
        _check_node_statuses(code_node_statuses, generated_dir)
    )

    # 8. Role separation (lead vs specialist)
    issues.extend(
        _check_role_separation(code_agent_roles, defaults_agents, generated_dir)
    )

    # 9. Source hierarchy consistency
    issues.extend(
        _check_source_hierarchy(defaults_source_hierarchy, generated_dir)
    )

    # 10. Documentation drift
    issues.extend(
        _check_documentation_drift(code_agent_roles, generated_dir)
    )

    return issues


def auto_fix(issues: list[ScanIssue]) -> list[ScanIssue]:
    """Apply auto-fixes for issues that have fix information.

    Returns the list of issues that were successfully fixed.
    """
    fixed: list[ScanIssue] = []
    for issue in issues:
        if issue.fix_path and issue.fix_old is not None and issue.fix_new is not None:
            path = Path(issue.fix_path)
            if path.exists():
                content = path.read_text()
                if issue.fix_old in content:
                    content = content.replace(issue.fix_old, issue.fix_new)
                    path.write_text(content)
                    fixed.append(issue)
    return fixed


def scan_loop(
    petri_dir: Path,
    generated_dir: Path | None = None,
    max_rounds: int = 10,
) -> list[ScanIssue]:
    """Run scan + auto-fix repeatedly until zero issues or max_rounds.

    Returns the issues from the final scan (empty = clean).
    """
    for _ in range(max_rounds):
        issues = scan(petri_dir, generated_dir)
        if not issues:
            return []
        fixable = [i for i in issues if i.fix_path]
        if not fixable:
            return issues
        auto_fix(fixable)
    return scan(petri_dir, generated_dir)


# ── Helpers ──────────────────────────────────────────────────────────────


def _load_yaml(path: Path) -> dict:
    """Load a YAML file, returning empty dict on failure."""
    if not path.exists():
        return {}
    try:
        return yaml.safe_load(path.read_text()) or {}
    except Exception:
        return {}


def _find_files(directory: Path, pattern: str = "*.md") -> list[Path]:
    """Recursively find files matching a pattern."""
    if not directory or not directory.exists():
        return []
    return sorted(directory.rglob(pattern))


def _extract_verdict_strings(content: str) -> set[str]:
    """Extract verdict-like strings (ALL_CAPS_UNDERSCORED) from file content."""
    return set(re.findall(r"\b[A-Z][A-Z_]{2,}\b", content))


# ── Category Checks ─────────────────────────────────────────────────────


def _check_agent_names(
    code_roles: dict[str, AgentRole],
    defaults_yaml: dict,
    generated_dir: Path | None,
) -> list[ScanIssue]:
    """Category 1: Agent names match between code, defaults, and generated files."""
    issues: list[ScanIssue] = []
    code_names = set(code_roles.keys())

    # Check defaults/agents.yaml
    defaults_names = set(defaults_yaml.get("agents", {}).keys())
    if code_names != defaults_names:
        missing_in_defaults = code_names - defaults_names
        extra_in_defaults = defaults_names - code_names
        if missing_in_defaults:
            issues.append(ScanIssue(
                category="agent_names",
                description=f"Agents in code but missing from defaults: {sorted(missing_in_defaults)}",
                authority="code",
            ))
        if extra_in_defaults:
            issues.append(ScanIssue(
                category="agent_names",
                description=f"Agents in defaults but not in code: {sorted(extra_in_defaults)}",
                authority="constitution",
            ))

    # Check generated agent files
    if generated_dir:
        agents_dir = generated_dir / "agents"
        if agents_dir.exists():
            generated_names = {
                p.stem for p in agents_dir.iterdir() if p.suffix == ".md"
            }
            missing_generated = code_names - generated_names
            extra_generated = generated_names - code_names
            if missing_generated:
                issues.append(ScanIssue(
                    category="agent_names",
                    description=f"Agents missing from generated config: {sorted(missing_generated)}",
                    authority="agents",
                ))
            if extra_generated:
                issues.append(ScanIssue(
                    category="agent_names",
                    description=f"Extra agents in generated config: {sorted(extra_generated)}",
                    authority="agents",
                ))

    return issues


def _check_verdict_vocabulary(
    code_roles: dict[str, AgentRole],
    defaults_yaml: dict,
    generated_dir: Path | None,
) -> list[ScanIssue]:
    """Category 2: Verdict strings consistent between code and config files."""
    issues: list[ScanIssue] = []

    # Build canonical verdict set from code
    canonical_verdicts: dict[str, set[str]] = {}
    for name, role in code_roles.items():
        canonical_verdicts[name] = set(role.verdicts_pass) | set(role.verdicts_block)

    # Check defaults
    for agent_name, cfg in defaults_yaml.get("agents", {}).items():
        if agent_name not in canonical_verdicts:
            continue
        defaults_verdicts = set(cfg.get("verdicts_pass", [])) | set(cfg.get("verdicts_block", []))
        expected = canonical_verdicts[agent_name]
        if defaults_verdicts != expected:
            missing = expected - defaults_verdicts
            extra = defaults_verdicts - expected
            if missing:
                issues.append(ScanIssue(
                    category="verdict_vocabulary",
                    description=(
                        f"Agent '{agent_name}' in defaults missing verdicts: {sorted(missing)}"
                    ),
                    authority="code",
                ))
            if extra:
                issues.append(ScanIssue(
                    category="verdict_vocabulary",
                    description=(
                        f"Agent '{agent_name}' in defaults has extra verdicts: {sorted(extra)}"
                    ),
                    authority="code",
                ))

    # Check generated agent files for verdict mentions
    if generated_dir:
        agents_dir = generated_dir / "agents"
        if agents_dir.exists():
            for agent_file in agents_dir.iterdir():
                if agent_file.suffix != ".md":
                    continue
                agent_name = agent_file.stem
                if agent_name not in canonical_verdicts:
                    continue
                content = agent_file.read_text()
                expected = canonical_verdicts[agent_name]
                found = _extract_verdict_strings(content) & (
                    expected | {v for vs in canonical_verdicts.values() for v in vs}
                )
                missing = expected - found
                if missing:
                    issues.append(ScanIssue(
                        category="verdict_vocabulary",
                        description=(
                            f"Agent file '{agent_name}.md' missing verdict references: "
                            f"{sorted(missing)}"
                        ),
                        file_path=str(agent_file),
                        authority="agents",
                    ))

    return issues


def _check_state_machine(
    code_states: set[str],
    code_transitions: dict[str, list[str]],
    generated_dir: Path | None,
) -> list[ScanIssue]:
    """Category 3: State machine states consistent."""
    issues: list[ScanIssue] = []

    if generated_dir:
        # Check skills/queue_update.md for state references
        queue_skill = generated_dir / "skills" / "queue_update.md"
        if queue_skill.exists():
            content = queue_skill.read_text()
            for state in code_states:
                if state not in content:
                    issues.append(ScanIssue(
                        category="state_machine",
                        description=f"Queue state '{state}' missing from queue_update skill",
                        file_path=str(queue_skill),
                        authority="skills",
                    ))

    return issues


def _check_event_types(
    code_types: set[str],
    generated_dir: Path | None,
) -> list[ScanIssue]:
    """Category 4: Event types consistent."""
    issues: list[ScanIssue] = []

    if generated_dir:
        write_skill = generated_dir / "skills" / "event_log_write.md"
        if write_skill.exists():
            content = write_skill.read_text()
            for et in code_types:
                if et not in content:
                    issues.append(ScanIssue(
                        category="event_types",
                        description=f"Event type '{et}' missing from event_log_write skill",
                        file_path=str(write_skill),
                        authority="skills",
                    ))

    return issues


def _check_queue_schema(petri_dir: Path) -> list[ScanIssue]:
    """Category 5: Queue JSON schema integrity."""
    issues: list[ScanIssue] = []
    queue_path = petri_dir / "queue.json"
    if not queue_path.exists():
        issues.append(ScanIssue(
            category="queue_schema",
            description="queue.json not found",
            file_path=str(queue_path),
            authority="code",
        ))
        return issues

    try:
        data = json.loads(queue_path.read_text())
    except json.JSONDecodeError as exc:
        issues.append(ScanIssue(
            category="queue_schema",
            description=f"queue.json is not valid JSON: {exc}",
            file_path=str(queue_path),
            authority="code",
        ))
        return issues

    if "entries" not in data:
        issues.append(ScanIssue(
            category="queue_schema",
            description="queue.json missing 'entries' key",
            file_path=str(queue_path),
            authority="code",
        ))

    # Validate queue state values
    for node_id, entry in data.get("entries", {}).items():
        state = entry.get("queue_state", "")
        if state and state not in {s.value for s in QueueState}:
            issues.append(ScanIssue(
                category="queue_schema",
                description=(
                    f"Node '{node_id}' has invalid queue state: '{state}'"
                ),
                file_path=str(queue_path),
                authority="code",
            ))

    return issues


def _check_convergence_logic(
    code_roles: dict[str, AgentRole],
    defaults_yaml: dict,
) -> list[ScanIssue]:
    """Category 6: Convergence blocking classification consistent."""
    issues: list[ScanIssue] = []

    for agent_name, cfg in defaults_yaml.get("agents", {}).items():
        if agent_name not in code_roles:
            continue
        code_role = code_roles[agent_name]
        defaults_blocking = str(cfg.get("blocking", "false")).lower()
        if defaults_blocking != code_role.blocking:
            issues.append(ScanIssue(
                category="convergence_logic",
                description=(
                    f"Agent '{agent_name}' blocking mismatch: "
                    f"code={code_role.blocking}, defaults={defaults_blocking}"
                ),
                authority="code",
            ))

    return issues


def _check_node_statuses(
    code_statuses: set[str],
    generated_dir: Path | None,
) -> list[ScanIssue]:
    """Category 7: Node status values consistent in generated files."""
    issues: list[ScanIssue] = []

    if generated_dir:
        # Check rules for node status references
        rules_dir = generated_dir / "rules"
        if rules_dir.exists():
            for rule_file in rules_dir.iterdir():
                if rule_file.suffix != ".md":
                    continue
                if "data" not in rule_file.stem and "feedback" not in rule_file.stem:
                    continue
                content = rule_file.read_text()
                found_statuses = _extract_verdict_strings(content) & code_statuses
                # Only check if the file references statuses at all
                if found_statuses and len(found_statuses) < len(code_statuses) // 2:
                    missing = code_statuses - found_statuses
                    issues.append(ScanIssue(
                        category="node_statuses",
                        description=(
                            f"Rule '{rule_file.stem}' references some but not all node "
                            f"statuses. Missing: {sorted(missing)}"
                        ),
                        file_path=str(rule_file),
                        authority="rules",
                    ))

    return issues


def _check_role_separation(
    code_roles: dict[str, AgentRole],
    defaults_yaml: dict,
    generated_dir: Path | None,
) -> list[ScanIssue]:
    """Category 8: Lead vs specialist role separation."""
    issues: list[ScanIssue] = []
    lead_names = {name for name, role in code_roles.items() if role.is_lead}

    # Check defaults
    for agent_name, cfg in defaults_yaml.get("agents", {}).items():
        if agent_name not in code_roles:
            continue
        is_lead_defaults = cfg.get("is_lead", False)
        is_lead_code = code_roles[agent_name].is_lead
        if is_lead_defaults != is_lead_code:
            issues.append(ScanIssue(
                category="role_separation",
                description=(
                    f"Agent '{agent_name}' is_lead mismatch: "
                    f"code={is_lead_code}, defaults={is_lead_defaults}"
                ),
                authority="code",
            ))

    # Check generated agent files for constitution re-read instruction
    if generated_dir:
        agents_dir = generated_dir / "agents"
        if agents_dir.exists():
            for agent_file in agents_dir.iterdir():
                if agent_file.suffix != ".md":
                    continue
                agent_name = agent_file.stem
                content = agent_file.read_text()
                is_lead = agent_name in lead_names
                has_reread = "re-read" in content.lower() or "reread" in content.lower()

                if is_lead and not has_reread:
                    issues.append(ScanIssue(
                        category="role_separation",
                        description=(
                            f"Lead agent '{agent_name}' missing constitution re-read instruction"
                        ),
                        file_path=str(agent_file),
                        authority="agents",
                    ))
                elif not is_lead and has_reread:
                    issues.append(ScanIssue(
                        category="role_separation",
                        description=(
                            f"Non-lead agent '{agent_name}' has constitution re-read "
                            f"instruction (should only be on leads)"
                        ),
                        file_path=str(agent_file),
                        authority="agents",
                    ))

    return issues


def _check_source_hierarchy(
    defaults_hierarchy: dict,
    generated_dir: Path | None,
) -> list[ScanIssue]:
    """Category 9: Source hierarchy consistent."""
    issues: list[ScanIssue] = []

    levels = defaults_hierarchy.get("levels", {})
    if not levels:
        issues.append(ScanIssue(
            category="source_hierarchy",
            description="No source hierarchy levels defined in defaults",
            authority="constitution",
        ))
        return issues

    min_level = defaults_hierarchy.get("minimum_terminal_level", 4)
    # Verify all levels up to min_level are terminal_eligible
    for level_num, info in levels.items():
        eligible = info.get("terminal_eligible", False)
        if int(level_num) <= min_level and not eligible:
            issues.append(ScanIssue(
                category="source_hierarchy",
                description=(
                    f"Source level {level_num} ({info.get('name', '?')}) should be "
                    f"terminal_eligible (level <= {min_level})"
                ),
                authority="constitution",
            ))
        if int(level_num) > min_level and eligible:
            issues.append(ScanIssue(
                category="source_hierarchy",
                description=(
                    f"Source level {level_num} ({info.get('name', '?')}) should NOT be "
                    f"terminal_eligible (level > {min_level})"
                ),
                authority="constitution",
            ))

    return issues


def _check_documentation_drift(
    code_roles: dict[str, AgentRole],
    generated_dir: Path | None,
) -> list[ScanIssue]:
    """Category 10: Documentation references match actual config."""
    issues: list[ScanIssue] = []

    if not generated_dir:
        return issues

    # Check that settings.json denies .jsonl writes
    settings_path = generated_dir / "settings.json"
    if settings_path.exists():
        try:
            settings = json.loads(settings_path.read_text())
            deny_patterns = settings.get("permissions", {}).get("deny", [])
            jsonl_denied = any(".jsonl" in str(p) for p in deny_patterns)
            if not jsonl_denied:
                issues.append(ScanIssue(
                    category="documentation_drift",
                    description=(
                        "settings.json does not deny agent write access to .jsonl files"
                    ),
                    file_path=str(settings_path),
                    authority="rules",
                ))
        except (json.JSONDecodeError, AttributeError):
            issues.append(ScanIssue(
                category="documentation_drift",
                description="settings.json is not valid JSON",
                file_path=str(settings_path),
                authority="rules",
            ))

    # Check that the number of agent files matches the roster
    agents_dir = generated_dir / "agents"
    if agents_dir.exists():
        agent_count = len([f for f in agents_dir.iterdir() if f.suffix == ".md"])
        expected_count = len(code_roles)
        if agent_count != expected_count:
            issues.append(ScanIssue(
                category="documentation_drift",
                description=(
                    f"Generated {agent_count} agent files but code defines "
                    f"{expected_count} agents"
                ),
                authority="agents",
            ))

    return issues
