"""Integration tests for the Claude Code adapter.

Verifies that the adapter generates a complete ``.claude/`` directory with
all expected files, correct verdict vocabularies, lead agent constitution
re-read instructions, and passes the contradiction scanner.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from petri.adapters.claude_code import ClaudeCodeAdapter
from petri.convergence import load_agent_roles
from petri.models import PetriConfig
from petri.scanner import scan


@pytest.fixture
def petri_env(tmp_path):
    """Set up a .petri/ directory with defaults from the package."""
    petri_dir = tmp_path / ".petri"
    petri_dir.mkdir()
    (petri_dir / "petri-dishes").mkdir()

    # Copy real defaults
    src_defaults = Path(__file__).parent.parent.parent / "petri" / "defaults"
    dst_defaults = petri_dir / "defaults"
    shutil.copytree(src_defaults, dst_defaults)

    # Write petri.yaml
    (petri_dir / "petri.yaml").write_text(
        "name: test-dish\n"
        "model:\n"
        "  name: gemma-3-4b-it\n"
        "  provider: local\n"
        "harness: claude-code\n"
        "max_iterations: 3\n"
        "max_concurrent: 4\n"
    )

    # Queue
    queue = {"version": 1, "last_updated": None, "entries": {}}
    (petri_dir / "queue.json").write_text(json.dumps(queue, indent=2) + "\n")

    return tmp_path


@pytest.fixture
def adapter(petri_env):
    """Create a ClaudeCodeAdapter with test config."""
    petri_dir = petri_env / ".petri"
    config = PetriConfig(
        name="test-dish",
        model={"name": "gemma-3-4b-it", "provider": "local"},
        harness="claude-code",
        max_iterations=3,
        max_concurrent=4,
    )
    return ClaudeCodeAdapter(config=config, petri_dir=petri_dir)


class TestClaudeCodeAdapterGeneration:
    """Test that the adapter generates the expected files."""

    def test_generate_creates_directory_structure(self, adapter, petri_env):
        output_dir = petri_env / ".claude"
        adapter.generate(output_dir)

        assert (output_dir / "rules").is_dir()
        assert (output_dir / "agents").is_dir()
        assert (output_dir / "skills").is_dir()
        assert (output_dir / "commands").is_dir()

    def test_generate_creates_all_rule_files(self, adapter, petri_env):
        output_dir = petri_env / ".claude"
        adapter.generate(output_dir)

        expected_rules = [
            "constitution.md",
            "data-model.md",
            "feedback-loop.md",
            "evidence-format.md",
            "research-methodology.md",
        ]
        for rule in expected_rules:
            assert (output_dir / "rules" / rule).exists(), f"Missing rule: {rule}"

    def test_generate_creates_all_agent_files(self, adapter, petri_env):
        output_dir = petri_env / ".claude"
        adapter.generate(output_dir)

        roles = load_agent_roles()
        agents_dir = output_dir / "agents"
        for agent_name in roles:
            assert (agents_dir / f"{agent_name}.md").exists(), (
                f"Missing agent: {agent_name}"
            )

    def test_generate_creates_13_agents(self, adapter, petri_env):
        output_dir = petri_env / ".claude"
        adapter.generate(output_dir)

        agent_files = list((output_dir / "agents").glob("*.md"))
        assert len(agent_files) == 13

    def test_generate_creates_all_skill_files(self, adapter, petri_env):
        output_dir = petri_env / ".claude"
        adapter.generate(output_dir)

        expected_skills = [
            "event_log_write.md",
            "event_log_read.md",
            "queue_update.md",
            "convergence_check.md",
            "read_node.md",
        ]
        for skill in expected_skills:
            assert (output_dir / "skills" / skill).exists(), (
                f"Missing skill: {skill}"
            )

    def test_generate_creates_all_command_files(self, adapter, petri_env):
        output_dir = petri_env / ".claude"
        adapter.generate(output_dir)

        expected_commands = [
            "seed.md", "grow.md", "check.md", "feed.md", "analyze.md", "stop.md",
        ]
        for cmd in expected_commands:
            assert (output_dir / "commands" / cmd).exists(), (
                f"Missing command: {cmd}"
            )

    def test_generate_creates_settings_json(self, adapter, petri_env):
        output_dir = petri_env / ".claude"
        adapter.generate(output_dir)

        settings_path = output_dir / "settings.json"
        assert settings_path.exists()

        settings = json.loads(settings_path.read_text())
        assert "permissions" in settings
        assert "deny" in settings["permissions"]
        # Must deny .jsonl writes
        deny = settings["permissions"]["deny"]
        assert any(".jsonl" in str(p) for p in deny)


class TestAgentVerdictVocabularies:
    """Test that generated agents contain correct verdict strings."""

    def test_agent_files_contain_pass_verdicts(self, adapter, petri_env):
        output_dir = petri_env / ".claude"
        adapter.generate(output_dir)

        roles = load_agent_roles()
        for name, role in roles.items():
            agent_path = output_dir / "agents" / f"{name}.md"
            content = agent_path.read_text()
            for verdict in role.verdicts_pass:
                assert verdict in content, (
                    f"Agent '{name}' missing pass verdict: {verdict}"
                )

    def test_agent_files_contain_block_verdicts(self, adapter, petri_env):
        output_dir = petri_env / ".claude"
        adapter.generate(output_dir)

        roles = load_agent_roles()
        for name, role in roles.items():
            for verdict in role.verdicts_block:
                content = (output_dir / "agents" / f"{name}.md").read_text()
                assert verdict in content, (
                    f"Agent '{name}' missing block verdict: {verdict}"
                )


class TestLeadAgentConstitutionReread:
    """Test that lead agents include constitution re-read instruction."""

    def test_lead_agents_have_reread_instruction(self, adapter, petri_env):
        output_dir = petri_env / ".claude"
        adapter.generate(output_dir)

        lead_names = ["decomposition_lead", "node_lead", "red_team_lead"]
        for name in lead_names:
            content = (output_dir / "agents" / f"{name}.md").read_text()
            assert "re-read" in content.lower() or "reread" in content.lower(), (
                f"Lead agent '{name}' missing constitution re-read instruction"
            )

    def test_non_lead_agents_no_reread_instruction(self, adapter, petri_env):
        output_dir = petri_env / ".claude"
        adapter.generate(output_dir)

        roles = load_agent_roles()
        for name, role in roles.items():
            if role.is_lead:
                continue
            content = (output_dir / "agents" / f"{name}.md").read_text()
            assert "re-read" not in content.lower(), (
                f"Non-lead agent '{name}' has constitution re-read instruction"
            )


class TestCustomConfigPropagation:
    """Test that custom names from config propagate consistently."""

    def test_get_generated_files_lists_all(self, adapter):
        files = adapter.get_generated_files()
        # 5 rules + 13 agents + 5 skills + 6 commands + 1 settings
        assert len(files) == 30

    def test_validate_returns_list(self, adapter, petri_env):
        output_dir = petri_env / ".claude"
        adapter.generate(output_dir)
        issues = adapter.validate(output_dir)
        assert isinstance(issues, list)


class TestScannerOnGeneratedConfig:
    """Test that the contradiction scanner finds zero issues on a fresh generate."""

    def test_scanner_clean_on_fresh_generate(self, adapter, petri_env):
        output_dir = petri_env / ".claude"
        adapter.generate(output_dir)

        petri_dir = petri_env / ".petri"
        issues = scan(petri_dir, output_dir)

        # Filter for actual errors (some informational issues may remain)
        errors = [
            i for i in issues
            if i.category not in ("node_statuses",)  # informational
        ]
        assert len(errors) == 0, (
            f"Scanner found {len(errors)} issues on fresh generate: "
            + "; ".join(f"[{i.category}] {i.description}" for i in errors)
        )
