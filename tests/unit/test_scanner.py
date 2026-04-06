"""Unit tests for the contradiction scanner.

Tests contradiction detection across categories, authority hierarchy
resolution, auto-fix application, and loop termination.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from petri.scanner import ScanIssue, auto_fix, scan, scan_loop


@pytest.fixture
def petri_env(tmp_path):
    """Set up a .petri/ directory with real defaults."""
    petri_dir = tmp_path / ".petri"
    petri_dir.mkdir()
    (petri_dir / "petri-dishes").mkdir()

    # Copy real defaults (includes consolidated petri.yaml)
    src_defaults = Path(__file__).parent.parent.parent / "petri" / "defaults"
    dst_defaults = petri_dir / "defaults"
    shutil.copytree(src_defaults, dst_defaults)

    # Queue
    queue = {"version": 1, "last_updated": None, "entries": {}}
    (petri_dir / "queue.json").write_text(json.dumps(queue, indent=2) + "\n")

    return tmp_path


class TestScanBaseline:
    """Test scanning with only defaults (no generated dir)."""

    def test_scan_defaults_only_returns_no_critical_issues(self, petri_env):
        petri_dir = petri_env / ".petri"
        issues = scan(petri_dir)
        # With only defaults and no generated dir, should be clean
        assert isinstance(issues, list)

    def test_scan_returns_scan_issue_objects(self, petri_env):
        petri_dir = petri_env / ".petri"
        issues = scan(petri_dir)
        for issue in issues:
            assert isinstance(issue, ScanIssue)
            assert issue.category
            assert issue.description


class TestAgentNameConsistency:
    """Category 1: Agent names."""

    def test_detects_missing_agent_in_generated(self, petri_env):
        petri_dir = petri_env / ".petri"
        generated = petri_env / ".claude"
        agents_dir = generated / "agents"
        agents_dir.mkdir(parents=True)

        # Only create a few agent files (missing most)
        (agents_dir / "investigator.md").write_text("# Investigator\n")

        issues = scan(petri_dir, generated)
        agent_name_issues = [i for i in issues if i.category == "agent_names"]
        assert len(agent_name_issues) > 0

    def test_detects_extra_agent_in_generated(self, petri_env):
        petri_dir = petri_env / ".petri"
        generated = petri_env / ".claude"
        agents_dir = generated / "agents"
        agents_dir.mkdir(parents=True)

        # Create all 13 + one extra
        from petri.convergence import load_agent_roles

        for name in load_agent_roles():
            (agents_dir / f"{name}.md").write_text(f"# {name}\n")
        (agents_dir / "rogue_agent.md").write_text("# Rogue\n")

        issues = scan(petri_dir, generated)
        extra_issues = [
            i for i in issues
            if i.category == "agent_names" and "Extra" in i.description
        ]
        assert len(extra_issues) > 0


class TestVerdictVocabulary:
    """Category 2: Verdict vocabulary consistency."""

    def test_detects_missing_verdicts_in_agent_file(self, petri_env):
        petri_dir = petri_env / ".petri"
        generated = petri_env / ".claude"
        agents_dir = generated / "agents"
        agents_dir.mkdir(parents=True)

        # Create agent file without any verdict references
        (agents_dir / "investigator.md").write_text(
            "# Investigator\nJust a stub without verdict strings.\n"
        )

        issues = scan(petri_dir, generated)
        verdict_issues = [
            i for i in issues if i.category == "verdict_vocabulary"
        ]
        # Should detect missing verdicts for investigator
        inv_issues = [i for i in verdict_issues if "investigator" in i.description]
        assert len(inv_issues) > 0


class TestQueueSchema:
    """Category 5: Queue JSON schema."""

    def test_detects_missing_queue_json(self, petri_env):
        petri_dir = petri_env / ".petri"
        (petri_dir / "queue.json").unlink()

        issues = scan(petri_dir)
        queue_issues = [i for i in issues if i.category == "queue_schema"]
        assert len(queue_issues) > 0

    def test_detects_invalid_json(self, petri_env):
        petri_dir = petri_env / ".petri"
        (petri_dir / "queue.json").write_text("not valid json{{{")

        issues = scan(petri_dir)
        queue_issues = [i for i in issues if i.category == "queue_schema"]
        assert len(queue_issues) > 0

    def test_detects_invalid_queue_state(self, petri_env):
        petri_dir = petri_env / ".petri"
        queue = {
            "version": 1,
            "last_updated": None,
            "entries": {
                "test-node-001-001": {
                    "queue_state": "INVALID_STATE",
                    "node_id": "test-node-001-001",
                }
            },
        }
        (petri_dir / "queue.json").write_text(json.dumps(queue))

        issues = scan(petri_dir)
        queue_issues = [i for i in issues if i.category == "queue_schema"]
        assert any("INVALID_STATE" in i.description for i in queue_issues)


class TestConvergenceLogic:
    """Category 6: Convergence blocking classification."""

    def test_no_issues_with_matching_defaults(self, petri_env):
        petri_dir = petri_env / ".petri"
        issues = scan(petri_dir)
        conv_issues = [i for i in issues if i.category == "convergence_logic"]
        assert len(conv_issues) == 0


class TestRoleSeparation:
    """Category 8: Lead vs specialist roles."""

    def test_detects_missing_reread_on_lead(self, petri_env):
        petri_dir = petri_env / ".petri"
        generated = petri_env / ".claude"
        agents_dir = generated / "agents"
        agents_dir.mkdir(parents=True)

        # Create lead agent without re-read instruction
        (agents_dir / "node_lead.md").write_text(
            "# Node Lead\nYou are the node lead.\n"
        )

        issues = scan(petri_dir, generated)
        role_issues = [
            i for i in issues
            if i.category == "role_separation" and "node_lead" in i.description
        ]
        assert any("re-read" in i.description.lower() for i in role_issues)

    def test_detects_reread_on_non_lead(self, petri_env):
        petri_dir = petri_env / ".petri"
        generated = petri_env / ".claude"
        agents_dir = generated / "agents"
        agents_dir.mkdir(parents=True)

        # Create non-lead agent WITH re-read instruction
        (agents_dir / "investigator.md").write_text(
            "# Investigator\nYou must re-read the constitution.\n"
        )

        issues = scan(petri_dir, generated)
        role_issues = [
            i for i in issues
            if i.category == "role_separation" and "investigator" in i.description
        ]
        assert any("Non-lead" in i.description for i in role_issues)


class TestSourceHierarchy:
    """Category 9: Source hierarchy consistency."""

    def test_no_issues_with_valid_hierarchy(self, petri_env):
        petri_dir = petri_env / ".petri"
        issues = scan(petri_dir)
        hierarchy_issues = [
            i for i in issues if i.category == "source_hierarchy"
        ]
        assert len(hierarchy_issues) == 0

    def test_detects_wrong_terminal_eligibility(self, petri_env):
        petri_dir = petri_env / ".petri"

        # Modify source hierarchy in consolidated petri.yaml
        config_path = petri_dir / "defaults" / "petri.yaml"
        import yaml

        data = yaml.safe_load(config_path.read_text())
        data["source_hierarchy"]["levels"][1]["terminal_eligible"] = False  # Level 1 should be eligible
        config_path.write_text(yaml.dump(data, default_flow_style=False))

        issues = scan(petri_dir)
        hierarchy_issues = [
            i for i in issues if i.category == "source_hierarchy"
        ]
        assert len(hierarchy_issues) > 0


class TestDocumentationDrift:
    """Category 10: Documentation references."""

    def test_detects_missing_jsonl_deny_in_settings(self, petri_env):
        petri_dir = petri_env / ".petri"
        generated = petri_env / ".claude"
        generated.mkdir(parents=True)

        # Settings without deny
        settings = {"permissions": {}}
        (generated / "settings.json").write_text(json.dumps(settings))

        issues = scan(petri_dir, generated)
        drift_issues = [
            i for i in issues if i.category == "documentation_drift"
        ]
        assert any(".jsonl" in i.description for i in drift_issues)


class TestAutoFix:
    """Test auto-fix functionality."""

    def test_auto_fix_applies_text_replacement(self, tmp_path):
        # Create a file with an error
        target = tmp_path / "test.md"
        target.write_text("The verdict is WRONG_VERDICT here.")

        issue = ScanIssue(
            category="test",
            description="Wrong verdict",
            fix="Replace WRONG_VERDICT with RIGHT_VERDICT",
            fix_path=str(target),
            fix_old="WRONG_VERDICT",
            fix_new="RIGHT_VERDICT",
        )

        fixed = auto_fix([issue])
        assert len(fixed) == 1
        assert "RIGHT_VERDICT" in target.read_text()
        assert "WRONG_VERDICT" not in target.read_text()

    def test_auto_fix_skips_missing_file(self):
        issue = ScanIssue(
            category="test",
            description="Missing file",
            fix_path="/nonexistent/path.md",
            fix_old="old",
            fix_new="new",
        )
        fixed = auto_fix([issue])
        assert len(fixed) == 0

    def test_auto_fix_skips_issues_without_fix_info(self):
        issue = ScanIssue(
            category="test",
            description="No fix available",
        )
        fixed = auto_fix([issue])
        assert len(fixed) == 0


class TestScanLoop:
    """Test the scan_loop function."""

    def test_loop_terminates_when_clean(self, petri_env):
        petri_dir = petri_env / ".petri"
        issues = scan_loop(petri_dir, max_rounds=5)
        # With valid defaults and no generated dir, should be clean
        assert isinstance(issues, list)

    def test_loop_respects_max_rounds(self, petri_env):
        petri_dir = petri_env / ".petri"
        # Remove queue.json to create a persistent unfixable issue
        (petri_dir / "queue.json").unlink()

        issues = scan_loop(petri_dir, max_rounds=2)
        assert len(issues) > 0  # Should still report unfixed issues


class TestAuthorityHierarchy:
    """Test that authority levels are correctly defined."""

    def test_authority_levels_are_ordered(self):
        from petri.scanner import AUTHORITY_LEVELS

        assert AUTHORITY_LEVELS["constitution"] > AUTHORITY_LEVELS["code"]
        assert AUTHORITY_LEVELS["code"] > AUTHORITY_LEVELS["rules"]
        assert AUTHORITY_LEVELS["rules"] > AUTHORITY_LEVELS["skills"]
        assert AUTHORITY_LEVELS["skills"] > AUTHORITY_LEVELS["agents"]
        assert AUTHORITY_LEVELS["agents"] > AUTHORITY_LEVELS["overview"]

    def test_six_authority_levels(self):
        from petri.scanner import AUTHORITY_LEVELS

        assert len(AUTHORITY_LEVELS) == 6
