"""Regression tests for the ``petri seed`` on-disk persistence flow.

These tests pin bug #4 from
``/Users/onthemarkdata/.claude/plans/quiet-doodling-pumpkin.md``: the
``petri seed`` command previously wrote ``dependencies: []`` to the center
node's on-disk ``metadata.json`` because the CLI and the decomposer each
constructed their own level-0 ``Node`` instance. The CLI serialized *its*
center (which was never mutated) while the decomposer wired dependencies
onto a throwaway local Node, so the on-disk center always looked like a
sink with no children.

The fix threads the CLI's ``center_node`` into ``decompose_claim`` via
the ``center=`` kwarg so the decomposer mutates the same object the CLI
later serializes.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from petri.cli import app
from petri.config import LLM_INFERENCE_MODEL, MAX_CONCURRENT, MAX_ITERATIONS
from petri.models import build_node_key

from tests.conftest import FakeProvider


# ── Fake provider tuned for the seed flow ───────────────────────────────


class SeedFakeProvider(FakeProvider):
    """A FakeProvider with a fixed 3-premise decomposition.

    The base ``FakeProvider`` in ``tests/conftest.py`` defaults to two
    level-1 premises; this subclass widens that to three so the test can
    assert exact membership against a known set. ``decompose_why``
    returns ``[]`` so recursion stops at level 1 — that keeps the on-disk
    structure small and predictable.
    """

    def __init__(self) -> None:
        super().__init__()
        self.decompose_response = {
            "nodes": [
                {"level": 1, "seq": 1, "claim_text": "First supporting premise"},
                {"level": 1, "seq": 2, "claim_text": "Second supporting premise"},
                {"level": 1, "seq": 3, "claim_text": "Third supporting premise"},
            ],
            "edges": [],
        }
        # Force atomic level-1 nodes — no Five Whys recursion.
        self.why_response = []


# ── Test fixture: a real .petri/ directory ready for `petri seed` ──────


@pytest.fixture
def petri_dir_for_seed(tmp_path, monkeypatch):
    """Build a minimal .petri/ directory and chdir into ``tmp_path``.

    This mirrors what ``petri init`` would lay down: a ``defaults/``
    directory with ``petri.yaml`` and ``constitution.md`` (copied from
    the package defaults so any code path that loads the constitution
    still works), an empty ``petri-dishes/``, and an empty
    ``queue.json``. The dish name matches the package canonical
    ``test-dish`` constant for consistency with the rest of the suite.
    """
    monkeypatch.chdir(tmp_path)

    petri_dir = tmp_path / ".petri"
    petri_dir.mkdir()
    (petri_dir / "petri-dishes").mkdir()

    src_defaults = Path(__file__).parent.parent.parent / "petri" / "defaults"
    dst_defaults = petri_dir / "defaults"
    shutil.copytree(src_defaults, dst_defaults)

    (dst_defaults / "petri.yaml").write_text(
        "name: test-dish\n"
        "model:\n"
        f"  name: {LLM_INFERENCE_MODEL}\n"
        "  provider: local\n"
        "harness: claude-code\n"
        f"max_iterations: {MAX_ITERATIONS}\n"
        f"max_concurrent: {MAX_CONCURRENT}\n"
    )

    queue_data = {"version": 1, "last_updated": None, "entries": {}}
    (petri_dir / "queue.json").write_text(json.dumps(queue_data, indent=2) + "\n")

    return tmp_path


# ── The regression test ─────────────────────────────────────────────────


def test_seed_persists_center_dependencies_to_disk(petri_dir_for_seed, monkeypatch):
    """Regression: bug #4 — center node's on-disk metadata.json had
    dependencies=[] because the CLI and decomposer created separate Node
    objects.

    Path chosen: drive the *full* CLI ``petri seed ... --no-questions``
    flow end-to-end via ``CliRunner``. ``--no-questions`` flips the
    seed command into non-interactive mode (skipping the substance
    check, the clarifying-questions wizard, and the
    Accept/Regenerate/Abort prompt), and ``_resolve_provider`` is
    monkeypatched to return a deterministic ``SeedFakeProvider`` so we
    never touch the real Claude Code CLI. After the command exits, we
    open ``metadata.json`` for the center node directly off disk and
    assert that ``dependencies`` is the exact set of level-1 IDs that
    the fake decomposition produced.

    This is the broadest possible regression: it tests the full
    CLI -> decomposer -> serializer handshake, not just the
    decomposer+serializer pair in isolation.
    """
    fake_provider = SeedFakeProvider()

    # Replace the live ClaudeCodeProvider resolver with one that always
    # hands the CLI our deterministic fake. The CLI imports
    # ``_resolve_provider`` from its own module, so patching the
    # attribute on ``petri.cli`` is sufficient.
    monkeypatch.setattr(
        "petri.cli._resolve_provider", lambda petri_dir: fake_provider
    )

    runner = CliRunner()
    claim_text = "The proposed approach reliably reduces system latency"
    invocation_result = runner.invoke(
        app,
        ["seed", claim_text, "--no-questions"],
        catch_exceptions=False,
    )

    assert invocation_result.exit_code == 0, (
        f"petri seed exited non-zero:\n{invocation_result.output}"
    )

    # Sanity: the FakeProvider's decompose_claim was actually called.
    assert fake_provider.decompose_calls, (
        "Expected the seed command to call decompose_claim on the "
        "injected provider, but no calls were recorded."
    )

    # ── Locate the center node's metadata.json on disk ──────────────
    petri_dir = petri_dir_for_seed / ".petri"
    dishes_dir = petri_dir / "petri-dishes"
    colony_dirs = [child for child in dishes_dir.iterdir() if child.is_dir()]
    assert len(colony_dirs) == 1, (
        f"Expected exactly one colony directory, found {len(colony_dirs)}: "
        f"{[child.name for child in colony_dirs]}"
    )
    colony_path = colony_dirs[0]

    # The center node lives at level 0, seq 0 — its directory name
    # starts with ``000-`` under a level dir whose name also starts
    # with ``000-``. Find it via a glob rather than hard-coding the
    # claim slug (which depends on the slugify implementation).
    metadata_files = list(colony_path.rglob("metadata.json"))
    assert metadata_files, (
        f"No metadata.json files found under {colony_path}"
    )

    center_metadata_path = None
    for metadata_path in metadata_files:
        loaded = json.loads(metadata_path.read_text())
        if loaded.get("level") == 0:
            center_metadata_path = metadata_path
            center_metadata = loaded
            break

    assert center_metadata_path is not None, (
        f"Could not find a level-0 metadata.json under {colony_path}; "
        f"candidates were: {[mp.relative_to(colony_path) for mp in metadata_files]}"
    )

    # ── The actual regression assertions ────────────────────────────
    persisted_dependencies = center_metadata.get("dependencies", [])
    assert isinstance(persisted_dependencies, list)
    assert persisted_dependencies, (
        "REGRESSION (bug #4): center node metadata.json was serialized "
        "with an empty dependencies list. The CLI and decomposer are "
        "operating on separate Node instances again — the decomposer's "
        "``center=`` kwarg fix has been undone or the CLI is no longer "
        "passing center_node through.\n"
        f"metadata.json contents: {json.dumps(center_metadata, indent=2)}"
    )

    # Build the exact set of level-1 IDs the FakeProvider should have
    # produced, then compare against what's on disk. The decomposer
    # constructs node keys via build_node_key(dish_id, colony_name,
    # level, seq), so we have to know the colony name. The CLI
    # auto-generates it from the claim text, but rather than reproduce
    # that logic, we recover it from the colony directory name.
    dish_id = "test-dish"
    colony_name = colony_path.name
    expected_level_one_ids = {
        build_node_key(dish_id, colony_name, 1, 1),
        build_node_key(dish_id, colony_name, 1, 2),
        build_node_key(dish_id, colony_name, 1, 3),
    }

    assert set(persisted_dependencies) == expected_level_one_ids, (
        "Center node's persisted dependencies do not match the "
        "level-1 nodes returned by the fake decomposition.\n"
        f"  expected: {sorted(expected_level_one_ids)}\n"
        f"  actual:   {sorted(persisted_dependencies)}"
    )

    # And cross-check that those exact level-1 metadata files exist
    # on disk too — i.e. the dependencies aren't pointing at phantoms.
    on_disk_ids = {
        json.loads(metadata_path.read_text())["id"]
        for metadata_path in metadata_files
    }
    for expected_id in expected_level_one_ids:
        assert expected_id in on_disk_ids, (
            f"Center metadata references {expected_id} but no "
            f"corresponding metadata.json exists on disk. "
            f"Found IDs: {sorted(on_disk_ids)}"
        )
