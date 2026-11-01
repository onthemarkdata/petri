"""Regression tests for the ``petri seed`` on-disk persistence flow.

These tests pin bug #4 from
``/Users/onthemarkdata/.claude/plans/quiet-doodling-pumpkin.md``: the
``petri seed`` command previously wrote ``dependencies: []`` to the center
cell's on-disk ``metadata.json`` because the CLI and the decomposer each
constructed their own level-0 ``Cell`` instance. The CLI serialized *its*
center (which was never mutated) while the decomposer wired dependencies
onto a throwaway local Cell, so the on-disk center always looked like a
sink with no children.

The fix threads the CLI's ``center_cell`` into ``decompose_claim`` via
the ``center=`` kwarg so the decomposer mutates the same object the CLI
later serializes.
"""

from __future__ import annotations


import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from petri.cli import app
from petri.config import load_config
import yaml
from petri.models import build_cell_key

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
        # Force atomic level-1 cells — no Five Whys recursion.
        self.why_response = []

    def initial_decomposition(self, request, on_progress=None):
        from petri.reasoning.decomposition_schemas import InitialDecomposition
        self.decompose_calls.append({"claim": request.claim})
        return InitialDecomposition.model_validate({"nodes": [
            {"claim_text": node["claim_text"]} for node in self.decompose_response["nodes"]
        ]})

    def expand_why(self, request, on_progress=None):
        from petri.reasoning.decomposition_schemas import WhyExpansion
        return WhyExpansion(is_atomic=True, bedrock_reason="A local fixture premise", sub_premises=[])


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

    values = load_config().model_dump(mode="json")
    values["name"] = "test-dish"
    (dst_defaults / "petri.yaml").write_text(yaml.safe_dump(values, sort_keys=False))

    return tmp_path


# ── The regression test ─────────────────────────────────────────────────


def test_seed_persists_center_dependencies_to_disk(petri_dir_for_seed, monkeypatch):
    """The CLI and decomposer must persist the center's actual dependencies.

    Run the command with a deterministic provider, then reconstruct its graph
    from database rows and compare the center's edges to the generated premises.
    """
    fake_provider = SeedFakeProvider()

    # Replace the live ClaudeCodeProvider resolver with one that always
    # hands the CLI our deterministic fake. The seed command imports
    # ``resolve_provider`` from ``petri.cli._bootstrap`` into its own
    # module namespace, so we patch the bound name on ``petri.cli.seed``.
    monkeypatch.setattr(
        "petri.cli.seed.resolve_provider", lambda petri_dir: fake_provider
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

    from petri.storage.colonies import list_colonies
    from petri.storage.paths import dish_db_path

    petri_dir = petri_dir_for_seed / ".petri"
    colonies = list_colonies(dish_db_path(petri_dir))
    assert len(colonies) == 1
    graph, colony = colonies[0]
    center = graph.get_cell(colony.center_cell_id)
    colony_name = colony.id.removeprefix("test-dish-")
    expected = {build_cell_key("test-dish", colony_name, 1, seq) for seq in (1, 2, 3)}
    assert set(center.dependencies) == expected
    assert expected <= {cell.id for cell in graph.get_all_cells()}
    assert not list(petri_dir.rglob("metadata.json"))
