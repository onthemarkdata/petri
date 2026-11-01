"""Shared test fixtures for Petri test suite."""

import json
import shutil
import subprocess
import sys
from collections.abc import Callable
from typing import NamedTuple
from pathlib import Path

import pytest

from petri.config import load_config
import yaml

from petri.graph.colony import ColonyGraph, serialize_colony
from petri.storage.colonies import save_colony
from petri.storage.paths import dish_db_path
from petri.models import Cell, CellStatus, Colony, Edge, Verdict, build_cell_key

LLM_INFERENCE_MODEL = load_config().model
MAX_CONCURRENT = load_config().execution.worker_concurrency
MAX_ITERATIONS = load_config().max_iterations

# ── Fake InferenceProvider ───────────────────────────────────────────────
# A dependency-injectable InferenceProvider used by unit tests. It is passed
# explicitly to functions under test (no monkeypatching). Anything that
# would require simulating the real `claude` CLI through monkeypatching is
# *not* automated — see learnings/ for the manual checks.


class FakeProvider:
    """Deterministic InferenceProvider for unit tests.

    Tests pass this directly to ``decompose_claim`` /
    ``generate_clarifying_questions``. Override any ``*_response`` attribute
    before calling the function under test. Calls are recorded so tests can
    assert what arguments were threaded through.
    """

    def __init__(self) -> None:
        self.substance_response: dict = {
            "is_substantive": True,
            "reason": "",
            "suggested_rewrite": "",
        }
        self.questions_response: list[dict] = [
            {"question": "What scope are you assuming?", "options": ["Global", "Local"]},
            {"question": "Any time horizon constraints?", "options": []},
        ]
        self.decompose_response: dict = {
            "nodes": [
                {"level": 1, "seq": 1, "claim_text": "premise A"},
                {"level": 1, "seq": 2, "claim_text": "premise B"},
            ],
            "edges": [],
        }
        self.why_response: list[dict] = []
        # When non-empty, each call to a streaming method emits these chunks
        # to ``on_progress`` (if supplied) before returning.
        self.progress_chunks: list[str] = []

        # Call recorders
        self.substance_calls: list[str] = []
        self.questions_calls: list[tuple[str, int]] = []
        self.decompose_calls: list[dict] = []
        self.why_calls: list[dict] = []

    def _emit_progress(self, on_progress) -> None:
        if on_progress is None:
            return
        for chunk in self.progress_chunks:
            on_progress(chunk)

    def assess_claim_substance(self, claim: str, on_progress=None) -> dict:
        self.substance_calls.append(claim)
        self._emit_progress(on_progress)
        return self.substance_response

    def generate_clarifying_questions(
        self, claim: str, max_questions: int = 5, on_progress=None
    ) -> list[dict]:
        self.questions_calls.append((claim, max_questions))
        self._emit_progress(on_progress)
        return self.questions_response

    def decompose_claim(
        self,
        claim: str,
        clarifications: list[dict],
        guidance: str = "",
        max_premises: int = 5,
        on_progress=None,
    ) -> dict:
        self.decompose_calls.append(
            {
                "claim": claim,
                "clarifications": clarifications,
                "guidance": guidance,
                "max_premises": max_premises,
            }
        )
        self._emit_progress(on_progress)
        return self.decompose_response

    def decompose_why(
        self,
        premise: str,
        parent_level: int,
        parent_seq: int,
        max_premises: int = 5,
        on_progress=None,
    ) -> list[dict]:
        self.why_calls.append(
            {
                "premise": premise,
                "parent_level": parent_level,
                "parent_seq": parent_seq,
                "max_premises": max_premises,
            }
        )
        self._emit_progress(on_progress)
        return self.why_response


class TypedFakeProvider(FakeProvider):
    """Use the existing scripted responses through the durable typed seed seam."""

    def initial_decomposition(self, request, on_progress=None):
        from petri.reasoning.decomposition_schemas import InitialDecomposition
        raw = self.decompose_claim(request.claim, request.clarifications, request.guidance,
                                   on_progress=on_progress)
        return InitialDecomposition.model_validate({"nodes": [
            {key: value for key, value in node.items() if key not in {"level", "seq"}}
            for node in raw["nodes"]
        ]})

    def expand_why(self, request, on_progress=None):
        from petri.reasoning.decomposition_schemas import WhyExpansion
        children = self.decompose_why(request.claim, request.parent_level, request.parent_seq,
                                      on_progress=on_progress)
        return WhyExpansion.model_validate({
            "is_atomic": not children,
            "bedrock_reason": "A scripted test premise" if not children else "",
            "sub_premises": children,
        })


# ── Canonical Constants ──────────────────────────────────────────────────
# Single source of truth for test IDs used across all test files.

CANONICAL_DISH_ID = "test-dish"
CANONICAL_COLONY_NAME = "colony"
CANONICAL_COLONY_ID = "test-dish-colony"
CANONICAL_CELL_IDS = {
    "center": "test-dish-colony-000-000",
    "premise1": "test-dish-colony-001-001",
    "premise2": "test-dish-colony-001-002",
    "cell1": "test-dish-colony-002-003",
    "cell2": "test-dish-colony-002-004",
}


# ── Shared Helpers ────────────────────────────────────────────────────────


def make_cell(
    dish: str,
    colony: str,
    level: int,
    seq: int,
    claim: str = "",
    status: CellStatus = CellStatus.NEW,
    dependencies: list[str] | None = None,
    dependents: list[str] | None = None,
) -> Cell:
    """Build a test Cell from composite-key parts."""
    cell_id = build_cell_key(dish, colony, level, seq)
    colony_id = f"{dish}-{colony}"
    return Cell(
        id=cell_id,
        colony_id=colony_id,
        claim_text=claim or f"claim-{level}-{seq}",
        level=level,
        status=status,
        dependencies=dependencies or [],
        dependents=dependents or [],
    )


def make_edge(from_id: str, to_id: str, edge_type: str = "intra_colony") -> Edge:
    """Build a test Edge."""
    return Edge(from_cell=from_id, to_cell=to_id, edge_type=edge_type)


def make_verdict(
    agent: str,
    verdict: str,
    cell_id: str = CANONICAL_CELL_IDS["center"],
) -> Verdict:
    """Build a test Verdict using canonical cell ID by default."""
    return Verdict(
        cell_id=cell_id, agent=agent, iteration=0, verdict=verdict, summary=""
    )


def make_event(
    *,
    cell_id: str = CANONICAL_CELL_IDS["premise1"],
    event_id: str | None = None,
    event_type: str = "verdict_issued",
    agent: str = "analyst",
    iteration: int = 1,
    timestamp: str = "2026-01-15T10:00:00+00:00",
    data: dict | None = None,
) -> dict:
    """Build a raw event dict (as it would appear serialised in JSONL).

    Defaults to canonical cell IDs from the shared colony fixture.
    """
    if event_id is None:
        event_id = f"{cell_id}-aabbccdd"
    if data is None:
        data = {"verdict": "VALIDATED", "summary": "Looks good"}
    return {
        "id": event_id,
        "cell_id": cell_id,
        "timestamp": timestamp,
        "type": event_type,
        "agent": agent,
        "iteration": iteration,
        "data": data,
    }


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_petri_dir(tmp_path):
    """Create a temporary .petri/ directory with default structure."""
    petri_dir = tmp_path / ".petri"
    petri_dir.mkdir()
    (petri_dir / "petri-dishes").mkdir()
    (petri_dir / "defaults").mkdir()

    # Default config
    config = {
        "name": "test-dish",
        "model": {"name": LLM_INFERENCE_MODEL, "provider": "local"},
        "harness": "claude-code",
        "max_iterations": MAX_ITERATIONS,
    }
    (petri_dir / "petri.yaml").write_text(
        "# Petri configuration\n"
        f"name: {config['name']}\n"
        f"model:\n  name: {config['model']['name']}\n  provider: {config['model']['provider']}\n"
        f"harness: {config['harness']}\n"
        f"max_iterations: {config['max_iterations']}\n"
    )

    return tmp_path


@pytest.fixture
def petri_defaults_dir():
    """Return the path to the package defaults directory."""
    return Path(__file__).parent.parent / "petri" / "defaults"


@pytest.fixture
def petri_env(tmp_path):
    """Set up a .petri/ directory with real defaults from the package.

    Used by scanner, adapter, and propagation tests. Includes:
    - petri.yaml config
    - queue.json (empty)
    - defaults/ copied from petri/defaults/
    - petri-dishes/ directory
    """
    petri_dir = tmp_path / ".petri"
    petri_dir.mkdir()
    (petri_dir / "petri-dishes").mkdir()

    # Copy real defaults (agents.yaml, debates.yaml, constitution, etc.)
    src_defaults = Path(__file__).parent.parent / "petri" / "defaults"
    dst_defaults = petri_dir / "defaults"
    shutil.copytree(src_defaults, dst_defaults)

    # Write petri.yaml
    config_values = load_config().model_dump(mode="json")
    config_values["name"] = CANONICAL_DISH_ID
    config_target = petri_dir / "defaults" / "petri.yaml"
    config_target.parent.mkdir(exist_ok=True)
    config_target.write_text(yaml.safe_dump(config_values, sort_keys=False))

    # Empty queue
    queue = {"version": 1, "last_updated": None, "entries": {}}
    (petri_dir / "queue.json").write_text(json.dumps(queue, indent=2) + "\n")

    return tmp_path


def _build_canonical_colony(
    cell_status: CellStatus = CellStatus.NEW,
    premise_status: CellStatus = CellStatus.NEW,
    center_status: CellStatus = CellStatus.NEW,
) -> dict:
    """Build the canonical 5-cell diamond DAG.

    Structure:
        center (L0, 000-000) depends on premise1, premise2
        premise1 (L1, 001-001) depends on cell1, cell2
        premise2 (L1, 001-002) depends on cell2 (shared = diamond)
        cell1 (L2, 002-003): leaf cell
        cell2 (L2, 002-004): leaf cell (shared dependency)
    """
    dish_id = CANONICAL_DISH_ID
    colony_name = CANONICAL_COLONY_NAME
    colony_id = CANONICAL_COLONY_ID

    center = make_cell(
        dish_id, colony_name, 0, 0, "Central thesis",
        status=center_status,
        dependencies=[
            build_cell_key(dish_id, colony_name, 1, 1),
            build_cell_key(dish_id, colony_name, 1, 2),
        ],
    )
    premise1 = make_cell(
        dish_id, colony_name, 1, 1, "First premise",
        status=premise_status,
        dependencies=[
            build_cell_key(dish_id, colony_name, 2, 3),
            build_cell_key(dish_id, colony_name, 2, 4),
        ],
    )
    premise2 = make_cell(
        dish_id, colony_name, 1, 2, "Second premise",
        status=premise_status,
        dependencies=[
            build_cell_key(dish_id, colony_name, 2, 4),
        ],
    )
    cell1 = make_cell(
        dish_id, colony_name, 2, 3, "Leaf premise of P1",
        status=cell_status,
    )
    cell2 = make_cell(
        dish_id, colony_name, 2, 4, "Shared leaf premise",
        status=cell_status,
    )

    graph = ColonyGraph(colony_id=colony_id)
    for cell in [center, premise1, premise2, cell1, cell2]:
        graph.add_cell(cell)

    graph.add_edge(make_edge(center.id, premise1.id))
    graph.add_edge(make_edge(center.id, premise2.id))
    graph.add_edge(make_edge(premise1.id, cell1.id))
    graph.add_edge(make_edge(premise1.id, cell2.id))
    graph.add_edge(make_edge(premise2.id, cell2.id))

    colony_model = Colony(
        id=colony_id,
        dish=dish_id,
        center_claim="Central thesis",
        center_cell_id=center.id,
        created_at="2026-01-01T00:00:00Z",
    )

    return {
        "dish_id": dish_id,
        "colony_name": colony_name,
        "colony_id": colony_id,
        "graph": graph,
        "colony_model": colony_model,
        "center": center,
        "premise1": premise1,
        "premise2": premise2,
        "cell1": cell1,
        "cell2": cell2,
    }


@pytest.fixture
def canonical_colony():
    """Build the canonical 5-cell diamond DAG (all cells NEW)."""
    return _build_canonical_colony()


@pytest.fixture
def canonical_colony_validated_cells():
    """Build the canonical diamond DAG with cells and premises VALIDATED.

    Used by propagation and dashboard tests where cells need to be
    re-openable (VALIDATED -> NEW).
    """
    return _build_canonical_colony(
        cell_status=CellStatus.VALIDATED,
        premise_status=CellStatus.VALIDATED,
        center_status=CellStatus.NEW,
    )


@pytest.fixture
def seeded_petri_dir(tmp_path, monkeypatch, canonical_colony):
    """Canonical end-to-end fixture for CLI tests.

    Builds a real ``.petri/`` directory containing:
      - ``defaults/petri.yaml`` and ``defaults/constitution.md`` (copied from
        the package defaults so commands that read config find what they
        expect)
      - ``petri-dishes/<canonical colony>/`` serialized via the same
        ``serialize_colony`` path the production seed command uses
      - the canonical colony in the dish's SQLite database

    Sets cwd to ``tmp_path`` so ``runner.invoke(app, [...])`` finds the
    petri dir without further plumbing. Returned dict mirrors
    ``petri_env_with_colony`` plus the resolved CLI cwd.

    This fixture is the single source of truth for "I need a populated
    petri dish to run a CLI command against". It does NOT call
    ``petri seed`` (which now requires the real Claude Code CLI) and does
    NOT monkeypatch any provider — the colony is laid down on disk
    directly. Tests that need a populated dish should depend on this
    fixture rather than reinventing setup.
    """
    monkeypatch.chdir(tmp_path)

    petri_dir = tmp_path / ".petri"
    petri_dir.mkdir()
    (petri_dir / "petri-dishes").mkdir()

    # Copy real package defaults so commands that read constitution etc work
    src_defaults = Path(__file__).parent.parent / "petri" / "defaults"
    dst_defaults = petri_dir / "defaults"
    shutil.copytree(src_defaults, dst_defaults)

    # Override petri.yaml with canonical test config (matches CANONICAL_DISH_ID)
    config_values = load_config().model_dump(mode="json")
    config_values["name"] = CANONICAL_DISH_ID
    config_target = petri_dir / "defaults" / "petri.yaml"
    config_target.parent.mkdir(exist_ok=True)
    config_target.write_text(yaml.safe_dump(config_values, sort_keys=False))

    # Serialize the canonical colony to disk via the production code path
    colony_path = petri_dir / "petri-dishes" / canonical_colony["colony_name"]
    serialize_colony(
        canonical_colony["graph"], canonical_colony["colony_model"], colony_path
    )

    save_colony(dish_db_path(petri_dir), canonical_colony["graph"], canonical_colony["colony_model"])

    return {
        "tmp_path": tmp_path,
        "petri_dir": petri_dir,
        "colony_path": colony_path,
        "dish_id": canonical_colony["dish_id"],
        "colony_name": canonical_colony["colony_name"],
        "colony_id": canonical_colony["colony_id"],
        "graph": canonical_colony["graph"],
        "colony_model": canonical_colony["colony_model"],
        "center": canonical_colony["center"],
        "premise1": canonical_colony["premise1"],
        "premise2": canonical_colony["premise2"],
        "cell1": canonical_colony["cell1"],
        "cell2": canonical_colony["cell2"],
    }


@pytest.fixture
def petri_env_with_colony(tmp_path, canonical_colony_validated_cells):
    """Full .petri/ environment with the canonical colony serialized to disk.

    Combines petri_env setup with colony serialization. Used by propagation,
    dashboard, and other tests that need on-disk colony state.
    """
    colony = canonical_colony_validated_cells
    petri_dir = tmp_path / ".petri"
    petri_dir.mkdir()
    config_values = load_config().model_dump(mode="json")
    config_values["name"] = CANONICAL_DISH_ID
    config_target = petri_dir / "defaults" / "petri.yaml"
    config_target.parent.mkdir(exist_ok=True)
    config_target.write_text(yaml.safe_dump(config_values, sort_keys=False))

    colony_path = petri_dir / "petri-dishes" / colony["colony_name"]
    serialize_colony(colony["graph"], colony["colony_model"], colony_path)
    save_colony(dish_db_path(petri_dir), colony["graph"], colony["colony_model"])

    return {
        "tmp_path": tmp_path,
        "petri_dir": petri_dir,
        "colony_path": colony_path,
        "dish_id": colony["dish_id"],
        "colony_name": colony["colony_name"],
        "colony_id": colony["colony_id"],
        "graph": colony["graph"],
        "colony_model": colony["colony_model"],
        "center": colony["center"],
        "premise1": colony["premise1"],
        "premise2": colony["premise2"],
        "cell1": colony["cell1"],
        "cell2": colony["cell2"],
    }


# ── pydantic-ai offline harness fixtures (v2) ─────────────────────────────
# Defined in tests/harness_utils.py; importing them here registers them as
# shared fixtures for the whole suite. See that module's docstring. Imported
# at the bottom (not the top) so FakeProvider stays anchored at lines 21-112;
# the suppression spans E402 (fires on ruff<0.16 for this placement) and
# RUF100 (fires on ruff>=0.16, where E402 is off and its code reads as unused).

from tests.harness_utils import (  # noqa: E402, F401, RUF100
    assessment_test_model,
    clarifying_questions_test_model,
    decomposition_test_model,
    evidence_matches_test_model,
    scripted_assessment_model,
)
# Capture before the harness offline guard patches subprocess construction.
_FAKE_PI_POPEN_INIT = subprocess.Popen.__init__


class FakePi(NamedTuple):
    """Executable path and helper that writes a scenario and selects it for this test."""

    path: Path
    scenario: Callable[..., Path]


@pytest.fixture
def fake_pi(tmp_path, monkeypatch):
    """Allow only the local fake pi executable through the subprocess guard.

    The scenario helper accepts one or more dictionaries; the final scenario
    repeats on subsequent prompts. Network and model API guards remain active.
    """
    path = Path(__file__).parent / "fixtures" / "fake_pi.py"
    path = path.resolve()

    def script(*scenarios):
        target = tmp_path / "fake-pi-scenario.json"
        target.write_text(json.dumps(list(scenarios) or [{}]), encoding="utf-8")
        monkeypatch.setenv("FAKE_PI_SCENARIO", str(target))
        return target

    def guarded_init(process, args, *positional, **kwargs):
        command = [os_arg if isinstance(os_arg, str) else str(os_arg) for os_arg in args]
        allowed = (
            command[:1] == [str(path)]
            or command[:2] == [sys.executable, str(path)]
        )
        if (
            not allowed or positional or kwargs.get("shell", False)
            or kwargs.get("executable") not in (None, command[0])
        ):
            raise RuntimeError("The fake_pi fixture permits only its local stub subprocess.")
        _FAKE_PI_POPEN_INIT(process, args, **kwargs)

    monkeypatch.setattr(subprocess.Popen, "__init__", guarded_init)
    script()
    return FakePi(path, script)
