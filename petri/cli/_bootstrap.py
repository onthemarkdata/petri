"""Shared bootstrap helpers for Petri CLI commands.

These helpers are package-private: the leading underscore on the module
name (``_bootstrap``) signals that. The public entry points live in the
individual command modules via their ``register(app)`` functions.
"""

from __future__ import annotations

from pathlib import Path

from petri.cli_ui import print_error_and_exit
from petri.config import load_dish_config


def resolve_provider(petri_dir: Path):
    """Resolve an InferenceProvider from petri.yaml config.

    All inference routes through Claude Code CLI, which handles auth
    and model routing via the Anthropic API.
    """
    config = load_dish_config(petri_dir)
    model_cfg = config.get("model", {})
    if isinstance(model_cfg, dict):
        model_name = model_cfg.get("name")
    else:
        model_name = str(model_cfg) if model_cfg else None

    if not model_name:
        return None

    from petri.reasoning.claude_code_provider import ClaudeCodeProvider
    return ClaudeCodeProvider(model=model_name)


def find_petri_dir(start: Path | None = None) -> Path:
    """Find the .petri directory from the current or given path."""
    start = start or Path.cwd()
    petri_dir = Path(start) / ".petri"
    if not petri_dir.exists():
        print_error_and_exit(
            "No petri dish found. Run `petri init` to create one.", code=3
        )
    return petri_dir


def get_dish_id(petri_dir: Path) -> str:
    """Get the dish ID from config or derive from directory name."""
    config = load_dish_config(petri_dir)
    return config.get("name", petri_dir.parent.name)


def load_colonies(petri_dir: Path, dish_id: str) -> list[tuple]:
    """Load all colonies from the petri-dishes directory."""
    from petri.graph.colony import deserialize_colony

    colonies: list[tuple] = []
    dishes_dir = petri_dir / "petri-dishes"
    if not dishes_dir.exists():
        return colonies
    for colony_dir in sorted(dishes_dir.iterdir()):
        if colony_dir.is_dir():
            try:
                graph, colony = deserialize_colony(colony_dir, dish_id)
                colonies.append((graph, colony))
            except Exception:
                continue
    return colonies


def detect_interactive_mode() -> tuple[bool, object]:
    """Return ``(is_interactive, questionary_module)``.

    ``is_interactive`` is True iff stdin is a TTY AND the ``questionary``
    package can be imported. ``questionary_module`` is the imported
    module or ``None``.
    """
    import sys

    if not sys.stdin.isatty():
        return False, None
    try:
        import questionary  # type: ignore

        return True, questionary
    except ImportError:
        return False, None
