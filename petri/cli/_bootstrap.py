"""Shared bootstrap helpers for Petri CLI commands.

These helpers are package-private: the leading underscore on the module
name (``_bootstrap``) signals that. The public entry points live in the
individual command modules via their ``register(app)`` functions.
"""

from __future__ import annotations

from pathlib import Path

from petri.storage.dish import resolve_dish_id

from petri.cli_ui import print_error_and_exit
from petri.config import load_dish_config

get_dish_id = resolve_dish_id


def resolve_provider(petri_dir: Path):
    """Resolve the typed inference bridge from this dish's validated settings."""
    from click import get_current_context
    from petri.harness.legacy_bridge import HarnessInferenceProvider
    from petri.harness.resolve import resolve_model

    config = load_dish_config(petri_dir)
    bootstrap_telemetry(petri_dir, config=config)
    provider = HarnessInferenceProvider(resolve_model(config.model_dump(mode="json")), config=config.model_dump(mode="json"))
    context = get_current_context(silent=True)
    if context is not None:
        context.call_on_close(provider.close)
    return provider


def bootstrap_telemetry(petri_dir: Path, *, config=None):
    """Initialize local tracing and agent instrumentation for the current dish."""
    from petri.config import get_telemetry_config
    from petri.telemetry import init_telemetry
    from petri.telemetry.instrument_agents import instrument_agents

    dish = config if config is not None else load_dish_config(petri_dir)
    provider = init_telemetry(
        get_telemetry_config(petri_dir), dish_id=dish.name,
        petri_dir=petri_dir,
    )
    instrument_agents(provider)
    return provider


def find_petri_dir(start: Path | None = None) -> Path:
    """Find the .petri directory from the current or given path."""
    start = start or Path.cwd()
    petri_dir = Path(start) / ".petri"
    if not petri_dir.exists():
        print_error_and_exit(
            "No petri dish found. Run `petri init` to create one.", code=3
        )
    return petri_dir


def load_colonies(petri_dir: Path, dish_id: str) -> list[tuple]:
    """Load v2 colonies from the configured dish database."""
    from petri.storage.colonies import list_colonies
    from petri.storage.paths import dish_db_path

    return list_colonies(dish_db_path(petri_dir, dish_id))


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
