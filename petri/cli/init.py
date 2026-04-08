"""petri init command."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import typer

from petri.cli_ui import print_error_and_exit
from petri.config import (
    LLM_INFERENCE_MODEL,
    MAX_CONCURRENT,
    MAX_ITERATIONS,
)

DEFAULTS_DIR = Path(__file__).parent.parent / "defaults"


@dataclass
class DishCreationResult:
    """Record of what create_petri_dish actually did on disk.

    Lets callers report accurate "created" vs "repaired" vs "already
    complete" state to the user instead of guessing.
    """

    petri_dir: Path
    created_root: bool
    created_defaults: bool
    created_config: bool
    created_constitution: bool
    created_dishes_dir: bool
    created_queue: bool

    @property
    def any_created(self) -> bool:
        return any(
            [
                self.created_root,
                self.created_defaults,
                self.created_config,
                self.created_constitution,
                self.created_dishes_dir,
                self.created_queue,
            ]
        )

    @property
    def fully_fresh(self) -> bool:
        """True iff we created every piece from nothing."""
        return all(
            [
                self.created_root,
                self.created_defaults,
                self.created_config,
                self.created_dishes_dir,
                self.created_queue,
            ]
        )


def create_petri_dish(
    petri_dir: Path,
    *,
    dish_name: str,
    model_name: str = LLM_INFERENCE_MODEL,
    max_concurrent: int = MAX_CONCURRENT,
    max_iterations: int = MAX_ITERATIONS,
) -> DishCreationResult:
    """Idempotently create or repair a petri dish directory.

    Only missing pieces are created — existing files are never
    overwritten, so this is safe to call against a partially
    initialized ``.petri/`` (e.g. one where sqlite exists but
    ``defaults/petri.yaml`` was deleted, or vice versa).

    This helper is shared by three call sites:

    * ``petri init`` CLI — creates a brand-new dish after the
      interactive questionary wizard collects answers.
    * ``petri launch`` CLI — auto-heals the dish on startup so the
      dashboard always boots against a consistent filesystem state.
    * ``POST /api/init`` — the web onboarding wizard's init step.

    All three pass their own ``dish_name`` / ``model_name`` / concurrency
    / iteration values; the defaults here match ``petri init --no-
    questions``.
    """
    result = DishCreationResult(
        petri_dir=petri_dir,
        created_root=False,
        created_defaults=False,
        created_config=False,
        created_constitution=False,
        created_dishes_dir=False,
        created_queue=False,
    )

    if not petri_dir.exists():
        result.created_root = True
    petri_dir.mkdir(parents=True, exist_ok=True)

    defaults_dest = petri_dir / "defaults"
    if not defaults_dest.exists():
        result.created_defaults = True
    defaults_dest.mkdir(exist_ok=True)

    config_path = defaults_dest / "petri.yaml"
    if not config_path.exists():
        result.created_config = True
        src_config = DEFAULTS_DIR / "petri.yaml"
        if src_config.exists():
            try:
                import yaml

                with open(src_config) as f:
                    config = yaml.safe_load(f) or {}
                config["name"] = dish_name
                config.setdefault("model", {})["name"] = model_name
                config["max_concurrent"] = max_concurrent
                config["max_iterations"] = max_iterations
                with open(config_path, "w") as f:
                    f.write("# Petri Dish Configuration\n")
                    f.write(f"# Initialized for dish: {dish_name}\n\n")
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            except ImportError:
                config_path.write_text(
                    f"name: {dish_name}\n"
                    f"model:\n  name: {model_name}\n"
                    f"max_concurrent: {max_concurrent}\n"
                    f"max_iterations: {max_iterations}\n",
                    encoding="utf-8",
                )
        else:
            config_path.write_text(
                f"name: {dish_name}\n"
                f"model:\n  name: {model_name}\n"
                f"max_concurrent: {max_concurrent}\n"
                f"max_iterations: {max_iterations}\n",
                encoding="utf-8",
            )

    constitution_dest = defaults_dest / "constitution.md"
    if not constitution_dest.exists():
        src_constitution = DEFAULTS_DIR / "constitution.md"
        if src_constitution.exists():
            shutil.copy2(src_constitution, constitution_dest)
            result.created_constitution = True

    dishes_dir = petri_dir / "petri-dishes"
    if not dishes_dir.exists():
        result.created_dishes_dir = True
    dishes_dir.mkdir(exist_ok=True)

    queue_path = petri_dir / "queue.json"
    if not queue_path.exists():
        result.created_queue = True
        queue_data = {"version": 1, "last_updated": None, "entries": {}}
        queue_path.write_text(
            json.dumps(queue_data, indent=2) + "\n", encoding="utf-8"
        )

    return result


def register(app: typer.Typer) -> None:
    @app.command()
    def init(
        path: str = typer.Argument(".", help="Directory to initialize"),
        name: Optional[str] = typer.Option(None, help="Dish name (default: directory name)"),
        no_questions: bool = typer.Option(
            False, "--no-questions", help="Skip interactive setup wizard"
        ),
    ) -> None:
        """Initialize a new petri dish with an interactive setup wizard."""
        target = Path(path).resolve()
        petri_dir = target / ".petri"

        if petri_dir.exists():
            print_error_and_exit(f"Error: .petri/ already exists at {target}")

        # Derive dish name
        dish_name = name or target.name

        # ── Interactive Setup Wizard ─────────────────────────────────────
        model_name = LLM_INFERENCE_MODEL
        max_concurrent = MAX_CONCURRENT
        max_iterations = MAX_ITERATIONS
        interactive = not no_questions

        if interactive:
            try:
                import questionary
                import sys

                if not sys.stdin.isatty():
                    interactive = False
            except ImportError:
                interactive = False

        if interactive:
            typer.echo("\n  Petri Setup Wizard\n")

            # Dish name
            wizard_name = questionary.text(
                "Dish name:",
                default=dish_name,
            ).ask()
            if wizard_name is None:
                typer.echo("Cancelled.")
                raise typer.Exit(code=1)
            dish_name = wizard_name

            # Model selection
            model_choice = questionary.select(
                "Inference model:",
                choices=[
                    questionary.Choice(
                        "claude-sonnet-4-6  (fast, cost-effective — default)",
                        value="claude-sonnet-4-6",
                    ),
                    questionary.Choice(
                        "claude-opus-4-6  (most capable, higher cost)",
                        value="claude-opus-4-6",
                    ),
                    questionary.Choice(
                        "Other (enter model name)",
                        value="_custom",
                    ),
                ],
            ).ask()
            if model_choice is None:
                typer.echo("Cancelled.")
                raise typer.Exit(code=1)

            if model_choice == "_custom":
                model_name = questionary.text(
                    "Model name (as recognized by Claude Code):",
                    default=LLM_INFERENCE_MODEL,
                ).ask()
                if model_name is None:
                    typer.echo("Cancelled.")
                    raise typer.Exit(code=1)
            else:
                model_name = model_choice

            # Max concurrent agents
            concurrent_input = questionary.text(
                "Max concurrent agents:",
                default=str(MAX_CONCURRENT),
            ).ask()
            if concurrent_input is None:
                typer.echo("Cancelled.")
                raise typer.Exit(code=1)
            try:
                max_concurrent = int(concurrent_input)
            except ValueError:
                max_concurrent = MAX_CONCURRENT

            # Max iterations per convergence cycle
            iterations_input = questionary.text(
                "Max iterations per convergence cycle:",
                default=str(MAX_ITERATIONS),
            ).ask()
            if iterations_input is None:
                typer.echo("Cancelled.")
                raise typer.Exit(code=1)
            try:
                max_iterations = int(iterations_input)
            except ValueError:
                max_iterations = MAX_ITERATIONS

            typer.echo("")

        # ── Create Dish ──────────────────────────────────────────────────
        try:
            create_petri_dish(
                petri_dir,
                dish_name=dish_name,
                model_name=model_name,
                max_concurrent=max_concurrent,
                max_iterations=max_iterations,
            )

            typer.echo(f"Initialized petri dish '{dish_name}' at {target}")
            typer.echo(f"  Model: {model_name}")
            typer.echo(f"  Max concurrent: {max_concurrent}")
            typer.echo(f"  Max iterations: {max_iterations}")

            # Non-blocking preflight warnings
            from petri.engine.preflight import run_preflight

            results = run_preflight()
            warnings = [result for result in results if not result.passed]
            if warnings:
                typer.echo("\n  Prerequisites:")
                for result in warnings:
                    typer.echo(f"  [x] {result.name}: {result.message}")
                typer.echo("\n  Run 'petri inspect' for details.\n")

        except OSError as exc:
            print_error_and_exit(f"Error creating petri dish: {exc}", code=2)
