"""petri init command."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import typer

from petri.cli_ui import print_error_and_exit
from petri.config import load_config, load_dish_config

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
    created_database: bool

    @property
    def any_created(self) -> bool:
        return any(
            [
                self.created_root,
                self.created_defaults,
                self.created_config,
                self.created_constitution,
                self.created_dishes_dir,
                self.created_database,
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
                self.created_database,
            ]
        )


def create_petri_dish(
    petri_dir: Path,
    *,
    dish_name: str,
    model_name: str | None = None,
    max_concurrent: int | None = None,
    max_iterations: int | None = None,
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
    defaults = load_config()
    model_name = model_name or defaults.model
    max_concurrent = max_concurrent or defaults.execution.worker_concurrency
    max_iterations = max_iterations or defaults.max_iterations
    result = DishCreationResult(
        petri_dir=petri_dir,
        created_root=False,
        created_defaults=False,
        created_config=False,
        created_constitution=False,
        created_dishes_dir=False,
        created_database=False,
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
        import yaml

        config = defaults.model_dump(mode="json")
        config["name"] = dish_name
        config["model"] = model_name
        config["harness"]["model"] = model_name
        config["execution"]["backend"] = "dbos"
        config["execution"]["worker_concurrency"] = max_concurrent
        config["max_iterations"] = max_iterations
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

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

    from petri.storage.database import connect
    from petri.storage.paths import dish_db_path, dish_dir

    path = dish_db_path(petri_dir)
    result.created_database = not path.exists()
    connect(path).close()
    (dish_dir(petri_dir) / "exports").mkdir(exist_ok=True)

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
        defaults = load_config()
        model_name = defaults.model
        max_concurrent = defaults.execution.worker_concurrency
        max_iterations = defaults.max_iterations
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
                    "Model name (as recognized by pi):",
                    default=defaults.model,
                ).ask()
                if model_name is None:
                    typer.echo("Cancelled.")
                    raise typer.Exit(code=1)
            else:
                model_name = model_choice

            # Max concurrent agents
            concurrent_input = questionary.text(
                "Max concurrent agents:",
                default=str(defaults.execution.worker_concurrency),
            ).ask()
            if concurrent_input is None:
                typer.echo("Cancelled.")
                raise typer.Exit(code=1)
            try:
                max_concurrent = int(concurrent_input)
            except ValueError:
                max_concurrent = defaults.execution.worker_concurrency

            # Max iterations per convergence cycle
            iterations_input = questionary.text(
                "Max iterations per convergence cycle:",
                default=str(defaults.max_iterations),
            ).ask()
            if iterations_input is None:
                typer.echo("Cancelled.")
                raise typer.Exit(code=1)
            try:
                max_iterations = int(iterations_input)
            except ValueError:
                max_iterations = defaults.max_iterations

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

            results = run_preflight(load_dish_config(petri_dir).model_dump(mode="json"))
            warnings = [result for result in results if not result.passed]
            if warnings:
                typer.echo("\n  Prerequisites:")
                for result in warnings:
                    typer.echo(f"  [x] {result.name}: {result.message}")
                typer.echo("\n  Run 'petri inspect' for details.\n")

        except OSError as exc:
            print_error_and_exit(f"Error creating petri dish: {exc}", code=2)
