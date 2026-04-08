"""petri init command."""

from __future__ import annotations

import json
import shutil
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
            petri_dir.mkdir(parents=True)

            defaults_dest = petri_dir / "defaults"
            defaults_dest.mkdir()

            # Load package defaults and apply wizard choices
            src_config = DEFAULTS_DIR / "petri.yaml"
            if src_config.exists():
                import yaml

                with open(src_config) as f:
                    config = yaml.safe_load(f)
                config["name"] = dish_name
                config.setdefault("model", {})["name"] = model_name
                config["max_concurrent"] = max_concurrent
                config["max_iterations"] = max_iterations
                with open(defaults_dest / "petri.yaml", "w") as f:
                    f.write("# Petri Dish Configuration\n")
                    f.write(f"# Initialized for dish: {dish_name}\n\n")
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            else:
                (defaults_dest / "petri.yaml").write_text(
                    f"name: {dish_name}\nmodel:\n  name: {model_name}\n"
                    f"max_concurrent: {max_concurrent}\nmax_iterations: {max_iterations}\n",
                    encoding="utf-8",
                )

            # Copy constitution.md
            src_constitution = DEFAULTS_DIR / "constitution.md"
            if src_constitution.exists():
                shutil.copy2(src_constitution, defaults_dest / "constitution.md")

            # Create petri-dishes directory
            (petri_dir / "petri-dishes").mkdir()

            # Create queue.json
            queue_data = {"version": 1, "last_updated": None, "entries": {}}
            queue_path = petri_dir / "queue.json"
            queue_path.write_text(
                json.dumps(queue_data, indent=2) + "\n", encoding="utf-8"
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
