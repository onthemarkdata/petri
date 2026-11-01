"""petri inspect command."""

from __future__ import annotations

from pathlib import Path

import typer
import yaml


def register(app: typer.Typer) -> None:
    @app.command()
    def inspect() -> None:
        """Check that all prerequisites are installed and working."""
        from petri.engine.preflight import run_preflight

        path = Path.cwd() / ".petri" / "defaults" / "petri.yaml"
        if path.exists():
            try:
                config = yaml.safe_load(path.read_text())
                if not isinstance(config, dict):
                    raise ValueError("Expected a configuration mapping.")
            except (OSError, ValueError, yaml.YAMLError) as error:
                typer.echo(f"Cannot read {path}: {error}", err=True)
                raise typer.Exit(code=1) from error
        else:
            config = {"harness": {"backend": "pi"}}
        results = run_preflight(config)

        typer.echo("\n  Petri Environment Check\n")
        all_passed = True
        for result in results:
            icon = "+" if result.passed else "x"
            typer.echo(f"  [{icon}] {result.name:<22} {result.message}")
            if not result.passed:
                all_passed = False

        typer.echo("")
        if all_passed:
            typer.echo("  All checks passed.\n")
        else:
            typer.echo("  Some checks failed. Fix the issues above before running 'petri grow'.\n")
            raise typer.Exit(code=1)
