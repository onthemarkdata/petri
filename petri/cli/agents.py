from __future__ import annotations

def register(app: typer.Typer) -> None:
    @app.command()
    def agents_list() -> None:
        """List all agents"""
        typer.echo("Agents list command")

    @app.command()
    def agents_check() -> None:
        """Check agents"""
        typer.echo("Agents check command")