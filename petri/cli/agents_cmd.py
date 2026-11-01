"""Inspect and validate v2 agent content without constructing inference models."""

import json
from pathlib import Path
from typing import Annotated

import typer

from petri.agents.config import AgentConfig, AgentConfigError, load_agent_config
from petri.agents.contract import AGENT_SPECS, AgentName
from petri.cli._bootstrap import find_petri_dir


def load_command_config(config: Path | None = None) -> tuple[Path, AgentConfig]:
    petri_dir = find_petri_dir()
    path = config if config is not None else petri_dir / "defaults" / "petri.yaml"
    try:
        return petri_dir, load_agent_config(path)
    except AgentConfigError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(1) from error


def register(app: typer.Typer) -> None:
    agents = typer.Typer(
        help="Inspect, validate, and run typed research agents.", no_args_is_help=True
    )
    app.add_typer(agents, name="agents")

    @agents.command("list")
    def list_agents(
        json_output: Annotated[
            bool, typer.Option("--json", help="Print machine-readable roster JSON.")
        ] = False,
        config: Annotated[
            Path | None, typer.Option("--config", help="Read an explicit v2 config file.")
        ] = None,
    ) -> None:
        """List the configured roster without making model calls."""
        _, cfg = load_command_config(config)
        rows: list[dict] = []
        for name in AgentName:
            spec, content = AGENT_SPECS[name], cfg.agents[name]
            rows.append(
                {
                    "name": name.value,
                    "display_name": spec.display_name,
                    "phase": spec.phase.value,
                    "blocking_mode": spec.blocking_mode.value,
                    "is_lead": spec.is_lead,
                    "verdicts_pass": content.verdicts_pass,
                    "verdicts_block": content.verdicts_block,
                    "model": content.model or cfg.harness.model or cfg.model,
                }
            )
        if json_output:
            typer.echo(json.dumps(rows, indent=2))
            return
        typer.echo("Agent | Phase | Blocking | Lead | Pass verdicts | Block verdicts | Model")
        for row in rows:
            typer.echo(
                f"{row['name']} | {row['phase']} | {row['blocking_mode']} | "
                f"{row['is_lead']} | {', '.join(row['verdicts_pass'])} | "
                f"{', '.join(row['verdicts_block']) or '-'} | {row['model']}"
            )

    @agents.command("check")
    def check_agents(
        config: Annotated[
            Path | None, typer.Option("--config", help="Read an explicit v2 config file.")
        ] = None,
    ) -> None:
        """Validate agent content before spending tokens."""
        _, cfg = load_command_config(config)
        typer.echo(f"Agent configuration is valid: {len(cfg.agents)} roster members.")
