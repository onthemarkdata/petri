"""Run a single typed specialist and report measured usage."""

import asyncio
import sys
from pathlib import Path
from typing import Annotated, Literal

import typer
from pydantic_ai import capture_run_messages
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.messages import ModelResponse, RetryPromptPart, ToolReturnPart
from pydantic_ai.usage import RunUsage

from petri.agents.config import AgentConfig
from petri.agents.contract import AgentName
from petri.agents.factory import AgentRunError, build_agents, run_agent
from petri.cli._bootstrap import bootstrap_telemetry, load_colonies
from petri.harness.errors import HarnessError


def resolve_claim(petri_dir: Path, config: AgentConfig, claim: str | None, cell: str | None) -> str:
    if claim and cell:
        raise ValueError("Choose either --claim or --cell.")
    if claim and claim.strip():
        return claim
    if cell:
        for graph, _ in load_colonies(petri_dir, config.name):
            try:
                return graph.get_cell(cell).claim_text
            except KeyError:
                continue
        raise ValueError(f"Cell '{cell}' not found.")
    raise ValueError("Provide a nonempty --claim or an existing --cell ID.")


def print_usage(usage: RunUsage) -> None:
    typer.echo(
        f"Usage: requests={usage.requests}, input_tokens={usage.input_tokens}, "
        f"output_tokens={usage.output_tokens}, "
        f"cache_read_tokens={usage.cache_read_tokens}, "
        f"cache_write_tokens={usage.cache_write_tokens}"
    )


async def execute_agent(
    name: AgentName,
    claim: str,
    config: AgentConfig,
    constitution: str,
    *,
    verbose: bool,
    output_mode: Literal["tool", "prompted"] | None,
) -> int:
    agents = build_agents(config, constitution=constitution, output_mode=output_mode)
    usage = RunUsage()
    failed = False
    with capture_run_messages() as messages:
        try:
            result = await run_agent(agents, name, claim, config, usage=usage)
        except (UsageLimitExceeded, AgentRunError, HarnessError) as error:
            failed = True
            label = "Over-budget" if isinstance(error, UsageLimitExceeded) else "Execution failed"
            typer.echo(f"{label}: {error}", err=True)
            if isinstance(error, HarnessError) and error.detail and error.detail.partial_output:
                typer.echo("Partial backend output (not a validated verdict):")
                typer.echo(error.detail.partial_output)
            responses = [message for message in messages if isinstance(message, ModelResponse)]
            if responses:
                typer.echo("Partial model output (not a validated verdict):")
                for part in responses[-1].parts:
                    if part.part_kind == "text":
                        typer.echo(part.content)
                    elif part.part_kind == "tool-call":
                        typer.echo(part.args_as_json_str())
        else:
            output = result.output
            typer.echo(f"{name.value}: {output.verdict}")
            typer.echo(f"Summary: {output.summary}")
            typer.echo("Sources:")
            for source in output.sources_cited:
                typer.echo(f"  [level {source.hierarchy_level}] {source.title or source.url}")
                if source.title:
                    typer.echo(f"    {source.url}")
            if not output.sources_cited:
                typer.echo("  (none)")
    if verbose:
        retries = sum(
            isinstance(part, RetryPromptPart)
            or part.part_kind == "retry-feedback"
            or (isinstance(part, ToolReturnPart) and part.outcome == "retried")
            for message in messages
            for part in message.parts
        )
        typer.echo(f"Retries: {retries}")
    print_usage(usage)
    return 1 if failed else 0


def register(agents: typer.Typer) -> None:
    @agents.command("run")
    def run(
        agent: str,
        claim: Annotated[str | None, typer.Option(help="Claim text to assess.")] = None,
        cell: Annotated[str | None, typer.Option(help="Assess an existing cell's claim.")] = None,
        model: Annotated[
            str | None, typer.Option(help="Override with a provider:model ID.")
        ] = None,
        max_tokens: Annotated[
            int | None, typer.Option(min=1, help="Maximum total input and output tokens.")
        ] = None,
        max_requests: Annotated[
            int | None, typer.Option(min=1, help="Maximum model requests, including retries.")
        ] = None,
        yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip the cost warning.")] = False,
        verbose: Annotated[bool, typer.Option(help="Include the output retry count.")] = False,
        config: Annotated[
            Path | None, typer.Option(help="Read an explicit v2 config file.")
        ] = None,
        output_mode: Annotated[
            str | None, typer.Option(help="Structured output mode: tool or prompted.")
        ] = None,
    ) -> None:
        """Run one specialist with the configured usage limits."""
        from petri.cli.agents_cmd import load_command_config

        try:
            name = AgentName(agent)
        except ValueError as error:
            typer.echo(f"Unknown agent '{agent}'. Valid agents: {', '.join(AgentName)}", err=True)
            raise typer.Exit(1) from error
        petri_dir, cfg = load_command_config(config)
        try:
            prompt = resolve_claim(petri_dir, cfg, claim, cell)
            if output_mode not in {None, "tool", "prompted"}:
                raise ValueError("--output-mode must be tool or prompted.")
            if model:
                cfg.agents[name].model = model
            if max_tokens is not None:
                cfg.limits.run.total_tokens_limit = max_tokens
            if max_requests is not None:
                cfg.limits.run.request_limit = max_requests
            constitution = (petri_dir / "defaults" / "constitution.md").read_text()
            if sys.stdin.isatty() and not yes:
                typer.confirm(
                    "This run may incur model costs. "
                    f"Limits: {cfg.limits.run.request_limit} requests, "
                    f"{cfg.limits.run.total_tokens_limit} tokens. Continue?",
                    abort=True,
                )
            bootstrap_telemetry(petri_dir)
            code = asyncio.run(
                execute_agent(
                    name,
                    prompt,
                    cfg,
                    constitution,
                    verbose=verbose,
                    output_mode=output_mode,  # type: ignore[arg-type]
                )
            )
        except (ValueError, OSError, RuntimeError, HarnessError) as error:
            typer.echo(f"Cannot run agent: {error}", err=True)
            raise typer.Exit(1) from error
        if code:
            raise typer.Exit(code)
