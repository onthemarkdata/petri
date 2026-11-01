"""Trace CLI roots after resolving their current dish configuration."""

from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import ParamSpec, TypeVar, cast, get_type_hints

from petri.telemetry.attributes import CommandName, root_command_span

P = ParamSpec("P")
R = TypeVar("R")


def traced_command(command: CommandName) -> Callable[[Callable[P, R]], Callable[P, R]]:
    def decorate(function: Callable[P, R]) -> Callable[P, R]:
        @wraps(function)
        def run(*args: P.args, **kwargs: P.kwargs) -> R:
            from petri.cli._bootstrap import bootstrap_telemetry, get_dish_id

            petri_dir = Path.cwd() / ".petri"
            bootstrap_telemetry(petri_dir)
            from click.exceptions import Exit

            with root_command_span(command, get_dish_id(petri_dir)):
                try:
                    return function(*args, **kwargs)
                except Exit as error:
                    if error.exit_code == 0:
                        return cast(R, None)
                    raise

        # Typer evaluates annotations in the wrapper's module. Resolve postponed
        # annotations in the command's original module before copying them.
        run.__annotations__ = get_type_hints(function)
        return run

    return decorate
