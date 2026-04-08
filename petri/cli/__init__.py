"""Typer CLI for the Petri Research Orchestration Framework.

11 commands: init, seed, check, grow, stop, feed, inspect, launch, scan,
graph, connect. All commands except ``init``, ``inspect``, and ``launch``
require a ``.petri/`` directory to exist.
"""

from __future__ import annotations

import typer

from petri.cli.check import register as register_check
from petri.cli.connect import register as register_connect
from petri.cli.feed import register as register_feed
from petri.cli.graph import register as register_graph
from petri.cli.grow import register as register_grow
from petri.cli.init import register as register_init
from petri.cli.inspect import register as register_inspect
from petri.cli.launch import register as register_launch
from petri.cli.scan import register as register_scan
from petri.cli.seed import register as register_seed
from petri.cli.stop import register as register_stop

app = typer.Typer(
    name="petri",
    help="Petri -- colony-based research orchestration framework",
    no_args_is_help=True,
)

_COMMAND_REGISTRARS = (
    register_init,
    register_seed,
    register_check,
    register_grow,
    register_stop,
    register_feed,
    register_inspect,
    register_launch,
    register_scan,
    register_graph,
    register_connect,
)

for register_command in _COMMAND_REGISTRARS:
    register_command(app)


def main() -> None:
    """Entry point for ``python -m petri``."""
    app()


__all__ = ["app", "main"]


if __name__ == "__main__":
    main()
