"""petri launch command."""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import typer


def _port_is_listening(port: int) -> bool:
    """Return True if something is accepting TCP connections on ``port``."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _is_petri_dashboard(port: int) -> bool:
    """Return True iff ``GET /api/health`` on ``port`` returns the petri shape.

    This is the self-identification check that lets us safely kill a
    prior dashboard instance without risking an unrelated dev server
    that happens to be on the same port.
    """
    url = f"http://127.0.0.1:{port}/api/health"
    try:
        with urllib.request.urlopen(url, timeout=1.0) as response:
            payload = json.loads(response.read())
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        return False
    return isinstance(payload, dict) and payload.get("status") == "ok"


def _listening_pid(port: int) -> int | None:
    """Return the PID of the process listening on ``port``, if lsof finds one."""
    try:
        result = subprocess.run(
            ["lsof", "-iTCP:" + str(port), "-sTCP:LISTEN", "-t"],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    first_line = result.stdout.strip().splitlines()
    if not first_line:
        return None
    try:
        return int(first_line[0])
    except ValueError:
        return None


def _free_port_or_exit(port: int) -> None:
    """Ensure ``port`` is free before we bind, killing a prior petri dashboard
    instance if that's what's holding it. Fails loudly on anything else.

    The sqlite index is disposable (rebuilt from JSONL on every launch), so
    SIGKILLing a stale dashboard is safe — no in-memory state to preserve.
    """
    if not _port_is_listening(port):
        return
    if not _is_petri_dashboard(port):
        typer.echo(
            f"ERROR: Port {port} is already in use, but the process on it "
            f"is not a petri dashboard (no valid /api/health response). "
            f"Free it manually:  lsof -iTCP:{port} -sTCP:LISTEN",
            err=True,
        )
        raise typer.Exit(code=1)
    pid = _listening_pid(port)
    if pid is None:
        typer.echo(
            f"ERROR: Port {port} is held by a petri dashboard, but lsof "
            f"could not find its PID. Free the port manually and retry.",
            err=True,
        )
        raise typer.Exit(code=1)
    typer.echo(f"Stopping prior petri dashboard (PID {pid})...")
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass  # already gone
    except PermissionError as exc:
        typer.echo(
            f"ERROR: Could not stop PID {pid}: {exc}. Free port {port} manually.",
            err=True,
        )
        raise typer.Exit(code=1) from exc
    # Wait briefly for the socket to actually release.
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        if not _port_is_listening(port):
            return
        time.sleep(0.1)
    typer.echo(
        f"ERROR: Port {port} still in use 3s after killing PID {pid}. "
        f"Something else may have grabbed it.",
        err=True,
    )
    raise typer.Exit(code=1)


def register(app: typer.Typer) -> None:
    @app.command()
    def launch(
        port: int = typer.Option(8090, "--port", help="Dashboard port"),
        host: str = typer.Option(
            "127.0.0.1",
            "--host",
            help=(
                "Bind address. Defaults to 127.0.0.1 because the Computer tab "
                "can run arbitrary petri commands. Pass --host 0.0.0.0 to "
                "expose the dashboard on the LAN (opt-in only)."
            ),
        ),
    ) -> None:
        """Launch the Petri Lab dashboard in your browser."""
        import threading
        import webbrowser

        import uvicorn

        from petri.dashboard.api import create_app
        from petri.dashboard.migrate import init_db, rebuild_sqlite

        # Free the port first so a stale dashboard from a prior session
        # doesn't block us. The sqlite index is disposable — killing is safe.
        _free_port_or_exit(port)

        petri_dir = Path.cwd() / ".petri"
        db_path = petri_dir / "petri.sqlite"

        if petri_dir.exists():
            typer.echo("Building event index...")
            count = rebuild_sqlite(petri_dir, db_path)
            typer.echo(f"Indexed {count} events.")
        else:
            # No dish yet -- the Computer tab will run the onboarding wizard.
            # Ensure the db directory + schema exist so SSE works.
            petri_dir.mkdir(parents=True, exist_ok=True)
            init_db(db_path)

        dashboard_app = create_app(petri_dir, db_path)

        # The browser always opens localhost even when we're also bound on
        # a wider interface, because localhost is always reachable from the
        # same machine and avoids surprising redirects to LAN addresses.
        url = f"http://localhost:{port}"
        typer.echo(f"Launching Petri Lab at {url}  (bound on {host}:{port})")
        if host == "0.0.0.0":
            typer.echo(
                "WARNING: --host 0.0.0.0 exposes the Computer tab's command "
                "runner to anything that can reach this machine on the LAN."
            )
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()

        uvicorn.run(dashboard_app, host=host, port=port, log_level="warning")
