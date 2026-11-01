"""Autouse offline guard for the ``tests/unit/harness/`` tree.

Every test in this package runs against pydantic-ai's ``TestModel`` /
``FunctionModel`` and must be fully offline: no network sockets, no real
``claude`` (or any other) CLI subprocess, no real model API calls. The
guard below enforces that loudly instead of letting a stray integration
dependency slip through silently.
"""

import os
import socket
import subprocess
from collections.abc import Iterator

import pytest
from pydantic_ai import models as pydantic_ai_models

_OFFLINE_MESSAGE = (
    "Offline guard (tests/unit/harness/conftest.py): {activity} blocked. "
    "Tests under tests/unit/harness/ must run fully offline against "
    "TestModel/FunctionModel. If this test genuinely needs the real network "
    "or a real CLI, mark it with @pytest.mark.integration and run that tier "
    "explicitly with `pytest -m integration`."
)


def _blocked(activity: str) -> RuntimeError:
    """Build the loud, actionable error raised on any escape attempt."""
    return RuntimeError(_OFFLINE_MESSAGE.format(activity=activity))


@pytest.fixture(autouse=True)
def offline_guard(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> Iterator[None]:
    """Fail loudly if a test in this tree touches the network or spawns a CLI.

    Applied automatically (autouse) to every test under ``tests/unit/harness/``:

    - outbound network: ``socket.socket.connect`` / ``connect_ex`` /
      ``sendto`` and ``socket.create_connection`` raise ``RuntimeError``
      (socket construction stays allowed so asyncio internals keep working);
    - real-CLI attempts: ``subprocess.Popen`` — and therefore
      ``subprocess.run`` / ``check_output`` / etc. — and ``os.system`` raise
      ``RuntimeError``;
    - real model APIs: ``pydantic_ai.models.ALLOW_MODEL_REQUESTS`` is forced
      to ``False`` for defence in depth.

    Escape hatch: decorate a test with ``@pytest.mark.integration`` (marker
    registered in pyproject.toml) to bypass this guard entirely, and run that
    tier explicitly with ``pytest -m integration``. Everything else in this
    tree stays offline.
    """
    if request.node.get_closest_marker("integration"):
        yield
        return

    monkeypatch.setattr(pydantic_ai_models, "ALLOW_MODEL_REQUESTS", False)

    class _GuardedSocket(socket.socket):
        """Socket whose outbound operations raise the offline-guard error."""

        def connect(self, *args: object, **kwargs: object) -> None:
            raise _blocked("network access (socket.connect)")

        def connect_ex(self, *args: object, **kwargs: object) -> int:
            raise _blocked("network access (socket.connect_ex)")

        def sendto(self, *args: object, **kwargs: object) -> int:
            raise _blocked("network access (socket.sendto)")

    def _blocked_create_connection(*args: object, **kwargs: object) -> None:
        raise _blocked("network access (socket.create_connection)")

    def _blocked_popen_init(self: object, *args: object, **kwargs: object) -> None:
        raise _blocked("subprocess/CLI launch (subprocess.Popen)")

    def _blocked_system(*args: object, **kwargs: object) -> None:
        raise _blocked("subprocess/CLI launch (os.system)")

    monkeypatch.setattr(socket, "socket", _GuardedSocket)
    monkeypatch.setattr(socket, "create_connection", _blocked_create_connection)
    monkeypatch.setattr(subprocess.Popen, "__init__", _blocked_popen_init)
    monkeypatch.setattr(os, "system", _blocked_system)
    yield
