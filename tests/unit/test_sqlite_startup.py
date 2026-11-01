"""Concurrent journal setup retries lock errors without extending its deadline."""

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from threading import Barrier

import pytest

from petri.storage import database


class ContendedConnection(sqlite3.Connection):
    remaining_failures = 0
    attempts = 0
    error_code = sqlite3.SQLITE_BUSY

    def execute(self, sql, parameters=()):
        if sql == "PRAGMA journal_mode = WAL":
            self.attempts += 1
            if self.remaining_failures:
                self.remaining_failures -= 1
                error = sqlite3.OperationalError("injected journal setup contention")
                error.sqlite_errorcode = self.error_code
                raise error
        return super().execute(sql, parameters)


class Clock:
    now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def install_clock(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(database.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(database.time, "sleep", clock.sleep)
    return clock


@pytest.mark.parametrize(
    "code", [sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED, sqlite3.SQLITE_BUSY | (2 << 8)]
)
def test_wal_setup_retries_only_lock_errors(tmp_path, monkeypatch, code):
    clock = install_clock(monkeypatch)
    with closing(sqlite3.connect(tmp_path / "retry.sqlite", factory=ContendedConnection)) as conn:
        conn.remaining_failures = 3
        conn.error_code = code
        database._enable_wal(conn)
        assert conn.attempts == 4
        assert clock.now == pytest.approx(0.035)
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == database.BUSY_TIMEOUT_MS


def test_wal_retry_uses_one_deadline_and_restores_timeout(tmp_path, monkeypatch):
    clock = install_clock(monkeypatch)
    monkeypatch.setattr(database, "BUSY_TIMEOUT_MS", 20)
    with closing(
        sqlite3.connect(tmp_path / "deadline.sqlite", factory=ContendedConnection)
    ) as conn:
        conn.remaining_failures = 100
        with pytest.raises(sqlite3.OperationalError, match="contention"):
            database._enable_wal(conn)
        assert clock.now == pytest.approx(0.020)
        assert conn.attempts == 4
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 20


def test_wal_setup_propagates_non_lock_errors_immediately(tmp_path, monkeypatch):
    clock = install_clock(monkeypatch)
    with closing(sqlite3.connect(tmp_path / "failure.sqlite", factory=ContendedConnection)) as conn:
        conn.remaining_failures = 1
        conn.error_code = sqlite3.SQLITE_READONLY
        with pytest.raises(sqlite3.OperationalError, match="contention"):
            database._enable_wal(conn)
        assert conn.attempts == 1
        assert clock.now == 0


def test_simultaneous_first_openers_migrate_one_database(tmp_path):
    for trial in range(10):
        gate = Barrier(32)
        path = tmp_path / f"fresh-{trial}.sqlite"

        def open_database(_):
            gate.wait(timeout=10)
            with closing(database.connect(path)) as conn:
                assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
                assert conn.execute("PRAGMA user_version").fetchone()[0] == database.SCHEMA_VERSION

        with ThreadPoolExecutor(max_workers=32) as pool:
            list(pool.map(open_database, range(32)))
        with closing(database.connect(path)) as conn:
            assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
