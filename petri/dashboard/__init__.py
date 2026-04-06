"""Petri Dashboard -- FastAPI REST + SSE with SQLite read index.

SQLite is a disposable read index rebuilt from JSONL event files.
Agents never write to SQLite; it is rebuilt on demand from the
append-only event logs.
"""
