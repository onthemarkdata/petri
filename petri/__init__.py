"""Petri — colony-based research orchestration framework."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("petri-grow")
except PackageNotFoundError:  # source checkout without installed package metadata
    __version__ = "0.0.0+unknown"
