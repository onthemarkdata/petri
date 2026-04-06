"""Abstract adapter interface for harness config generation.

Each adapter generates a harness-specific config directory from a PetriDish
definition. The base class defines the contract; concrete adapters (e.g.
``ClaudeCodeAdapter``) implement generation and validation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from petri.models import PetriConfig


class AbstractAdapter(ABC):
    """Base class for all harness adapters.

    Parameters
    ----------
    config:
        Parsed ``PetriConfig`` from the dish's ``petri.yaml``.
    petri_dir:
        Path to the ``.petri/`` directory.
    """

    def __init__(self, config: PetriConfig, petri_dir: Path) -> None:
        self.config = config
        self.petri_dir = petri_dir

    @abstractmethod
    def generate(self, output_dir: Path) -> list[Path]:
        """Generate harness config files into *output_dir*.

        Returns a list of all created file paths.
        """

    @abstractmethod
    def validate(self, config_dir: Path) -> list[str]:
        """Check generated files for consistency.

        Returns a list of issue descriptions. An empty list means the
        config directory is valid.
        """

    @abstractmethod
    def get_generated_files(self) -> list[str]:
        """Return relative paths of all files that ``generate`` will create.

        Useful for previewing changes before writing to disk.
        """
