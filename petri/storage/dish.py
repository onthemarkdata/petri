"""Resolve the current dish identity consistently across Petri."""

from pathlib import Path

from petri.config import load_dish_config


def resolve_dish_id(petri_dir: Path) -> str:
    """Read the name from the dish config, or use the project directory name.

    The current config location is ``.petri/defaults/petri.yaml``, as defined
    by ``load_dish_config``. The older ``.petri/petri.yaml`` is a fallback only
    when that file is absent; it never overrides the canonical file.
    Packaged defaults never supply the identity of a user's dish.
    """
    name = load_dish_config(petri_dir).get("name")
    if name is None or name == "":
        return petri_dir.parent.name
    if not isinstance(name, str):
        raise ValueError("Dish config 'name' must be a string.")
    return name
