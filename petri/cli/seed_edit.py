"""Editor and inline controls for the seed approval prompt."""

import os

import typer
import yaml

from petri.reasoning.seed_edit import dump_draft, save_draft_edit, validate_draft
from petri.storage.backup import snapshot_before_destructive


def _inline_edit(text: str, questionary) -> str | None:
    document = yaml.safe_load(text)
    while True:
        rows = document["premises"]
        for index, row in enumerate(rows, 1):
            typer.echo(f"{index}. [{row['relation']}] {row['claim_text']}")
        command = questionary.text(
            "edit N <text>, drop N, relation N <supports|limits|rebuts>, atomic N <reason>, done, cancel"
        ).ask()
        if command is None or command.strip() == "cancel":
            return None
        if command.strip() == "done":
            return yaml.safe_dump(document, sort_keys=False, allow_unicode=True)
        try:
            operation, number, *rest = command.split(maxsplit=2)
            index = int(number) - 1
            if index < 0 or index >= len(rows):
                raise ValueError("Choose one of the displayed premise numbers.")
            value = rest[0].strip() if rest else ""
            if operation == "drop" and not value:
                rows.pop(index)
            elif operation == "edit" and value:
                rows[index]["claim_text"] = value
            elif operation == "relation" and value:
                rows[index]["relation"] = value
            elif operation == "atomic" and value:
                rows[index].update(is_atomic=True, bedrock_reason=value)
            else:
                raise ValueError("Use one of the displayed commands.")
        except (ValueError, TypeError) as error:
            typer.echo(f"Invalid edit: {error}")


def edit_pending_result(result, settings, path, petri_dir, dish_id, questionary):
    """Validate the whole draft before backing up or replacing any rows."""
    original_text = dump_draft(result)
    text = original_text
    while True:
        if os.environ.get("EDITOR") or os.environ.get("VISUAL"):
            edited_text = typer.edit(text, extension=".yaml", require_save=False)
        else:
            edited_text = _inline_edit(text, questionary)
        if edited_text is None:
            typer.echo("Edit cancelled; draft retained.")
            return result
        if edited_text == original_text:
            return result
        try:
            edited = validate_draft(edited_text, result, settings)
        except (ValueError, yaml.YAMLError) as error:
            typer.echo(f"Invalid edit: {error}")
            text = edited_text
            continue
        if edited == result:
            return result
        save_draft_edit(
            path,
            result,
            edited,
            settings,
            original_text,
            edited_text,
            lambda: snapshot_before_destructive(petri_dir, dish_id),
        )
        return edited
