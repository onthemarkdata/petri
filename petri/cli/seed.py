"""petri seed command — decompose a claim into a colony."""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import typer

from petri.cli._bootstrap import (
    find_petri_dir,
    get_dish_id,
    resolve_provider,
)
from petri.cli_ui import print_error_and_exit
from petri.storage.paths import (
    events_path as events_file_path,
    node_dir as node_dir_for,
    parse_node_id,
)


# ── Module-private helpers (extracted from nested closures) ──────────────


def _run_substance_check(
    claim: str,
    provider,
    questionary,
    *,
    force_plain: bool = False,
):
    """Agentic substance check before the clarifying-question wizard.

    Returns (claim, skip_wizard, aborted). If the model classifies the claim
    as non-substantive, the user is offered Continue / Edit / Abort. Editing
    re-runs the check on the new text. ``skip_wizard`` is True when the user
    chooses Continue (so the wizard is bypassed).
    """
    from petri.cli_ui import Spinner

    while True:
        try:
            with Spinner("substance check", force_plain=force_plain) as spinner:
                assessment = provider.assess_claim_substance(
                    claim, on_progress=spinner.update
                )
        except Exception as exc:
            typer.echo(
                f"Warning: substance check failed ({exc}); continuing anyway.",
                err=True,
            )
            return claim, False, False

        if assessment.get("is_substantive", True):
            return claim, False, False

        reason = assessment.get("reason", "")
        suggested = assessment.get("suggested_rewrite", "")

        typer.echo("This does not look like a substantive research claim.")
        if reason:
            typer.echo(f"  Reason: {reason}")
        if suggested:
            typer.echo(f"  Suggested rewrite: {suggested}")

        action = questionary.select(
            "How do you want to proceed?",
            choices=["Edit claim", "Continue anyway", "Abort"],
            default="Edit claim",
        ).ask()

        if action is None or action == "Abort":
            return claim, False, True

        if action == "Continue anyway":
            return claim, True, False

        # Edit claim — re-prompt and loop
        new_claim = questionary.text(
            "Edit the claim:", default=suggested or claim
        ).ask()
        if new_claim is None:
            return claim, False, True
        claim = new_claim.strip() or claim


def _events_path_for(colony_path: Path, colony_model, node_id: str) -> Path:
    """Resolve a node's events.jsonl path.

    Prefers the colony model's ``node_paths`` mapping when available,
    falling back to parsing the node ID via ``storage.paths``.
    """
    relative_path = colony_model.node_paths.get(node_id)
    if relative_path:
        return colony_path / relative_path / "events.jsonl"
    _, _, level, seq = parse_node_id(node_id)
    return events_file_path(node_dir_for(colony_path, level, seq))


def _log_node_event(
    events_path: Path,
    node_id: str,
    event_type: str,
    data: dict | None = None,
) -> None:
    """Append an event to a node's events.jsonl with error handling."""
    from petri.storage.event_log import append_event

    try:
        append_event(
            events_path=events_path,
            node_id=node_id,
            event_type=event_type,
            agent="decomposition_lead",
            iteration=0,
            data=data or {},
        )
    except Exception as exc:
        typer.echo(
            f"Warning: failed to log event {event_type} for {node_id}: {exc}",
            err=True,
        )


def _make_node_created_callback(
    *,
    graph,
    colony_model,
    colony_path: Path,
    spinner,
    log_event: Callable[[str, str, dict], None],
) -> Callable:
    """Return a callback for ``decompose_claim`` that persists each new node.

    The callback adds the node + edges to the graph, re-serializes the
    colony, logs a ``node_created`` event, and prints the new claim above
    the live spinner so the user retains a permanent record.
    """
    from petri.graph.colony import serialize_colony

    def _on_node_created(node, new_edges) -> None:
        try:
            graph.add_node(node)
            for edge in new_edges:
                graph.add_edge(edge)
            # Re-serialize incrementally — serialize_colony preserves
            # existing events.jsonl files (touch only on first create).
            serialize_colony(graph, colony_model, colony_path)
            log_event(
                node.id,
                "node_created",
                {
                    "level": node.level,
                    "parents": [edge.from_node for edge in new_edges],
                },
            )
            # Print the new claim as a permanent line above the live
            # spinner so the user has a record of what was just created.
            spinner.print_line(node.claim_text)
        except Exception as exc:
            typer.echo(
                f"Warning: failed to persist node {node.id}: {exc}",
                err=True,
            )

    return _on_node_created


# ── Command registration ────────────────────────────────────────────────


def register(app: typer.Typer) -> None:
    @app.command()
    def seed(
        source: str = typer.Argument(
            ..., help="Claim text, URL, or file path to decompose"
        ),
        colony: Optional[str] = typer.Option(
            None, help="Colony name (default: auto-generated)"
        ),
        no_questions: bool = typer.Option(
            False, "--no-questions", help="Skip clarifying questions and auto-accept"
        ),
    ) -> None:
        """Seed a new colony from a claim, URL, or file. Requires Claude Code CLI."""
        petri_dir = find_petri_dir()
        dish_id = get_dish_id(petri_dir)

        from petri.cli_ui import Spinner
        from petri.graph.colony import ColonyGraph, serialize_colony
        from petri.models import Colony, Node, build_node_key
        from petri.reasoning.decomposer import (
            decompose_claim,
            format_colony_display,
            generate_clarifying_questions,
            generate_colony_name,
        )
        from petri.reasoning.ingest import ingest

        # ── Resolve provider up front; hard-fail if unavailable ──
        try:
            provider = resolve_provider(petri_dir)
        except FileNotFoundError as exc:
            print_error_and_exit(
                f"Error: {exc}\nRun 'petri inspect' to verify prerequisites.",
                code=2,
            )

        if provider is None:
            print_error_and_exit(
                "Error: petri seed requires an LLM provider. "
                "Configure 'model' in .petri/defaults/petri.yaml.",
                code=2,
            )

        # ── Ingest the source to extract content ──
        ingested = ingest(source)
        source_type = ingested.source_type

        if source_type != "text":
            typer.echo(f"Ingested {source_type}: {ingested.title or source}")
            meta = ingested.metadata
            if meta.get("page_count"):
                typer.echo(f"  Pages: {meta['page_count']}")
            typer.echo(f"  Content length: {len(ingested.content)} characters\n")

        content = ingested.content
        title = ingested.title

        if source_type == "text":
            claim = content
        elif title:
            claim = f"{title}\n\nSource content:\n{content}"
        else:
            claim = content

        # ── Detect interactive TTY for the wizard ──
        # NOTE: We intentionally inline this instead of using
        # detect_interactive_mode() from _bootstrap because seed needs
        # to distinguish "no TTY" from "TTY but questionary missing" in
        # order to reproduce the original warning-only-on-import-failure
        # behaviour.
        import sys

        interactive = sys.stdin.isatty() and not no_questions
        force_plain_spinner = not interactive
        questionary = None
        if interactive:
            try:
                import questionary as _questionary  # type: ignore

                questionary = _questionary
            except ImportError:
                typer.echo(
                    "Warning: questionary not installed. Running non-interactively.",
                    err=True,
                )
                interactive = False
                force_plain_spinner = True

        # ── Agentic substance check ──
        skip_wizard = no_questions
        if interactive and not skip_wizard:
            claim, skip_wizard, aborted = _run_substance_check(
                claim, provider, questionary, force_plain=force_plain_spinner
            )
            if aborted:
                typer.echo("Aborted.")
                raise typer.Exit(code=1)

        # ── Print finalized claim so the user has a permanent record ──
        typer.echo()
        typer.echo(f"Claim: {claim}")
        typer.echo()

        # ── Phase C: Pre-create colony directory + center node ──
        colony_name = colony or generate_colony_name(claim)
        colony_id = f"{dish_id}-{colony_name}"
        center_id = build_node_key(dish_id, colony_name, 0, 0)
        colony_path = petri_dir / "petri-dishes" / colony_name

        # Wipe any leftover from a previous run before we start writing
        if colony_path.exists():
            shutil.rmtree(colony_path)

        center_node = Node(
            id=center_id,
            colony_id=colony_id,
            claim_text=claim,
            level=0,
        )
        graph = ColonyGraph(colony_id=colony_id)
        graph.add_node(center_node)

        colony_model = Colony(
            id=colony_id,
            dish=dish_id,
            center_claim=claim,
            center_node_id=center_id,
            clarifications=[],
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        serialize_colony(graph, colony_model, colony_path)

        def log_for_this_run(
            node_id: str, event_type: str, data: dict | None = None
        ) -> None:
            events_file = _events_path_for(colony_path, colony_model, node_id)
            _log_node_event(events_file, node_id, event_type, data)

        log_for_this_run(
            center_id, "seed_started", {"claim": claim, "no_questions": no_questions}
        )

        # ── Phase D: Clarifying questions wizard ──
        clarifications: list[dict] = []
        if interactive and not skip_wizard:
            log_for_this_run(center_id, "clarifying_questions_requested")
            try:
                with Spinner(
                    "clarifying questions", force_plain=force_plain_spinner
                ) as spinner:
                    questions = generate_clarifying_questions(
                        claim, provider=provider, on_progress=spinner.update
                    )
            except Exception as exc:
                log_for_this_run(
                    center_id, "clarifying_questions_failed", {"error": str(exc)}
                )
                print_error_and_exit(
                    f"Failed to generate clarifying questions: {exc}", code=2
                )
            log_for_this_run(
                center_id,
                "clarifying_questions_received",
                {"count": len(questions)},
            )

            for question in questions:
                question_text = question.question
                options = question.options

                if options:
                    choices = [*options, "Skip"]
                    answer = questionary.select(
                        question_text, choices=choices
                    ).ask()
                else:
                    answer = questionary.text(
                        f"{question_text} (leave blank to skip)"
                    ).ask()

                if answer is None:
                    typer.echo("Cancelled.")
                    raise typer.Exit(code=1)

                if answer == "Skip" or not answer.strip():
                    log_for_this_run(
                        center_id,
                        "clarification_skipped",
                        {"question": question_text},
                    )
                    continue

                clarifications.append(
                    {"question": question_text, "answer": answer}
                )
                log_for_this_run(
                    center_id,
                    "clarification_recorded",
                    {"question": question_text, "answer": answer},
                )

            # Persist user clarifications onto the colony so a kill mid-flow
            # doesn't lose them.
            colony_model.clarifications = clarifications
            serialize_colony(graph, colony_model, colony_path)

        # ── Phase E: Decompose / Accept / Regenerate / Abort loop ──
        guidance = ""
        result = None
        while True:
            # Each re-roll wipes the colony dir and rebuilds — no stale state
            if colony_path.exists():
                shutil.rmtree(colony_path)
            graph = ColonyGraph(colony_id=colony_id)
            graph.add_node(center_node)
            serialize_colony(graph, colony_model, colony_path)

            log_for_this_run(
                center_id,
                "decomposition_started",
                {"guidance": guidance or None},
            )

            try:
                with Spinner(
                    "decomposing claim", force_plain=force_plain_spinner
                ) as spinner:
                    on_node_created = _make_node_created_callback(
                        graph=graph,
                        colony_model=colony_model,
                        colony_path=colony_path,
                        spinner=spinner,
                        log_event=log_for_this_run,
                    )
                    result = decompose_claim(
                        claim=claim,
                        clarifications=clarifications,
                        dish_id=dish_id,
                        colony_name=colony_name,
                        provider=provider,
                        guidance=guidance,
                        on_progress=spinner.update,
                        on_node_created=on_node_created,
                        center=center_node,
                    )
            except Exception as exc:
                log_for_this_run(
                    center_id,
                    "decomposition_failed",
                    {"error": str(exc)},
                )
                print_error_and_exit(f"Decomposition failed: {exc}", code=2)

            log_for_this_run(
                center_id,
                "decomposition_completed",
                {"node_count": len(result.nodes)},
            )

            typer.echo(format_colony_display(result))

            if not interactive:
                break  # non-interactive: auto-accept (used by --no-questions)

            action = questionary.select(
                "What would you like to do?",
                choices=["Accept", "Regenerate with guidance", "Abort"],
                default="Accept",
            ).ask()

            if action is None or action == "Abort":
                if colony_path.exists():
                    shutil.rmtree(colony_path)
                typer.echo("Decomposition rejected.")
                raise typer.Exit(code=1)

            if action == "Accept":
                break

            # Regenerate path — Claude-Code-style free-text feedback
            guidance = questionary.text(
                "What should change? (free text, press Enter to re-roll as-is)"
            ).ask()
            if guidance is None:
                if colony_path.exists():
                    shutil.rmtree(colony_path)
                typer.echo("Cancelled.")
                raise typer.Exit(code=1)
            # Loop iterates: top of the loop wipes the colony dir and rebuilds

        # Final colony.json rewrite — ensures node_paths is consistent and
        # persists the user's clarifications alongside the approved decomposition.
        colony_model.clarifications = clarifications
        serialize_colony(graph, colony_model, colony_path)
        log_for_this_run(
            center_id,
            "colony_approved",
            {"node_count": len(result.nodes) if result else 0},
        )

        typer.echo(
            f"\nColony '{colony_name}' created with "
            f"{len(result.nodes) if result else 0} nodes."
        )
