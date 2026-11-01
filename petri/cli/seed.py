"""petri seed command — decompose a claim into a colony."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

import typer

from petri.telemetry.commands import traced_command

from petri.cli._bootstrap import (
    find_petri_dir,
    get_dish_id,
    resolve_provider,
)
from petri.cli_ui import print_error_and_exit
from petri.execution import resolve_backend
from petri.storage.colonies import (
    clear_unapproved_colony,
    colony_exists,
    reserve_colony,
    save_colony,
)
from petri.storage.seed_guard import ColonyExistsError
from petri.storage.backup import snapshot_before_destructive
from petri.storage.database import SQLiteEventLog
from petri.storage.paths import dish_db_path

if TYPE_CHECKING:
    from petri.models import ClarifyingQuestion


# ── Module-private helpers (extracted from nested closures) ──────────────


def _clarifications_to_models(clarifications: list[dict]) -> list[ClarifyingQuestion]:
    """Convert stored clarification dicts into ClarifyingQuestion models.

    seed persists clarifications as plain dicts (Colony.clarifications is
    list[dict]), but decompose_claim consumes ClarifyingQuestion objects and
    reads .question/.answer/.options. Passing raw dicts raised AttributeError
    once a user answered a clarifying question; converting here fixes that.
    """
    from petri.models import ClarifyingQuestion

    return [
        ClarifyingQuestion(
            question=clar.get("question", ""),
            answer=clar.get("answer"),
            options=clar.get("options", []),
        )
        for clar in clarifications
    ]


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
                assessment = provider.assess_claim_substance(claim, on_progress=spinner.update)
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
        new_claim = questionary.text("Edit the claim:", default=suggested or claim).ask()
        if new_claim is None:
            return claim, False, True
        claim = new_claim.strip() or claim


def _events_path_for(colony_path: Path, colony_model, cell_id: str) -> SQLiteEventLog:
    """Select one cell's event stream in the dish database."""
    return SQLiteEventLog(colony_path, cell_id)


def _log_cell_event(
    events_path: SQLiteEventLog,
    cell_id: str,
    event_type: str,
    data: dict | None = None,
    *,
    sequence: int = 0,
) -> None:
    """Persist a seed event, failing if its history cannot be saved."""
    from petri.storage.event_log import append_event

    append_event(
        events_path=events_path,
        cell_id=cell_id,
        event_type=event_type,
        agent="decomposition_lead",
        iteration=0,
        data=data or {},
        node="seed",
        sequence=sequence,
    )


def _make_cell_created_callback(
    *,
    graph,
    colony_model,
    colony_path: Path,
    spinner,
    log_event: Callable[[str, str, dict], None],
) -> Callable:
    """Persist each created cell and its edges before reporting progress."""

    def _on_cell_created(cell, new_edges) -> None:
        graph.add_cell(cell)
        for edge in new_edges:
            graph.add_edge(edge)
        save_colony(colony_path, graph, colony_model)
        log_event(
            cell.id,
            "cell_created",
            {
                "level": cell.level,
                "parents": [edge.from_cell for edge in new_edges],
            },
        )
        spinner.print_line(cell.claim_text)

    return _on_cell_created


class _IncompleteSeed(RuntimeError):
    pass


class _CompletedSeedRunner:
    """Rebuild committed output events and signal if a new model call is needed."""

    def initial_decomposition(self, request, on_progress=None):
        raise _IncompleteSeed()

    def expand_why(self, request, on_progress=None):
        raise _IncompleteSeed()


def _resume_colony(
    petri_dir: Path, dish_id: str, colony_name: str, no_questions: bool, backend
) -> None:
    """Resume the retained seed draft through the selected execution backend."""
    from petri.graph.colony import ColonyGraph
    from petri.reasoning.decomposer import decompose_claim, format_colony_display
    from petri.reasoning.decomposition_settings import DecompositionSettings
    from petri.storage import database
    from petri.storage.colonies import load_colony

    path = dish_db_path(petri_dir, dish_id)
    colony_id = f"{dish_id}-{colony_name}"
    if not colony_exists(path, colony_id):
        print_error_and_exit(f"Colony '{colony_name}' does not exist; seed it first.", code=1)
    graph, colony_model = load_colony(path, colony_id)
    center_id = colony_model.center_cell_id
    events = database.read_events(path, cell_id=center_id)
    if any(event["type"] == "colony_approved" for event in events):
        typer.echo(f"Colony '{colony_name}': nothing to resume.")
        return
    arguments = dict(
        claim=colony_model.center_claim,
        clarifications=_clarifications_to_models(colony_model.clarifications),
        dish_id=dish_id,
        colony_name=colony_name,
        settings=DecompositionSettings.from_dish(petri_dir),
        db_path=path,
        resume=True,
    )
    provider = None
    try:
        decompose_claim(**arguments, provider=None, agent_runner=_CompletedSeedRunner())
        runner = _CompletedSeedRunner()
    except _IncompleteSeed:
        provider = resolve_provider(petri_dir)
        if provider is None or not hasattr(provider, "initial_decomposition"):
            print_error_and_exit("Seed resume requires the configured typed agent backend.", code=2)
        runner = provider
    try:
        result = backend.seed_colony(
            petri_dir=petri_dir, **arguments, provider=runner, agent_runner=runner
        )
    except Exception as error:
        print_error_and_exit(f"Seed resume failed: {error}", code=2)
    finally:
        if provider is not None and hasattr(provider, "close"):
            provider.close()
    graph = ColonyGraph(colony_id=colony_id)
    for cell in result.cells:
        graph.add_cell(cell)
    for edge in result.edges:
        graph.add_edge(edge)
    save_colony(path, graph, colony_model)
    typer.echo(format_colony_display(result))
    import sys

    if (
        sys.stdin.isatty()
        and not no_questions
        and not typer.confirm("Accept this colony?", default=True)
    ):
        typer.echo("Draft retained. Run seed --resume to review it again.")
        return
    _log_cell_event(
        SQLiteEventLog(path), center_id, "colony_approved", {"cell_count": len(result.cells)}
    )
    typer.echo(f"Colony '{colony_name}' resumed with {len(result.cells)} cells.")


# ── Command registration ────────────────────────────────────────────────


def register(app: typer.Typer) -> None:
    @app.command()
    @traced_command("seed")
    def seed(
        source: str = typer.Argument("", help="Claim text, URL, or file path to decompose"),
        colony: Optional[str] = typer.Option(None, help="Colony name (default: auto-generated)"),
        resume: bool = typer.Option(
            False, "--resume", help="Continue an interrupted seed; requires --colony"
        ),
        no_questions: bool = typer.Option(
            False, "--no-questions", help="Skip clarifying questions and auto-accept"
        ),
    ) -> None:
        """Seed a colony from a claim, URL, or file. Seed foundations first so later colonies can reference their premises."""
        petri_dir = find_petri_dir()
        dish_id = get_dish_id(petri_dir)
        try:
            backend = resolve_backend(petri_dir)
        except ValueError as error:
            print_error_and_exit(str(error), code=2)
        typer.echo(f"Execution backend: {getattr(backend, 'backend_name', 'custom')}")
        if resume:
            if not colony:
                print_error_and_exit("Use --colony to identify the seed to resume.", code=2)
            _resume_colony(petri_dir, dish_id, colony, no_questions, backend)
            return
        if not source:
            print_error_and_exit("Provide a claim, URL, or file path to seed.", code=2)
        from petri.reasoning.decomposition_settings import DecompositionSettings
        from petri.harness.legacy_bridge import HarnessInferenceProvider

        settings = DecompositionSettings.from_dish(petri_dir)

        from petri.cli_ui import Spinner
        from petri.graph.colony import ColonyGraph
        from petri.models import Cell, Colony, build_cell_key
        from petri.reasoning.decomposer import (
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
        #
        # `interactive` controls whether the questionary wizard runs —
        # that's a stdin concern. `force_plain_spinner` controls whether
        # the Spinner uses its animated braille output — that's a
        # stdout concern and must be decoupled, otherwise launching
        # `petri seed` with stdin redirected (or from the dashboard's
        # PTY bridge where stdin=DEVNULL but stdout is a PTY slave)
        # silently falls back to the plain `→` fallback even though
        # the terminal is perfectly capable of rendering the animation.
        import sys

        interactive = sys.stdin.isatty() and not no_questions
        force_plain_spinner = not sys.stdout.isatty()
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

        # ── Phase C: Reserve the colony and persist its center ──
        colony_name = colony or generate_colony_name(claim)
        colony_id = f"{dish_id}-{colony_name}"
        center_id = build_cell_key(dish_id, colony_name, 0, 0)
        colony_path = dish_db_path(petri_dir, dish_id)
        if colony_exists(colony_path, colony_id):
            print_error_and_exit(
                f"Colony '{colony_name}' already exists. Choose another name; "
                "use seed --resume --colony with this name to continue an interrupted seed.",
                code=1,
            )

        center_cell = Cell(
            id=center_id,
            colony_id=colony_id,
            claim_text=claim,
            level=0,
        )
        graph = ColonyGraph(colony_id=colony_id)
        graph.add_cell(center_cell)

        colony_model = Colony(
            id=colony_id,
            dish=dish_id,
            center_claim=claim,
            center_cell_id=center_id,
            clarifications=[],
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        try:
            reserve_colony(colony_path, colony_model)
        except ColonyExistsError as error:
            print_error_and_exit(str(error), code=1)
        save_colony(colony_path, graph, colony_model)

        event_sequences: dict[tuple[str, str], int] = {}

        def log_for_this_run(cell_id: str, event_type: str, data: dict | None = None) -> None:
            events_file = _events_path_for(colony_path, colony_model, cell_id)
            key = (cell_id, event_type)
            sequence = event_sequences.get(key, 0)
            _log_cell_event(events_file, cell_id, event_type, data, sequence=sequence)
            event_sequences[key] = sequence + 1

        log_for_this_run(center_id, "seed_started", {"claim": claim, "no_questions": no_questions})

        # ── Phase D: Clarifying questions wizard ──
        clarifications: list[dict] = []
        if interactive and not skip_wizard:
            # interactive is only True when the questionary import above
            # succeeded, so narrow away the Optional for mypy.
            assert questionary is not None
            log_for_this_run(center_id, "clarifying_questions_requested")
            try:
                with Spinner("clarifying questions", force_plain=force_plain_spinner) as spinner:
                    questions = generate_clarifying_questions(
                        claim, provider=provider, on_progress=spinner.update
                    )
            except Exception as exc:
                log_for_this_run(center_id, "clarifying_questions_failed", {"error": str(exc)})
                print_error_and_exit(f"Failed to generate clarifying questions: {exc}", code=2)
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
                    answer = questionary.select(question_text, choices=choices).ask()
                else:
                    answer = questionary.text(f"{question_text} (leave blank to skip)").ask()

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

                clarifications.append({"question": question_text, "answer": answer})
                log_for_this_run(
                    center_id,
                    "clarification_recorded",
                    {"question": question_text, "answer": answer},
                )

            # Persist user clarifications onto the colony so a kill mid-flow
            # doesn't lose them.
            colony_model.clarifications = clarifications
            save_colony(colony_path, graph, colony_model)

        # ── Phase E: Decompose / Accept / Regenerate / Abort loop ──
        guidance = ""
        result = None
        while True:
            # Only this run's cells are replaced during regeneration.
            snapshot_before_destructive(petri_dir, dish_id)
            clear_unapproved_colony(colony_path, colony_id)
            graph = ColonyGraph(colony_id=colony_id)
            graph.add_cell(center_cell)
            save_colony(colony_path, graph, colony_model)

            log_for_this_run(
                center_id,
                "decomposition_started",
                {"guidance": guidance or None},
            )

            try:
                with Spinner("decomposing claim", force_plain=force_plain_spinner) as spinner:
                    on_cell_created = _make_cell_created_callback(
                        graph=graph,
                        colony_model=colony_model,
                        colony_path=colony_path,
                        spinner=spinner,
                        log_event=log_for_this_run,
                    )
                    result = backend.seed_colony(
                        petri_dir=petri_dir,
                        claim=claim,
                        clarifications=_clarifications_to_models(clarifications),
                        dish_id=dish_id,
                        colony_name=colony_name,
                        provider=provider,
                        guidance=guidance,
                        on_progress=spinner.update,
                        on_cell_created=on_cell_created,
                        center=center_cell,
                        settings=settings,
                        agent_runner=provider
                        if isinstance(provider, HarnessInferenceProvider)
                        else None,
                        db_path=colony_path,
                    )
            except Exception as exc:
                log_for_this_run(
                    center_id,
                    "decomposition_failed",
                    {"error": str(exc)},
                )
                print_error_and_exit(f"Decomposition failed: {exc}", code=2)

            graph = ColonyGraph(colony_id=colony_id)
            for cell_result in result.cells:
                graph.add_cell(cell_result)
            for edge_result in result.edges:
                graph.add_edge(edge_result)

            log_for_this_run(
                center_id,
                "decomposition_completed",
                {"cell_count": len(result.cells)},
            )

            typer.echo(format_colony_display(result))

            if not interactive:
                break  # non-interactive: auto-accept (used by --no-questions)

            # Reaching here means interactive is True, so the questionary
            # import above succeeded; narrow away the Optional for mypy.
            assert questionary is not None
            action = questionary.select(
                "What would you like to do?",
                choices=["Accept", "Regenerate with guidance", "Abort"],
                default="Accept",
            ).ask()

            if action is None or action == "Abort":
                # Only this run's cells are removed when this run is rejected.
                snapshot_before_destructive(petri_dir, dish_id)
                clear_unapproved_colony(colony_path, colony_id)
                typer.echo("Decomposition rejected.")
                raise typer.Exit(code=1)

            if action == "Accept":
                break

            # Regenerate path — Claude-Code-style free-text feedback
            guidance = questionary.text(
                "What should change? (free text, press Enter to re-roll as-is)"
            ).ask()
            if guidance is None:
                # Only this run's cells are removed when this run is rejected.
                snapshot_before_destructive(petri_dir, dish_id)
                clear_unapproved_colony(colony_path, colony_id)
                typer.echo("Cancelled.")
                raise typer.Exit(code=1)
            # The next iteration replaces draft rows and retains their events.

        # Persist clarifications alongside the approved cells.
        colony_model.clarifications = clarifications
        save_colony(colony_path, graph, colony_model)
        log_for_this_run(
            center_id,
            "colony_approved",
            {"cell_count": len(result.cells) if result else 0},
        )

        typer.echo(
            f"\nColony '{colony_name}' created with {len(result.cells) if result else 0} cells."
        )
