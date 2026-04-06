"""Typer CLI for the Petri Research Orchestration Framework.

7 commands: init, seed, check, analyze, grow, stop, feed.
All commands except ``init`` require a ``.petri/`` directory to exist.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(
    name="petri",
    help="Petri -- colony-based research orchestration framework",
    no_args_is_help=True,
)

DEFAULTS_DIR = Path(__file__).parent / "defaults"


# ── Helper Functions ─────────────────────────────────────────────────────


def _resolve_provider(petri_dir: Path):
    """Resolve an InferenceProvider from petri.yaml config.

    Returns None if no provider can be constructed (caller must handle).
    """
    config = _load_dish_config(petri_dir)
    # TODO: wire up actual inference providers based on config
    # e.g. harness=claude-code, model=gemma-3-4b-it, etc.
    return None


def _find_petri_dir(start: Path | None = None) -> Path:
    """Find the .petri directory from the current or given path."""
    start = start or Path.cwd()
    petri_dir = Path(start) / ".petri"
    if not petri_dir.exists():
        typer.echo(
            "No petri dish found. Run `petri init` to create one.", err=True
        )
        raise typer.Exit(code=3)
    return petri_dir


def _load_dish_config(petri_dir: Path) -> dict:
    """Load petri.yaml config from .petri/defaults/petri.yaml."""
    config_path = petri_dir / "defaults" / "petri.yaml"
    if not config_path.exists():
        return {}
    try:
        import yaml

        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        # Fallback: simple line parsing for basic fields
        config: dict = {}
        for line in config_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            if ":" in line and not line.startswith(" "):
                key, _, val = line.partition(":")
                config[key.strip()] = val.strip()
        return config


def _get_dish_id(petri_dir: Path) -> str:
    """Get the dish ID from config or derive from directory name."""
    config = _load_dish_config(petri_dir)
    return config.get("name", petri_dir.parent.name)


def _load_colonies(petri_dir: Path, dish_id: str) -> list[tuple]:
    """Load all colonies from the petri-dishes directory."""
    from petri.colony import deserialize_colony

    colonies = []
    dishes_dir = petri_dir / "petri-dishes"
    if not dishes_dir.exists():
        return colonies
    for colony_dir in sorted(dishes_dir.iterdir()):
        if colony_dir.is_dir():
            try:
                graph, colony = deserialize_colony(colony_dir, dish_id)
                colonies.append((graph, colony))
            except Exception:
                continue
    return colonies


# ── Commands ─────────────────────────────────────────────────────────────


@app.command()
def init(
    path: str = typer.Argument(".", help="Directory to initialize"),
    name: Optional[str] = typer.Option(None, help="Dish name (default: directory name)"),
) -> None:
    """Initialize a new petri dish directory structure."""
    target = Path(path).resolve()
    petri_dir = target / ".petri"

    if petri_dir.exists():
        typer.echo(f"Error: .petri/ already exists at {target}", err=True)
        raise typer.Exit(code=1)

    # Derive dish name
    dish_name = name or target.name

    try:
        # Create .petri/ directory structure
        petri_dir.mkdir(parents=True)

        # Copy defaults (petri.yaml + constitution.md)
        defaults_dest = petri_dir / "defaults"
        defaults_dest.mkdir()

        # Copy petri.yaml from package defaults and set dish name
        src_config = DEFAULTS_DIR / "petri.yaml"
        if src_config.exists():
            import yaml

            with open(src_config) as f:
                config = yaml.safe_load(f)
            config["name"] = dish_name
            with open(defaults_dest / "petri.yaml", "w") as f:
                f.write("# Petri Dish Configuration\n")
                f.write(f"# Initialized for dish: {dish_name}\n\n")
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        else:
            # Minimal fallback
            (defaults_dest / "petri.yaml").write_text(
                f"name: {dish_name}\n", encoding="utf-8"
            )

        # Copy constitution.md
        src_constitution = DEFAULTS_DIR / "constitution.md"
        if src_constitution.exists():
            shutil.copy2(src_constitution, defaults_dest / "constitution.md")

        # Create petri-dishes directory
        (petri_dir / "petri-dishes").mkdir()

        # Create queue.json
        queue_data = {"version": 1, "last_updated": None, "entries": {}}
        queue_path = petri_dir / "queue.json"
        queue_path.write_text(
            json.dumps(queue_data, indent=2) + "\n", encoding="utf-8"
        )

        typer.echo(f"Initialized petri dish '{dish_name}' at {target}")

    except OSError as exc:
        typer.echo(f"Error creating petri dish: {exc}", err=True)
        raise typer.Exit(code=2)


@app.command()
def seed(
    claim: str = typer.Argument(..., help="The thesis to decompose"),
    colony: Optional[str] = typer.Option(
        None, help="Colony name (default: auto-generated)"
    ),
    no_questions: bool = typer.Option(
        False, "--no-questions", help="Skip clarifying questions"
    ),
) -> None:
    """Seed a new colony from a claim."""
    petri_dir = _find_petri_dir()
    dish_id = _get_dish_id(petri_dir)

    from petri.decomposer import (
        decompose_claim,
        format_colony_display,
        generate_clarifying_questions,
        generate_colony_name,
    )

    # Gather clarifications
    clarifications: list[dict] = []

    if not no_questions:
        try:
            import questionary
            import sys

            if not sys.stdin.isatty():
                no_questions = True
        except ImportError:
            typer.echo(
                "Warning: questionary not installed. Skipping clarifying questions.",
                err=True,
            )
            no_questions = True

        if not no_questions:
            questions = generate_clarifying_questions(claim)
            for q in questions:
                question_text = q.get("question", "")
                options = q.get("options", [])

                if options:
                    answer = questionary.select(
                        question_text, choices=options
                    ).ask()
                else:
                    answer = questionary.text(question_text).ask()

                if answer is None:
                    # User cancelled (Ctrl+C)
                    typer.echo("Cancelled.")
                    raise typer.Exit(code=1)

                clarifications.append(
                    {"question": question_text, "answer": answer}
                )

    # Generate colony name if not provided
    colony_name = colony or generate_colony_name(claim)

    # Decompose the claim
    try:
        result = decompose_claim(
            claim=claim,
            clarifications=clarifications,
            dish_id=dish_id,
            colony_name=colony_name,
        )
    except Exception as exc:
        typer.echo(f"Decomposition failed: {exc}", err=True)
        raise typer.Exit(code=2)

    # Display the colony
    display = format_colony_display(result)
    typer.echo(display)

    # Prompt for approval
    try:
        import questionary
        import sys

        if sys.stdin.isatty():
            approved = questionary.confirm(
                "Approve this decomposition?", default=True
            ).ask()
        else:
            # Non-interactive terminal: auto-approve
            approved = True
    except (ImportError, OSError):
        # Non-interactive fallback: auto-approve
        approved = True

    if approved is None or not approved:
        typer.echo("Decomposition rejected.")
        raise typer.Exit(code=1)

    # Serialize colony to filesystem
    from petri.colony import ColonyGraph, serialize_colony
    from petri.event_log import append_event
    from petri.models import Colony, Edge, Node

    now = datetime.now(timezone.utc).isoformat()
    colony_id = f"{dish_id}-{colony_name}"

    # Build the graph
    graph = ColonyGraph(colony_id=colony_id)
    nodes = result.nodes
    edges = result.edges

    for node in nodes:
        graph.add_node(node)
    for edge in edges:
        graph.add_edge(edge)

    # Create Colony model
    center_node_id = nodes[0].id if nodes else ""
    colony_model = Colony(
        id=colony_id,
        dish=dish_id,
        center_claim=claim,
        center_node_id=center_node_id,
        clarifications=clarifications,
        created_at=now,
    )

    # Write to filesystem
    colony_path = petri_dir / "petri-dishes" / colony_name
    serialize_colony(graph, colony_model, colony_path)

    # Log decomposition_created events for each node
    for node in nodes:
        child_ids = [
            e.from_node for e in edges if e.to_node == node.id
        ]
        node_rel_path = colony_model.node_paths.get(node.id, f"{node.id.split('-')[-2]}-{node.id.split('-')[-1]}")
        events_path = colony_path / node_rel_path / "events.jsonl"
        append_event(
            events_path=events_path,
            node_id=node.id,
            event_type="decomposition_created",
            agent="decomposition_lead",
            iteration=0,
            data={
                "parent_node_id": node.id,
                "child_node_ids": child_ids,
            },
        )

    typer.echo(f"\nColony '{colony_name}' created with {len(nodes)} nodes.")


@app.command()
def check(
    colony_name: Optional[str] = typer.Option(
        None, "--colony", help="Filter to one colony"
    ),
    node: Optional[str] = typer.Option(
        None, "--node", help="Detailed view of a single node"
    ),
    json_output: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """Show current state of the petri dish."""
    petri_dir = _find_petri_dir()
    dish_id = _get_dish_id(petri_dir)
    colonies = _load_colonies(petri_dir, dish_id)

    if colony_name:
        colonies = [
            (g, c) for g, c in colonies if c.id.endswith(f"-{colony_name}")
        ]

    if not colonies:
        typer.echo("No colonies found.")
        raise typer.Exit(code=0)

    # Load queue for state info
    from petri.queue import load_queue

    queue_path = petri_dir / "queue.json"
    queue = load_queue(queue_path)
    queue_entries = queue.get("entries", {})

    # Detailed node view
    if node:
        found = False
        for graph, col in colonies:
            try:
                n = graph.get_node(node)
            except KeyError:
                continue
            found = True
            q_entry = queue_entries.get(node, {})

            if json_output:
                detail = {
                    "node_id": n.id,
                    "colony_id": n.colony_id,
                    "claim_text": n.claim_text,
                    "level": n.level,
                    "status": n.status.value,
                    "dependencies": n.dependencies,
                    "dependents": n.dependents,
                    "queue_state": q_entry.get("queue_state", ""),
                    "iteration": q_entry.get("iteration", 0),
                }
                typer.echo(json.dumps(detail, indent=2))
            else:
                typer.echo(f"Node: {n.id}")
                typer.echo(f"  Colony:       {n.colony_id}")
                typer.echo(f"  Claim:        {n.claim_text}")
                typer.echo(f"  Level:        {n.level}")
                typer.echo(f"  Status:       {n.status.value}")
                typer.echo(f"  Dependencies: {', '.join(n.dependencies) or '(none)'}")
                typer.echo(f"  Dependents:   {', '.join(n.dependents) or '(none)'}")
                if q_entry:
                    typer.echo(f"  Queue State:  {q_entry.get('queue_state', '')}")
                    typer.echo(f"  Iteration:    {q_entry.get('iteration', 0)}")

                # Show events
                from petri.event_log import load_events

                colony_slug = n.colony_id.replace(f"{dish_id}-", "", 1)
                colony_base = petri_dir / "petri-dishes" / colony_slug
                # Look up path from colony.json
                node_rel = col.node_paths.get(n.id)
                if not node_rel:
                    parts = n.id.split("-")
                    node_rel = f"{parts[-2]}-{parts[-1]}"
                events_path = colony_base / node_rel / "events.jsonl"
                events = load_events(events_path)
                if events:
                    typer.echo(f"  Events:       {len(events)}")
                    for evt in events[-5:]:
                        typer.echo(
                            f"    [{evt.get('type')}] {evt.get('agent')} "
                            f"iter={evt.get('iteration')} "
                            f"{evt.get('timestamp', '')[:19]}"
                        )
            break

        if not found:
            typer.echo(f"Node '{node}' not found.", err=True)
            raise typer.Exit(code=0)
        return

    # Table output
    all_data: list[dict] = []
    for graph, col in colonies:
        for n in graph.get_nodes():
            q_entry = queue_entries.get(n.id, {})
            all_data.append(
                {
                    "colony": col.id,
                    "level": n.level,
                    "node_id": n.id,
                    "claim": n.claim_text,
                    "status": n.status.value,
                    "queue_state": q_entry.get("queue_state", ""),
                }
            )

    if json_output:
        typer.echo(json.dumps(all_data, indent=2))
        return

    # Group by level
    levels: dict[int, list[dict]] = {}
    for item in all_data:
        levels.setdefault(item["level"], []).append(item)

    for level in sorted(levels.keys()):
        typer.echo(f"\nLevel {level}:")
        typer.echo(f"  {'Node ID':<40} {'Status':<16} {'Queue State':<16}")
        typer.echo(f"  {'-' * 40} {'-' * 16} {'-' * 16}")
        for item in levels[level]:
            typer.echo(
                f"  {item['node_id']:<40} {item['status']:<16} "
                f"{item['queue_state'] or '-':<16}"
            )
    typer.echo("")


@app.command()
def analyze(
    dashboard: bool = typer.Option(False, "--dashboard", help="Launch dashboard"),
    scan: bool = typer.Option(False, "--scan", help="Run contradiction scanner"),
    graph: bool = typer.Option(False, "--graph", help="Show colony graph"),
    connect: Optional[list[str]] = typer.Option(
        None, "--connect", help="Connect two nodes"
    ),
    colony_name: Optional[str] = typer.Option(
        None, "--colony", help="Filter to colony"
    ),
    format: str = typer.Option("text", "--format", help="Output format: text or dot"),
    fix: bool = typer.Option(False, "--fix", help="Auto-fix issues (with --scan)"),
    loop: bool = typer.Option(False, "--loop", help="Repeat until clean (with --scan)"),
    port: int = typer.Option(8090, "--port", help="Dashboard port"),
) -> None:
    """Analysis and visualization tools."""
    petri_dir = _find_petri_dir()
    dish_id = _get_dish_id(petri_dir)

    has_flag = dashboard or scan or graph or (connect is not None)

    # If no flags, present a menu
    if not has_flag:
        try:
            import questionary

            choice = questionary.select(
                "What would you like to do?",
                choices=["Graph", "Connect", "Dashboard", "Scan"],
            ).ask()

            if choice is None:
                raise typer.Exit(code=0)
            elif choice == "Graph":
                graph = True
            elif choice == "Connect":
                connect = []  # Will prompt below
            elif choice == "Dashboard":
                dashboard = True
            elif choice == "Scan":
                scan = True
        except ImportError:
            typer.echo(
                "No flag specified. Use --graph, --connect, --dashboard, or --scan.",
                err=True,
            )
            raise typer.Exit(code=1)

    # --dashboard
    if dashboard:
        from petri.dashboard.migrate import rebuild_sqlite
        from petri.dashboard.api import create_app
        import uvicorn

        db_path = petri_dir / "petri.sqlite"
        typer.echo("Building event index...")
        count = rebuild_sqlite(petri_dir, db_path)
        typer.echo(f"Indexed {count} events.")
        typer.echo(f"Starting dashboard at http://localhost:{port}")

        app_instance = create_app(petri_dir, db_path)
        uvicorn.run(app_instance, host="0.0.0.0", port=port, log_level="info")
        return

    # --scan
    if scan:
        from petri.scanner import auto_fix as scanner_auto_fix
        from petri.scanner import scan as run_scan
        from petri.scanner import scan_loop

        # Determine generated config dir (default: .claude/ in project root)
        generated_dir = petri_dir.parent / ".claude"
        if not generated_dir.exists():
            generated_dir = None

        if loop:
            issues = scan_loop(petri_dir, generated_dir)
        elif fix:
            issues = run_scan(petri_dir, generated_dir)
            fixable = [i for i in issues if i.fix_path]
            if fixable:
                fixed = scanner_auto_fix(fixable)
                typer.echo(f"Auto-fixed {len(fixed)} issues.")
            issues = run_scan(petri_dir, generated_dir)
        else:
            issues = run_scan(petri_dir, generated_dir)

        if not issues:
            typer.echo("No inconsistencies found.")
        else:
            typer.echo(f"\nFound {len(issues)} inconsistencies:\n")
            for i, issue in enumerate(issues, 1):
                typer.echo(f"  {i}. [{issue.category}] {issue.description}")
                if issue.file_path:
                    typer.echo(f"     File: {issue.file_path}")

        raise typer.Exit(code=0 if not issues else 1)

    # --graph
    if graph:
        colonies = _load_colonies(petri_dir, dish_id)

        if colony_name:
            colonies = [
                (g, c) for g, c in colonies if c.id.endswith(f"-{colony_name}")
            ]

        if not colonies:
            typer.echo("No colonies found.")
            raise typer.Exit(code=0)

        for g, col in colonies:
            if format == "dot":
                _render_dot(g, col)
            else:
                _render_text_tree(g, col)

        raise typer.Exit(code=0)

    # --connect
    if connect is not None:
        if len(connect) != 2:
            try:
                import questionary

                node1 = questionary.text("First node ID:").ask()
                node2 = questionary.text("Second node ID:").ask()
                if not node1 or not node2:
                    typer.echo("Cancelled.")
                    raise typer.Exit(code=0)
                connect = [node1, node2]
            except ImportError:
                typer.echo(
                    "Usage: petri analyze --connect NODE1 --connect NODE2",
                    err=True,
                )
                raise typer.Exit(code=1)

        node1_id, node2_id = connect[0], connect[1]
        colonies = _load_colonies(petri_dir, dish_id)

        # Find the nodes across colonies
        from petri.colony import serialize_colony
        from petri.models import Edge

        source_graph = None
        source_colony = None
        target_found = False

        for g, col in colonies:
            try:
                g.get_node(node1_id)
                source_graph = g
                source_colony = col
            except KeyError:
                pass
            try:
                g.get_node(node2_id)
                target_found = True
            except KeyError:
                pass

        if source_graph is None:
            typer.echo(f"Node '{node1_id}' not found.", err=True)
            raise typer.Exit(code=1)
        if not target_found:
            typer.echo(f"Node '{node2_id}' not found.", err=True)
            raise typer.Exit(code=1)

        edge = Edge(
            from_node=node1_id,
            to_node=node2_id,
            edge_type="cross_colony",
        )

        try:
            source_graph.add_edge(edge)
        except ValueError as exc:
            typer.echo(f"Cannot create edge: {exc}", err=True)
            raise typer.Exit(code=1)

        # Re-serialize the colony
        colony_slug = source_colony.id.replace(f"{dish_id}-", "", 1)
        colony_path = petri_dir / "petri-dishes" / colony_slug
        serialize_colony(source_graph, source_colony, colony_path)

        typer.echo(
            f"Cross-colony edge created: {node1_id} -> {node2_id}"
        )
        raise typer.Exit(code=0)


def _render_text_tree(graph, colony) -> None:
    """Render a colony as an indented text tree."""
    from petri.colony import ColonyGraph

    typer.echo(f"\nColony: {colony.id}")
    typer.echo(f"Center: {colony.center_claim}")
    typer.echo("")

    nodes = graph.get_nodes()
    if not nodes:
        typer.echo("  (empty)")
        return

    for node in nodes:
        indent = "  " * (node.level + 1)
        deps = graph.get_dependencies(node.id)
        dep_arrow = ""
        if deps:
            dep_arrow = f" -> [{', '.join(deps)}]"
        typer.echo(f"{indent}[L{node.level}] {node.id}: {node.claim_text}{dep_arrow}")

    typer.echo("")


def _render_dot(graph, colony) -> None:
    """Render a colony as Graphviz DOT format."""
    typer.echo(f'digraph "{colony.id}" {{')
    typer.echo("  rankdir=TB;")
    typer.echo(f'  label="{colony.center_claim}";')
    typer.echo("")

    for node in graph.get_nodes():
        label = node.claim_text.replace('"', '\\"')
        if len(label) > 50:
            label = label[:47] + "..."
        typer.echo(f'  "{node.id}" [label="{label}\\nL{node.level}"];')

    typer.echo("")

    for edge in graph.get_edges():
        style = ""
        if edge.edge_type == "cross_colony":
            style = ' [style=dashed, color=blue]'
        typer.echo(f'  "{edge.from_node}" -> "{edge.to_node}"{style};')

    typer.echo("}")


# ── Stub Commands (Later Phases) ─────────────────────────────────────────


@app.command()
def grow(
    nodes: Optional[list[str]] = typer.Argument(None, help="Node IDs to grow"),
    colony_name: Optional[str] = typer.Option(
        None, "--colony", help="Grow all in colony"
    ),
    all_nodes: bool = typer.Option(False, "--all", help="Grow all eligible"),
    max_concurrent: int = typer.Option(
        4, "--max-concurrent", help="Max parallel nodes"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would process"
    ),
) -> None:
    """Enqueue nodes and process through validation pipeline."""
    petri_dir = _find_petri_dir()
    dish_id = _get_dish_id(petri_dir)

    from petri.processor import NoProviderError, process_queue

    # Resolve the inference provider from petri.yaml config
    provider = _resolve_provider(petri_dir)

    try:
        result = process_queue(
            petri_dir=petri_dir,
            provider=provider,
            max_concurrent=max_concurrent,
            node_ids=nodes,
            colony_filter=colony_name,
            all_nodes=all_nodes,
            dry_run=dry_run,
        )
    except NoProviderError:
        typer.echo(
            "Error: No inference provider configured.\n"
            "Set model and harness in .petri/petri.yaml to enable processing.",
            err=True,
        )
        raise typer.Exit(code=1)

    if dry_run:
        would_process = result.get("would_process", [])
        if would_process:
            typer.echo(f"Would process {len(would_process)} nodes:")
            for nid in would_process:
                typer.echo(f"  {nid}")
        else:
            typer.echo("No eligible nodes found.")
        raise typer.Exit(code=0)

    processed = result.get("processed", 0)
    succeeded = result.get("succeeded", 0)
    failed = result.get("failed", 0)
    stalled = result.get("stalled", 0)

    typer.echo(f"\nProcessing complete: {processed} nodes processed")
    typer.echo(f"  Succeeded: {succeeded}")
    if stalled:
        typer.echo(f"  Stalled:   {stalled}")
    if failed:
        typer.echo(f"  Failed:    {failed}")

    # Show per-node results
    for r in result.get("results", []):
        node_id = r.get("node_id", "?")
        final = r.get("final_state", "?")
        iters = r.get("iterations", 0)
        events = r.get("events_logged", 0)
        typer.echo(f"  {node_id}: {final} ({iters} iterations, {events} events)")

    if stalled:
        raise typer.Exit(code=1)
    raise typer.Exit(code=0)


@app.command()
def stop(
    force: bool = typer.Option(False, "--force", help="Immediate stop"),
) -> None:
    """Gracefully stop all running tasks."""
    petri_dir = _find_petri_dir()

    from petri.processor import request_stop
    from petri.queue import list_queue, update_state
    from petri.event_log import append_event

    queue_path = petri_dir / "queue.json"

    # Signal the processor to stop
    request_stop()

    # Find active nodes and stall them
    active_states = {"phase1_active", "phase2_active", "mediating", "red_team_active", "evaluating"}
    entries = list_queue(queue_path)
    stopped_nodes: list[str] = []

    for entry in entries:
        state = entry.get("queue_state", "")
        node_id = entry.get("node_id", "")
        if state in active_states and node_id:
            try:
                if force:
                    # Force stop: stall immediately
                    update_state(queue_path, node_id, "stalled")
                else:
                    # Graceful: stall
                    update_state(queue_path, node_id, "stalled")

                # Log stop event
                dish_id = _get_dish_id(petri_dir)
                parts = node_id.split("-")
                if len(parts) >= 4:
                    seq_str = parts[-1]
                    level_str = parts[-2]
                    # Find colony dir
                    dishes_dir = petri_dir / "petri-dishes"
                    for colony_dir in dishes_dir.iterdir():
                        if colony_dir.is_dir():
                            events_path = colony_dir / f"{level_str}-{seq_str}" / "events.jsonl"
                            metadata_path = colony_dir / f"{level_str}-{seq_str}" / "metadata.json"
                            if metadata_path.exists():
                                try:
                                    import json as _json
                                    meta = _json.loads(metadata_path.read_text())
                                    if meta.get("id") == node_id:
                                        append_event(
                                            events_path=events_path,
                                            node_id=node_id,
                                            event_type="verdict_issued",
                                            agent="node_lead",
                                            iteration=entry.get("iteration", 0),
                                            data={
                                                "verdict": "PIPELINE_STALLED",
                                                "summary": "Stop requested by user"
                                                + (" (force)" if force else ""),
                                            },
                                        )
                                        break
                                except Exception:
                                    pass

                stopped_nodes.append(node_id)
            except ValueError:
                pass  # Transition not valid

    if stopped_nodes:
        typer.echo(f"Stopped {len(stopped_nodes)} nodes:")
        for nid in stopped_nodes:
            typer.echo(f"  {nid}: stalled")
    else:
        typer.echo("No active nodes to stop.")

    # Also check for queued nodes
    queued_nodes = [
        e for e in entries if e.get("queue_state") == "queued"
    ]
    if queued_nodes:
        typer.echo(f"\n{len(queued_nodes)} nodes remain queued (not started).")

    raise typer.Exit(code=0)


@app.command()
def feed(
    source: str = typer.Argument(..., help="URL, file path, or - for stdin"),
    colony_name: Optional[str] = typer.Option(
        None, "--colony", help="Limit matching to colony"
    ),
    auto_reopen: bool = typer.Option(
        False, "--auto-reopen", help="Skip re-open confirmation"
    ),
) -> None:
    """Provide new evidence to the colony."""
    petri_dir = _find_petri_dir()
    dish_id = _get_dish_id(petri_dir)

    # Load colonies
    colonies = _load_colonies(petri_dir, dish_id)
    if not colonies:
        typer.echo("No colonies found.", err=True)
        raise typer.Exit(code=1)

    # Read source content
    if source == "-":
        import sys

        content = sys.stdin.read()
    elif Path(source).exists():
        content = Path(source).read_text()
    else:
        # Assume URL or text -- store as-is
        content = source

    # Without a InferenceProvider, we can't do intelligent matching.
    # Instead, show all nodes and let the user choose which to re-open.
    all_nodes = []
    for graph, colony in colonies:
        if colony_name and colony.id.split("-", 1)[-1] != colony_name:
            continue
        for node in graph.get_nodes():
            all_nodes.append((graph, colony, node))

    # Display nodes that could be affected
    typer.echo(f"\nSource: {source}")
    typer.echo(f"Content length: {len(content)} characters\n")

    from petri.models import NodeStatus

    reopenable = [
        (g, c, n)
        for g, c, n in all_nodes
        if n.status
        in (
            NodeStatus.VALIDATED,
            NodeStatus.DISPROVEN,
            NodeStatus.DEFER_OPEN,
        )
        or n.status.value
        in ("VALIDATED", "DISPROVEN", "DEFER_OPEN")
    ]

    if not reopenable:
        typer.echo(
            "No nodes are in a re-openable state "
            "(VALIDATED, DISPROVEN, or DEFER_OPEN)."
        )
        raise typer.Exit(code=1)

    typer.echo(f"Re-openable nodes ({len(reopenable)}):")
    for i, (g, c, n) in enumerate(reopenable, 1):
        status_val = (
            n.status.value if isinstance(n.status, NodeStatus) else str(n.status)
        )
        typer.echo(f"  {i}. [{status_val}] {n.id}: {n.claim_text}")

    # Re-open nodes
    from petri.propagation import (
        get_impact_report,
        propagate_upward,
        reopen_node,
    )

    if auto_reopen:
        # Re-open all
        for g, c, n in reopenable:
            reopen_node(petri_dir, n.id, trigger=f"New evidence from: {source}")
            flagged = propagate_upward(petri_dir, n.id, g, dish_id)
            typer.echo(
                f"  Re-opened {n.id}, flagged {len(flagged)} dependents"
            )
    else:
        # In non-interactive mode, just show the impact report
        import sys

        if not sys.stdin.isatty():
            typer.echo(
                "\nRun with --auto-reopen to re-open all affected nodes."
            )
            raise typer.Exit(code=0)

        # Interactive: ask for each node
        try:
            import questionary

            for g, c, n in reopenable:
                report = get_impact_report(petri_dir, n.id, g, dish_id)
                typer.echo(f"\n{n.id}: {n.claim_text}")
                typer.echo(
                    f"  Would affect {report['total_affected']} dependent nodes"
                )

                if questionary.confirm(
                    f"Re-open {n.id}?", default=False
                ).ask():
                    reopen_node(
                        petri_dir,
                        n.id,
                        trigger=f"New evidence from: {source}",
                    )
                    flagged = propagate_upward(petri_dir, n.id, g, dish_id)
                    typer.echo(
                        f"  Re-opened. Flagged {len(flagged)} dependents."
                    )
        except (ImportError, OSError):
            typer.echo(
                "\nRun with --auto-reopen in non-interactive mode."
            )
            raise typer.Exit(code=0)

    typer.echo("\nEvidence feed complete.")


# ── Entry Point ──────────────────────────────────────────────────────────


def main() -> None:
    """Entry point for ``python -m petri``."""
    app()


if __name__ == "__main__":
    main()
