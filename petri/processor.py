"""Pipeline processor for the Petri Research Orchestration Framework.

Processes nodes through the full validation pipeline:
research -> critique -> convergence -> red team -> evaluation.

Pure library logic -- no CLI, no UI.  The CLI (``petri grow``) calls this.
"""

from __future__ import annotations

import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from petri.convergence import (
    check_convergence,
    compute_circuit_breaker,
    evaluate_short_circuits,
    identify_weakest_link,
    load_agent_roles,
)
from petri.debate import load_debate_pairings, log_debate, mediate_debate
from petri.event_log import append_event, get_verdicts
from petri.models import NodeStatus, QueueState, InferenceProvider
from petri.queue import (
    add_to_queue,
    get_next,
    list_queue,
    new_cycle,
    set_focused_directive,
    set_iteration,
    set_weakest_link,
    update_state,
)
from petri.validators import validate_terminal_sources

logger = logging.getLogger(__name__)

# ── Graceful Stop ────────────────────────────────────────────────────────

_stop_event = threading.Event()


def request_stop() -> None:
    """Signal the processor to stop after current phases complete."""
    _stop_event.set()


def is_stop_requested() -> bool:
    """Check if a stop has been requested."""
    return _stop_event.is_set()


def reset_stop() -> None:
    """Clear the stop signal."""
    _stop_event.clear()


# ── Path Helpers ─────────────────────────────────────────────────────────


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _queue_path(petri_dir: Path) -> Path:
    return petri_dir / "queue.json"


def _get_dish_id(petri_dir: Path) -> str:
    """Get the dish ID from config or derive from directory name."""
    config_path = petri_dir / "defaults" / "petri.yaml"
    if config_path.exists():
        try:
            import yaml

            with open(config_path) as f:
                config = yaml.safe_load(f) or {}
            return config.get("name", petri_dir.parent.name)
        except Exception:
            pass
    return petri_dir.parent.name


def _colony_slug(node_id: str, dish_id: str) -> str:
    """Extract the colony slug from a node's composite key."""
    dish_prefix = dish_id + "-"
    if node_id.startswith(dish_prefix):
        colony_and_rest = node_id[len(dish_prefix):]
        return colony_and_rest.rsplit("-", 2)[0]
    parts = node_id.split("-")
    return "-".join(parts[:-2])


def _load_node_paths(petri_dir: Path, node_id: str, dish_id: str) -> dict[str, str]:
    """Load node_paths from colony.json for the node's colony."""
    slug = _colony_slug(node_id, dish_id)
    colony_json = petri_dir / "petri-dishes" / slug / "colony.json"
    if colony_json.exists():
        data = json.loads(colony_json.read_text())
        return data.get("node_paths", {})
    return {}


def _node_dir(petri_dir: Path, node_id: str, dish_id: str) -> Path:
    """Derive the filesystem path for a node.

    Looks up the human-readable path from colony.json's node_paths mapping.
    Falls back to flat {level}-{seq} layout for backwards compatibility.
    """
    slug = _colony_slug(node_id, dish_id)
    colony_base = petri_dir / "petri-dishes" / slug

    # Try node_paths from colony.json
    node_paths = _load_node_paths(petri_dir, node_id, dish_id)
    if node_id in node_paths:
        return colony_base / node_paths[node_id]

    # Fallback: flat layout
    parts = node_id.split("-")
    return colony_base / f"{parts[-2]}-{parts[-1]}"


def _events_path(petri_dir: Path, node_id: str, dish_id: str) -> Path:
    return _node_dir(petri_dir, node_id, dish_id) / "events.jsonl"


def _metadata_path(petri_dir: Path, node_id: str, dish_id: str) -> Path:
    return _node_dir(petri_dir, node_id, dish_id) / "metadata.json"


def _load_node_metadata(petri_dir: Path, node_id: str, dish_id: str) -> dict:
    """Load a node's metadata.json."""
    path = _metadata_path(petri_dir, node_id, dish_id)
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


# ── Node Status Update ──────────────────────────────────────────────────


def _update_node_status(petri_dir: Path, node_id: str, new_status: str) -> None:
    """Update a node's status in its metadata.json file."""
    dish_id = _get_dish_id(petri_dir)
    path = _metadata_path(petri_dir, node_id, dish_id)
    if not path.exists():
        logger.warning("metadata.json not found for node %s at %s", node_id, path)
        return
    with open(path) as f:
        metadata = json.load(f)
    metadata["status"] = new_status
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")


# ── Provider Guard ───────────────────────────────────────────────────────


class NoProviderError(RuntimeError):
    """Raised when the processor is invoked without a InferenceProvider."""

    def __init__(self) -> None:
        super().__init__(
            "No InferenceProvider configured. "
            "Set model and harness in petri.yaml to enable processing."
        )


# ── Phase Runners ────────────────────────────────────────────────────────


def _run_phase1(
    node_id: str,
    claim_text: str,
    petri_dir: Path,
    dish_id: str,
    iteration: int,
    provider: InferenceProvider,
    agent_roles: dict,
    queue_entry: dict,
) -> list[dict]:
    """Phase 1: Research -- investigator, freshness_checker, dependency_auditor."""
    events_p = _events_path(petri_dir, node_id, dish_id)
    qp = _queue_path(petri_dir)
    phase1_agents = ["investigator", "freshness_checker", "dependency_auditor"]
    verdicts_collected: list[dict] = []

    context = {
        "iteration": iteration,
        "weakest_link": queue_entry.get("weakest_link"),
        "focused_directive": queue_entry.get("focused_directive"),
    }

    for agent_name in phase1_agents:
        result = provider.assess_node(
            node_id, claim_text, context, agent_name
        )

        # Log verdict_issued event
        append_event(
            events_path=events_p,
            node_id=node_id,
            event_type="verdict_issued",
            agent=agent_name,
            iteration=iteration,
            data={
                "verdict": result.get("verdict", "PASS"),
                "summary": result.get("summary", ""),
            },
        )
        verdicts_collected.append(result)

    # Transition to phase2_active
    update_state(qp, node_id, QueueState.phase2_active.value)
    return verdicts_collected


def _run_phase2(
    node_id: str,
    claim_text: str,
    petri_dir: Path,
    dish_id: str,
    iteration: int,
    provider: InferenceProvider,
    agent_roles: dict,
    debate_pairings: list | None,
    queue_entry: dict,
) -> list[dict]:
    """Phase 2: Critique -- skeptic, champion, pragmatist, simplifier, triage, impact_assessor."""
    events_p = _events_path(petri_dir, node_id, dish_id)
    qp = _queue_path(petri_dir)
    phase2_agents = [
        "skeptic", "champion", "pragmatist",
        "simplifier", "triage", "impact_assessor",
    ]
    verdicts_collected: list[dict] = []
    agent_outputs: dict[str, dict] = {}

    context = {
        "iteration": iteration,
        "weakest_link": queue_entry.get("weakest_link"),
        "focused_directive": queue_entry.get("focused_directive"),
    }

    for agent_name in phase2_agents:
        result = provider.assess_node(
            node_id, claim_text, context, agent_name
        )

        # Log verdict_issued event
        append_event(
            events_path=events_p,
            node_id=node_id,
            event_type="verdict_issued",
            agent=agent_name,
            iteration=iteration,
            data={
                "verdict": result.get("verdict", "PASS"),
                "summary": result.get("summary", ""),
            },
        )
        verdicts_collected.append(result)
        agent_outputs[agent_name] = result

    # Run debate mediation
    pairings = debate_pairings or load_debate_pairings()
    for debate in pairings:
        agent_a_name = debate.pair[0]
        agent_b_name = debate.pair[1]
        agent_a_output = agent_outputs[agent_a_name]
        agent_b_output = agent_outputs[agent_b_name]

        debate_result = mediate_debate(
            agent_a_output=agent_a_output,
            agent_b_output=agent_b_output,
            debate=debate,
            provider=provider,
        )
        log_debate(
            events_path=events_p,
            node_id=node_id,
            iteration=iteration,
            debate_result=debate_result,
        )

    # Transition to mediating
    update_state(qp, node_id, QueueState.mediating.value)
    return verdicts_collected


def _run_convergence(
    node_id: str,
    petri_dir: Path,
    dish_id: str,
    iteration: int,
    agent_roles: dict,
    queue_entry: dict,
) -> dict:
    """Convergence check -- determines converged, iterate, or stalled."""
    events_p = _events_path(petri_dir, node_id, dish_id)
    qp = _queue_path(petri_dir)

    # Gather all verdicts for this node at the current iteration
    verdicts = get_verdicts(events_p, node_id=node_id, iteration=iteration)

    # Check for short circuits first
    short_circuit = evaluate_short_circuits(verdicts, agent_roles)
    if short_circuit:
        sc_type = short_circuit["type"]
        append_event(
            events_path=events_p,
            node_id=node_id,
            event_type="convergence_checked",
            agent="node_lead",
            iteration=iteration,
            data={
                "converged": False,
                "blocking_verdicts": {"short_circuit": short_circuit},
                "weakest_link": short_circuit.get("agent"),
                "focused_directive": f"Short-circuit: {sc_type}",
            },
        )
        if sc_type == "needs_experiment":
            _update_node_status(petri_dir, node_id, NodeStatus.NEEDS_EXPERIMENT.value)
            update_state(qp, node_id, QueueState.stalled.value)
            update_state(qp, node_id, QueueState.needs_human.value)
            return {"outcome": "short_circuit", "type": sc_type}
        elif sc_type == "defer_open":
            _update_node_status(petri_dir, node_id, NodeStatus.DEFER_OPEN.value)
            update_state(qp, node_id, QueueState.converged.value)
            update_state(qp, node_id, QueueState.deferred_open.value)
            return {"outcome": "short_circuit", "type": sc_type}

    # Normal convergence check
    convergence = check_convergence(verdicts, agent_roles)

    # Check circuit breaker
    max_iter = queue_entry.get("max_iterations", 3)
    cycle_start = queue_entry.get("cycle_start_iteration", 0)
    breaker_fires = compute_circuit_breaker(iteration, cycle_start, max_iter)

    # Log convergence event
    append_event(
        events_path=events_p,
        node_id=node_id,
        event_type="convergence_checked",
        agent="node_lead",
        iteration=iteration,
        data={
            "converged": convergence["converged"],
            "blocking_verdicts": {
                k: v.get("verdict") for k, v in convergence.get("blocking_results", {}).items()
            } or None,
            "weakest_link": convergence.get("weakest_link"),
        },
    )

    if convergence["converged"]:
        update_state(qp, node_id, QueueState.converged.value)
        return {"outcome": "converged"}

    # Not converged -- iterate or stall
    if breaker_fires:
        update_state(qp, node_id, QueueState.stalled.value)
        update_state(qp, node_id, QueueState.needs_human.value)
        _update_node_status(petri_dir, node_id, NodeStatus.STALLED.value)
        return {"outcome": "circuit_breaker"}

    # Iterate: increment iteration, set weakest link, back to phase1_active
    # Note: we use set_iteration (not new_cycle) because this is a retry
    # within the same convergence cycle, not a fresh cycle start.
    wl = convergence.get("weakest_link")

    set_iteration(qp, node_id, iteration + 1)
    if wl:
        set_weakest_link(qp, node_id, wl)
        directive = f"Focus on addressing {wl} concerns"
        set_focused_directive(qp, node_id, directive)

    update_state(qp, node_id, QueueState.phase1_active.value)
    return {"outcome": "iterate", "weakest_link": wl}


def _run_red_team(
    node_id: str,
    claim_text: str,
    petri_dir: Path,
    dish_id: str,
    iteration: int,
    provider: InferenceProvider,
    agent_roles: dict,
) -> dict:
    """Red Team phase -- red_team_lead attempts disproval."""
    events_p = _events_path(petri_dir, node_id, dish_id)
    qp = _queue_path(petri_dir)

    update_state(qp, node_id, QueueState.red_team_active.value)

    context = {"iteration": iteration, "phase": "red_team"}
    result = provider.assess_node(node_id, claim_text, context, "red_team_lead")

    append_event(
        events_path=events_p,
        node_id=node_id,
        event_type="verdict_issued",
        agent="red_team_lead",
        iteration=iteration,
        data={
            "verdict": result.get("verdict", "NO_CONTRADICTING_EVIDENCE"),
            "summary": result.get("summary", ""),
        },
    )

    update_state(qp, node_id, QueueState.evaluating.value)
    return result


def _run_evaluation(
    node_id: str,
    claim_text: str,
    petri_dir: Path,
    dish_id: str,
    iteration: int,
    provider: InferenceProvider,
    agent_roles: dict,
) -> dict:
    """Evidence Evaluation -- final verdict based on source hierarchy."""
    events_p = _events_path(petri_dir, node_id, dish_id)
    qp = _queue_path(petri_dir)

    # Validate terminal sources
    source_validation = validate_terminal_sources(events_p, node_id)

    context = {
        "iteration": iteration,
        "phase": "evaluation",
        "source_validation": source_validation,
    }
    result = provider.assess_node(
        node_id, claim_text, context, "evidence_evaluator"
    )

    append_event(
        events_path=events_p,
        node_id=node_id,
        event_type="verdict_issued",
        agent="evidence_evaluator",
        iteration=iteration,
        data={
            "verdict": result.get("verdict", "EVIDENCE_CONFIRMS"),
            "summary": result.get("summary", ""),
        },
    )

    # Determine final node status from evaluator verdict
    evaluator_verdict = result.get("verdict", "EVIDENCE_CONFIRMS")
    if evaluator_verdict == "EVIDENCE_CONFIRMS":
        final_status = NodeStatus.VALIDATED.value
    elif evaluator_verdict == "EVIDENCE_REFUTES":
        final_status = NodeStatus.DISPROVEN.value
    else:
        # EVIDENCE_INCONCLUSIVE or unknown
        final_status = NodeStatus.DEFER_OPEN.value

    _update_node_status(petri_dir, node_id, final_status)
    update_state(qp, node_id, QueueState.done.value)

    return {"verdict": evaluator_verdict, "final_status": final_status}


# ── Main Pipeline ────────────────────────────────────────────────────────


def process_node(
    node_id: str,
    petri_dir: Path,
    provider: InferenceProvider,
    agent_roles: dict | None = None,
    debate_pairings: list | None = None,
) -> dict:
    """Process a single node through the validation pipeline.

    Drives the node from its current queue state through successive phases
    until it reaches a terminal state (done, needs_human, deferred) or a
    graceful stop is requested.

    Returns a status dict:
        {node_id, final_state, iterations, events_logged, ...}
    """
    dish_id = _get_dish_id(petri_dir)
    qp = _queue_path(petri_dir)
    events_p = _events_path(petri_dir, node_id, dish_id)

    # Load agent roles if not provided
    if agent_roles is None:
        agent_roles = load_agent_roles()

    # Load node metadata for claim_text
    metadata = _load_node_metadata(petri_dir, node_id, dish_id)
    claim_text = metadata.get("claim_text", "")

    # Track processing stats
    iterations_run = 0
    events_before = len(get_verdicts(events_p, node_id=node_id))

    # Load queue entry
    from petri.queue import load_queue

    queue = load_queue(qp)
    if node_id not in queue.get("entries", {}):
        return {
            "node_id": node_id,
            "final_state": "not_in_queue",
            "iterations": 0,
            "events_logged": 0,
            "error": f"Node {node_id} not found in queue",
        }

    # Drive the state machine until we reach a terminal state
    max_loops = 50  # safety limit to prevent infinite loops
    loop_count = 0

    while loop_count < max_loops:
        loop_count += 1

        # Reload queue entry each loop iteration to get fresh state
        queue = load_queue(qp)
        entry = queue["entries"].get(node_id, {})
        current_state = entry.get("queue_state", "done")
        iteration = entry.get("iteration", 0)

        # Check for graceful stop between phases
        if is_stop_requested():
            logger.info("Graceful stop requested for node %s", node_id)
            if current_state not in ("done", "needs_human", "deferred_open", "deferred_closed"):
                try:
                    update_state(qp, node_id, QueueState.stalled.value)
                except ValueError:
                    pass  # Already in a non-stallable state
            break

        # Terminal states -- we are done
        if current_state in ("done", "needs_human", "deferred_open", "deferred_closed"):
            break

        # Dispatch based on current state
        if current_state == QueueState.queued.value:
            _update_node_status(petri_dir, node_id, NodeStatus.RESEARCH.value)
            update_state(qp, node_id, QueueState.phase1_active.value)

        elif current_state == QueueState.phase1_active.value:
            _run_phase1(
                node_id, claim_text, petri_dir, dish_id,
                iteration, provider, agent_roles, entry,
            )
            iterations_run += 1

        elif current_state == QueueState.phase2_active.value:
            _run_phase2(
                node_id, claim_text, petri_dir, dish_id,
                iteration, provider, agent_roles, debate_pairings, entry,
            )

        elif current_state == QueueState.mediating.value:
            result = _run_convergence(
                node_id, petri_dir, dish_id,
                iteration, agent_roles, entry,
            )
            if result["outcome"] == "iterate":
                iterations_run += 1

        elif current_state == QueueState.converged.value:
            _update_node_status(petri_dir, node_id, NodeStatus.RED_TEAM.value)
            _run_red_team(
                node_id, claim_text, petri_dir, dish_id,
                iteration, provider, agent_roles,
            )

        elif current_state == QueueState.evaluating.value:
            _update_node_status(petri_dir, node_id, NodeStatus.EVALUATE.value)
            _run_evaluation(
                node_id, claim_text, petri_dir, dish_id,
                iteration, provider, agent_roles,
            )

        elif current_state == QueueState.stalled.value:
            update_state(qp, node_id, QueueState.needs_human.value)

        else:
            # Unknown or unhandled state
            logger.warning("Unhandled queue state %s for node %s", current_state, node_id)
            break

    # Compute final state
    queue = load_queue(qp)
    final_entry = queue["entries"].get(node_id, {})
    final_state = final_entry.get("queue_state", "unknown")

    events_after = len(get_verdicts(events_p, node_id=node_id))
    events_logged = events_after - events_before

    return {
        "node_id": node_id,
        "final_state": final_state,
        "iterations": iterations_run,
        "events_logged": events_logged,
        "final_iteration": final_entry.get("iteration", 0),
    }


# ── Eligible Node Discovery ─────────────────────────────────────────────


def find_eligible_nodes(
    petri_dir: Path,
    dish_id: str,
    node_ids: list[str] | None = None,
    colony_filter: str | None = None,
    all_nodes: bool = False,
) -> list[str]:
    """Find nodes eligible for validation.

    Eligible = cell nodes (or nodes whose deps are all VALIDATED) in NEW status.

    When *node_ids* is provided, only those specific nodes are considered.
    When *colony_filter* is provided, only nodes from that colony are scanned.
    When *all_nodes* is True, all colonies are scanned.
    """
    from petri.colony import deserialize_colony

    dishes_dir = petri_dir / "petri-dishes"
    if not dishes_dir.exists():
        return []

    eligible: list[str] = []

    # If specific node IDs are given, return them directly (caller knows best)
    if node_ids:
        return list(node_ids)

    # Scan colonies
    for colony_dir in sorted(dishes_dir.iterdir()):
        if not colony_dir.is_dir():
            continue

        colony_slug = colony_dir.name
        if colony_filter and colony_slug != colony_filter:
            continue

        try:
            graph, colony = deserialize_colony(colony_dir, dish_id)
        except Exception:
            continue

        # Build status map from metadata files
        nodes_status: dict[str, NodeStatus] = {}
        for node in graph.get_nodes():
            nodes_status[node.id] = node.status

        # Get eligible nodes from graph
        for node in graph.get_eligible_for_validation(nodes_status):
            eligible.append(node.id)

        # If not processing all, stop after first colony
        if not all_nodes and not colony_filter:
            break

    return eligible


# ── Queue Processor ──────────────────────────────────────────────────────


def process_queue(
    petri_dir: Path,
    provider: InferenceProvider | None = None,
    max_concurrent: int = 4,
    node_ids: list[str] | None = None,
    colony_filter: str | None = None,
    all_nodes: bool = False,
    dry_run: bool = False,
) -> dict:
    """Process the queue, running eligible nodes through the pipeline.

    Uses ``ThreadPoolExecutor`` for concurrent processing.

    Returns summary dict:
        {processed, succeeded, failed, stalled, results, ...}
    """
    if not dry_run and provider is None:
        raise NoProviderError()

    reset_stop()

    dish_id = _get_dish_id(petri_dir)
    qp = _queue_path(petri_dir)

    # Find eligible nodes
    eligible = find_eligible_nodes(
        petri_dir, dish_id,
        node_ids=node_ids,
        colony_filter=colony_filter,
        all_nodes=all_nodes,
    )

    if dry_run:
        return {
            "processed": 0,
            "succeeded": 0,
            "failed": 0,
            "stalled": 0,
            "dry_run": True,
            "would_process": eligible,
        }

    if not eligible:
        return {
            "processed": 0,
            "succeeded": 0,
            "failed": 0,
            "stalled": 0,
            "results": [],
        }

    # Enqueue eligible nodes that are not already in the queue
    existing_queue = {e["node_id"] for e in list_queue(qp)}
    for nid in eligible:
        if nid not in existing_queue:
            try:
                add_to_queue(qp, nid)
            except ValueError:
                pass  # Already in queue (race condition)

    # Load shared config
    agent_roles = load_agent_roles()
    debate_pairings = load_debate_pairings()

    # Process concurrently
    results: list[dict] = []
    succeeded = 0
    failed = 0
    stalled = 0

    def _process_one(nid: str) -> dict:
        try:
            return process_node(
                nid, petri_dir,
                provider=provider,
                agent_roles=agent_roles,
                debate_pairings=debate_pairings,
            )
        except Exception as exc:
            logger.exception("Error processing node %s", nid)
            return {
                "node_id": nid,
                "final_state": "error",
                "iterations": 0,
                "events_logged": 0,
                "error": str(exc),
            }

    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        futures = {executor.submit(_process_one, nid): nid for nid in eligible}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            final = result.get("final_state", "")
            if final == "done":
                succeeded += 1
            elif final in ("needs_human", "stalled"):
                stalled += 1
            elif final == "error":
                failed += 1
            else:
                # deferred_open, deferred_closed, or other
                succeeded += 1

    return {
        "processed": len(results),
        "succeeded": succeeded,
        "failed": failed,
        "stalled": stalled,
        "results": results,
    }
