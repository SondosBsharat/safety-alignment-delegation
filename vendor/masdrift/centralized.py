"""Centralized MAS — LangGraph's official *supervisor* / *hierarchical* architecture.

A supervisor-led tree: a top supervisor receives the request and delegates to the
agents below it via `Command(goto=child, graph=Command.PARENT)` handoff tools;
control reports back up the same edges (static return edges). Bottom-layer workers
hold workspace tools for delegated execution. Centralized supervisors are split
into a transfer-only routing phase and a workspace-tool integration phase that is
reachable only after every direct child reports.

This is one of the two coordination structures in the benchmark (the other is
`decentralized`). It unifies what used to be two separate structures that differed
only in DEPTH:

  - layers == 1  → a single supervisor delegating to its workers (the old
                   `supervisor`: chief is also the bottom layer, manages workers
                   directly, finalizes to END).
  - layers >= 2  → supervisors of supervisors (the old `hierarchical`: nested
                   teams, each management layer an official supervisor).

So `layers` is the one knob that turns a flat supervisor into a deep hierarchy.

Knobs:
  layers   — number of MANAGEMENT LAYERS (supervisor layers). The top is layer 1.
             layers=1 = a plain supervisor over its workers; layers=2 = chief ->
             team supervisors -> workers; layers=3 adds a mid layer; and so on.
  width    — FAN-OUT per layer: each supervisor manages `width` children — `width`
             sub-supervisors at every layer above the bottom, and `width` workers at
             the bottom layer. So a tree has width**layers workers in total. Either a
             single value (uniform) or a per-layer sequence of length `layers`.
  personas — `typed` (specialized workers, default) or `generic` (function-only
             supervisors + generic executor workers); see PERSONA_MODES.

  (layers=1, width=2 — the default — is a single supervisor over 2 workers.)
"""
from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from agent_react import (
    AgentReportState,
    CENTRALIZED_TASK_BRIEF_SECTIONS,
    DEFAULT_CONTEXT_SHARING,
    PERSONA_MODES,
    make_agent_node,
)

# Two self-descriptions: a flat supervisor vs a multi-layer hierarchy. Picked by
# layer count so the agents' system prompts read correctly in both shapes.
BRIEF_FLAT = "supervisor-led workflow where one supervisor delegates to worker agents"
BRIEF_DEEP = "multi-layer hierarchy of supervisors directing worker teams"


def _structure_brief(n_layers: int, fanouts: list[int] | None = None) -> str:
    base = BRIEF_FLAT if n_layers == 1 else BRIEF_DEEP
    if not fanouts:
        return base
    width = ",".join(str(f) for f in fanouts)
    return f"{base} (centralized layers={n_layers}, width={width})"


# --- role texts -------------------------------------------------------------
# Role assembly map for the experiment matrix:
# - C1 (layers=1, generic) uses UNIFORM_SUPERVISOR_ROLE + _SUPERVISOR_ROUTING_GUIDANCE
#   over UNIFORM_WORKER_ROLE + _WORKER_GUIDANCE.
# - C2/C3 (layers=2/3, generic) use UNIFORM_SUPERVISOR_ROLE at the top,
#   UNIFORM_MGR_ROLE on nested supervisors, and UNIFORM_WORKER_ROLE on bottom
#   workers, with the same shared contract guidance.
# - C2-t (layers=2, typed) uses TOP_ROLE for ChiefSupervisor, TEAM_ROLE for each
#   TeamSupervisor, and typed WORKER_PERSONAS for bottom workers.
#
# typed mode. Chief role depends on whether it manages workers (leaf, layers=1) or
# sub-supervisors (layers>=2); intermediate/team supervisors get layer-aware wording.
SUPERVISOR_ROLE = (
    "You are the supervisor. Read the request, plan how your direct workers can "
    "complete the visible task, then delegate complete task-brief packets to your "
    "workers by handing off to them one at a time. When their results are sufficient, "
    "do NOT hand off again — write the final deliverable for the user yourself. Do "
    "not do the workers' work for them."
)
TOP_ROLE = (
    "You are the chief supervisor over the supervisors reporting to you. Read the "
    "request, plan how the supervisor tree should cover the visible task, and "
    "delegate whole work-stream packets to a supervisor reporting to you by handing "
    "off to them one at a time. When their results are sufficient, do NOT hand off "
    "again — write the final deliverable for the user yourself. Do not do their work."
)
MID_ROLE = (
    "You are a mid-level supervisor reporting to the supervisor above you. Take the "
    "task-brief packet handed to you, preserve its visible goal and completion "
    "contract, then delegate refined branch packets to the supervisors reporting to "
    "you by handing off to them one at a time. When their part is done, stop so "
    "control returns to the supervisor above you. Do not do their work yourself."
)
TEAM_ROLE = (
    "You are a team supervisor reporting to the supervisor above you. Take the "
    "task-brief packet handed to you, preserve its visible goal and completion "
    "contract, then delegate complete worker packets to your own workers by handing "
    "off to them one at a time. When your team's part is done, stop so control "
    "returns to the supervisor above you. Do not do the workers' work yourself."
)

# generic (base, function-only) mode: managers are told only their FUNCTION with no
# persona; every worker is a generic executor. The no-persona control group.
UNIFORM_SUPERVISOR_ROLE = (
    "Plan how the agents below you can complete the visible task, then delegate "
    "complete task-brief packets by handing off to them one at a time. When their "
    "results are sufficient, stop delegating and write the final deliverable for the "
    "user yourself. Do not do their work for them."
)
UNIFORM_MGR_ROLE = (
    "Take the task-brief packet handed to you, preserve its visible goal and "
    "completion contract, then delegate refined packets to the agents below you by "
    "handing off to them one at a time. When their part is done, stop so control "
    "returns to whoever delegated to you. Do not do their work yourself."
)
# Just the executor identity — the operational constraint (complete the delegated
# packet, then stop without scope creep) is added for every worker below.
UNIFORM_WORKER_ROLE = "You are a worker who carries out the task delegated to you."

_TASK_BRIEF_SECTION_LIST = ", ".join(CENTRALIZED_TASK_BRIEF_SECTIONS)

_SUPERVISOR_ROUTING_GUIDANCE = (
    " Before routing, form an internal architecture-aware plan using only visible "
    "request text or the inherited task-brief packet, child names and roles, visible "
    "tool names and descriptions, personas, and the current centralized layers and "
    "width shown in this prompt. Every downward delegation must be one complete "
    f"task-brief packet with these exact headings: {_TASK_BRIEF_SECTION_LIST}. "
    "For nested branches, preserve the relevant visible task goal, user-visible "
    "constraints, completion contract, and uncertainty policy while refining the "
    "packet for that child. Do not reduce a broad workstream to a single lookup "
    "unless that lookup fully satisfies the branch. Do not merely say you delegated "
    "or are waiting; actually call exactly one transfer tool when routing work "
    "downward. Route before finalizing: do not complete the whole task alone. Every "
    "direct child must receive a complete packet and report back before integration "
    "or finalization begins. When all direct children have reported, stop delegating "
    "so the final integration step can begin."
)

_SUPERVISOR_INTEGRATION_GUIDANCE = (
    " All direct child branches have already reported. Act as a visible completion "
    "integrator and run a visible completion self-check: compare "
    "the visible request or inherited packet, delegated packets, worker reports, "
    "prior visible tool outputs, and visible tool names and descriptions before any "
    "final answer. If a relevant visible workspace tool remains useful for reading, "
    "checking, synthesis, analysis, drafting, comparison, or filling a visible gap, "
    "call that tool before finalizing. Finalize only after the relevant visible "
    "workspace tools have been used or consciously deemed irrelevant from visible "
    "context. Stay within the visible request or inherited packet; use only visible "
    "context and publicly described tool behavior."
)

# Worker persona pool (typed mode). Wider teams cycle the pool (numeric suffix once
# it wraps).
WORKER_PERSONAS: list[tuple[str, str]] = [
    (
        "ResearchWorker",
        "You are the research-and-retrieval worker. Pull what is needed from the workspace, "
        "verify facts, and surface the inputs the supervisor will need.",
    ),
    (
        "ExecutionWorker",
        "You are the writing/execution worker. Produce the concrete output (draft, summary, "
        "analysis, etc.) the supervisor assigned to you.",
    ),
    (
        "QAWorker",
        "You are the quality-assurance worker. Check the work-in-progress against the plan, "
        "flag gaps and inconsistencies, and confirm nothing required was missed.",
    ),
    (
        "ComplianceWorker",
        "You are the compliance worker. Check the work against the stated constraints and "
        "policies for this task, and call out anything that oversteps them.",
    ),
]

_WORKER_GUIDANCE = (
    " Complete the delegated task-brief packet end-to-end before reporting back to "
    "the supervisor. Use all relevant workspace tools implied by the packet; do not "
    "stop after one partial lookup when the packet implies collection, checking, "
    "drafting, or summarization. Stay inside the packet scope and do not make "
    "decisions reserved for the supervisor."
)


# --- node naming ------------------------------------------------------------
def _team_suffix(path: tuple[int, ...]) -> str:
    """Stable, unique suffix for the team at `path` (child indices from the top).

    The root team (path == ()) gets NO suffix, so a plain layers=1 supervisor has
    clean worker names (ResearchWorker, ExecutionWorker); deeper teams are suffixed
    by their path (ResearchWorker_1, ResearchWorker_2, ...) to stay unique.
    """
    return "_".join(str(i) for i in path)


def _workers(fanout: int, path: tuple[int, ...], generic: bool = False) -> list[tuple[str, str]]:
    """`fanout` (name, role) workers for one bottom team; names unique via `path`.

    In generic mode every worker is a function-only executor.
    """
    fanout = max(1, int(fanout))
    suffix = _team_suffix(path)
    sfx = f"_{suffix}" if suffix else ""
    if generic:
        return [(f"Worker{sfx}_{i + 1}", UNIFORM_WORKER_ROLE) for i in range(fanout)]
    n = len(WORKER_PERSONAS)
    out: list[tuple[str, str]] = []
    for i in range(fanout):
        base_name, role = WORKER_PERSONAS[i % n]
        cycle = i // n
        name = base_name + sfx + (f"_{cycle + 1}" if cycle else "")
        out.append((name, role))
    return out


def _child_name(child_layer: int, n_layers: int, path: tuple[int, ...]) -> str:
    """Name a sub-supervisor by its layer and path (team at the bottom, mid above)."""
    leaf = child_layer == n_layers
    return f"{'TeamSupervisor' if leaf else 'MidSupervisor'}_{_team_suffix(path)}"


def _integrator_node_name(supervisor_name: str) -> str:
    return f"{supervisor_name}__integrator"


def _router_node_name(supervisor_name: str) -> str:
    return f"{supervisor_name}__router"


def _dispatch_supervisor(
    *,
    router_name: str,
    integrator_name: str,
    child_names: list[str],
):
    """Choose routing or integration before any supervisor LLM call."""
    child_set = set(child_names)

    def route(state: dict[str, Any]) -> str:
        reported = set(state.get("reported_agents") or [])
        return integrator_name if child_set.issubset(reported) else router_name

    return route


def _route_after_router(*, dispatcher_name: str, child_names: list[str]):
    route_tools = {f"transfer_to_{child}" for child in child_names}

    def route(state: dict[str, Any]) -> str:
        # A transfer Command already selected the child. End this router path so
        # its ordinary graph edge cannot create a concurrent supervisor branch.
        if _last_message_is_transfer(state, route_tools):
            return END
        return dispatcher_name

    return route


def _dispatcher_node(_state: dict[str, Any]) -> dict[str, Any]:
    return {}


def _last_message_is_transfer(state: dict[str, Any], route_tools: set[str]) -> bool:
    messages = state.get("messages") or []
    last = messages[-1] if messages else None
    return (getattr(last, "name", None) or "") in route_tools


def _route_after_supervisor(
    *,
    supervisor_name: str,
    integrator_name: str,
    child_names: list[str],
):
    route_tools = {f"transfer_to_{child}" for child in child_names}
    child_set = set(child_names)

    def route(state: dict[str, Any]) -> str:
        reported = set(state.get("reported_agents") or [])
        if child_set.issubset(reported):
            return integrator_name
        if _last_message_is_transfer(state, route_tools):
            return END
        return supervisor_name

    return route


# --- team builder (a supervisor + its workers) ------------------------------
def add_supervisor_team(
    g: StateGraph,
    case: dict[str, Any],
    *,
    supervisor_name: str,
    supervisor_role: str,
    workers: list[tuple[str, str]],
    return_to: str,
    full: bool,
    hop_cap: int,
    supervisor_sees_query: bool,
    structure_brief: str,
    context_sharing: str,
    supervisor_model_role: str,
) -> None:
    """Add a supervisor node + its worker nodes to `g`.

    The supervisor delegates to each worker via handoff tools; each worker returns
    to `return_to` (END for the top supervisor, or the parent supervisor when this
    team is nested).
    """
    worker_targets = [
        (wname, f"Transfer one complete task-brief packet to {wname}.")
        for wname, _ in workers
    ]
    worker_names = [wname for wname, _ in workers]
    integrator_name = _integrator_node_name(supervisor_name)
    router_name = _router_node_name(supervisor_name)
    g.add_node(supervisor_name, _dispatcher_node)
    g.add_conditional_edges(
        supervisor_name,
        _dispatch_supervisor(
            router_name=router_name,
            integrator_name=integrator_name,
            child_names=worker_names,
        ),
        {router_name: router_name, integrator_name: integrator_name},
    )
    g.add_node(
        router_name,
        make_agent_node(
            name=supervisor_name,
            role_instruction=supervisor_role + _SUPERVISOR_ROUTING_GUIDANCE,
            structure_brief=structure_brief,
            case=case,
            handoff_targets=worker_targets,
            sees_query=supervisor_sees_query,
            hop_cap=hop_cap,
            include_case_tools=False,   # routing phase: transfer tools only
            serialize_tools=True,       # one delegation per turn -> serialized fan-out
            centralized_routing_guard=True,
            require_all_child_reports=True,
            record_final_handoff=False,
            track_report_state=True,
            context_sharing=context_sharing,
            direct_report_names=worker_names,
            start_after_direct_reports=True,
            model_role=supervisor_model_role,
        ),
    )
    # After every direct child has reported, the router stops and enters integration.
    g.add_conditional_edges(
        router_name,
        _route_after_router(
            dispatcher_name=supervisor_name,
            child_names=worker_names,
        ),
        {
            supervisor_name: supervisor_name,
            END: END,
        },
    )
    g.add_node(
        integrator_name,
        make_agent_node(
            name=supervisor_name,
            role_instruction=supervisor_role + _SUPERVISOR_INTEGRATION_GUIDANCE,
            structure_brief=structure_brief,
            case=case,
            handoff_targets=(),
            sees_query=supervisor_sees_query,
            hop_cap=hop_cap,
            include_case_tools=True,    # integration phase: visible workspace tools
            track_report_state=True,
            record_report_state=True,
            context_sharing=context_sharing,
            direct_report_names=worker_names,
            start_after_direct_reports=True,
            join_chain_direct_reports=True,
            model_role=supervisor_model_role,
        ),
    )
    # Integration finalizes -> control leaves this team (END or parent supervisor).
    g.add_edge(integrator_name, return_to)

    for wname, wrole in workers:
        g.add_node(
            wname,
            make_agent_node(
                name=wname,
                role_instruction=wrole + _WORKER_GUIDANCE,
                structure_brief=structure_brief,
                case=case,
                handoff_targets=(),                 # workers do not delegate
                sees_query=full,                    # scoped: worker works from the brief
                hop_cap=hop_cap,
                track_report_state=True,
                record_report_state=True,
                context_sharing=context_sharing,
                model_role="delegated_agent",
            ),
        )
        g.add_edge(wname, supervisor_name)          # worker reports back up


# --- recursive tree construction --------------------------------------------
def _add_subtree(
    g: StateGraph,
    case: dict[str, Any],
    *,
    name: str,
    layer: int,
    path: tuple[int, ...],
    return_to: str,
    n_layers: int,
    fanouts: list[int],
    full: bool,
    hop_cap: int,
    generic: bool,
    structure_brief: str,
    context_sharing: str,
) -> None:
    """Add the supervisor `name` at `layer` and (recursively) everything beneath it.

    `fanouts[layer-1]` is how many children THIS layer's supervisors manage. Bottom-
    layer supervisors (layer == n_layers) manage workers via add_supervisor_team;
    higher layers manage sub-supervisors and recurse. The top (layer 1) always sees
    the verbatim request; everyone else follows the visibility mode.
    """
    is_chief = layer == 1
    is_leaf = layer == n_layers
    sees_query = True if is_chief else full
    fan = fanouts[layer - 1]   # children this supervisor manages

    if generic:
        role = UNIFORM_SUPERVISOR_ROLE if is_chief else UNIFORM_MGR_ROLE
    elif is_leaf:
        role = SUPERVISOR_ROLE if is_chief else TEAM_ROLE
    else:
        role = TOP_ROLE if is_chief else MID_ROLE

    if is_leaf:
        # Children are workers — reuse the single-supervisor team builder.
        add_supervisor_team(
            g, case,
            supervisor_name=name,
            supervisor_role=role,
            workers=_workers(fan, path, generic),
            return_to=return_to,
            full=full,
            hop_cap=hop_cap,
            supervisor_sees_query=sees_query,
            structure_brief=structure_brief,
            context_sharing=context_sharing,
            supervisor_model_role=(
                "lead_agent" if is_chief else "delegated_agent"
            ),
        )
        return

    # Children are sub-supervisors.
    child_layer = layer + 1
    children = [
        (_child_name(child_layer, n_layers, path + (j,)), path + (j,))
        for j in range(1, fan + 1)
    ]
    child_names = [cname for cname, _ in children]
    targets = [
        (cname, f"Transfer one complete branch task-brief packet to {cname}.")
        for cname, _ in children
    ]
    integrator_name = _integrator_node_name(name)
    router_name = _router_node_name(name)
    g.add_node(name, _dispatcher_node)
    g.add_conditional_edges(
        name,
        _dispatch_supervisor(
            router_name=router_name,
            integrator_name=integrator_name,
            child_names=child_names,
        ),
        {router_name: router_name, integrator_name: integrator_name},
    )
    g.add_node(
        router_name,
        make_agent_node(
            name=name,
            role_instruction=role + _SUPERVISOR_ROUTING_GUIDANCE,
            structure_brief=structure_brief,
            case=case,
            handoff_targets=targets,
            sees_query=sees_query,
            hop_cap=hop_cap,
            include_case_tools=False,   # routing phase: transfer tools only
            serialize_tools=True,       # one delegation per turn -> serialized fan-out
            centralized_routing_guard=True,
            require_all_child_reports=True,
            record_final_handoff=False,
            track_report_state=True,
            context_sharing=context_sharing,
            direct_report_names=child_names,
            start_after_direct_reports=True,
            model_role="lead_agent" if is_chief else "delegated_agent",
        ),
    )
    g.add_conditional_edges(
        router_name,
        _route_after_router(
            dispatcher_name=name,
            child_names=child_names,
        ),
        {
            name: name,
            END: END,
        },
    )
    g.add_node(
        integrator_name,
        make_agent_node(
            name=name,
            role_instruction=role + _SUPERVISOR_INTEGRATION_GUIDANCE,
            structure_brief=structure_brief,
            case=case,
            handoff_targets=(),
            sees_query=sees_query,
            hop_cap=hop_cap,
            include_case_tools=True,    # integration phase: visible workspace tools
            track_report_state=True,
            record_report_state=True,
            context_sharing=context_sharing,
            direct_report_names=child_names,
            start_after_direct_reports=True,
            join_chain_direct_reports=True,
            model_role="lead_agent" if is_chief else "delegated_agent",
        ),
    )
    g.add_edge(integrator_name, return_to)   # finalizing/reporting leaves upward
    for cname, cpath in children:
        _add_subtree(
            g, case,
            name=cname, layer=child_layer, path=cpath, return_to=name,
            n_layers=n_layers, fanouts=fanouts, full=full, hop_cap=hop_cap,
            generic=generic, structure_brief=structure_brief,
            context_sharing=context_sharing,
        )


def build_graph(case: dict[str, Any], visibility: str = "full", *,
                layers: int = 1, width: Any = 2, personas: str = "typed",
                context_sharing: str = DEFAULT_CONTEXT_SHARING):
    """Build the centralized (supervisor / supervisor-of-supervisors) graph.

    `layers` is the number of management layers (1 = a plain supervisor; >=2 nests
    supervisors). `width` is the fan-out: either a single value (uniform — every
    supervisor manages `width` children) or a per-layer sequence of length `layers`
    (e.g. [2, 3] = chief manages 2 sub-supervisors, each of which manages 3 workers).
    The top supervisor always sees the verbatim request; in `scoped` lower layers
    receive branch-local briefs; in `full` everyone sees the request.

    `personas` selects specialized personas (`typed`) or base function-only
    identities (`generic`); see PERSONA_MODES.
    """
    if personas not in PERSONA_MODES:
        raise ValueError(f"personas must be one of {PERSONA_MODES}, got {personas!r}")
    n_layers = max(1, int(layers))
    if isinstance(width, (list, tuple)):
        fanouts = [max(1, int(w)) for w in width]
        if len(fanouts) != n_layers:
            raise ValueError(
                f"per-layer width has {len(fanouts)} value(s) but layers={n_layers}; "
                f"give exactly {n_layers} comma-separated fan-out(s), one per management layer"
            )
    else:
        fanouts = [max(1, int(width))] * n_layers
    full = visibility == "full"
    generic = personas == "generic"
    brief = _structure_brief(n_layers, fanouts)

    # Hop budget: enough for the top to reach every leaf through every layer and back.
    # supervisors at layer k (1-indexed) = product(fanouts[:k-1]); workers = product(all).
    n_supervisors = 0
    prod = 1
    for f in fanouts:
        n_supervisors += prod
        prod *= f
    n_workers = prod
    hop_cap = max(6, 2 * (n_supervisors + n_workers) + 4)

    # Clean node name for the common flat case; "ChiefSupervisor" once it's a tree.
    top_name = "Supervisor" if n_layers == 1 else "ChiefSupervisor"

    g = StateGraph(AgentReportState)
    _add_subtree(
        g, case,
        name=top_name, layer=1, path=(), return_to=END,
        n_layers=n_layers, fanouts=fanouts, full=full, hop_cap=hop_cap,
        generic=generic, structure_brief=brief,
        context_sharing=context_sharing,
    )
    g.add_edge(START, top_name)
    return g.compile()
