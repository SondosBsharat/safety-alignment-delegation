r"""Dataset-driven runner for the two MAS structures (centralized / decentralized).

Default dataset is the 600-task pilot with MAS ground truth
(dataset/pilot_tasks_mas.json), loaded in the task/environment split format
(pilot_tasks_mas.json + environments/). For the 24-task MAS pilot, pass
--dataset dataset\pilot_tasks_mas_pilot.json explicitly.

Each run is written to out-dir as soon as it finishes (atomic temp-file + rename),
so a large batch is not lost on a crash; --resume picks up where it stopped.

Usage (PowerShell):
    $env:OPENAI_API_KEY = "sk-..."
    python main.py                                          # all cases x all structures
    python main.py --structure decentralized                 # all cases, one structure
    python main.py --case-id pilot_hc_referral_letter_001 --structure centralized --layers 2
    python main.py --dataset dataset\pilot_tasks_mas.json --out-dir results\raw
    python main.py --resume                                  # re-run only missing runs

Switch provider:
    $env:MAS_LLM_PROVIDER = "google"
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import centralized
import decentralized
import single
from agent_react import DEFAULT_CONTEXT_SHARING, PERSONA_MODES, VISIBILITY_MODES, build_entry_message
from dataset import load_cases
from defense import DefenseConfig, use_defense
from llm import (
    apply_role_profile,
    get_active_endpoint,
    get_agent_model_routing_metadata,
    get_endpoint_pool,
    get_role_endpoint_pool,
    role_profiles_configured,
    use_agent_model_routes,
    use_endpoint,
)
from tracing import Recorder, use_recorder

# Serializes console output from concurrent workers so per-run lines stay intact.
_PRINT_LOCK = threading.Lock()


def _log(msg: str) -> None:
    with _PRINT_LOCK:
        print(msg)


def _token_totals(token_stats: dict | None) -> tuple[int, int]:
    """(worker_tokens, defense_added_tokens) from a trace's per-stage token_stats.

    Defense-added = authorizer:* + chain:* stages; worker = baseline agent reasoning.
    Judge tokens never appear here (they are spent at eval time, not in the run).
    """
    ts = token_stats or {}
    worker = ts.get("worker", {}).get("total_tokens", 0)
    defense = sum(
        v.get("total_tokens", 0)
        for k, v in ts.items()
        if k.startswith("authorizer") or k.startswith("chain")
    )
    return worker, defense

# Coordination structures — what `--structure all` expands to. `single` is a
# non-coordination baseline (one agent, no handoff): a valid explicit choice but
# deliberately NOT part of `all`, so the default matrix run keeps its meaning.
STRUCTURES = ["centralized", "decentralized"]
SINGLE = "single"
ALL_STRUCTURE_CHOICES = STRUCTURES + [SINGLE]

# personas (typed vs generic identities) applies to both structures.
_PERSONA_STRUCTURES = {"centralized", "decentralized"}


def _parse_width(s: str):
    """`--width` value: a single int (uniform) or a comma list (per-layer, centralized
    only), e.g. "4" -> 4, "2,3" -> (2, 3)."""
    parts = [p.strip() for p in str(s).split(",") if p.strip() != ""]
    vals = [int(p) for p in parts]   # argparse surfaces ValueError as a clean CLI error
    if not vals:
        raise ValueError("empty --width")
    return vals[0] if len(vals) == 1 else tuple(vals)

# Safety backstop on graph super-steps. The structures converge on their own via the
# per-structure hop cap (max_hops); this only guards against a pathological non-converging
# run and is set well above any real run's step count.
RECURSION_LIMIT = 200

_ENV_VAR_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _normalize_openai_base_url(value: str) -> str:
    """Return an OpenAI-compatible API root, never a resource endpoint."""
    raw = value.strip().rstrip("/")
    parsed = urlsplit(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("--llm-base-url must be an absolute http(s) URL")
    if parsed.query or parsed.fragment:
        raise ValueError("--llm-base-url must not contain a query string or fragment")
    path = parsed.path.rstrip("/")
    if path.lower().endswith("/chat/completions"):
        raise ValueError(
            "--llm-base-url must be the API root (for example https://relay.example.com/v1), "
            "not the full /v1/chat/completions endpoint"
        )
    if not path:
        path = "/v1"
    elif not path.lower().endswith("/v1"):
        path += "/v1"
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def _configure_worker_llm(args, environ: dict[str, str] | None = None) -> None:
    """Apply explicit raw-worker CLI overrides without touching MAS_EVAL_* vars."""
    env = os.environ if environ is None else environ
    if args.llm_provider is not None:
        env["MAS_LLM_PROVIDER"] = args.llm_provider
    effective_provider = args.llm_provider or env.get("MAS_LLM_PROVIDER", "openai")
    if args.llm_model is not None:
        model_var = {
            "openai": "OPENAI_MODEL",
            "deepseek": "DEEPSEEK_MODEL",
            "google": "GEMINI_MODEL",
        }[effective_provider]
        env[model_var] = args.llm_model
    if args.llm_base_url is not None:
        if effective_provider != "openai":
            raise ValueError(
                "--llm-base-url is only valid with --llm-provider openai; official "
                "Google provider uses its native SDK endpoint"
            )
        env["OPENAI_BASE_URL"] = _normalize_openai_base_url(args.llm_base_url)
    if args.llm_api_keys_env is not None:
        name = args.llm_api_keys_env.strip()
        if not _ENV_VAR_NAME_RE.fullmatch(name):
            raise ValueError("--llm-api-keys-env must be a valid environment-variable name")
        value = env.get(name, "").strip()
        if not value:
            raise ValueError(
                f"--llm-api-keys-env names missing or empty environment variable {name!r}"
            )
        keys = [key.strip() for key in value.split(",") if key.strip()]
        if not keys:
            raise ValueError(f"environment variable {name!r} contains no usable API keys")
        if effective_provider == "openai":
            env["OPENAI_API_KEYS"] = ",".join(keys)
            # An explicit CLI key pool takes precedence over an ambient heterogeneous pool.
            env.pop("MAS_LLM_POOL", None)
        else:
            # MAS_LLM_POOL is provider-neutral; native clients consume the active
            # endpoint's key/model through the same per-run ContextVar.
            env["MAS_LLM_POOL"] = json.dumps(
                [{"api_key": key} for key in keys], ensure_ascii=False
            )
            env.pop("OPENAI_API_KEYS", None)
    _configure_role_llms(args, env)


def _configure_role_llms(args, env: dict[str, str]) -> None:
    """Apply optional lead/delegate overrides without exposing literal credentials."""
    uniform_provider = env.get("MAS_LLM_PROVIDER", "openai")
    for role, attr_prefix, env_prefix in (
        ("lead", "lead_llm", "MAS_LEAD_LLM"),
        ("delegate", "delegate_llm", "MAS_DELEGATE_LLM"),
    ):
        provider = getattr(args, f"{attr_prefix}_provider", None)
        model = getattr(args, f"{attr_prefix}_model", None)
        base_url = getattr(args, f"{attr_prefix}_base_url", None)
        key_env = getattr(args, f"{attr_prefix}_api_keys_env", None)
        effort = getattr(args, f"{attr_prefix}_reasoning_effort", None)
        if not any((provider, model, base_url, key_env, effort)):
            continue
        effective_provider = provider or uniform_provider
        if provider:
            env[f"{env_prefix}_PROVIDER"] = provider
        if model:
            env[f"{env_prefix}_MODEL"] = model
        if base_url:
            if effective_provider != "openai":
                raise ValueError(
                    f"--{role}-llm-base-url is only valid for provider openai"
                )
            env[f"{env_prefix}_BASE_URL"] = _normalize_openai_base_url(base_url)
        if effort:
            env[f"{env_prefix}_REASONING_EFFORT"] = effort
        if key_env:
            name = key_env.strip()
            if not _ENV_VAR_NAME_RE.fullmatch(name):
                raise ValueError(
                    f"--{role}-llm-api-keys-env must name a valid environment variable"
                )
            value = env.get(name, "").strip()
            if not value:
                raise ValueError(
                    f"--{role}-llm-api-keys-env names missing or empty variable {name!r}"
                )
            keys = [key.strip() for key in value.split(",") if key.strip()]
            if not keys:
                raise ValueError(f"environment variable {name!r} contains no usable keys")
            env[f"{env_prefix}_API_KEYS"] = ",".join(keys)


def _apply_worker_pool_overrides(
    pool: list[dict[str, Any]], *, model: str | None, base_url: str | None
) -> list[dict[str, Any]]:
    """Make explicit CLI model/base overrides win over per-entry ambient pool fields."""
    normalized_base = _normalize_openai_base_url(base_url) if base_url else None
    if not model and not normalized_base:
        return pool
    overridden: list[dict[str, Any]] = []
    for endpoint in pool:
        item = dict(endpoint)
        if model:
            item["model"] = model
        if normalized_base:
            item["base_url"] = normalized_base
        overridden.append(item)
    return overridden


def _active_worker_llm_metadata() -> dict[str, str | None]:
    """Secret-free endpoint provenance for traces and reproducibility checks."""
    provider = os.environ.get("MAS_LLM_PROVIDER", "openai")
    endpoint = get_active_endpoint()
    if provider == "openai":
        model = endpoint.get("model") or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        base_url = endpoint.get("base_url") or os.environ.get("OPENAI_BASE_URL")
    elif provider == "deepseek":
        model = endpoint.get("model") or os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
        base_url = endpoint.get("base_url") or os.environ.get(
            "DEEPSEEK_BASE_URL", "https://api.deepseek.com"
        )
    else:
        model = endpoint.get("model") or os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
        base_url = None
    return {"provider": provider, "model": model, "base_url": base_url}


def _explicit_worker_llm_resume_key(args) -> str | None:
    """Differentiate explicit model experiments while preserving legacy no-flag resume."""
    role_values = [
        getattr(args, f"{role}_llm_{field}", None)
        for role in ("lead", "delegate")
        for field in ("provider", "model", "base_url", "api_keys_env", "reasoning_effort")
    ]
    if not any(
        (args.llm_provider, args.llm_model, args.llm_base_url, args.llm_api_keys_env,
         *role_values)
    ):
        return None
    payload: dict[str, Any] = {"uniform": _active_worker_llm_metadata()}
    routing = get_agent_model_routing_metadata()
    if routing:
        payload["agent_model_routing"] = routing
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def _initial_state(structure: str, case: dict[str, Any]) -> dict[str, Any]:
    # Every official structure shares one message channel: the run opens with the
    # entry user message and agents append to it as they act / hand off.
    return {"messages": [build_entry_message(case)]}


# Per-structure size knobs. width = agents per level; max_hops = hand-off
# cap (decentralized only); layers = management layers (centralized only).
# An axis absent here is not applicable to that structure.
STRUCTURE_DEFAULTS: dict[str, dict[str, int]] = {
    "decentralized": {"width": 3, "max_hops": 6},
    "centralized": {"width": 2, "layers": 1},      # 1 layer, 2 workers (= old supervisor)
    "single": {},                                  # baseline: one agent, no size axes
}


def _resolve_dims(structure: str, width: int | None, max_hops: int | None,
                  layers: int | None) -> dict[str, int]:
    """Merge CLI overrides over the structure's defaults, dropping unsupported axes."""
    dims = dict(STRUCTURE_DEFAULTS[structure])
    for name, value in (("width", width), ("max_hops", max_hops), ("layers", layers)):
        if value is not None and name in dims:
            dims[name] = value
    return dims


# --- Resume support ---------------------------------------------------------
# A "run key" uniquely identifies one (case, structure, params) combination. On
# --resume we scan out-dir, key every already-written trace, and skip any planned
# run whose key is already present — so an interrupted 800-run batch picks up
# exactly where it stopped instead of redoing finished work.

def _defense_sig(defense: dict[str, Any] | None) -> str | None:
    """Canonical run-key signature for defense configuration only.

    Trace metadata also stores the extracted boundary_state, which is case-specific.
    Resume keys should distinguish the enabled method(s), not fragment on extracted
    state details that are already implied by the case.
    """
    if not defense:
        return None
    components = defense.get("components") if isinstance(defense, dict) else None
    if not components:
        return None
    payload = {
        "schema": "boundary_state_defense_config_v1",
        "components": list(components),
    }
    if "chain" in components or "chain-advisory" in components:
        # Missing fields intentionally identify a prototype Chain trace.  A new v2
        # command therefore cannot silently resume coarse AuthorizationScope output.
        if defense.get("authorization_policy_schema"):
            payload["authorization_policy_schema"] = defense["authorization_policy_schema"]
        if defense.get("chain_generation"):
            payload["chain_generation"] = defense["chain_generation"]
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def _personas_key(structure: str, personas: str) -> str | None:
    """personas only differentiates structures that consume it; None elsewhere.

    Both current structures consume it, but the guard is kept so a future axis-free
    structure would not fragment its run keys on personas."""
    return personas if structure in _PERSONA_STRUCTURES else None


def _run_key(case_id, structure, visibility, width, max_hops, layers,
             personas, defense_sig=None, context_sharing=DEFAULT_CONTEXT_SHARING,
             worker_llm_resume_key=None) -> tuple:
    # centralized per-layer width is a tuple in-process but a list after a JSON
    # round-trip (trace files) — coerce so the key is hashable and both forms match.
    if isinstance(width, list):
        width = tuple(width)
    return (case_id, structure, visibility, width, max_hops, layers,
            personas, defense_sig, context_sharing,
            worker_llm_resume_key)


def _completed_keys(out_dir: Path) -> set[tuple]:
    """Keys of every complete trace already in out_dir.

    Tolerant of corrupt/partial files (e.g. a previous crash mid-write): those are
    simply ignored, so the corresponding run is re-done rather than skipped.
    """
    keys: set[tuple] = set()
    if not out_dir.exists():
        return keys
    for p in out_dir.glob("*.json"):
        try:
            with p.open(encoding="utf-8") as f:
                t = json.load(f)
        except Exception:
            continue
        if not (isinstance(t, dict) and t.get("case_id") and t.get("structure")):
            continue
        if t.get("status") == "failed":
            continue
        structure = t.get("structure")
        personas = t.get("personas")
        # Legacy traces predate the personas axis and were all typed; normalize a
        # missing value to "typed" so re-running the default still skips them.
        if personas is None and structure in _PERSONA_STRUCTURES:
            personas = "typed"
        # Traces from before the connectivity sweep have no cut_edges/graph_seed; they
        # were all the complete graph (cut 0), so normalize missing to 0 to keep the
        # default decentralized run matching them.
        keys.add(_run_key(
            t.get("case_id"), structure, t.get("visibility"),
            t.get("width"), t.get("max_hops"), t.get("layers"),
            _personas_key(structure, personas),
            _defense_sig(t.get("defense")),
            t.get("context_sharing") or DEFAULT_CONTEXT_SHARING,
            t.get("worker_llm_resume_key"),
        ))
    return keys


def _build_graph(structure: str, case: dict[str, Any], visibility: str = "full", *,
                 width: int | None = None, max_hops: int | None = None,
                 layers: int | None = None, personas: str = "typed",
                 context_sharing: str = DEFAULT_CONTEXT_SHARING):
    dims = _resolve_dims(structure, width, max_hops, layers)
    kwargs: dict[str, Any] = dict(dims)
    kwargs["personas"] = personas   # both structures consume personas
    kwargs["context_sharing"] = context_sharing
    builder = {
        "centralized": centralized.build_graph,
        "decentralized": decentralized.build_graph,
        "single": single.build_graph,
    }[structure]
    return builder(case, visibility, **kwargs)


def _final_message(structure: str, final_state: dict[str, Any]) -> str:
    # All official structures share one message channel; the final user-facing answer
    # is the last AI message in it.
    from agent_react import extract_final_message
    return extract_final_message(final_state)


def _atomic_write_trace(
    trace: dict[str, Any],
    *,
    out_dir: Path,
    case_id: str,
    structure: str,
    suffix: str = "",
) -> Path:
    """Crash-safe per-run trace write, including content-free failure diagnostics."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    out_path = out_dir / f"{case_id}__{structure}__{ts}{suffix}.json"
    tmp_path = out_path.with_suffix(".json.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(trace, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, out_path)
    return out_path


def run_one(
    case: dict[str, Any],
    structure: str,
    out_dir: Path,
    visibility: str = "full",
    width: int | None = None,
    max_hops: int | None = None,
    layers: int | None = None,
    personas: str = "typed",
    defense_config: DefenseConfig | None = None,
    context_sharing: str = DEFAULT_CONTEXT_SHARING,
    worker_llm_resume_key: str | None = None,
) -> dict[str, Any]:
    defense_config = defense_config or DefenseConfig()
    # Each run gets its own workspace: tools now mutate case["workspace_state"]
    # (drafts created, state committed, preconditions satisfied) and those writes
    # must persist across this run's agents but never bleed into the next
    # case/structure run, which reuses the same loaded case object.
    case = copy.deepcopy(case)
    recorder = Recorder(case_id=case["case_id"], structure=structure)
    defense_meta: dict[str, Any] | None = None
    dims = _resolve_dims(structure, width, max_hops, layers)
    try:
        with use_recorder(recorder), use_defense(case, defense_config) as defense:
            if defense is not None:
                defense_meta = defense.metadata()
            graph = _build_graph(structure, case, visibility, width=width,
                                 max_hops=max_hops, layers=layers, personas=personas,
                                 context_sharing=context_sharing)
            final_state = graph.invoke(
                _initial_state(structure, case),
                {"recursion_limit": RECURSION_LIMIT},
            )
    except Exception as exc:
        failure_trace = recorder.to_dict()
        failure_trace.update({
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "visibility": visibility,
            "context_sharing": context_sharing,
            "width": dims.get("width"),
            "max_hops": dims.get("max_hops"),
            "layers": dims.get("layers"),
            "personas": _personas_key(structure, personas),
            "worker_llm": _active_worker_llm_metadata(),
            "tool_call_count": len(failure_trace.get("tool_calls") or []),
            "handoff_count": len(failure_trace.get("handoffs") or []),
        })
        routing_meta = get_agent_model_routing_metadata()
        if routing_meta:
            failure_trace["agent_model_routing"] = routing_meta
        if not defense_config.is_empty():
            failure_trace["defense"] = defense_meta or defense_config.to_dict()
        if worker_llm_resume_key is not None:
            failure_trace["worker_llm_resume_key"] = worker_llm_resume_key
        _atomic_write_trace(
            failure_trace,
            # Keep diagnostics outside the top-level result glob used by evaluate.
            out_dir=out_dir / "_failed",
            case_id=case["case_id"],
            structure=structure,
            suffix="__failed",
        )
        raise

    trace = recorder.to_dict()
    trace["status"] = "success"
    trace["visibility"] = visibility
    trace["context_sharing"] = context_sharing
    trace["width"] = dims.get("width")       # # peers / fan-out per layer
    trace["max_hops"] = dims.get("max_hops")  # None for centralized (auto hop cap)
    trace["layers"] = dims.get("layers")      # None for decentralized
    trace["personas"] = _personas_key(structure, personas)
    trace["worker_llm"] = _active_worker_llm_metadata()
    routing_meta = get_agent_model_routing_metadata()
    if routing_meta:
        trace["agent_model_routing"] = routing_meta
    if worker_llm_resume_key is not None:
        trace["worker_llm_resume_key"] = worker_llm_resume_key
    if not defense_config.is_empty():  # omit when off -> baseline trace shape remains close
        trace["defense"] = defense_meta or defense_config.to_dict()
    trace["final_message"] = _final_message(structure, final_state)
    trace["tool_call_count"] = len(trace["tool_calls"])
    trace["handoff_count"] = len(trace["handoffs"])

    out_path = _atomic_write_trace(
        trace,
        out_dir=out_dir,
        case_id=case["case_id"],
        structure=structure,
    )

    bits = []
    if dims.get("width") is not None:
        bits.append(f"w={dims['width']}")
    if dims.get("max_hops") is not None:
        bits.append(f"h={dims['max_hops']}")
    if dims.get("layers") is not None:
        bits.append(f"L={dims['layers']}")
    if structure in _PERSONA_STRUCTURES:
        bits.append(personas)
    bits.append(f"ctx={context_sharing}")
    if not defense_config.is_empty():
        bits.append("def=" + "+".join(defense_config.components))
    dims_str = ",".join(bits)
    worker_tok, defense_tok = _token_totals(trace.get("token_stats"))
    if worker_tok or defense_tok:
        if defense_tok:
            ov = f", +{round(100 * defense_tok / worker_tok, 1)}%" if worker_tok else ""
            tok_str = f", tokens={worker_tok}(+{defense_tok} def{ov})"
        else:
            tok_str = f", tokens={worker_tok}"
    else:
        tok_str = ""
    _log(
        f"[{structure:>12}] {case['case_id']} ({dims_str}): "
        f"tool_calls={trace['tool_call_count']}, "
        f"handoffs={trace['handoff_count']}{tok_str}, "
        f"saved={out_path}"
    )
    return trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", default=str(Path("dataset") / "pilot_tasks_mas.json")
    )
    parser.add_argument(
        "--structure",
        default="all",
        choices=ALL_STRUCTURE_CHOICES + ["all"],
    )
    parser.add_argument("--case-id", default=None)
    parser.add_argument(
        "--visibility",
        default="scoped",
        choices=list(VISIBILITY_MODES),
        help="How much downstream agents see. full: every agent sees the request and "
             "full history. scoped: downstream agents receive branch-local task context "
             "without the verbatim user request.",
    )
    parser.add_argument(
        "--personas",
        default="typed",
        choices=list(PERSONA_MODES),
        help="Agent identity mode (applies to both structures). typed: specialized "
             "personas — decentralized peers (Analyst/Skeptic/...), worker specialists "
             "(Research/Execution/...), the default. generic: base identities — generic "
             "equal peers / function-only managers + generic executor workers (the "
             "no-persona control group).",
    )
    parser.add_argument(
        "--defense",
        default="none",
        help="Boundary-State Defense components to enable. Use none (default), "
             "authorizer, chain, both, or a comma list such as authorizer,chain. "
             "authorizer is the paper's Source defense: an LLM authorization agent "
             "over user_query + public "
             "tool names/descriptions; chain propagates a complete v2 Tool-Scope-Stage "
             "policy lineage through handoffs.",
    )
    parser.add_argument(
        "--width", type=_parse_width, default=None,
        help="Horizontal size (parallel agents). decentralized: # peers. centralized: "
             "fan-out per management layer — a single value (uniform) OR a comma list of "
             "length = --layers for per-layer fan-out, e.g. '2,3' = chief manages 2 "
             "sub-supervisors, each manages 3 workers. Omit for the per-structure default.",
    )
    parser.add_argument(
        "--max-hops", type=int, default=None,
        help="Hop cap: max hand-offs before agents are told to finalize. Applies to "
             "decentralized only; centralized auto-computes its own cap from the tree "
             "size, so this is ignored there. Omit for the default (6).",
    )
    parser.add_argument(
        "--layers", type=int, default=None,
        help="centralized only: number of management layers (top = layer 1; layers=1 "
             "is a plain supervisor, layers>=2 nests supervisors). Ignored by "
             "decentralized. Omit for the default (1).",
    )
    parser.add_argument("--out-dir", default=str(Path("results") / "default"))
    parser.add_argument(
        "--llm-provider", choices=["openai", "deepseek", "google"],
        default=None,
        help="Raw-worker transport provider override. For an OpenAI-compatible relay, "
             "use openai for models served through an OpenAI-compatible relay.",
    )
    parser.add_argument(
        "--llm-model", default=None,
        help="Raw-worker model override (for example gpt-5.4-mini or deepseek-v4-flash).",
    )
    parser.add_argument(
        "--llm-base-url", default=None,
        help="Raw-worker OpenAI-compatible API root. An origin such as "
             "https://relay.example.com is normalized to /v1. Do not include /chat/completions.",
    )
    parser.add_argument(
        "--llm-api-keys-env", default=None, metavar="ENV_VAR",
        help="Name of an environment variable containing one or more comma-separated "
             "raw-worker API keys. The secret value stays out of the command line.",
    )
    for role in ("lead", "delegate"):
        parser.add_argument(
            f"--{role}-llm-provider",
            choices=["openai", "deepseek", "google"],
            default=None,
            help=f"Optional {role} agent provider; inherits --llm-provider when omitted.",
        )
        parser.add_argument(
            f"--{role}-llm-model", default=None,
            help=f"Optional {role} agent model override.",
        )
        parser.add_argument(
            f"--{role}-llm-base-url", default=None,
            help=f"Optional OpenAI-compatible API root for the {role} profile.",
        )
        parser.add_argument(
            f"--{role}-llm-api-keys-env", default=None, metavar="ENV_VAR",
            help=f"Named comma-separated key environment for the {role} profile.",
        )
        parser.add_argument(
            f"--{role}-llm-reasoning-effort", default=None,
            help=f"Optional OpenAI reasoning effort for the {role} profile.",
        )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Run this many (case, structure) runs concurrently in a thread pool "
             "(LLM calls are I/O-bound, so threads give real speedup). Each run is "
             "independent and writes its own file, so no data is run twice. With a "
             "multi-endpoint pool (MAS_LLM_POOL / OPENAI_API_KEYS) runs round-robin "
             "across the keys/relays. Default 1 = sequential (original behaviour).",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip (case, structure) runs already present in --out-dir, matching on "
             "case_id + structure + visibility + width + max_hops + layers + personas "
             "+ defense. Lets an interrupted batch "
             "(network drop, etc.) resume without "
             "redoing finished runs. Corrupt/partial trace files are ignored and re-run.",
    )
    args = parser.parse_args()
    try:
        _configure_worker_llm(args)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        defense_config = DefenseConfig.from_spec(args.defense)
    except ValueError as exc:
        raise SystemExit(str(exc))

    cases = load_cases(args.dataset, case_id=args.case_id)
    structures = STRUCTURES if args.structure == "all" else [args.structure]
    if isinstance(args.width, tuple) and any(s != "centralized" for s in structures):
        raise SystemExit("--width as a per-layer list (e.g. 2,3) is only valid for "
                         "--structure centralized")
    out_dir = Path(args.out_dir)
    defense_sig = defense_config.signature()
    completed = _completed_keys(out_dir) if args.resume else set()
    worker_llm_resume_key = _explicit_worker_llm_resume_key(args)
    if args.resume:
        print(f"[resume] found {len(completed)} completed run(s) in {out_dir}")

    # Build the job grid ONCE (each (case, structure) appears exactly once, so no
    # data is ever run twice), dropping anything already completed on --resume.
    jobs: list[tuple[dict[str, Any], str]] = []
    skipped = 0
    for case in cases:
        for structure in structures:
            if args.resume:
                dims = _resolve_dims(structure, args.width, args.max_hops, args.layers)
                key = _run_key(
                    case["case_id"], structure,
                    args.visibility,
                    dims.get("width"), dims.get("max_hops"), dims.get("layers"),
                    _personas_key(structure, args.personas),
                    defense_sig,
                    DEFAULT_CONTEXT_SHARING,
                    worker_llm_resume_key,
                )
                if key in completed:
                    skipped += 1
                    continue
            jobs.append((case, structure))

    # Endpoint pool: each run is pinned (round-robin) to one key/relay so several
    # keys share the load. Empty pool => every run uses the single OPENAI_* default.
    pool = _apply_worker_pool_overrides(
        get_endpoint_pool(), model=args.llm_model, base_url=args.llm_base_url
    )
    lead_pool = get_role_endpoint_pool("lead_agent")
    delegate_pool = get_role_endpoint_pool("delegated_agent")
    workers = max(1, int(args.workers))
    if pool:
        print(f"[pool] {len(pool)} endpoint(s); runs round-robin across them")
    print(f"[run] {len(jobs)} run(s) to do, {workers} worker(s)"
          + (f", {skipped} skipped (resume)" if args.resume and skipped else ""))

    # RELAY-WORKAROUND (中转 API): one bad run must not abort the whole batch — log it
    # and continue. The exception is caught per run so a single failure is isolated.
    failures: list[tuple[str, str, str]] = []
    # Batch token tally (run-time cost visibility): summed across this invocation's runs.
    batch_tokens = {"worker": 0, "defense": 0, "runs": 0}

    def _do(idx: int, case: dict[str, Any], structure: str) -> None:
        endpoint = pool[idx % len(pool)] if pool else None
        routes = None
        if role_profiles_configured():
            lead_base = lead_pool[idx % len(lead_pool)] if lead_pool else endpoint
            delegate_base = (
                delegate_pool[idx % len(delegate_pool)] if delegate_pool else endpoint
            )
            routes = {
                "lead_agent": apply_role_profile(lead_base, "lead_agent"),
                "delegated_agent": apply_role_profile(
                    delegate_base, "delegated_agent"
                ),
            }
        try:
            with use_endpoint(endpoint), use_agent_model_routes(routes):
                trace = run_one(
                    case, structure, out_dir,
                    visibility=args.visibility,
                    width=args.width, max_hops=args.max_hops, layers=args.layers,
                    personas=args.personas,
                    defense_config=defense_config,
                    context_sharing=DEFAULT_CONTEXT_SHARING,
                    worker_llm_resume_key=worker_llm_resume_key,
                )
            # run_one already logs its own success line; here we only tally tokens.
            w, d = _token_totals(trace.get("token_stats"))
            with _PRINT_LOCK:
                batch_tokens["worker"] += w
                batch_tokens["defense"] += d
                batch_tokens["runs"] += 1
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            with _PRINT_LOCK:
                failures.append((case["case_id"], structure, msg))
            _log(f"[  FAILED  ] {case['case_id']} / {structure}: {msg[:200]}")

    if workers == 1:
        for i, (case, structure) in enumerate(jobs):
            _do(i, case, structure)
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_do, i, case, structure)
                       for i, (case, structure) in enumerate(jobs)]
            for f in as_completed(futures):
                f.result()  # _do swallows run errors; this surfaces only bugs in _do

    if batch_tokens["runs"]:
        w, d = batch_tokens["worker"], batch_tokens["defense"]
        line = f"\n[tokens] {batch_tokens['runs']} run(s): worker={w}"
        if d:
            ov = f", +{round(100 * d / w, 1)}%" if w else ""
            line += f", defense-added={d}{ov} (within-run overhead)"
        print(line + "  — per-cell breakdown + cost/benefit at eval time")

    if args.resume and skipped:
        print(f"\n[resume] skipped {skipped} already-completed run(s).")
    if failures:
        print(f"\n{len(failures)} run(s) failed and were skipped:")
        for cid, st, err in failures:
            print(f"  {cid} / {st}: {err[:160]}")
        if args.resume:
            print("Re-run the same command to retry only the failed/missing runs.")


if __name__ == "__main__":
    main()
