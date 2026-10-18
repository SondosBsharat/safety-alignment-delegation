#!/usr/bin/env python3
"""Run a secret-safe, resumable cross-model defense matrix.

This launcher owns raw generation only and invokes ``main.py`` one task cell at
a time, so ``--workers`` is the total concurrency instead of a multiplier
across cells.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlsplit

from dotenv import dotenv_values


REPO = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = REPO / "dataset" / "pilot_tasks_mas.json"
DEFAULT_SMOKE_CASE = "pilot_hc_referral_letter_001"
EXPECTED_CASES = 600
TRANSIENT_PATTERN = re.compile(
    r"(?:\b429\b|rate.?limit|too many requests|timeout|timed out|"
    r"connection(?:error|reset|aborted)?|transport|temporar(?:y|ily)|"
    r"service unavailable|bad gateway|gateway timeout)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Condition:
    name: str
    structure: str
    extra: tuple[str, ...]


CONDITIONS: tuple[Condition, ...] = (
    Condition("S1", "single", ()),
    Condition("N2", "decentralized", ("--width", "2", "--personas", "generic")),
    Condition("N4", "decentralized", ("--width", "4", "--personas", "generic")),
    Condition("N8", "decentralized", ("--width", "8", "--personas", "generic")),
    Condition("C1", "centralized", ("--layers", "1", "--width", "2", "--personas", "generic")),
    Condition("C2", "centralized", ("--layers", "2", "--width", "2", "--personas", "generic")),
    Condition("C3", "centralized", ("--layers", "3", "--width", "2", "--personas", "generic")),
    Condition("N4-t", "decentralized", ("--width", "4", "--personas", "typed")),
    Condition("C2-t", "centralized", ("--layers", "2", "--width", "2", "--personas", "typed")),
)
TREATMENTS = (("no_defense", "none"), ("authorizer", "authorizer"))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def model_slug(model: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", model.strip()).strip("-.")
    if not slug:
        raise ValueError("model must contain at least one safe path character")
    return slug


def experiment_slug(args: argparse.Namespace) -> str:
    if args.lead_model or args.delegate_model:
        return model_slug(
            f"lead-{args.lead_model or args.model}__delegate-"
            f"{args.delegate_model or args.model}"
        )
    return model_slug(args.model)


def endpoint_origin(base_url: str | None) -> str:
    if not base_url:
        return "https://api.openai.com"
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("--base-url must be an absolute HTTP(S) URL")
    return f"{parsed.scheme}://{parsed.netloc}"


def expected_ids(dataset: Path) -> set[str]:
    payload = json.loads(dataset.read_text(encoding="utf-8"))
    rows = payload if isinstance(payload, list) else payload.get("tasks", [])
    result = {
        str(row.get("case_id") or row.get("task_id"))
        for row in rows
        if isinstance(row, Mapping) and (row.get("case_id") or row.get("task_id"))
    }
    if not result:
        raise ValueError(f"no case IDs found in {dataset}")
    return result


def trace_matches(
    trace: Mapping[str, Any],
    *,
    condition: Condition,
    defense: str,
    model: str,
    expected_origin: str,
    lead_model: str | None = None,
    delegate_model: str | None = None,
) -> list[str]:
    errors: list[str] = []
    if trace.get("status") != "success":
        errors.append("status")
    if trace.get("structure") != condition.structure:
        errors.append("structure")
    if trace.get("visibility") != "scoped":
        errors.append("visibility")
    if trace.get("context_sharing") != "branch-local":
        errors.append("context_sharing")
    worker = trace.get("worker_llm")
    if not isinstance(worker, Mapping):
        errors.append("worker_llm")
    else:
        if worker.get("provider") != "openai":
            errors.append("worker_llm.provider")
        if worker.get("model") != model:
            errors.append("worker_llm.model")
        observed_base = worker.get("base_url")
        observed_origin = endpoint_origin(str(observed_base)) if observed_base else "https://api.openai.com"
        if observed_origin != expected_origin:
            errors.append("worker_llm.base_url")
    defense_meta = trace.get("defense")
    if defense == "none":
        if defense_meta:
            errors.append("unexpected defense")
    else:
        components = defense_meta.get("components") if isinstance(defense_meta, Mapping) else None
        if components != ["authorizer"]:
            errors.append("defense.components")
        schema = defense_meta.get("authorization_policy_schema") if isinstance(defense_meta, Mapping) else None
        if schema != "structured_authorization_policy_v1_1":
            errors.append("defense.policy")
    if lead_model or delegate_model:
        expected_models = {
            "lead_agent": lead_model or model,
            "delegated_agent": delegate_model or model,
        }
        routing = trace.get("agent_model_routing")
        profiles = routing.get("profiles") if isinstance(routing, Mapping) else None
        if not isinstance(profiles, Mapping):
            errors.append("agent_model_routing")
        else:
            for role, expected_model in expected_models.items():
                profile = profiles.get(role)
                if not isinstance(profile, Mapping) or profile.get("model") != expected_model:
                    errors.append(f"agent_model_routing.{role}.model")
        bindings = trace.get("model_bindings")
        if not isinstance(bindings, list) or not bindings:
            errors.append("model_bindings")
        else:
            observed_roles: set[str] = set()
            for binding in bindings:
                if not isinstance(binding, Mapping):
                    errors.append("model_bindings.row")
                    continue
                role = str(binding.get("role") or "")
                observed_roles.add(role)
                if role not in expected_models or binding.get("model") != expected_models.get(role):
                    errors.append("model_bindings.assignment")
            if "lead_agent" not in observed_roles:
                errors.append("model_bindings.lead_agent")
            if condition.name != "S1" and "delegated_agent" not in observed_roles:
                errors.append("model_bindings.delegated_agent")
            if condition.name == "S1" and "delegated_agent" in observed_roles:
                errors.append("model_bindings.unexpected_delegate")
    return errors


def audit_cell(
    directory: Path,
    ids: set[str],
    *,
    condition: Condition,
    defense: str,
    model: str,
    expected_origin: str,
    lead_model: str | None = None,
    delegate_model: str | None = None,
) -> dict[str, Any]:
    valid: set[str] = set()
    duplicates: set[str] = set()
    invalid: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in sorted(directory.glob("*.json")) if directory.exists() else ():
        try:
            trace = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            invalid.append({"file": path.name, "errors": [f"json:{type(exc).__name__}"]})
            continue
        case_id = str(trace.get("case_id") or "")
        if case_id in seen:
            duplicates.add(case_id)
        seen.add(case_id)
        errors = trace_matches(
            trace,
            condition=condition,
            defense=defense,
            model=model,
            expected_origin=expected_origin,
            lead_model=lead_model,
            delegate_model=delegate_model,
        )
        if case_id not in ids:
            errors.append("case_id")
        if errors:
            invalid.append({"file": path.name, "case_id": case_id, "errors": errors})
        else:
            valid.add(case_id)
    missing = sorted(ids - valid)
    return {
        "directory": str(directory),
        "expected": len(ids),
        "valid": len(valid),
        "missing_count": len(missing),
        "missing_ids": missing,
        "duplicates": sorted(duplicates),
        "invalid_count": len(invalid),
        "invalid": invalid[:50],
        "complete": len(valid) == len(ids) and not duplicates and not invalid,
    }


def build_command(
    args: argparse.Namespace,
    *,
    condition: Condition,
    defense: str,
    out_dir: Path,
    workers: int,
    case_id: str | None = None,
    dataset: Path | None = None,
) -> list[str]:
    selected_dataset = dataset or args.dataset
    command = [
        str(args.python), str(REPO / "main.py"),
        "--dataset", str(selected_dataset),
        "--structure", condition.structure,
        *condition.extra,
        "--visibility", "scoped",
        "--defense", defense,
        "--out-dir", str(out_dir),
        "--workers", str(max(1, workers)),
        "--llm-provider", "openai",
        "--llm-model", args.model,
        "--llm-api-keys-env", args.key_env,
        "--resume",
    ]
    if args.lead_model:
        command.extend(("--lead-llm-model", args.lead_model))
    if args.delegate_model:
        command.extend(("--delegate-llm-model", args.delegate_model))
    if args.base_url:
        command.extend(("--llm-base-url", args.base_url))
    if case_id:
        command.extend(("--case-id", case_id))
    return command


def subprocess_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    secret = env.get(args.key_env, "").strip()
    if not secret:
        # Read only the explicitly named secret from the repository configuration.
        # This keeps unrelated ambient provider/base/model values out of the launcher.
        secret = str(dotenv_values(REPO / ".env").get(args.key_env) or "").strip()
    if not secret:
        raise ValueError(f"missing or empty credential environment variable {args.key_env!r}")
    if args.key_index is not None:
        keys = [item.strip() for item in secret.split(",") if item.strip()]
        if not (0 <= args.key_index < len(keys)):
            raise ValueError(
                f"--key-index {args.key_index} is out of range for {args.key_env!r} "
                f"({len(keys)} key(s))"
            )
        secret = keys[args.key_index]
    env[args.key_env] = secret
    env["MAS_LLM_PROVIDER"] = "openai"
    env["OPENAI_MODEL"] = args.model
    env.pop("MAS_LLM_POOL", None)
    env.pop("OPENAI_API_KEYS", None)
    # An empty existing variable prevents python-dotenv from restoring the ambient
    # DeepSeek relay URL while ChatOpenAI still interprets it as the official default.
    env["OPENAI_BASE_URL"] = args.base_url or ""
    env["OPENAI_REASONING_EFFORT"] = args.reasoning_effort
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def invoke(command: Sequence[str], env: Mapping[str, str], log_path: Path) -> tuple[int, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = log_path.stat().st_size if log_path.exists() else 0
    with log_path.open("a", encoding="utf-8", newline="\n") as log:
        log.write(f"\n[{utc_now()}] START {' '.join(command)}\n")
        log.flush()
        completed = subprocess.run(command, cwd=REPO, env=dict(env), stdout=log, stderr=subprocess.STDOUT)
        log.write(f"[{utc_now()}] EXIT {completed.returncode}\n")
    with log_path.open("r", encoding="utf-8", errors="replace") as stream:
        stream.seek(start)
        segment = stream.read()
    return completed.returncode, segment


def run_smoke(args: argparse.Namespace, state_path: Path, env: Mapping[str, str]) -> int:
    condition = next(item for item in CONDITIONS if item.name == args.smoke_condition)
    smoke_ids = {args.smoke_case}
    slug = experiment_slug(args)
    root = args.output_root / slug / "smoke"
    for treatment, defense in TREATMENTS:
        out_dir = root / treatment / condition.name
        log_path = args.log_root / slug / "smoke" / f"{treatment}.log"
        command = build_command(
            args,
            condition=condition,
            defense=defense,
            out_dir=out_dir,
            workers=1,
            case_id=args.smoke_case,
        )
        atomic_json(state_path, {
            "schema": "model_defense_matrix_state_v1",
            "status": "smoke_running",
            "model": args.model,
            "treatment": treatment,
            "condition": condition.name,
            "updated_at": utc_now(),
        })
        code, segment = invoke(command, env, log_path)
        audit = audit_cell(
            out_dir,
            smoke_ids,
            condition=condition,
            defense=defense,
            model=args.model,
            expected_origin=endpoint_origin(args.base_url),
            lead_model=args.lead_model,
            delegate_model=args.delegate_model,
        )
        tool_ok = False
        for path in out_dir.glob("*.json"):
            try:
                trace = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if trace.get("case_id") == args.smoke_case and int(trace.get("tool_call_count") or 0) > 0:
                tool_ok = True
        if code != 0 or not audit["complete"] or not tool_ok:
            atomic_json(state_path, {
                "schema": "model_defense_matrix_state_v1",
                "status": "smoke_failed",
                "model": args.model,
                "treatment": treatment,
                "condition": condition.name,
                "exit_code": code,
                "audit": audit,
                "tool_call_observed": tool_ok,
                "transient_errors": len(TRANSIENT_PATTERN.findall(segment)),
                "updated_at": utc_now(),
            })
            return 2
    atomic_json(state_path, {
        "schema": "model_defense_matrix_state_v1",
        "status": "smoke_complete",
        "model": args.model,
        "smoke_case": args.smoke_case,
        "smoke_condition": args.smoke_condition,
        "updated_at": utc_now(),
    })
    return 0


def selected_conditions(names: Iterable[str] | None) -> list[Condition]:
    wanted = set(names or ())
    selected = [condition for condition in CONDITIONS if not wanted or condition.name in wanted]
    unknown = wanted - {condition.name for condition in CONDITIONS}
    if unknown:
        raise ValueError(f"unknown condition(s): {', '.join(sorted(unknown))}")
    return selected


def run_full(args: argparse.Namespace, state_path: Path, env: Mapping[str, str]) -> int:
    default_ids = expected_ids(args.dataset)
    if len(default_ids) != args.expected:
        raise ValueError(f"dataset has {len(default_ids)} tasks; expected {args.expected}")
    conditions = selected_conditions(args.only)
    treatments = [item for item in TREATMENTS if not args.treatment or item[0] == args.treatment]
    root = args.output_root / experiment_slug(args)
    current_workers = args.workers
    cell_results: dict[str, Any] = {}
    for treatment, defense in treatments:
        for condition in conditions:
            cell_dataset = args.dataset
            ids = default_ids
            key = f"{treatment}/{condition.name}"
            out_dir = root / treatment / condition.name
            log_path = args.log_root / experiment_slug(args) / treatment / f"{condition.name}.log"
            audit = audit_cell(
                out_dir, ids, condition=condition, defense=defense,
                model=args.model, expected_origin=endpoint_origin(args.base_url),
                lead_model=args.lead_model, delegate_model=args.delegate_model,
            )
            if audit["complete"]:
                cell_results[key] = audit
                continue
            attempts = 0
            while not audit["complete"] and attempts < args.max_attempts:
                attempts += 1
                atomic_json(state_path, {
                    "schema": "model_defense_matrix_state_v1",
                    "status": "running",
                    "model": args.model,
                    "endpoint_origin": endpoint_origin(args.base_url),
                    "dataset": str(cell_dataset),
                    "treatment": treatment,
                    "condition": condition.name,
                    "workers": current_workers,
                    "fallback_workers": args.fallback_workers,
                    "attempt": attempts,
                    "valid_before": audit["valid"],
                    "output": str(out_dir),
                    "log": str(log_path),
                    "completed_cells": cell_results,
                    "updated_at": utc_now(),
                })
                command = build_command(
                    args, condition=condition, defense=defense,
                    out_dir=out_dir, workers=current_workers, dataset=cell_dataset,
                )
                code, segment = invoke(command, env, log_path)
                after = audit_cell(
                    out_dir, ids, condition=condition, defense=defense,
                    model=args.model, expected_origin=endpoint_origin(args.base_url),
                    lead_model=args.lead_model, delegate_model=args.delegate_model,
                )
                transient_errors = len(TRANSIENT_PATTERN.findall(segment))
                progress = after["valid"] - audit["valid"]
                audit = after
                if (
                    current_workers > args.fallback_workers
                    and not audit["complete"]
                    and transient_errors >= args.fallback_threshold
                ):
                    current_workers = args.fallback_workers
            cell_results[key] = audit
            if not audit["complete"]:
                atomic_json(state_path, {
                    "schema": "model_defense_matrix_state_v1",
                    "status": "incomplete",
                    "model": args.model,
                    "failed_cell": key,
                    "workers": current_workers,
                    "audit": audit,
                    "completed_cells": cell_results,
                    "updated_at": utc_now(),
                })
                return 3
    atomic_json(state_path, {
        "schema": "model_defense_matrix_state_v1",
        "status": "complete",
        "model": args.model,
        "endpoint_origin": endpoint_origin(args.base_url),
        "workers_final": current_workers,
        "cells": cell_results,
        "updated_at": utc_now(),
    })
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--lead-model", default=None,
        help="Optional lead-agent model; transport/key pool inherits --model settings.",
    )
    parser.add_argument(
        "--delegate-model", default=None,
        help="Optional delegated-agent model; transport/key pool inherits --model settings.",
    )
    parser.add_argument("--key-env", required=True)
    parser.add_argument(
        "--key-index", type=int, default=None,
        help="Use one zero-based key from a comma-separated named key pool without exposing it.",
    )
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--reasoning-effort", default="none")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--output-root", type=Path, default=REPO / "results" / "model_transfer")
    parser.add_argument("--log-root", type=Path, default=REPO / "run_logs" / "model_transfer")
    parser.add_argument("--state", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--fallback-workers", type=int, default=4)
    parser.add_argument("--fallback-threshold", type=int, default=3)
    parser.add_argument("--max-attempts", type=int, default=8)
    parser.add_argument("--expected", type=int, default=EXPECTED_CASES)
    parser.add_argument("--smoke-case", default=DEFAULT_SMOKE_CASE)
    parser.add_argument("--smoke-condition", default="C1")
    parser.add_argument("--treatment", choices=("no_defense", "authorizer"), default=None)
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args(argv)
    if args.workers < 1 or args.fallback_workers < 1:
        parser.error("worker counts must be positive")
    if args.fallback_workers > args.workers:
        parser.error("--fallback-workers must not exceed --workers")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", args.key_env):
        parser.error("--key-env must be a valid environment-variable name")
    args.dataset = args.dataset.resolve()
    args.output_root = args.output_root.resolve()
    args.log_root = args.log_root.resolve()
    if args.state is None:
        args.state = args.log_root / experiment_slug(args) / "state.json"
    else:
        args.state = args.state.resolve()
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        env = subprocess_env(args)
        if args.mode == "smoke":
            return run_smoke(args, args.state, env)
        return run_full(args, args.state, env)
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        atomic_json(args.state, {
            "schema": "model_defense_matrix_state_v1",
            "status": "failed_preflight",
            "model": args.model,
            "error": f"{type(exc).__name__}: {exc}",
            "updated_at": utc_now(),
        })
        print(f"[preflight failed] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
