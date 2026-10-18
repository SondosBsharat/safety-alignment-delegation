#!/usr/bin/env python3
"""Validate the mas-eval judge against human-reviewed gold labels."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluate import (
    DEFAULT_DATASET,
    DEFAULT_MAS_EVAL_EMBEDDING_DIM,
    DEFAULT_MAS_EVAL_EMBEDDING_MODEL,
    _case_for,
    llm_judge_eval,
)
from llm import get_eval_endpoint_pool, use_endpoint

ROOT = Path(__file__).resolve().parent

DEFAULT_CANONICAL_GOLD = Path("labeling") / "canonical_validation_v1" / "gold.jsonl"
DEFAULT_SUPPLEMENTAL_GOLD = Path("labeling") / "c2b_negative_control_v1" / "gold.jsonl"
DEFAULT_COMBINED_GOLD = Path("labeling") / "judge_validation_v1" / "gold.jsonl"
CONTRASTIVE_ENRICHMENT_NOTE = (
    "The supplemental stratum is contrastive class enrichment. Combined-set results "
    "estimate agreement/discrimination only; they are not estimates of deployment "
    "prevalence, violation rate, calibration, or population base rate."
)

CRITERIA = {
    "c1": "c1_unauthorized_action",
    "c2a": "c2a_over_disclosure",
    "c2b": "c2b_constraint_lost",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        row = json.loads(text)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_no} is not a JSON object")
        rows.append(row)
    return rows


def resolve_trace_path(results_root: Path, row: dict[str, Any]) -> Path:
    result_relpath = str(row.get("result_relpath") or "").strip()
    if result_relpath:
        path = Path(result_relpath)
        if not path.is_absolute():
            path = ROOT / path
        return path

    result_file = str(row.get("result_file") or "")
    if not result_file:
        raise ValueError(f"{row.get('uid')}: missing result_file")
    path = Path(result_file)
    if not path.is_absolute():
        condition = str(row.get("condition") or "")
        path = results_root / condition / result_file
    return path


def load_trace(results_root: Path, row: dict[str, Any]) -> dict[str, Any]:
    path = resolve_trace_path(results_root, row)
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"trace is not a JSON object: {path}")
    return payload


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_stratum(row: dict[str, Any], *, source_role: str) -> str:
    raw = str(row.get("validation_stratum") or row.get("gold_stratum") or "").strip().lower()
    if raw in {"original", "canonical", "primary"}:
        return "original"
    if raw in {"supplemental", "contrastive", "c2b_negative_control"}:
        return "supplemental"
    family = str(row.get("family") or "").strip().lower()
    if "negative_control" in family or family == "c2b-control":
        return "supplemental"
    return source_role


def human_class_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for crit in CRITERIA:
        values = [
            int((row.get("human") or {})[crit])
            for row in rows
            if (row.get("applicability") or {}).get(crit, True)
        ]
        n1 = sum(values)
        counts[crit] = {"0": len(values) - n1, "1": n1}
    return counts


def load_validation_gold(
    primary_gold: Path,
    supplemental_gold: list[Path] | None = None,
    *,
    split: str = "validation",
    results_root: Path = Path("results") / "baseline",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load, validate, and combine raw-backed human gold strata.

    The loader intentionally verifies the complete selected split before any caller
    applies ``--limit``. This prevents a smoke run from silently accepting duplicate
    ids, missing traces, or stale hashes elsewhere in the gold artifact.
    """
    primary_gold = Path(primary_gold)
    supplemental_gold = [Path(path) for path in (supplemental_gold or [])]
    sources = [(primary_gold, "original"), *[(path, "supplemental") for path in supplemental_gold]]

    resolved_sources: set[Path] = set()
    rows: list[dict[str, Any]] = []
    source_meta: list[dict[str, Any]] = []
    for source, source_role in sources:
        resolved = source.resolve()
        if resolved in resolved_sources:
            raise ValueError(f"gold source loaded more than once: {source}")
        resolved_sources.add(resolved)
        if not source.exists():
            raise FileNotFoundError(source)
        selected = [row for row in read_jsonl(source) if row.get("split") == split]
        source_meta.append({
            "path": source.as_posix(),
            "sha256": file_sha256(source),
            "count": len(selected),
            "source_role": source_role,
        })
        for source_row in selected:
            row = dict(source_row)
            row["_gold_stratum"] = _normalise_stratum(row, source_role=source_role)
            row["_gold_source"] = source.as_posix()
            rows.append(row)

    if not rows:
        raise ValueError(f"no rows with split={split!r} in configured gold source(s)")

    seen_trace_ids: set[str] = set()
    seen_case_ids: set[str] = set()
    for row in rows:
        uid = str(row.get("uid") or "<missing uid>")
        trace_id = str(row.get("trace_id") or "").strip()
        case_id = str(row.get("case_id") or "").strip()
        if not trace_id or not case_id:
            raise ValueError(f"{uid}: trace_id and case_id are required")
        if trace_id in seen_trace_ids:
            raise ValueError(f"duplicate trace_id in combined gold: {trace_id}")
        seen_trace_ids.add(trace_id)
        seen_case_ids.add(case_id)

        human = row.get("human")
        if not isinstance(human, dict):
            raise ValueError(f"{uid}: missing human labels")
        for crit in CRITERIA:
            if human.get(crit) not in {0, 1, False, True}:
                raise ValueError(f"{uid}: human.{crit} must be 0 or 1")

        trace_path = resolve_trace_path(results_root, row)
        if not trace_path.exists():
            raise FileNotFoundError(trace_path)
        expected_hash = str(row.get("trace_sha256") or "").strip().lower()
        if len(expected_hash) != 64:
            raise ValueError(f"{uid}: missing or invalid trace_sha256")
        actual_hash = file_sha256(trace_path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"{uid}: raw trace hash mismatch for {trace_path}: "
                f"expected {expected_hash}, got {actual_hash}"
            )
        trace = load_trace(results_root, row)
        raw_case_id = str(trace.get("case_id") or "").strip()
        if raw_case_id and raw_case_id != case_id:
            raise ValueError(f"{uid}: raw trace case_id {raw_case_id!r} != gold {case_id!r}")

    strata: dict[str, dict[str, Any]] = {}
    for name in ("original", "supplemental"):
        stratum_rows = [row for row in rows if row["_gold_stratum"] == name]
        strata[name] = {
            "count": len(stratum_rows),
            "criterion_class_counts": human_class_counts(stratum_rows) if stratum_rows else {
                crit: {"0": 0, "1": 0} for crit in CRITERIA
            },
        }
    unexpected = sorted({str(row["_gold_stratum"]) for row in rows} - set(strata))
    if unexpected:
        raise ValueError(f"unsupported validation stratum name(s): {unexpected}")

    fingerprint_rows = [
        {
            "trace_id": row["trace_id"],
            "case_id": row["case_id"],
            "trace_sha256": row["trace_sha256"],
            "human": row["human"],
            "stratum": row["_gold_stratum"],
        }
        for row in rows
    ]
    gold_set_sha256 = hashlib.sha256(
        json.dumps(fingerprint_rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    metadata = {
        "schema": "combined_human_reviewed_gold_v1",
        "total_count": len(rows),
        "strata": strata,
        "criterion_class_counts": human_class_counts(rows),
        "unique_trace_ids": len(seen_trace_ids),
        "unique_case_ids": len(seen_case_ids),
        "all_raw_traces_exist": True,
        "all_trace_hashes_match": True,
        "sources": source_meta,
        "gold_set_sha256": gold_set_sha256,
        "contrastive_enrichment_note": CONTRASTIVE_ENRICHMENT_NOTE,
    }
    return rows, metadata


def label_revision_rates(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    rates: dict[str, dict[str, Any]] = {}
    for crit in CRITERIA:
        drafted = [
            row for row in rows
            if isinstance(row.get("draft"), dict)
            and (row.get("draft") or {}).get(crit) in {0, 1, False, True}
        ]
        flips = sum(1 for row in drafted if bool((row.get("revision") or {}).get(crit)))
        n = len(drafted)
        rates[crit] = {
            "n": len(rows),
            "n_drafted": n,
            "flips": flips,
            "rate": flips / n if n else None,
            "near_zero_flag": flips == 0 if n else None,
            "unavailable": n == 0,
        }
    return rates


def criterion_metrics(gold: list[int], pred: list[int], *, low_n_threshold: int) -> dict[str, Any]:
    if len(gold) != len(pred):
        raise ValueError("gold and pred lengths differ")
    tp = sum(1 for g, p in zip(gold, pred) if g == 1 and p == 1)
    tn = sum(1 for g, p in zip(gold, pred) if g == 0 and p == 0)
    fp = sum(1 for g, p in zip(gold, pred) if g == 0 and p == 1)
    fn = sum(1 for g, p in zip(gold, pred) if g == 1 and p == 0)
    n = len(gold)
    pos = sum(gold)
    neg = n - pos
    minority = min(pos, neg) if n else 0
    both_classes_present = pos > 0 and neg > 0
    selection_eligible = both_classes_present and minority >= low_n_threshold
    ineligible_reasons: list[str] = []
    if not both_classes_present:
        ineligible_reasons.append("single_class_gold")
    elif minority < low_n_threshold:
        ineligible_reasons.append("minority_class_below_threshold")
    accuracy = (tp + tn) / n if n else None
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and (precision + recall)
        else None
    )
    if n and both_classes_present:
        p_gold_pos = (tp + fn) / n
        p_gold_neg = (tn + fp) / n
        p_pred_pos = (tp + fp) / n
        p_pred_neg = (tn + fn) / n
        expected = p_gold_pos * p_pred_pos + p_gold_neg * p_pred_neg
        kappa = None if expected == 1 else (accuracy - expected) / (1 - expected)
    else:
        kappa = None
    return {
        "n": n,
        "positive_count": pos,
        "negative_count": neg,
        "class_counts": {"0": neg, "1": pos},
        "minority_class_count": minority,
        "both_classes_present": both_classes_present,
        "selection_eligible": selection_eligible,
        "selection_ineligible_reasons": ineligible_reasons,
        "accuracy": accuracy,
        "cohen_kappa": kappa,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "low_n": not selection_eligible,
        "low_n_threshold": low_n_threshold,
    }


def macro_average(metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ("accuracy", "cohen_kappa", "f1", "precision", "recall"):
        vals = [m[key] for m in metrics.values() if m.get(key) is not None]
        out[key] = sum(vals) / len(vals) if vals else None
    return out


def compute_agreement(rows: list[dict[str, Any]], *, low_n_threshold: int) -> dict[str, Any]:
    metrics: dict[str, dict[str, Any]] = {}
    for crit in CRITERIA:
        usable = [
            row for row in rows
            if (row.get("pred") or {}).get(crit) is not None
            and (row.get("applicability") or {}).get(crit, True)
        ]
        gold = [int((row.get("human") or {})[crit]) for row in usable]
        pred = [int((row.get("pred") or {})[crit]) for row in usable]
        metrics[crit] = criterion_metrics(gold, pred, low_n_threshold=low_n_threshold)
    ineligible = [crit for crit, metric in metrics.items() if not metric["selection_eligible"]]
    return {
        "criteria": metrics,
        "macro": macro_average(metrics),
        "selection_eligible": not ineligible,
        "selection_ineligible_criteria": ineligible,
    }


def endpoint_for_model(endpoint: dict[str, Any] | None, model: str | None) -> dict[str, Any] | None:
    """Clone an endpoint and make an explicit CLI model authoritative.

    Eval-pool entries may carry their own model. Without this copy-and-override,
    ``--model deepseek-v4-pro`` can silently execute a pool's flash model instead.
    The original endpoint (which may contain a secret key) is never mutated or logged.
    """
    if endpoint is None and model is None:
        return None
    selected = dict(endpoint or {})
    if model:
        selected["model"] = model
    return selected


def evaluate_one(
    row: dict[str, Any],
    *,
    dataset: str,
    results_root: Path,
    memory_bank: str,
    top_k: int,
    embedding_model: str,
    embedding_dim: int,
) -> dict[str, Any]:
    out = {
        "uid": row.get("uid"),
        "trace_id": row.get("trace_id"),
        "case_id": row.get("case_id"),
        "family": row.get("family"),
        "condition": row.get("condition"),
        "structure": row.get("structure"),
        "result_file": row.get("result_file"),
        "result_relpath": row.get("result_relpath"),
        "trace_sha256": row.get("trace_sha256"),
        "validation_stratum": row.get("_gold_stratum") or row.get("validation_stratum"),
        "human": row.get("human"),
        "applicability": row.get("applicability") or {crit: True for crit in CRITERIA},
        "pred": {},
        "reasoning": {},
        "error": None,
    }
    try:
        trace = load_trace(results_root, row)
        case = _case_for(dataset, str(row.get("case_id") or trace.get("case_id")))
        if case is None:
            raise ValueError(f"case not found in dataset: {row.get('case_id')}")
        judged = llm_judge_eval(
            trace,
            case,
            judge_backend="mas-eval",
            memory_bank_path=memory_bank,
            mas_eval_top_k=top_k,
            mas_eval_embedding_model=embedding_model,
            mas_eval_embedding_dim=embedding_dim,
        )
        if "error" in judged:
            raise RuntimeError(str(judged["error"]))
        applicability = row.get("applicability") or {crit: True for crit in CRITERIA}
        for crit, key in CRITERIA.items():
            if not applicability.get(crit, True):
                out["pred"][crit] = None
                out["reasoning"][crit] = "structural N/A excluded from criterion metrics"
                continue
            node = judged.get(key) or {}
            if node.get("status") in {"invalid", "insufficient_evidence"} or node.get("violation") is None:
                raise ValueError(f"{crit} returned invalid/insufficient evidence")
            out["pred"][crit] = 1 if node.get("violation") else 0
            out["reasoning"][crit] = node.get("reasoning") or node.get("evidence") or ""
        out["raw_judge"] = judged
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
        for crit in CRITERIA:
            out["pred"][crit] = None
    return out


def _default_gold() -> Path:
    return DEFAULT_COMBINED_GOLD if (ROOT / DEFAULT_COMBINED_GOLD).exists() else DEFAULT_CANONICAL_GOLD


def validation_row_key(row: dict[str, Any], model: str | None) -> str:
    identity = {
        "uid": row.get("uid"),
        "trace_id": row.get("trace_id"),
        "case_id": row.get("case_id"),
        "trace_sha256": row.get("trace_sha256"),
        "model": model,
    }
    return hashlib.sha256(
        json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _stratum_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    strata: dict[str, dict[str, Any]] = {}
    for name in ("original", "supplemental"):
        selected = [row for row in rows if row.get("_gold_stratum") == name]
        strata[name] = {
            "count": len(selected),
            "criterion_class_counts": human_class_counts(selected) if selected else {
                crit: {"0": 0, "1": 0} for crit in CRITERIA
            },
        }
    return {
        "count": len(rows),
        "strata": strata,
        "criterion_class_counts": human_class_counts(rows),
    }


def judge_request_config() -> dict[str, Any]:
    """Return the deterministic request contract used by mas-eval."""
    return {
        "temperature": float(os.environ.get("MAS_EVAL_TEMPERATURE", "0")),
        "thinking": os.environ.get("MAS_EVAL_THINKING", "disabled").strip().lower(),
        "response_format": os.environ.get(
            "MAS_EVAL_RESPONSE_FORMAT", "json_object"
        ).strip().lower(),
        "max_tokens": int(os.environ.get("MAS_EVAL_MAX_TOKENS", "800")),
        "timeout_seconds": int(os.environ.get("MAS_EVAL_TIMEOUT", "120")),
        "parse_retries": max(1, int(os.environ.get("MAS_EVAL_PARSE_RETRIES", "2"))),
    }


def validation_code_hashes() -> dict[str, str]:
    paths = {
        "validate_judge": Path(__file__).resolve(),
        "evaluate": ROOT / "evaluate.py",
        "mas_eval_simple_judge": (
            ROOT / "mas_eval" / "mas_eval" / "simple_judge.py"
        ),
    }
    return {name: file_sha256(path) for name, path in paths.items()}


def _run_fingerprint(args: argparse.Namespace, gold_set_sha256: str, model: str | None) -> str:
    memory_bank_path = Path(args.memory_bank)
    config = {
        "gold_set_sha256": gold_set_sha256,
        "model": model,
        "split": args.split,
        "dataset": args.dataset,
        "memory_bank": args.memory_bank,
        "memory_bank_sha256": file_sha256(memory_bank_path),
        "top_k": args.mas_eval_top_k,
        "embedding_model": args.mas_eval_embedding_model,
        "embedding_dim": args.mas_eval_embedding_dim,
        "input_max_chars": args.mas_eval_input_max_chars,
        "judge_request_config": judge_request_config(),
        "validation_code_sha256": validation_code_hashes(),
    }
    return hashlib.sha256(
        json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _build_validation_payload(
    *,
    args: argparse.Namespace,
    model: str | None,
    rows: list[dict[str, Any]],
    gold_rows: list[dict[str, Any]],
    gold_metadata: dict[str, Any],
    run_fingerprint: str,
    complete: bool,
    pending_count: int,
    resumed_count: int,
    started_at: str,
) -> dict[str, Any]:
    errors = [row for row in rows if row.get("error")]
    agreement = compute_agreement(rows, low_n_threshold=args.low_n_threshold)
    return {
        "schema": "judge_validation_agreement_v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "started_at": started_at,
        "complete": complete,
        "completed_count": len(rows),
        "pending_count": pending_count,
        "resumed_count": resumed_count,
        "run_fingerprint": run_fingerprint,
        "judge_backend": "mas-eval",
        "model": model,
        "memory_bank": args.memory_bank,
        "memory_bank_sha256": file_sha256(Path(args.memory_bank)),
        "judge_request_config": judge_request_config(),
        "validation_code_sha256": validation_code_hashes(),
        "gold": args.gold,
        "supplemental_gold": [
            source["path"] for source in gold_metadata.get("sources", [])[1:]
        ],
        "split": args.split,
        "results_root": args.results_root,
        "dataset": args.dataset,
        "low_n_threshold": args.low_n_threshold,
        "mas_eval_input_max_chars": args.mas_eval_input_max_chars,
        "gold_metadata": gold_metadata,
        "evaluated_gold": _stratum_summary(gold_rows),
        "interpretation": {
            "contrastive_enrichment": bool(
                ((gold_metadata.get("strata") or {}).get("supplemental") or {}).get("count")
            ),
            "prevalence_or_calibration_estimate": False,
            "note": CONTRASTIVE_ENRICHMENT_NOTE,
        },
        "revision_rates": label_revision_rates(gold_rows),
        "agreement": agreement,
        "selection_eligible": agreement["selection_eligible"],
        "selection_ineligible_criteria": agreement["selection_ineligible_criteria"],
        "error_count": len(errors),
        "errors": [{"uid": row.get("uid"), "error": row.get("error")} for row in errors],
        "rows": rows,
    }


def run_validation(args: argparse.Namespace) -> dict[str, Any]:
    primary_gold = Path(args.gold)
    supplemental_arg = getattr(args, "supplemental_gold", None)
    if supplemental_arg is None and primary_gold.resolve() == (ROOT / DEFAULT_CANONICAL_GOLD).resolve():
        supplemental_paths = [ROOT / DEFAULT_SUPPLEMENTAL_GOLD] if (ROOT / DEFAULT_SUPPLEMENTAL_GOLD).exists() else []
    else:
        supplemental_paths = [Path(path) for path in (supplemental_arg or [])]
    all_gold_rows, gold_metadata = load_validation_gold(
        primary_gold,
        supplemental_paths,
        split=args.split,
        results_root=Path(args.results_root),
    )
    rows = all_gold_rows
    if args.limit:
        rows = rows[: int(args.limit)]
    if not rows:
        raise SystemExit(f"no rows with split={args.split!r} in {args.gold}")

    if args.model:
        os.environ["MAS_EVAL_MODEL"] = args.model
    if args.mas_eval_input_max_chars:
        os.environ["MAS_EVAL_INPUT_MAX_CHARS"] = str(args.mas_eval_input_max_chars)

    model = args.model or os.environ.get("MAS_EVAL_MODEL")
    pool = get_eval_endpoint_pool()
    workers = max(1, int(args.workers))
    run_fingerprint = _run_fingerprint(args, gold_metadata["gold_set_sha256"], model)
    started_at = datetime.now(timezone.utc).isoformat()

    resume_arg = getattr(args, "resume", None)
    checkpoint_path = Path(args.out) if getattr(args, "out", None) else None
    resume_path: Path | None = None
    if resume_arg:
        if resume_arg == "__OUT__":
            if checkpoint_path is None:
                raise ValueError("--resume without a path requires --out")
            resume_path = checkpoint_path
        else:
            resume_path = Path(resume_arg)
            if checkpoint_path is None:
                checkpoint_path = resume_path

    judged_by_index: dict[int, dict[str, Any]] = {}
    resumed_count = 0
    if resume_path is not None and resume_path.exists():
        previous = json.loads(resume_path.read_text(encoding="utf-8"))
        if previous.get("run_fingerprint") != run_fingerprint:
            raise ValueError(
                f"resume checkpoint does not match model/gold/config: {resume_path}"
            )
        previous_by_key = {
            str(row.get("resume_key")): row
            for row in (previous.get("rows") or [])
            if row.get("resume_key") and not row.get("error")
            and all((row.get("pred") or {}).get(crit) in {0, 1, False, True} for crit in CRITERIA)
        }
        for idx, row in enumerate(rows):
            key = validation_row_key(row, model)
            cached = previous_by_key.get(key)
            if cached is not None:
                judged_by_index[idx] = cached
                resumed_count += 1

    def _work(idx_row: tuple[int, dict[str, Any]]) -> dict[str, Any]:
        idx, row = idx_row
        ep = pool[idx % len(pool)] if pool else None
        # Explicit --model wins over a model embedded in MAS_EVAL_POOL.
        with use_endpoint(endpoint_for_model(ep, args.model)):
            judged = evaluate_one(
                row,
                dataset=args.dataset,
                results_root=Path(args.results_root),
                memory_bank=args.memory_bank,
                top_k=args.mas_eval_top_k,
                embedding_model=args.mas_eval_embedding_model,
                embedding_dim=args.mas_eval_embedding_dim,
            )
        judged["resume_key"] = validation_row_key(row, model)
        return judged

    def _checkpoint(*, complete: bool) -> dict[str, Any]:
        completed_indices = sorted(judged_by_index)
        completed_rows = [judged_by_index[idx] for idx in completed_indices]
        completed_gold = [rows[idx] for idx in completed_indices]
        payload = _build_validation_payload(
            args=args,
            model=model,
            rows=completed_rows,
            gold_rows=completed_gold,
            gold_metadata=gold_metadata,
            run_fingerprint=run_fingerprint,
            complete=complete,
            pending_count=len(rows) - len(completed_rows),
            resumed_count=resumed_count,
            started_at=started_at,
        )
        if checkpoint_path is not None:
            write_json_atomic(checkpoint_path, payload)
        return payload

    pending = [(idx, row) for idx, row in enumerate(rows) if idx not in judged_by_index]
    if resumed_count:
        print(f"[judge-validation] resumed {resumed_count}/{len(rows)} successful trace(s)")
    if checkpoint_path is not None:
        _checkpoint(complete=not pending)
    if workers == 1:
        for pair in pending:
            judged_by_index[pair[0]] = _work(pair)
            if checkpoint_path is not None:
                _checkpoint(complete=False)
    elif pending:
        print(f"[judge-validation] {len(pending)} pending trace(s), workers={workers}")
        if pool:
            print(f"[judge-validation] {len(pool)} endpoint(s); calls round-robin")
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_work, pair): pair[0] for pair in pending}
            for future in as_completed(futures):
                idx = futures[future]
                judged_by_index[idx] = future.result()
                if checkpoint_path is not None:
                    _checkpoint(complete=False)

    return _checkpoint(complete=True)


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def model_label(path: Path, payload: dict[str, Any]) -> str:
    model = payload.get("model")
    if model:
        return str(model)
    return path.stem


def select_best(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    def eligibility(payload: dict[str, Any]) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        if payload.get("complete") is False:
            reasons.append("validation_incomplete")
        if int(payload.get("error_count") or 0):
            reasons.append("judge_errors_present")
        criteria = ((payload.get("agreement") or {}).get("criteria") or {})
        threshold = int(payload.get("low_n_threshold") or 8)
        for crit in CRITERIA:
            metric = criteria.get(crit) or {}
            if "selection_eligible" in metric:
                eligible = bool(metric["selection_eligible"])
            else:
                n = int(metric.get("n") or 0)
                n1 = int(metric.get("positive_count") or 0)
                n0 = int(metric.get("negative_count", n - n1) or 0)
                eligible = n0 > 0 and n1 > 0 and min(n0, n1) >= threshold
            if not eligible:
                reasons.append(f"{crit}_class_support_ineligible")
        return not reasons, reasons

    def score(payload: dict[str, Any]) -> tuple[float, float]:
        agreement = payload.get("agreement") or {}
        criteria = agreement.get("criteria") or {}
        macro = agreement.get("macro") or {}
        c2b = (criteria.get("c2b") or {}).get("cohen_kappa")
        macro_kappa = macro.get("cohen_kappa")
        return (
            float("-inf") if c2b is None else float(c2b),
            float("-inf") if macro_kappa is None else float(macro_kappa),
        )

    eligible_payloads = [payload for payload in payloads if eligibility(payload)[0]]
    if not eligible_payloads:
        return {
            "model": None,
            "score": {"c2b_cohen_kappa": None, "macro_cohen_kappa": None},
            "selection_eligible": False,
            "selection_rule": "highest c2b cohen_kappa, tie-break macro cohen_kappa",
            "ineligible_models": {
                str(payload.get("model") or "<unknown>"): eligibility(payload)[1]
                for payload in payloads
            },
        }

    best = max(eligible_payloads, key=score)
    return {
        "model": best.get("model"),
        "score": {
            "c2b_cohen_kappa": score(best)[0],
            "macro_cohen_kappa": score(best)[1],
        },
        "selection_eligible": True,
        "selection_rule": "highest c2b cohen_kappa, tie-break macro cohen_kappa",
    }


def write_summary(inputs: list[Path], out_path: Path) -> dict[str, Any]:
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in inputs]
    lines = [
        "# mas-eval judge validation summary",
        "",
        "| model | criterion | n | negatives | positives | minority | kappa | f1 | accuracy | precision | recall | selection eligible |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for path, payload in zip(inputs, payloads):
        label = model_label(path, payload)
        criteria = ((payload.get("agreement") or {}).get("criteria") or {})
        for crit in ("c1", "c2a", "c2b"):
            m = criteria.get(crit) or {}
            lines.append(
                "| {model} | {crit} | {n} | {neg} | {pos} | {minority} | {kappa} | {f1} | {acc} | {prec} | {rec} | {eligible} |".format(
                    model=label,
                    crit=crit,
                    n=m.get("n", 0),
                    neg=m.get("negative_count", 0),
                    pos=m.get("positive_count", 0),
                    minority=m.get("minority_class_count", 0),
                    kappa=fmt(m.get("cohen_kappa")),
                    f1=fmt(m.get("f1")),
                    acc=fmt(m.get("accuracy")),
                    prec=fmt(m.get("precision")),
                    rec=fmt(m.get("recall")),
                    eligible="yes" if m.get("selection_eligible") else "no",
                )
            )
        macro = ((payload.get("agreement") or {}).get("macro") or {})
        lines.append(
            f"| {label} | macro |  |  |  |  | {fmt(macro.get('cohen_kappa'))} | "
            f"{fmt(macro.get('f1'))} | {fmt(macro.get('accuracy'))} | "
            f"{fmt(macro.get('precision'))} | {fmt(macro.get('recall'))} |  |"
        )
    selection = select_best(payloads)
    lines += ["", "## Selection", ""]
    if selection["selection_eligible"]:
        lines += [
            f"Selected model: `{selection['model']}`",
            "",
            "Rule: highest C2b Cohen's kappa; tie-break macro Cohen's kappa.",
            "",
            f"C2b kappa: {fmt(selection['score']['c2b_cohen_kappa'])}",
            f"Macro kappa: {fmt(selection['score']['macro_cohen_kappa'])}",
        ]
    else:
        lines += [
            "No configuration selected: one or more criteria lack both human-gold "
            "classes or the required minority-class support.",
            "",
            f"Ineligibility: `{json.dumps(selection['ineligible_models'], sort_keys=True)}`",
        ]
    lines += ["", f"> {CONTRASTIVE_ENRICHMENT_NOTE}", ""]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return selection


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--gold",
        default=str(_default_gold()),
        help=(
            "Primary/combined human gold JSONL. Defaults to the physical combined "
            "artifact when present, otherwise the canonical 60-row artifact."
        ),
    )
    ap.add_argument(
        "--supplemental-gold",
        action="append",
        default=None,
        help=(
            "Supplemental human-reviewed gold JSONL (repeatable). When canonical gold "
            "is the default primary, the standard C2b control gold is added automatically "
            "if present; a physical combined --gold is never loaded twice."
        ),
    )
    ap.add_argument("--split", default="validation", choices=("base", "validation"))
    ap.add_argument("--results-root", default=str(Path("results") / "baseline"))
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--memory-bank", default="mas_eval/memory_bank_human_v1.json")
    ap.add_argument("--model", default=os.environ.get("MAS_EVAL_MODEL"))
    ap.add_argument("--mas-eval-top-k", type=int, default=int(os.environ.get("MAS_EVAL_TOP_K", "3")))
    ap.add_argument("--mas-eval-embedding-model", default=os.environ.get(
        "MAS_EVAL_EMBEDDING_MODEL", DEFAULT_MAS_EVAL_EMBEDDING_MODEL,
    ))
    ap.add_argument("--mas-eval-embedding-dim", type=int, default=int(os.environ.get(
        "MAS_EVAL_EMBEDDING_DIM", str(DEFAULT_MAS_EVAL_EMBEDDING_DIM),
    )))
    ap.add_argument(
        "--mas-eval-input-max-chars",
        type=int,
        default=int(os.environ.get("MAS_EVAL_INPUT_MAX_CHARS", "8000")),
        help="Cap judge input text before retrieval/prompting. Use the same value in evaluate.py.",
    )
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--low-n-threshold", type=int, default=8)
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--resume",
        nargs="?",
        const="__OUT__",
        default=None,
        metavar="CHECKPOINT",
        help=(
            "Reuse successful rows from a matching atomic checkpoint. With no path, "
            "resume from --out; with a path and no --out, continue writing that path."
        ),
    )
    ap.add_argument(
        "--summary-inputs",
        nargs="+",
        help="Read agreement JSON files and write the side-by-side markdown summary.",
    )
    ap.add_argument("--summary-out", default=str(Path("evals") / "validation" / "agreement_summary.md"))
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.summary_inputs:
        selection = write_summary([Path(p) for p in args.summary_inputs], Path(args.summary_out))
        print(f"[summary] {args.summary_out}")
        print(f"[selected] {selection['model']}")
        return

    payload = run_validation(args)
    if args.out:
        write_json_atomic(Path(args.out), payload)
        print(f"[agreement] {args.out}")
    print(json.dumps({
        "model": payload.get("model"),
        "error_count": payload.get("error_count"),
        "agreement": payload.get("agreement"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
