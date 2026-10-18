#!/usr/bin/env python3
"""Combine compatible criterion-selective evaluator artifacts after validation."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import evaluate


CRITERIA = evaluate._CRITERION_ORDER


def load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"not a JSON object: {path}")
    return payload


def _compatibility_contract(payload: dict[str, Any]) -> dict[str, Any]:
    checkpoint = payload.get("checkpoint") or {}
    fingerprint = checkpoint.get("fingerprint_payload")
    if not isinstance(fingerprint, dict):
        progress_path = checkpoint.get("progress_path")
        if not progress_path:
            raise ValueError("input lacks checkpoint progress provenance")
        progress = load(Path(progress_path))
        fingerprint = progress.get("fingerprint_payload") or {}
    if not isinstance(fingerprint, dict):
        raise ValueError("input lacks fingerprint_payload")
    contract = copy.deepcopy(fingerprint)
    contract.pop("selected_criteria", None)
    contract.pop("criterion_contract", None)
    return contract


def combine(paths: list[Path]) -> dict[str, Any]:
    if not paths:
        raise ValueError("at least one --input is required")
    payloads = [load(path) for path in paths]
    coverage: dict[str, tuple[dict[str, Any], Path]] = {}
    reference_contract = _compatibility_contract(payloads[0])
    for path, payload in zip(paths, payloads):
        if _compatibility_contract(payload) != reference_contract:
            raise ValueError(f"incompatible corrected fingerprint contract: {path}")
        selected = payload.get("selected_criteria") or []
        if not isinstance(selected, list) or not selected:
            raise ValueError(f"missing selected_criteria: {path}")
        for criterion in selected:
            if criterion not in CRITERIA:
                raise ValueError(f"unknown criterion {criterion}: {path}")
            if criterion in coverage:
                raise ValueError(f"criterion {criterion} appears in multiple inputs")
            coverage[criterion] = (payload, path)
    missing = [criterion for criterion in CRITERIA if criterion not in coverage]
    if missing:
        raise ValueError(f"missing criterion artifacts: {','.join(missing)}")

    run_maps: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    for criterion, (payload, path) in coverage.items():
        runs = payload.get("runs") or []
        if not isinstance(runs, list):
            raise ValueError(f"runs must be a list: {path}")
        mapping: dict[tuple[str, str], dict[str, Any]] = {}
        for run in runs:
            key = (str(run.get("case_id") or ""), str(run.get("result_file") or ""))
            if not all(key) or key in mapping:
                raise ValueError(f"invalid or duplicate run identity in {path}: {key}")
            status = evaluate._run_checkpoint_status(run, True, (criterion,))
            if status != "success":
                raise ValueError(f"{path}:{key} is {status}; cannot combine")
            mapping[key] = run
        run_maps[criterion] = mapping

    identities = set(run_maps["c1"])
    for criterion in CRITERIA[1:]:
        if set(run_maps[criterion]) != identities:
            raise ValueError(f"run identity mismatch for {criterion}")

    runs: list[dict[str, Any]] = []
    for key in sorted(identities):
        base = copy.deepcopy(run_maps["c1"][key])
        base_llm = base.get("llm") or {}
        base_raw = base_llm.get("mas_eval") or {}
        for criterion in CRITERIA:
            source = run_maps[criterion][key]
            if (
                source.get("structure") != base.get("structure")
                or source.get("rule") != base.get("rule")
                or source.get("c1_deterministic") != base.get("c1_deterministic")
            ):
                raise ValueError(f"non-semantic run fields differ for {key} ({criterion})")
            source_llm = source.get("llm") or {}
            llm_key = evaluate._LLM_KEY[criterion]
            base_llm[llm_key] = copy.deepcopy(source_llm[llm_key])
            if criterion in (source_llm.get("mas_eval") or {}):
                base_raw[criterion] = copy.deepcopy(source_llm["mas_eval"][criterion])
        base_llm["mas_eval"] = base_raw
        base_llm["selected_criteria"] = list(CRITERIA)
        base_llm["prompt_versions"] = dict(evaluate._PROMPT_VERSIONS)
        base_llm["renderer_versions"] = dict(evaluate._RENDERER_VERSIONS)
        base["llm"] = base_llm
        runs.append(base)

    summary = evaluate.aggregate(runs, True, CRITERIA)
    summary["runs"] = runs
    summary["skipped"] = 0
    summary["selected_criteria"] = list(CRITERIA)
    sources = [
        {
            "path": str(path.resolve()),
            "criteria": payload.get("selected_criteria"),
            "fingerprint": (payload.get("checkpoint") or {}).get("fingerprint"),
        }
        for path, payload in zip(paths, payloads)
    ]
    summary["combined_provenance"] = {
        "schema": "combined_criterion_eval_v1",
        "sources": sources,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = combine(args.input)
    evaluate._atomic_write_json(args.out, payload)
    print(f"combined {len(payload['runs'])} runs -> {args.out}")


if __name__ == "__main__":
    main()
