#!/usr/bin/env python3
"""Rebuild normalized semantic verdicts from preserved structured Judge output."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import evaluate


def renormalize(source: Path, destination: Path) -> dict[str, Any]:
    payload = json.loads(source.read_text(encoding="utf-8"))
    criteria = tuple(payload.get("selected_criteria") or ())
    if len(criteria) != 1 or criteria[0] not in evaluate._CRITERION_ORDER:
        raise ValueError(f"{source}: expected one selected semantic criterion")
    criterion = criteria[0]
    llm_key = evaluate._LLM_KEY[criterion]
    corrected = 0
    applicable = 0
    runs = payload.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError(f"{source}: missing runs")
    for index, run in enumerate(runs):
        llm = run.get("llm") if isinstance(run, dict) else None
        if not isinstance(llm, dict) or llm.get("error"):
            raise ValueError(f"{source}: run {index} has no successful Judge result")
        current = llm.get(llm_key)
        if criterion == "c2b" and isinstance(current, dict) and current.get("status") == "structural_not_applicable":
            continue
        raw = (llm.get("mas_eval") or {}).get(criterion)
        if not isinstance(raw, dict) or not isinstance(raw.get("structured_output"), dict):
            raise ValueError(f"{source}: run {index} lacks preserved structured output")
        normalized = evaluate._normalize_semantic_result(criterion, raw)
        llm[llm_key] = normalized
        applicable += 1
        corrected += int(bool(normalized.get("consistency_corrected")))

    aggregate = evaluate.aggregate(runs, True, criteria)
    for key, value in aggregate.items():
        payload[key] = value
    payload["runs"] = runs
    payload["selected_criteria"] = list(criteria)
    payload["renormalization_provenance"] = {
        "schema": "semantic_renormalization_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(source.resolve()),
        "criterion": criterion,
        "applicable_runs": applicable,
        "consistency_corrected_runs": corrected,
        "method": "deterministic rebuild from preserved mas_eval.structured_output",
    }
    evaluate._atomic_write_json(destination, payload)
    return payload["renormalization_provenance"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    provenance = renormalize(args.input, args.out)
    print(json.dumps(provenance, ensure_ascii=False))


if __name__ == "__main__":
    main()
