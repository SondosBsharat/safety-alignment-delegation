#!/usr/bin/env python3
"""Audit authorizer confirmations and estimate an all-confirmations-denied counterfactual."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from evaluate import (  # noqa: E402
    _annotate_gate_outcomes,
    _case_for,
    _completion_eval,
    _gt,
)


CONDITIONS = ("S1", "N2", "N4", "N8", "C1", "C2", "C3", "N4-t", "C2-t")


def _rate(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def _completion_totals() -> Counter[str]:
    return Counter({
        "runs": 0,
        "strict_completed": 0,
        "tool_completed": 0,
        "steps_completed": 0,
        "steps_total": 0,
        "weighted_score_sum": 0,
        "weighted_score_runs": 0,
    })


def _add_completion(totals: Counter[str], completion: dict[str, Any]) -> None:
    totals["runs"] += 1
    totals["strict_completed"] += int(bool(completion.get("completed")))
    totals["tool_completed"] += int(bool(completion.get("tool_completed")))
    totals["steps_completed"] += int(completion.get("required_steps_completed") or 0)
    totals["steps_total"] += int(completion.get("required_steps_total") or 0)
    score = completion.get("required_step_weighted_score")
    if score is not None:
        totals["weighted_score_sum"] += float(score)
        totals["weighted_score_runs"] += 1


def _completion_summary(totals: Counter[str]) -> dict[str, Any]:
    return {
        "runs": totals["runs"],
        "strict_all_steps_complete_rate": _rate(
            totals["strict_completed"], totals["runs"]
        ),
        "tool_only_completion_rate": _rate(totals["tool_completed"], totals["runs"]),
        "required_step_coverage": _rate(totals["steps_completed"], totals["steps_total"]),
        "weighted_required_step_score": _rate(
            totals["weighted_score_sum"], totals["weighted_score_runs"]
        ),
        "counts": dict(totals),
    }


def analyze_cell(path: Path, dataset: str) -> dict[str, Any]:
    ask_classes: Counter[str] = Counter()
    failure_dimensions: Counter[str] = Counter()
    ask_tools: Counter[str] = Counter()
    baseline = _completion_totals()
    all_deny = _completion_totals()
    runs_with_confirmation = 0

    traces = sorted(
        item for item in path.glob("*.json") if not item.name.lower().startswith("eval")
    )
    for trace_path in traces:
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        case = _case_for(dataset, str(trace.get("case_id") or ""))
        if case is None:
            raise ValueError(f"case not found: {trace.get('case_id')}")
        specs = {
            str(tool.get("name") or ""): tool
            for tool in case.get("available_tools", [])
            if isinstance(tool, dict)
        }
        required, forbidden = _gt(case)
        calls = list(trace.get("tool_calls") or [])
        annotated, invalid = _annotate_gate_outcomes(trace, calls)
        if invalid or any(not row.get("valid") for row in annotated):
            raise ValueError(f"invalid gate alignment: {trace_path}")

        original_completion = _completion_eval(trace, case, specs)
        _add_completion(baseline, original_completion)

        counterfactual = copy.deepcopy(trace)
        counterfactual_calls = counterfactual.get("tool_calls") or []
        confirmation_rows = [
            row for row in annotated
            if row.get("authorization_outcome") == "needs_user_confirmation"
        ]
        runs_with_confirmation += int(bool(confirmation_rows))
        events = trace.get("defense_events") or []
        for row in confirmation_rows:
            tool = str(row.get("tool") or "")
            if tool in forbidden:
                ask_classes["forbidden_current_request"] += 1
            elif tool in required:
                ask_classes["required_task_tool"] += 1
            else:
                ask_classes["neither_required_nor_forbidden"] += 1
            ask_classes["total"] += 1
            ask_classes["executed_by_auto_consent"] += int(bool(row.get("effect_executed")))
            ask_tools[tool] += 1
            event_index = row.get("gate_event_index")
            payload = (
                events[event_index].get("payload") or {}
                if isinstance(event_index, int) and 0 <= event_index < len(events)
                else {}
            )
            failure_dimensions[str(payload.get("failure_dimension") or "none")] += 1

            blocked = {
                "status": "blocked",
                "code": "counterfactual_all_confirmation_denied",
                "operation": tool,
            }
            call = counterfactual_calls[int(row["call_index"])]
            call["result"] = blocked
            call["result_preview"] = blocked

        denied_completion = _completion_eval(counterfactual, case, specs)
        _add_completion(all_deny, denied_completion)

    total = ask_classes["total"]
    return {
        "trace_count": len(traces),
        "runs_with_confirmation": runs_with_confirmation,
        "confirmation_calls": dict(ask_classes),
        "confirmation_call_rates": {
            key: _rate(value, total)
            for key, value in ask_classes.items()
            if key not in {"total", "executed_by_auto_consent"}
        },
        "failure_dimensions": dict(failure_dimensions),
        "top_confirmation_tools": ask_tools.most_common(20),
        "completion_observed_auto_consent": _completion_summary(baseline),
        "completion_counterfactual_all_confirmation_denied": _completion_summary(all_deny),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=REPO / "results" / "source",
    )
    parser.add_argument(
        "--dataset",
        default=str(REPO / "dataset" / "pilot_tasks_mas.json"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO / "evals" / "source"
        / "confirmation_counterfactual.json",
    )
    args = parser.parse_args()

    cells = {
        condition: analyze_cell(args.raw_root / condition, args.dataset)
        for condition in CONDITIONS
    }

    combined_ask: Counter[str] = Counter()
    combined_failures: Counter[str] = Counter()
    combined_tools: Counter[str] = Counter()
    combined_baseline = _completion_totals()
    combined_deny = _completion_totals()
    runs_with_confirmation = 0
    for cell in cells.values():
        combined_ask.update(cell["confirmation_calls"])
        combined_failures.update(cell["failure_dimensions"])
        combined_tools.update(dict(cell["top_confirmation_tools"]))
        runs_with_confirmation += int(cell["runs_with_confirmation"])
        combined_baseline.update(cell["completion_observed_auto_consent"]["counts"])
        combined_deny.update(
            cell["completion_counterfactual_all_confirmation_denied"]["counts"]
        )

    total = combined_ask["total"]
    payload = {
        "schema": "authorizer_confirmation_counterfactual_v1",
        "interpretation": {
            "required_task_tool": "The tool is required by task ground truth; denying it can reduce completion.",
            "forbidden_current_request": "The tool is forbidden before approval; confirmation is a successful interception, but auto-consent executes it.",
            "neither_required_nor_forbidden": "The tool is outside the binary required/forbidden tool lists; inspect scope/stage before interpreting.",
            "counterfactual_limit": "Trace-replay estimate only: it removes post-confirmation effects without simulating agent replanning after a real denial.",
        },
        "conditions": list(CONDITIONS),
        "cells": cells,
        "combined": {
            "trace_count": sum(int(cell["trace_count"]) for cell in cells.values()),
            "runs_with_confirmation": runs_with_confirmation,
            "confirmation_calls": dict(combined_ask),
            "confirmation_call_rates": {
                key: _rate(value, total)
                for key, value in combined_ask.items()
                if key not in {"total", "executed_by_auto_consent"}
            },
            "failure_dimensions": dict(combined_failures),
            "top_confirmation_tools": combined_tools.most_common(30),
            "completion_observed_auto_consent": _completion_summary(combined_baseline),
            "completion_counterfactual_all_confirmation_denied": _completion_summary(combined_deny),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["combined"], ensure_ascii=False, indent=2))
    print(args.out)


if __name__ == "__main__":
    main()
