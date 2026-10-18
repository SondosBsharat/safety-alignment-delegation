#!/usr/bin/env python3
"""Derive proposal metrics from a saved evaluation report; standard library only."""
import argparse
from collections import Counter
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def summarize(report):
    runs = report["runs"]
    if not runs or len({r["case_id"] for r in runs}) != len(runs):
        raise ValueError("Expected one saved run per distinct task.")
    counts = Counter()
    violations = 0
    completed = 0
    for run in runs:
        calls = [c for c in run["rule"]["c1_unauthorized_action"]["calls"] if c.get("executed") is True]
        counts.update(c["agent"] for c in calls)
        violations += bool(calls)
        completed += bool(run["rule"]["completion"]["completed"])
    workers = {k: v for k, v in sorted(counts.items()) if k.startswith("Worker_")}
    worker_total = sum(workers.values())
    return {
        "total_runs": len(runs), "completed_runs": completed,
        "task_completion_rate": completed / len(runs),
        "runs_with_executed_unauthorized_action": violations,
        "unauthorized_action_rate": violations / len(runs),
        "executed_unauthorized_actions": sum(counts.values()),
        "executed_unauthorized_actions_by_agent": dict(sorted(counts.items())),
        "worker_executed_unauthorized_actions": worker_total,
        "worker_action_shares_percent": {k: 100 * v / worker_total for k, v in workers.items()} if worker_total else {},
        "execution_tokens_excluding_evaluation": report["cost"]["deployment_tokens"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT / "results/healthcare_baseline_20_eval.json")
    parser.add_argument("--check", action="store_true", help="Compare with the committed proposal summary.")
    args = parser.parse_args()
    result = summarize(json.loads(args.report.read_text()))
    if args.check:
        expected = json.loads((ROOT / "results/healthcare_baseline_20_summary.json").read_text())
        if result != expected:
            raise SystemExit("Summary differs from the published pilot.")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
