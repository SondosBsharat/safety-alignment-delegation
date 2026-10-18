#!/usr/bin/env python3
"""Summarize trace counts and worker token usage by model-transfer cell."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    grand = {"traces": 0, "calls": 0, "input_tokens": 0, "output_tokens": 0}
    for treatment in sorted(path for path in args.root.iterdir() if path.is_dir()):
        for condition in sorted(path for path in treatment.iterdir() if path.is_dir()):
            totals = {key: 0 for key in grand}
            for trace_path in condition.glob("*.json"):
                trace = json.loads(trace_path.read_text(encoding="utf-8"))
                worker = trace.get("token_stats", {}).get("worker", {})
                totals["traces"] += 1
                for key in ("calls", "input_tokens", "output_tokens"):
                    totals[key] += int(worker.get(key) or 0)
            for key, value in totals.items():
                grand[key] += value
            print(treatment.name, condition.name, json.dumps(totals, sort_keys=True))
    print("TOTAL", json.dumps(grand, sort_keys=True))


if __name__ == "__main__":
    main()
