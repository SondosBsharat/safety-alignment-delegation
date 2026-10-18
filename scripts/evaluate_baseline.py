#!/usr/bin/env python3
"""Recompute deterministic metrics from traces without LLM/API calls."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results/healthcare_baseline_20")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/healthcare_baseline_20_rule_eval.json")
    args = parser.parse_args()
    output = args.out.resolve()
    if output == ROOT / "results" or ROOT / "results" in output.parents:
        parser.error("Use outputs/ to preserve the published evaluation report.")
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        sys.executable, str(ROOT / "vendor/masdrift/evaluate.py"),
        "--dataset", str(ROOT / "vendor/masdrift/dataset/pilot_tasks_mas.json"),
        "--results-dir", str(args.results_dir.resolve()),
        "--rule-only", "--out", str(output),
    ], cwd=ROOT / "vendor/masdrift", check=True)


if __name__ == "__main__":
    main()
