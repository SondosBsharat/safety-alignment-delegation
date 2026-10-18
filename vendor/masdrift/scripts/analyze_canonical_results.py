"""Paired statistical analysis for matched experiment results.

The script compares the no-defense and authorizer conditions by case_id, prints
JSON to stdout, and uses a fixed seed for reproducible paired bootstrap CIs.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any


CONDITIONS = ("C1", "C2", "C3")
METRICS = ("attempt", "executed", "c2a", "completion")


def load_runs(directory: Path, condition: str) -> dict[str, dict[str, Any]]:
    path = directory / condition / f"eval_{condition}_rule.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {run["case_id"]: run for run in payload["runs"]}


def binary_metric(run: dict[str, Any], metric: str) -> bool:
    rule = run["rule"]
    if metric == "attempt":
        return bool(rule["c1_unauthorized_action"]["violation"])
    if metric == "executed":
        calls = rule["c1_unauthorized_action"].get("calls", [])
        return any(bool(call.get("executed")) for call in calls)
    if metric == "c2a":
        return bool(rule["c2a_over_disclosure_proxy"]["signal"])
    if metric == "completion":
        return bool(rule["completion"]["completed"])
    raise ValueError(f"Unknown metric: {metric}")


def step_coverage(run: dict[str, Any]) -> float:
    completion = run["rule"]["completion"]
    total = completion["required_steps_total"]
    return completion["required_steps_completed"] / total if total else 1.0


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total + z * z / (4 * total * total)
        )
        / denominator
    )
    return center - half_width, center + half_width


def exact_mcnemar(discordant_a: int, discordant_b: int) -> float:
    total = discordant_a + discordant_b
    if total == 0:
        return 1.0
    tail = sum(math.comb(total, k) for k in range(min(discordant_a, discordant_b) + 1))
    return min(1.0, 2 * tail / (2**total))


def bootstrap_mean_ci(
    values: list[float], seed: int, replicates: int
) -> tuple[float, float]:
    rng = random.Random(seed)
    size = len(values)
    estimates = [
        sum(values[rng.randrange(size)] for _ in range(size)) / size
        for _ in range(replicates)
    ]
    estimates.sort()
    return estimates[int(0.025 * replicates)], estimates[int(0.975 * replicates)]


def analyze(root: Path, replicates: int) -> dict[str, Any]:
    baseline_root = root / "evals" / "raw"
    defense_root = root / "evals" / "source"
    output: dict[str, Any] = {}

    for condition_index, condition in enumerate(CONDITIONS):
        baseline = load_runs(baseline_root, condition)
        defense = load_runs(defense_root, condition)
        paired_ids = sorted(baseline.keys() & defense.keys())
        condition_output: dict[str, Any] = {
            "n_baseline": len(baseline),
            "n_defense": len(defense),
            "n_paired": len(paired_ids),
            "metrics": {},
        }

        for metric_index, metric in enumerate(METRICS):
            baseline_values = [binary_metric(baseline[case_id], metric) for case_id in paired_ids]
            defense_values = [binary_metric(defense[case_id], metric) for case_id in paired_ids]
            baseline_count = sum(baseline_values)
            defense_count = sum(defense_values)
            baseline_yes_defense_no = sum(
                before and not after
                for before, after in zip(baseline_values, defense_values, strict=True)
            )
            baseline_no_defense_yes = sum(
                not before and after
                for before, after in zip(baseline_values, defense_values, strict=True)
            )
            differences = [
                int(after) - int(before)
                for before, after in zip(baseline_values, defense_values, strict=True)
            ]
            ci_low, ci_high = bootstrap_mean_ci(
                differences,
                1729 + condition_index * 100 + metric_index,
                replicates,
            )
            baseline_ci = wilson_interval(baseline_count, len(paired_ids))
            defense_ci = wilson_interval(defense_count, len(paired_ids))
            condition_output["metrics"][metric] = {
                "baseline_count": baseline_count,
                "defense_count": defense_count,
                "baseline_percent": 100 * baseline_count / len(paired_ids),
                "defense_percent": 100 * defense_count / len(paired_ids),
                "baseline_wilson_95ci_percent": [100 * value for value in baseline_ci],
                "defense_wilson_95ci_percent": [100 * value for value in defense_ci],
                "paired_difference_pp": 100 * sum(differences) / len(paired_ids),
                "paired_bootstrap_95ci_pp": [100 * ci_low, 100 * ci_high],
                "baseline_yes_defense_no": baseline_yes_defense_no,
                "baseline_no_defense_yes": baseline_no_defense_yes,
                "mcnemar_exact_p": exact_mcnemar(
                    baseline_yes_defense_no, baseline_no_defense_yes
                ),
            }

        baseline_coverage = [step_coverage(baseline[case_id]) for case_id in paired_ids]
        defense_coverage = [step_coverage(defense[case_id]) for case_id in paired_ids]
        coverage_differences = [
            after - before
            for before, after in zip(baseline_coverage, defense_coverage, strict=True)
        ]
        ci_low, ci_high = bootstrap_mean_ci(
            coverage_differences,
            1729 + condition_index * 100 + 99,
            replicates,
        )
        condition_output["step_coverage"] = {
            "baseline_percent": 100 * sum(baseline_coverage) / len(paired_ids),
            "defense_percent": 100 * sum(defense_coverage) / len(paired_ids),
            "paired_difference_pp": 100 * sum(coverage_differences) / len(paired_ids),
            "paired_bootstrap_95ci_pp": [100 * ci_low, 100 * ci_high],
        }
        condition_output["baseline_only_case_ids"] = sorted(baseline.keys() - defense.keys())
        condition_output["defense_only_case_ids"] = sorted(defense.keys() - baseline.keys())
        output[condition] = condition_output

    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--bootstrap-replicates", type=int, default=5_000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = analyze(args.root.resolve(), args.bootstrap_replicates)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
