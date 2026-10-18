#!/usr/bin/env python3
"""Verify source hashes, task coverage, and evaluation provenance."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads((ROOT / path).read_text())


def require(condition, message):
    if not condition:
        raise SystemExit(message)


def main():
    config = load("configs/healthcare_baseline_20.json")
    manifest = load("configs/upstream_files.sha256.json")
    for name, expected in manifest.items():
        path = ROOT / "vendor/masdrift" / name
        require(path.is_file(), f"Missing upstream file: {name}")
        require(hashlib.sha256(path.read_bytes()).hexdigest() == expected,
                f"Upstream file changed: {name}")
    ids = load(config["task_ids"])
    require(len(ids) == config["task_count"] == len(set(ids)), "Invalid task selection.")
    tasks = load(config["dataset"])["tasks"]
    require(set(ids) <= {t["task_id"] for t in tasks}, "Task absent from benchmark.")
    require(hashlib.sha256((ROOT / config["dataset"]).read_bytes()).hexdigest()
            == config["dataset_sha256"], "Dataset checksum differs.")
    traces = [json.loads(p.read_text()) for p in (ROOT / "results/healthcare_baseline_20").glob("*.json")]
    require(len(traces) == len(ids) and {t["case_id"] for t in traces} == set(ids),
            "Trace coverage differs from saved task list.")
    for trace in traces:
        for field in ["structure", "layers", "width", "personas", "visibility", "context_sharing"]:
            require(trace[field] == config[field], f"Trace setting differs: {field}")
        require(trace["worker_llm"]["model"] == config["model"], "Worker model differs.")
        require(trace["status"] == "success", "Unsuccessful trace in pilot.")
        require(not (trace.get("defense") or {}).get("components"), "Defended trace in baseline.")
    report = load("results/healthcare_baseline_20_eval.json")
    require(len(report["runs"]) == len(ids) and {r["case_id"] for r in report["runs"]} == set(ids),
            "Evaluation task coverage differs.")
    source_map = {"evaluate": "evaluate.py", "dataset": "dataset.py", "llm": "llm.py",
                  "mas_eval_simple_judge": "mas_eval/mas_eval/simple_judge.py"}
    for key, name in source_map.items():
        require(manifest[name] == report["checkpoint"]["fingerprint_payload"]["code_sha256"][key],
                f"Evaluator source differs from original report: {name}")
    print(f"Verified {len(manifest)} upstream files, {len(ids)} task IDs/traces, and evaluator provenance.")


if __name__ == "__main__":
    main()
