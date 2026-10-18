#!/usr/bin/env python3
"""Create a deterministic, domain-balanced subset of the MAS task dataset."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--per-domain", type=int, required=True)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    manifest = args.manifest.resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    groups: dict[str, list[dict]] = defaultdict(list)
    for task in payload["tasks"]:
        domain = str(task.get("metadata", {}).get("domain") or "").strip()
        if not domain:
            raise ValueError(f"task {task.get('task_id')} has no metadata.domain")
        groups[domain].append(task)

    rng = random.Random(args.seed)
    selected_ids: dict[str, list[str]] = {}
    chosen: set[str] = set()
    for domain in sorted(groups):
        rows = sorted(groups[domain], key=lambda row: row["task_id"])
        if len(rows) < args.per_domain:
            raise ValueError(f"domain {domain!r} has only {len(rows)} tasks")
        ids = sorted(row["task_id"] for row in rng.sample(rows, args.per_domain))
        selected_ids[domain] = ids
        chosen.update(ids)

    subset = dict(payload)
    subset["dataset_id"] = f"{payload.get('dataset_id', 'dataset')}_domain_balanced_{len(chosen)}"
    subset["notes"] = (
        f"Frozen domain-balanced sample: seed={args.seed}, "
        f"{args.per_domain} tasks from each of {len(groups)} metadata.domain groups."
    )
    subset["tasks"] = [task for task in payload["tasks"] if task["task_id"] in chosen]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(subset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest.write_text(json.dumps({
        "schema": "domain_balanced_sample_manifest_v1",
        "seed": args.seed,
        "per_domain": args.per_domain,
        "total": len(chosen),
        "source": str(source),
        "sample": str(output),
        "selected_task_ids": selected_ids,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
