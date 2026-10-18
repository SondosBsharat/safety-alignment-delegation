#!/usr/bin/env python3
"""Freeze an exact-size, near-equal sample across metadata.domain groups."""
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
    parser.add_argument(
        "--include",
        type=Path,
        help="Optional frozen subset whose task IDs must all be retained.",
    )
    parser.add_argument("--total", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    manifest = args.manifest.resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    tasks = payload.get("tasks", [])
    if not 0 < args.total <= len(tasks):
        raise ValueError(f"--total must be in 1..{len(tasks)}")

    groups: dict[str, list[dict]] = defaultdict(list)
    for task in tasks:
        domain = str((task.get("metadata") or {}).get("domain") or "").strip()
        if not domain:
            raise ValueError(f"task {task.get('task_id')} has no metadata.domain")
        groups[domain].append(task)

    domains = sorted(groups)
    base, remainder = divmod(args.total, len(domains))
    rng = random.Random(args.seed)
    # Rotate the one-item remainders reproducibly instead of privileging alphabetic
    # domain order.  Every domain differs by at most one selected task.
    extra_domains = set(rng.sample(domains, remainder))
    allocations = {d: base + int(d in extra_domains) for d in domains}
    if any(allocations[d] > len(groups[d]) for d in domains):
        raise ValueError("a domain is too small for the equal allocation")

    source_by_id = {str(task["task_id"]): task for task in tasks}
    included_ids: set[str] = set()
    included_by_domain: dict[str, set[str]] = defaultdict(set)
    include = args.include.resolve() if args.include else None
    if include is not None:
        include_payload = json.loads(include.read_text(encoding="utf-8"))
        include_tasks = include_payload.get("tasks", [])
        for task in include_tasks:
            task_id = str(task["task_id"])
            source_task = source_by_id.get(task_id)
            if source_task is None:
                raise ValueError(f"included task {task_id} is absent from --source")
            domain = str((source_task.get("metadata") or {}).get("domain") or "").strip()
            included_ids.add(task_id)
            included_by_domain[domain].add(task_id)
        if len(included_ids) != len(include_tasks):
            raise ValueError("--include contains duplicate task IDs")
        if len(included_ids) > args.total:
            raise ValueError("--include contains more tasks than --total")
        for domain in domains:
            if len(included_by_domain[domain]) > allocations[domain]:
                raise ValueError(
                    f"--include has {len(included_by_domain[domain])} {domain} tasks, "
                    f"above the target allocation {allocations[domain]}"
                )

    selected_ids: dict[str, list[str]] = {}
    chosen: set[str] = set(included_ids)
    for domain in domains:
        rows = sorted(groups[domain], key=lambda row: row["task_id"])
        remaining = [row for row in rows if str(row["task_id"]) not in included_ids]
        needed = allocations[domain] - len(included_by_domain[domain])
        picked = rng.sample(remaining, needed)
        ids = sorted(included_by_domain[domain] | {str(row["task_id"]) for row in picked})
        selected_ids[domain] = ids
        chosen.update(ids)

    subset = dict(payload)
    subset["dataset_id"] = f"{payload.get('dataset_id', 'dataset')}_uniform_domain_{args.total}"
    subset["notes"] = (
        f"Frozen near-equal domain sample: seed={args.seed}, total={args.total}, "
        f"allocations={allocations}, retained={len(included_ids)}. "
        "Original task order retained."
    )
    subset["tasks"] = [task for task in tasks if task["task_id"] in chosen]
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(subset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest.write_text(json.dumps({
        "schema": "uniform_domain_sample_manifest_v1",
        "seed": args.seed,
        "total": args.total,
        "allocations": allocations,
        "source": str(source),
        "sample": str(output),
        "include": str(include) if include else None,
        "included_task_count": len(included_ids),
        "selected_task_ids": selected_ids,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
