#!/usr/bin/env python3
"""Audit whether a dataset can measure completion beyond tool invocation."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from dataset import load_cases  # noqa: E402
from evaluate import _step_spec_issues  # noqa: E402


def audit(dataset: str | Path) -> dict[str, Any]:
    cases = load_cases(str(dataset))
    counts = Counter()
    mismatch_counts = Counter()
    write_steps_without_content: list[dict[str, str]] = []
    mismatch_rows: list[dict[str, Any]] = []

    for case in cases:
        counts["tasks"] += 1
        steps = ((case.get("ground_truth") or {}).get("required_steps") or [])
        counts["tasks_with_required_steps"] += int(bool(steps))
        specs = {str(t.get("name")): t for t in case.get("available_tools", [])}
        for step in steps:
            counts["required_steps"] += 1
            counts["steps_with_target_refs"] += int(bool(step.get("target_refs")))
            counts["steps_with_expected_writes"] += int(bool(step.get("expected_writes")))
            counts["steps_with_must_include"] += int(bool(step.get("must_include")))
            counts["steps_with_must_exclude"] += int(bool(step.get("must_exclude")))
            tool = str(step.get("tool") or step.get("id") or "")
            spec = specs.get(tool)
            issues = _step_spec_issues(step, spec)
            for issue in issues:
                mismatch_counts[issue] += 1
            if issues:
                mismatch_rows.append({
                    "case_id": case["case_id"],
                    "step": str(step.get("id") or tool),
                    "issues": issues,
                })
            write_like = bool(step.get("expected_writes")) or step.get("action_type") == "draft"
            if write_like:
                counts["write_or_draft_steps"] += 1
                if not step.get("must_include") and not step.get("must_exclude"):
                    write_steps_without_content.append({
                        "case_id": case["case_id"],
                        "step": str(step.get("id") or tool),
                    })

    content_constrained = counts["steps_with_must_include"] + counts["steps_with_must_exclude"]
    return {
        "schema": "completion_measurement_audit_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(Path(dataset).resolve()),
        **dict(counts),
        "static_mismatch_counts": dict(sorted(mismatch_counts.items())),
        "static_mismatch_total": sum(mismatch_counts.values()),
        "static_mismatch_rows": mismatch_rows,
        "write_or_draft_steps_without_content_constraints": len(write_steps_without_content),
        "write_or_draft_steps_without_content_constraint_rows": write_steps_without_content,
        "content_sensitive_completion_meaningful": content_constrained > 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    payload = audit(args.dataset)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
