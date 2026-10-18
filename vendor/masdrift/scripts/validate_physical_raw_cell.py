from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    args = parser.parse_args()

    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    expected = {str(task["task_id"]) for task in dataset["tasks"]}
    seen: set[str] = set()
    malformed: list[str] = []
    for path in sorted(args.path.glob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
            malformed.append(path.name)
            continue
        if isinstance(payload, dict):
            case_id = payload.get("case_id")
            if isinstance(case_id, str) and case_id in expected:
                seen.add(case_id)

    missing = sorted(expected - seen)
    result = {
        "unique_expected_cases": len(seen),
        "missing_cases": missing,
        "malformed_json": malformed,
        "complete": len(seen) == len(expected) and not missing and not malformed,
    }
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
