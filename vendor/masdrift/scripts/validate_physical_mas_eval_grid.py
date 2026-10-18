from __future__ import annotations

import argparse
import json
from pathlib import Path

from monitor_mas_eval_judge import run_llm_incomplete

CONDITIONS = ("S1", "C1", "C2", "C3", "C2-t", "N2", "N4", "N8", "N4-t")


def cell_inventory(path: Path) -> dict[str, object]:
    files = sorted(item for item in path.rglob("*") if item.is_file())
    total = sum(item.stat().st_size for item in files)
    return {"file_count": len(files), "total_bytes": total}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--views", nargs=2, required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    cells: list[dict[str, object]] = []
    all_complete = True
    for view in args.views:
        for condition in CONDITIONS:
            cell = args.root / view / condition
            output = cell / f"eval_{condition}_mas_eval_pro.json"
            reason = "complete"
            runs: list[object] = []
            skipped = 0
            errors = 0
            try:
                payload = json.loads(output.read_text(encoding="utf-8"))
                raw_runs = payload.get("runs")
                if not isinstance(raw_runs, list):
                    reason = "missing runs list"
                else:
                    runs = raw_runs
                    skipped = int(payload.get("skipped") or 0)
                    errors = sum(
                        1 for run in runs if not isinstance(run, dict) or run_llm_incomplete(run)
                    )
                    if len(runs) != 600:
                        reason = f"run count {len(runs)} != 600"
                    elif skipped:
                        reason = f"skipped={skipped}"
                    elif errors:
                        reason = f"invalid verdict rows={errors}"
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
            complete = reason == "complete"
            all_complete &= complete
            inventory = cell_inventory(cell) if cell.exists() else {
                "file_count": 0, "total_bytes": 0
            }
            cells.append({
                "view": view,
                "architecture": condition,
                "path": str(cell.resolve()) if cell.exists() else str(cell),
                "output": str(output),
                "runs": len(runs),
                "skipped": skipped,
                "invalid_verdict_rows": errors,
                "complete": complete,
                "reason": reason,
                **inventory,
            })

    result = {
        "schema": "physical_mas_eval_grid_v1",
        "root": str(args.root.resolve()),
        "views": args.views,
        "all_complete": all_complete,
        "complete_cells": sum(bool(cell["complete"]) for cell in cells),
        "cells": cells,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("all_complete", "complete_cells", "root")}))
    return 0 if all_complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
