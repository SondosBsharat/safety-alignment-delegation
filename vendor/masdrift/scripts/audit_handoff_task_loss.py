#!/usr/bin/env python3
"""Audit whether branch-local transfers lose the delegated task text."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any


GENERIC_FINAL = re.compile(
    r"^\s*(?:how can i help|what can i help|please (?:provide|share)|"
    r"could you (?:provide|share)|i(?:'m| am) ready to help)",
    re.IGNORECASE,
)
ASK_TOOL = re.compile(r"(?:ask|confirm|clarif|approval)", re.IGNORECASE)


def text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "\n".join(
            str(part.get("text") or "")
            for part in value
            if isinstance(part, dict) and part.get("type") == "text"
        ).strip()
    return str(value or "").strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--eval-summary", type=Path)
    args = parser.parse_args()

    traces: list[dict[str, Any]] = []
    traces_by_file: dict[str, dict[str, Any]] = {}
    invalid = 0
    for path in sorted(args.directory.glob("*.json")):
        try:
            trace = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            invalid += 1
            continue
        if trace.get("status") == "success":
            traces.append(trace)
            traces_by_file[path.name] = trace

    transfer_count = 0
    empty_transfer_count = 0
    empty_transfer_cases = 0
    no_tools_after_first_transfer = 0
    generic_final_cases = 0
    ask_tool_calls = 0
    ask_tool_names: Counter[str] = Counter()
    auto_boundary_hops = 0
    message_lengths: list[int] = []
    total_tool_calls = 0

    for trace in traces:
        transfers = [h for h in trace.get("handoffs", []) if h.get("to") is not None]
        empty_in_case = False
        for hop in transfers:
            transfer_count += 1
            payload = hop.get("payload") or {}
            message = text(payload.get("message"))
            message_lengths.append(len(message))
            if not message:
                empty_transfer_count += 1
                empty_in_case = True
            context = text(payload.get("branch_context"))
            if (
                "USER AUTHORIZATION BOUNDARY" in context
                or "CURRENT AUTHORIZATION STATE" in context
            ):
                auto_boundary_hops += 1
        empty_transfer_cases += int(empty_in_case)

        calls = trace.get("tool_calls", [])
        total_tool_calls += len(calls)
        for call in calls:
            tool_name = str(call.get("tool") or "")
            if ASK_TOOL.search(tool_name):
                ask_tool_calls += 1
                ask_tool_names[tool_name] += 1
        if transfers:
            first_ts = float(transfers[0].get("ts") or 0)
            if not any(float(call.get("ts") or 0) > first_ts for call in calls):
                no_tools_after_first_transfer += 1
        generic_final_cases += int(bool(GENERIC_FINAL.search(text(trace.get("final_message")))))

    n = len(traces)
    result = {
        "directory": str(args.directory),
        "valid_success_traces": n,
        "invalid_files": invalid,
        "transfer_hops": transfer_count,
        "empty_transfer_hops": empty_transfer_count,
        "empty_transfer_hop_rate": empty_transfer_count / transfer_count if transfer_count else 0,
        "cases_with_empty_transfer": empty_transfer_cases,
        "case_empty_transfer_rate": empty_transfer_cases / n if n else 0,
        "median_transfer_message_chars": median(message_lengths) if message_lengths else 0,
        "cases_without_tools_after_first_transfer": no_tools_after_first_transfer,
        "no_post_transfer_tool_rate": no_tools_after_first_transfer / n if n else 0,
        "generic_final_cases": generic_final_cases,
        "generic_final_rate": generic_final_cases / n if n else 0,
        "ask_like_tool_calls": ask_tool_calls,
        "ask_like_tool_names": dict(ask_tool_names.most_common()),
        "mean_tool_calls": total_tool_calls / n if n else 0,
        "auto_boundary_hops": auto_boundary_hops,
    }
    if args.eval_summary:
        summary = json.loads(args.eval_summary.read_text(encoding="utf-8"))
        rows = []
        for run in summary.get("runs", []):
            trace = traces_by_file.get(str(run.get("result_file") or ""))
            if trace is None:
                continue
            transfers = [h for h in trace.get("handoffs", []) if h.get("to") is not None]
            calls = trace.get("tool_calls", [])
            first_ts = float(transfers[0].get("ts") or 0) if transfers else None
            post_tools = bool(
                first_ts is not None
                and any(float(call.get("ts") or 0) > first_ts for call in calls)
            )
            rows.append({
                "complete": bool(((run.get("rule") or {}).get("completion") or {}).get("complete")),
                "transferred": bool(transfers),
                "post_tools": post_tools,
                "ask_user": any(str(call.get("tool") or "") == "ask_user" for call in calls),
            })

        def grouped(field: str) -> dict[str, Any]:
            output: dict[str, Any] = {}
            for value in (False, True):
                selected = [row for row in rows if row[field] is value]
                completed = sum(row["complete"] for row in selected)
                output[str(value).lower()] = {
                    "runs": len(selected),
                    "completed": completed,
                    "completion_rate": completed / len(selected) if selected else None,
                }
            return output

        result["completion_slices"] = {
            "has_transfer": grouped("transferred"),
            "has_post_transfer_tools": grouped("post_tools"),
            "called_ask_user": grouped("ask_user"),
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
