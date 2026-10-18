"""Dataset loader with visibility filter.

Returns one dict per case containing ONLY the fields agents should see.
Evaluator-only fields (authorization / ground_truth / key_constraint /
sensitive_content / workspace_state and per-tool action_type/recipient/effect)
are dropped. Within tool specs only name + description reach the agent.

Supports two on-disk formats, normalized into the same internal "case" shape:
  - single-file (legacy): {"cases": [...]}
  - task/environment split (pilot_tasks*.json + environments/<id>.json):
    {"tasks": [...], "generic_environment": "..."} — each task is merged with
    its environment (tools + workspace_state) and the shared generic environment.

Use full_case(...) when you need the unfiltered record for evaluator-side work.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_AGENT_META_KEEP = {"domain", "default_tool_rounds"}


def _public_tool(tool: dict[str, Any]) -> dict[str, Any]:
    # Only name + description reach the agent; action_type / object / recipient /
    # effect / tier / impact are evaluator/runtime metadata and must not leak.
    return {"name": tool["name"], "description": tool.get("description", "")}


def _public_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in meta.items() if k in _AGENT_META_KEEP}


def visible_view(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": case["case_id"],
        "project": case.get("project"),
        "task_type": case.get("task_type"),
        "user_query": case["user_query"],
        "recommended_agents": case.get("recommended_agents", []),
        "collaboration_context": case.get("collaboration_context", {}),
        "visible_artifacts": case.get("visible_artifacts", []),
        "available_tools": [_public_tool(t) for t in case.get("available_tools", [])],
        "metadata": _public_metadata(case.get("metadata", {})),
    }


def load_dataset(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def _build_case_from_task(
    task: dict[str, Any], tasks_path: Path, data: dict[str, Any]
) -> dict[str, Any]:
    """Merge a task with its environment (+ shared generic env) into one case."""
    base = tasks_path.parent
    env = load_dataset(base / task["environment"])
    tools = list(env.get("tools", []))
    generic = data.get("generic_environment")
    if generic:
        tools = tools + list(load_dataset(base / generic).get("tools", []))
    return {
        "case_id": task["task_id"],
        "user_query": task["user_query"],
        "available_tools": tools,                       # env tools + generic, carry effect (runtime-only)
        "workspace_state": env.get("workspace_state", {}),
        "ground_truth": task.get("ground_truth", {}),   # evaluator-only
        "key_constraint": task.get("key_constraint"),   # evaluator-only
        "sensitive_content": env.get("sensitive_content", []),  # evaluator-only
        "metadata": task.get("metadata", {}),
    }


def load_cases(path: str | Path, case_id: str | None = None) -> list[dict[str, Any]]:
    data = load_dataset(path)
    if "tasks" in data:  # task/environment split format
        p = Path(path)
        cases = [_build_case_from_task(t, p, data) for t in data["tasks"]]
    else:
        cases = data["cases"]
    if case_id is not None:
        match = [c for c in cases if c["case_id"] == case_id]
        if not match:
            available = "\n  ".join(c["case_id"] for c in cases)
            raise ValueError(
                f"case_id {case_id!r} not found in {path}\n"
                f"Available case_ids ({len(cases)}):\n  {available}"
            )
        cases = match
    return cases


def list_case_ids(path: str | Path) -> list[str]:
    return [c["case_id"] for c in load_cases(path)]


def get_visible_cases(
    path: str | Path, case_id: str | None = None
) -> list[dict[str, Any]]:
    return [visible_view(c) for c in load_cases(path, case_id=case_id)]


def full_case(path: str | Path, case_id: str) -> dict[str, Any]:
    return load_cases(path, case_id=case_id)[0]
