"""Out-of-band trace recorder.

State lives in ContextVar — never inside LangGraph state, never in any prompt.
The agent system has no signal that recording is happening: tool calls and
inter-agent handoffs return through normal LangGraph channels, and the
recorder simply observes from the side.

Two traces are captured per run:
  - tool_calls : [{ts, agent, tool, args, result_preview}]
  - handoffs   : [{ts, from, to, payload}]
  - defense_events : optional control-plane events emitted by defense modules
"""
from __future__ import annotations

import contextvars
import json
import threading
import time
from contextlib import contextmanager
from typing import Any, Iterator


_recorder: contextvars.ContextVar["Recorder | None"] = contextvars.ContextVar(
    "mas_recorder", default=None
)
_current_agent: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "mas_current_agent", default=None
)
# Pipeline stage that the current LLM call belongs to (for token accounting). Default
# "worker" = ordinary agent reasoning; defense/eval code sets it around their own calls
# (authorizer:policy, authorizer:scope, chain:propose, chain:gate, judge).
_current_stage: contextvars.ContextVar[str] = contextvars.ContextVar(
    "mas_current_stage", default="worker"
)
_current_model_role: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "mas_current_model_role", default=None
)


class Recorder:
    def __init__(self, case_id: str, structure: str) -> None:
        self.case_id = case_id
        self.structure = structure
        self.started_at = time.time()
        self.tool_calls: list[dict[str, Any]] = []
        self.handoffs: list[dict[str, Any]] = []
        self.defense_events: list[dict[str, Any]] = []
        # Content-free audit rows for branch-local model projections.  These
        # deliberately record only counts and agent names, never prompt text.
        self.context_projections: list[dict[str, Any]] = []
        self.routing_controls: list[dict[str, Any]] = []
        self.model_bindings: list[dict[str, Any]] = []
        # Token accounting, aggregated per stage (compact — not one row per call):
        # stage -> {calls, input_tokens, output_tokens, total_tokens, estimated_calls}.
        self.token_stats: dict[str, dict[str, int]] = {}
        # Additive role-level accounting; token_stats["worker"] remains the
        # compatibility aggregate used by existing reports.
        self.agent_role_token_stats: dict[str, dict[str, int]] = {}
        self._lock = threading.Lock()

    def _ts(self) -> float:
        return round(time.time() - self.started_at, 4)

    def add_tool_call(
        self, *, agent: str, tool: str, args: dict[str, Any], result: Any,
        observed_effect: dict[str, Any] | None = None,
    ) -> None:
        try:
            preview = json.dumps(result, ensure_ascii=False, default=str)
        except Exception:
            preview = str(result)
        if len(preview) > 1500:
            preview = preview[:1500] + "...(truncated)"
        with self._lock:
            row = {
                    "ts": self._ts(),
                    "agent": agent,
                    "tool": tool,
                    "args": args,
                    "result_preview": preview,
                }
            if observed_effect is not None:
                row["observed_effect"] = observed_effect
            self.tool_calls.append(row)

    def add_handoff(
        self,
        *,
        from_agent: str,
        to_agent: str | None,
        payload: dict[str, Any],
    ) -> None:
        with self._lock:
            self.handoffs.append(
                {
                    "ts": self._ts(),
                    "from": from_agent,
                    "to": to_agent,
                    "payload": payload,
                }
            )

    def add_defense_event(
        self,
        *,
        component: str,
        event: str,
        payload: dict[str, Any],
    ) -> None:
        with self._lock:
            self.defense_events.append(
                {
                    "ts": self._ts(),
                    "component": component,
                    "event": event,
                    "payload": payload,
                }
            )

    def add_context_projection(
        self,
        *,
        agent: str,
        shared_message_count: int,
        model_message_count: int,
        excluded_message_count: int,
        incoming_transfer: bool,
        direct_reports: list[str],
        retained_tool_results: int,
    ) -> None:
        with self._lock:
            self.context_projections.append(
                {
                    "ts": self._ts(),
                    "agent": agent,
                    "shared_message_count": int(shared_message_count),
                    "model_message_count": int(model_message_count),
                    "excluded_message_count": int(excluded_message_count),
                    "incoming_transfer": bool(incoming_transfer),
                    "direct_reports": list(direct_reports),
                    "retained_tool_results": int(retained_tool_results),
                }
            )

    def add_routing_control(
        self,
        *,
        agent: str,
        event: str,
        payload: dict[str, Any],
    ) -> None:
        with self._lock:
            self.routing_controls.append(
                {
                    "ts": self._ts(),
                    "agent": agent,
                    "event": event,
                    "payload": dict(payload),
                }
            )

    def add_token_usage(
        self, *, stage: str, input_tokens: int, output_tokens: int,
        total_tokens: int, estimated: bool, model_role: str | None = None,
    ) -> None:
        with self._lock:
            s = self.token_stats.setdefault(
                stage,
                {"calls": 0, "input_tokens": 0, "output_tokens": 0,
                 "total_tokens": 0, "estimated_calls": 0},
            )
            s["calls"] += 1
            s["input_tokens"] += int(input_tokens or 0)
            s["output_tokens"] += int(output_tokens or 0)
            s["total_tokens"] += int(total_tokens or 0)
            if estimated:
                s["estimated_calls"] += 1
            if stage == "worker" and model_role:
                role_stats = self.agent_role_token_stats.setdefault(
                    model_role,
                    {"calls": 0, "input_tokens": 0, "output_tokens": 0,
                     "total_tokens": 0, "estimated_calls": 0},
                )
                role_stats["calls"] += 1
                role_stats["input_tokens"] += int(input_tokens or 0)
                role_stats["output_tokens"] += int(output_tokens or 0)
                role_stats["total_tokens"] += int(total_tokens or 0)
                if estimated:
                    role_stats["estimated_calls"] += 1

    def add_model_binding(
        self, *, agent: str, role: str, provider: str | None,
        model: str | None, base_url: str | None,
    ) -> None:
        row = {
            "agent": agent,
            "role": role,
            "provider": provider,
            "model": model,
            "base_url": base_url,
        }
        with self._lock:
            if row not in self.model_bindings:
                self.model_bindings.append(row)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "case_id": self.case_id,
            "structure": self.structure,
            "started_at": self.started_at,
            "duration_sec": round(time.time() - self.started_at, 3),
            "tool_calls": list(self.tool_calls),
            "handoffs": list(self.handoffs),
        }
        if self.defense_events:
            payload["defense_events"] = list(self.defense_events)
        if self.context_projections:
            payload["context_projections"] = list(self.context_projections)
        if self.routing_controls:
            payload["routing_controls"] = list(self.routing_controls)
        if self.model_bindings:
            payload["model_bindings"] = list(self.model_bindings)
        if self.token_stats:
            payload["token_stats"] = {k: dict(v) for k, v in self.token_stats.items()}
        if self.agent_role_token_stats:
            payload["agent_role_token_stats"] = {
                k: dict(v) for k, v in self.agent_role_token_stats.items()
            }
        return payload


@contextmanager
def use_recorder(recorder: Recorder) -> Iterator[Recorder]:
    tok = _recorder.set(recorder)
    try:
        yield recorder
    finally:
        _recorder.reset(tok)


@contextmanager
def current_agent(name: str) -> Iterator[str]:
    tok = _current_agent.set(name)
    try:
        yield name
    finally:
        _current_agent.reset(tok)


@contextmanager
def use_stage(stage: str) -> Iterator[str]:
    """Tag LLM calls made inside this block with `stage` (token accounting)."""
    tok = _current_stage.set(stage)
    try:
        yield stage
    finally:
        _current_stage.reset(tok)


@contextmanager
def use_model_role(role: str | None) -> Iterator[str | None]:
    """Attribute worker-stage LLM usage to a stable logical model role."""
    tok = _current_model_role.set(role)
    try:
        yield role
    finally:
        _current_model_role.reset(tok)


def current_stage() -> str:
    return _current_stage.get()


def record_tool_call(
    *, tool: str, args: dict[str, Any], result: Any, agent: str | None = None,
    observed_effect: dict[str, Any] | None = None,
) -> None:
    rec = _recorder.get()
    if rec is None:
        return
    rec.add_tool_call(
        agent=agent or _current_agent.get() or "unknown",
        tool=tool,
        args=args,
        result=result,
        observed_effect=observed_effect,
    )


def record_handoff(
    *, from_agent: str, to_agent: str | None, payload: dict[str, Any]
) -> None:
    rec = _recorder.get()
    if rec is None:
        return
    rec.add_handoff(from_agent=from_agent, to_agent=to_agent, payload=payload)


def record_defense_event(
    *, component: str, event: str, payload: dict[str, Any]
) -> None:
    rec = _recorder.get()
    if rec is None:
        return
    rec.add_defense_event(component=component, event=event, payload=payload)


def record_context_projection(
    *,
    agent: str,
    shared_message_count: int,
    model_message_count: int,
    excluded_message_count: int,
    incoming_transfer: bool,
    direct_reports: list[str],
    retained_tool_results: int,
) -> None:
    """Record content-free evidence that branch-local filtering was applied."""
    rec = _recorder.get()
    if rec is None:
        return
    rec.add_context_projection(
        agent=agent,
        shared_message_count=shared_message_count,
        model_message_count=model_message_count,
        excluded_message_count=excluded_message_count,
        incoming_transfer=incoming_transfer,
        direct_reports=direct_reports,
        retained_tool_results=retained_tool_results,
    )


def record_routing_control(
    *, agent: str, event: str, payload: dict[str, Any]
) -> None:
    """Record content-free deterministic routing enforcement metadata."""
    rec = _recorder.get()
    if rec is None:
        return
    rec.add_routing_control(agent=agent, event=event, payload=payload)


def record_token_usage(
    *, input_tokens: int, output_tokens: int, total_tokens: int,
    estimated: bool = False, stage: str | None = None,
) -> None:
    rec = _recorder.get()
    if rec is None:
        return
    rec.add_token_usage(
        stage=stage or _current_stage.get(),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        estimated=estimated,
        model_role=_current_model_role.get(),
    )


def record_model_binding(
    *, agent: str, role: str, provider: str | None,
    model: str | None, base_url: str | None,
) -> None:
    """Record secret-free evidence of the model client bound to one logical agent."""
    rec = _recorder.get()
    if rec is None:
        return
    rec.add_model_binding(
        agent=agent, role=role, provider=provider, model=model, base_url=base_url,
    )
