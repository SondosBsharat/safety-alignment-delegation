"""Paper-aligned authorization-carrying delegation (Chain v2).

Every handoff carries a complete, total Tool-Scope-Stage policy.  Child policies
are constructed by code from immutable inherited clauses plus valid additive
restrictions; an LLM can propose attenuation but cannot replace or remove authority
constraints.  This module never stores or receives the original user request.
"""
from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Iterable

from authorization_policy import (
    AUTHORIZATION_LINEAGE_SCHEMA,
    AuthorizationAssessment,
    AuthorizationPolicy,
    AuthorizationRule,
    AuthorizationStageState,
    ScopeClause,
    make_rule,
    max_stage,
    monitor_authorization_call,
    normalize_stage,
    total_invalid_fallback,
)
from tracing import record_defense_event, use_stage


@dataclass(frozen=True)
class PolicyLineageEnvelope:
    lineage_id: str
    edge_id: str
    activation_id: str
    parent_policy_id: str | None
    policy: AuthorizationPolicy
    schema: str = AUTHORIZATION_LINEAGE_SCHEMA

    @property
    def policy_id(self) -> str:
        return self.policy.policy_id

    def to_dict(self, *, agent_visible: bool = False) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "lineage_id": self.lineage_id,
            "edge_id": self.edge_id,
            "activation_id": self.activation_id,
            "parent_policy_id": self.parent_policy_id,
            "policy_id": self.policy_id,
            "policy": self.policy.to_dict(agent_visible=agent_visible),
        }


@dataclass
class ProposalDiagnostics:
    valid: bool = True
    parse_error: str | None = None
    accepted_clauses: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    rejected_widening: list[dict[str, Any]] = field(default_factory=list)
    stage_clamps: list[dict[str, Any]] = field(default_factory=list)
    fallback_tools: dict[str, str] = field(default_factory=dict)
    unknown_tools: list[str] = field(default_factory=list)
    duplicate_clauses_removed: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_valid": self.valid,
            "parse_error": self.parse_error,
            "accepted_additive_clauses": self.accepted_clauses,
            "rejected_widening": self.rejected_widening,
            "stage_clamps": self.stage_clamps,
            "fallback_rules": self.fallback_tools,
            "unknown_tools": self.unknown_tools,
            "duplicate_clauses_removed": self.duplicate_clauses_removed,
        }


_PROPOSE_SYSTEM = (
    "You attenuate a complete downstream authorization policy for one delegated "
    "subtask. You receive only the inherited policy, delegated subtask, downstream "
    "public role/name, and public tools. Return exactly one entry for every inherited "
    "tool and no unknown tools. Each entry must explicitly inherit the parent rule and "
    "may only ADD conjunctive restrictions. Apply least privilege to the delegated "
    "subtask: set unrelated tools to never, preserve a usable stage only for tools "
    "needed by that subtask, and add exact object/content/recipient/operation limits "
    "when the subtask supplies them. Do not repeat an inherited clause as a new clause; "
    "that is not semantic attenuation. It may propose the same or a more restrictive "
    "stage; current_request < post_approval < never. Never replace, delete, negate, or "
    "relax inherited clauses. Return JSON only: "
    "{\"schema\":\"chain_attenuation_proposal_v2\",\"rules\":["
    "{\"tool\":\"...\",\"inherit\":true,\"additional_scope_clauses\":["
    "{\"dimension\":\"general|object|content|recipient|operation\",\"text\":\"...\"}],"
    "\"proposed_stage\":\"current_request|post_approval|never\"}]}"
)


class ChainState:
    """Per-run immutable policy contexts keyed by activation and lineage.

    An agent name is only a routing hint to resolve a pending activation.  Policies
    themselves live in ``_contexts`` and are never overwritten when a named agent is
    revisited.
    """

    def __init__(
        self,
        root_policy: AuthorizationPolicy,
        public_tools: list[dict[str, str]],
        *,
        enforced: bool,
    ) -> None:
        self.root = root_policy
        self.public_tools = [
            {"name": str(t.get("name", "")), "description": str(t.get("description", ""))}
            for t in public_tools
        ]
        self.enforced = enforced
        self._lock = threading.RLock()
        self._counter = 0
        root_lineage = self._new_id("lineage")
        self.root_envelope = PolicyLineageEnvelope(
            lineage_id=root_lineage,
            edge_id=self._new_id("edge_root"),
            activation_id=self._new_id("activation_root"),
            parent_policy_id=None,
            policy=root_policy,
        )
        self._contexts: dict[str, PolicyLineageEnvelope] = {
            self.root_envelope.activation_id: self.root_envelope
        }
        self._active_by_agent: dict[str, str] = {}
        self._pending_by_agent: dict[str, list[str]] = {}
        self._last_handoff_by_agent: dict[str, str] = {}
        self._known_targets: set[str] = set()
        self._joined_by_parents: dict[tuple[str, tuple[str, ...]], str] = {}

        record_defense_event(
            component="chain",
            event="root_policy_initialized",
            payload={
                "trace_schema": "chain_decision_trace_v2",
                "root_coverage": list(root_policy.tool_names),
                "envelope": self.root_envelope.to_dict(),
            },
        )

    def _new_id(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}_{self._counter:06d}"

    def envelope_for_activation(self, activation_id: str) -> PolicyLineageEnvelope:
        with self._lock:
            return self._contexts[activation_id]

    def active_envelope(self, agent: str | None) -> PolicyLineageEnvelope:
        """Bind a call to one activation or construct a conservative ambiguous fallback."""
        if not agent:
            return self.root_envelope
        with self._lock:
            pending = list(self._pending_by_agent.get(agent, ()))
            if len(pending) == 1:
                activation_id = pending[0]
                self._active_by_agent[agent] = activation_id
                self._pending_by_agent[agent] = []
                return self._contexts[activation_id]
            if len(pending) > 1:
                parents = [self._contexts[activation_id] for activation_id in pending]
                fallback = total_invalid_fallback(
                    _meet_policies([parent.policy for parent in parents]),
                    reason="ambiguous active authorization lineage",
                    parent_policy_ids=[parent.policy_id for parent in parents],
                )
                envelope = PolicyLineageEnvelope(
                    lineage_id=self._new_id("lineage_ambiguous"),
                    edge_id=self._new_id("edge_ambiguous"),
                    activation_id=self._new_id("activation_ambiguous"),
                    parent_policy_id=None,
                    policy=fallback,
                )
                self._contexts[envelope.activation_id] = envelope
                self._active_by_agent[agent] = envelope.activation_id
                self._pending_by_agent[agent] = []
                record_defense_event(
                    component="chain",
                    event="ambiguous_lineage_fallback",
                    payload={
                        "agent": agent,
                        "parent_activation_ids": pending,
                        "envelope": envelope.to_dict(),
                    },
                )
                return envelope
            active_id = self._active_by_agent.get(agent)
            if active_id is not None:
                return self._contexts[active_id]
            if agent in self._known_targets:
                prior_activation_id = self._last_handoff_by_agent.get(agent)
                prior = (
                    self._contexts[prior_activation_id]
                    if prior_activation_id in self._contexts
                    else self.root_envelope
                )
                fallback = _total_denied_fallback(
                    prior.policy,
                    reason="missing active authorization lineage",
                )
                envelope = PolicyLineageEnvelope(
                    lineage_id=self._new_id("lineage_missing"),
                    edge_id=self._new_id("edge_missing"),
                    activation_id=self._new_id("activation_missing"),
                    parent_policy_id=prior.policy_id,
                    policy=fallback,
                )
                self._contexts[envelope.activation_id] = envelope
                self._active_by_agent[agent] = envelope.activation_id
                record_defense_event(
                    component="chain",
                    event="missing_lineage_fallback",
                    payload={"agent": agent, "envelope": envelope.to_dict()},
                )
                return envelope
            # Entry/root agents have no incoming handoff. Missing non-root bindings are
            # observable only through explicit test/integration seams; root is the safe
            # initial execution context, not an opportunistic union.
            self._active_by_agent[agent] = self.root_envelope.activation_id
            return self.root_envelope

    def on_handoff(
        self, *, from_agent: str, to_agent: str, message: str
    ) -> PolicyLineageEnvelope:
        parent = self.active_envelope(from_agent)
        raw = _propose_downstream_policy(
            parent.policy,
            delegated_subtask=(message or "")[:1500],
            downstream_agent=to_agent,
            public_tools=self.public_tools,
        )
        child, diagnostics = attenuate_policy(parent.policy, raw)
        with self._lock:
            edge_id = self._new_id("edge")
            activation_id = self._new_id("activation")
            envelope = PolicyLineageEnvelope(
                lineage_id=parent.lineage_id,
                edge_id=edge_id,
                activation_id=activation_id,
                parent_policy_id=parent.policy_id,
                policy=child,
            )
            self._contexts[activation_id] = envelope
            self._pending_by_agent.setdefault(to_agent, []).append(activation_id)
            self._last_handoff_by_agent[to_agent] = activation_id
            self._known_targets.add(to_agent)
        record_defense_event(
            component="chain",
            event="policy_attenuated",
            payload={
                "trace_schema": "chain_decision_trace_v2",
                "from": from_agent,
                "to": to_agent,
                "parent_policy": parent.policy.to_dict(),
                "child_policy": child.to_dict(),
                "edge_id": edge_id,
                "lineage_id": envelope.lineage_id,
                "activation_id": activation_id,
                **diagnostics.to_dict(),
            },
        )
        return envelope

    def join_active_agents(
        self, *, agent: str, parent_agents: Iterable[str]
    ) -> PolicyLineageEnvelope:
        """Meet the active activation of every direct report exactly once.

        Centralized integrators call this before their first model turn.  The key is
        based on concrete activation IDs rather than reusable agent names, so a later
        activation cannot accidentally reuse an earlier join.
        """
        parent_names = tuple(parent_agents)
        if not parent_names:
            raise ValueError("authorization join requires direct-report agents")
        parents = [self.active_envelope(name) for name in parent_names]
        ids = tuple(parent.activation_id for parent in parents)
        key = (agent, ids)
        with self._lock:
            existing_id = self._joined_by_parents.get(key)
            if existing_id is not None:
                return self._contexts[existing_id]
        envelope = self.explicit_join(agent=agent, activation_ids=ids)
        with self._lock:
            self._joined_by_parents[key] = envelope.activation_id
        return envelope

    def latest_handoff_for(self, agent: str) -> PolicyLineageEnvelope | None:
        with self._lock:
            activation_id = self._last_handoff_by_agent.get(agent)
            return self._contexts.get(activation_id) if activation_id else None

    def explicit_join(
        self, *, agent: str, activation_ids: Iterable[str]
    ) -> PolicyLineageEnvelope:
        ids = tuple(activation_ids)
        if not ids:
            raise ValueError("explicit authorization join requires at least one parent")
        with self._lock:
            parents = [self._contexts[activation_id] for activation_id in ids]
            joined = _meet_policies([parent.policy for parent in parents])
            envelope = PolicyLineageEnvelope(
                lineage_id=self._new_id("lineage_join"),
                edge_id=self._new_id("edge_join"),
                activation_id=self._new_id("activation_join"),
                parent_policy_id=None,
                policy=joined,
            )
            self._contexts[envelope.activation_id] = envelope
            self._pending_by_agent[agent] = [envelope.activation_id]
            self._known_targets.add(agent)
        record_defense_event(
            component="chain",
            event="policy_joined",
            payload={
                "agent": agent,
                "parent_activation_ids": list(ids),
                "parent_policy_ids": [parent.policy_id for parent in parents],
                "joined_envelope": envelope.to_dict(),
            },
        )
        return envelope

    def release_activation(self, agent: str) -> None:
        """Narrow integration seam for an explicit branch-context teardown."""
        with self._lock:
            self._active_by_agent.pop(agent, None)
            self._pending_by_agent[agent] = []

    def assess(
        self,
        agent: str | None,
        tool_spec: dict[str, Any],
        args: dict[str, Any],
        stage_state: AuthorizationStageState,
        scope_matcher,
    ) -> tuple[AuthorizationAssessment, PolicyLineageEnvelope]:
        envelope = self.active_envelope(agent)
        # Source and Chain literally share P_0.  Root approval is therefore bound to
        # the same rule/scope provenance (lineage None); descendants remain lineage-local.
        approval_lineage = None if envelope.policy_id == self.root.policy_id else envelope.lineage_id
        assessment = monitor_authorization_call(
            envelope.policy,
            stage_state,
            tool_spec,
            args,
            scope_matcher,
            lineage_id=approval_lineage,
        )
        return assessment, envelope

    def find_rule(self, rule_id: str) -> tuple[AuthorizationRule, str | None] | None:
        with self._lock:
            for envelope in self._contexts.values():
                for rule in envelope.policy.rules.values():
                    if rule.rule_id == rule_id:
                        return rule, envelope.lineage_id
        return None


def _fallback_rule(parent: AuthorizationRule, reason: str) -> AuthorizationRule:
    return make_rule(
        tool=parent.tool,
        scope=parent.scope,
        scope_clauses=parent.scope_clauses,
        stage=max_stage(parent.stage, "post_approval"),
        reason=reason,
        valid=False,
        parent_rule_ids=(parent.rule_id,),
        id_salt="proposal_fallback",
    )


def _total_denied_fallback(
    parent: AuthorizationPolicy, *, reason: str
) -> AuthorizationPolicy:
    """Fail closed without re-anchoring a lost descendant to the root policy."""
    rules = {
        name: make_rule(
            tool=name,
            scope=inherited.scope,
            scope_clauses=inherited.scope_clauses,
            stage="never",
            reason=reason,
            valid=False,
            parent_rule_ids=(inherited.rule_id,),
            id_salt="missing_lineage_deny",
        )
        for name, inherited in parent.rules.items()
    }
    return AuthorizationPolicy(rules=rules, parent_policy_ids=(parent.policy_id,))


def _dedupe_clauses(
    clauses: Iterable[ScopeClause],
) -> tuple[tuple[ScopeClause, ...], tuple[ScopeClause, ...]]:
    """Remove semantically identical conjunctions, ignoring provenance labels."""
    kept: list[ScopeClause] = []
    removed: list[ScopeClause] = []
    seen: set[tuple[str, str]] = set()
    for clause in clauses:
        key = (clause.dimension.strip().lower(), " ".join(clause.text.lower().split()))
        if key in seen:
            removed.append(clause)
            continue
        seen.add(key)
        kept.append(clause)
    return tuple(kept), tuple(removed)


def _parse_clause(value: Any, *, source: str) -> ScopeClause:
    if isinstance(value, str):
        return ScopeClause(text=value, source=source)
    if not isinstance(value, dict):
        raise ValueError("scope clause must be a string or object")
    allowed = {"text", "dimension"}
    if set(value) - allowed:
        raise ValueError("scope clause contains unsupported operations")
    return ScopeClause(
        text=str(value.get("text", "")),
        dimension=str(value.get("dimension", "general")),
        source=source,
    )


def attenuate_policy(
    parent: AuthorizationPolicy, raw_proposal: str | dict[str, Any] | None
) -> tuple[AuthorizationPolicy, ProposalDiagnostics]:
    """Code-enforced structural attenuation with total conservative fallbacks."""
    diagnostics = ProposalDiagnostics()
    try:
        obj = raw_proposal if isinstance(raw_proposal, dict) else _loads_json_obj(str(raw_proposal or ""))
        rows = obj.get("rules") if isinstance(obj, dict) else None
        if not isinstance(rows, list):
            raise ValueError("proposal must contain a rules list")
    except Exception as exc:
        diagnostics.valid = False
        diagnostics.parse_error = f"{type(exc).__name__}: {exc}"
        for name in parent.rules:
            diagnostics.fallback_tools[name] = "unparseable proposal"
        return total_invalid_fallback(parent, reason="unparseable Chain proposal"), diagnostics

    by_tool: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if not isinstance(row, dict):
            diagnostics.valid = False
            continue
        name = str(row.get("tool", "")).strip()
        if name not in parent.rules:
            diagnostics.valid = False
            diagnostics.unknown_tools.append(name)
            continue
        by_tool.setdefault(name, []).append(row)

    child_rules: dict[str, AuthorizationRule] = {}
    for name, inherited in parent.rules.items():
        entries = by_tool.get(name, [])
        if len(entries) != 1:
            why = "missing proposal entry" if not entries else "duplicate proposal entries"
            diagnostics.valid = False
            diagnostics.fallback_tools[name] = why
            child_rules[name] = _fallback_rule(inherited, why)
            continue
        row = entries[0]
        try:
            if row.get("inherit") is not True:
                raise ValueError("entry must explicitly inherit parent rule")
            widening_keys = sorted(
                set(row) & {"scope", "replacement_scope", "remove_clauses", "allowed_kinds", "external_allowed"}
            )
            if widening_keys:
                diagnostics.rejected_widening.append({"tool": name, "keys": widening_keys})
            additions_raw = row.get("additional_scope_clauses", [])
            if not isinstance(additions_raw, list):
                raise ValueError("additional_scope_clauses must be a list")
            additions = tuple(
                _parse_clause(value, source=f"handoff:{parent.policy_id}")
                for value in additions_raw
            )
            proposed = normalize_stage(str(row.get("proposed_stage", inherited.stage)))
            stage = max_stage(inherited.stage, proposed)
            if stage != proposed:
                diagnostics.stage_clamps.append(
                    {"tool": name, "parent": inherited.stage, "proposed": proposed, "effective": stage}
                )
            clauses, removed = _dedupe_clauses(inherited.scope_clauses + additions)
            accepted_additions = tuple(clause for clause in additions if clause not in removed)
            child_rules[name] = make_rule(
                tool=name,
                scope=" AND ".join(clause.text for clause in clauses) or "*",
                scope_clauses=clauses,
                stage=stage,
                reason="structurally attenuated Chain rule",
                valid=inherited.valid,
                parent_rule_ids=(inherited.rule_id,),
                id_salt=parent.policy_id,
            )
            diagnostics.accepted_clauses[name] = [
                clause.to_dict() for clause in accepted_additions
            ]
            if removed:
                diagnostics.duplicate_clauses_removed[name] = [
                    clause.to_dict() for clause in removed
                ]
        except Exception as exc:
            diagnostics.valid = False
            diagnostics.fallback_tools[name] = str(exc)
            child_rules[name] = _fallback_rule(inherited, f"invalid proposal entry: {exc}")

    child = AuthorizationPolicy(rules=child_rules, parent_policy_ids=(parent.policy_id,))
    return child, diagnostics


def _meet_policies(parents: list[AuthorizationPolicy]) -> AuthorizationPolicy:
    if not parents:
        raise ValueError("policy meet requires parents")
    tool_set = set(parents[0].rules)
    if any(set(parent.rules) != tool_set for parent in parents[1:]):
        raise ValueError("policy meet requires identical exposed tool coverage")
    rules: dict[str, AuthorizationRule] = {}
    for name in parents[0].rules:
        inputs = [parent.rules[name] for parent in parents]
        clauses, _removed = _dedupe_clauses(
            clause for rule in inputs for clause in rule.scope_clauses
        )
        rules[name] = make_rule(
            tool=name,
            scope=" AND ".join(clause.text for clause in clauses) or "*",
            scope_clauses=clauses,
            stage=max_stage(*(rule.stage for rule in inputs)),
            reason="explicit per-tool meet of parent policies",
            valid=all(rule.valid for rule in inputs),
            parent_rule_ids=(rule.rule_id for rule in inputs),
            id_salt="explicit_meet",
        )
    return AuthorizationPolicy(
        rules=rules,
        parent_policy_ids=tuple(parent.policy_id for parent in parents),
    )


def _propose_downstream_policy(
    parent: AuthorizationPolicy,
    *,
    delegated_subtask: str,
    downstream_agent: str,
    public_tools: list[dict[str, str]],
) -> str:
    from langchain_core.messages import HumanMessage, SystemMessage
    from llm import get_llm, invoke_llm

    payload = {
        "inherited_policy": parent.to_dict(agent_visible=True),
        "delegated_subtask": delegated_subtask,
        "downstream_agent": downstream_agent,
        "public_tools": public_tools,
    }
    try:
        with use_stage("chain:propose"):
            ai = invoke_llm(
                get_llm(temperature=0.0),
                [
                    SystemMessage(content=_PROPOSE_SYSTEM),
                    HumanMessage(content=json.dumps(payload, ensure_ascii=False, default=str)),
                ],
            )
        return _message_content_text(ai)
    except Exception as exc:
        record_defense_event(
            component="chain",
            event="proposal_failed",
            payload={"to": downstream_agent, "error": f"{type(exc).__name__}: {exc}"},
        )
        return ""


def _message_content_text(message_or_content: Any) -> str:
    content = getattr(message_or_content, "content", message_or_content)
    if isinstance(content, str):
        return content
    if isinstance(content, dict) and isinstance(content.get("text"), str):
        return content["text"]
    if isinstance(content, list):
        parts = [
            str(block.get("text"))
            for block in content
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        ]
        return "\n".join(parts) if parts else json.dumps(content, ensure_ascii=False, default=str)
    return str(content)


def _loads_json_obj(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        obj = json.loads(match.group(0))
    if not isinstance(obj, dict):
        raise ValueError("proposal must be a JSON object")
    return obj


def proposal_fixture_for_inheritance(policy: AuthorizationPolicy) -> dict[str, Any]:
    """Deterministic total proposal used by offline tests and smoke fixtures."""
    return {
        "schema": "chain_attenuation_proposal_v2",
        "rules": [
            {
                "tool": name,
                "inherit": True,
                "additional_scope_clauses": [],
                "proposed_stage": rule.stage,
            }
            for name, rule in policy.rules.items()
        ],
    }
