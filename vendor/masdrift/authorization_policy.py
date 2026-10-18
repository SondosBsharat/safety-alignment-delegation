"""Shared Tool-Scope-Stage authorization policy and reference monitor.

This module is deliberately independent of MAS topology and tool execution.  It
contains the paper-facing policy objects used by both Source (the authorizer) and
Chain.  Only root compilation may inspect the user request; the monitor consumes a
compiled policy, a public tool description, actual arguments, and genuine approval
state.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable


AUTHORIZATION_POLICY_SCHEMA = "authorization_policy_v2"
AUTHORIZATION_RULE_SCHEMA = "authorization_rule_v2"
AUTHORIZATION_STAGE_SCHEMA = "authorization_stage_v2"
AUTHORIZATION_LINEAGE_SCHEMA = "authorization_policy_lineage_v2"
CHAIN_GENERATION = "paper_aligned_chain_v2"

STAGE_ORDER = {"current_request": 0, "post_approval": 1, "never": 2}
RULE_STAGES = frozenset(STAGE_ORDER)

FORMAL_ALLOW = "Allow"
FORMAL_CONFIRM = "Require Confirmation"
FORMAL_DENY = "Deny"
FORMAL_VERDICTS = frozenset({FORMAL_ALLOW, FORMAL_CONFIRM, FORMAL_DENY})

FORMAL_TO_GATEWAY = {
    FORMAL_ALLOW: "allow",
    FORMAL_CONFIRM: "needs_user_confirmation",
    FORMAL_DENY: "deny",
}
FORMAL_TO_TRACE = {
    FORMAL_ALLOW: "allow",
    FORMAL_CONFIRM: "needs_confirm",
    FORMAL_DENY: "block",
}
GATEWAY_TO_FORMAL = {value: key for key, value in FORMAL_TO_GATEWAY.items()}


def normalize_stage(stage: str) -> str:
    """Return a canonical v2 stage name."""
    value = str(stage or "").strip().lower()
    if value not in RULE_STAGES:
        raise ValueError(f"invalid authorization rule stage: {stage!r}")
    return value


def max_stage(*stages: str) -> str:
    canonical = [normalize_stage(stage) for stage in stages]
    return max(canonical, key=STAGE_ORDER.__getitem__)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def _stable_id(prefix: str, value: Any) -> str:
    digest = hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()[:20]
    return f"{prefix}_{digest}"


def normalize_scope_text(scope: str) -> str:
    return " ".join(str(scope or "").strip().lower().split())


@dataclass(frozen=True)
class ScopeClause:
    """One immutable conjunct in an effective authorization scope."""

    text: str
    dimension: str = "general"
    clause_id: str = ""
    source: str = "root"
    schema: str = "authorization_scope_clause_v2"

    def __post_init__(self) -> None:
        text = str(self.text or "").strip()
        dimension = str(self.dimension or "general").strip().lower()
        if not text:
            raise ValueError("scope clause text must be non-empty")
        if dimension not in {"general", "object", "content", "recipient", "operation"}:
            raise ValueError(f"invalid scope clause dimension: {self.dimension!r}")
        object.__setattr__(self, "text", text)
        object.__setattr__(self, "dimension", dimension)
        if not self.clause_id:
            object.__setattr__(
                self,
                "clause_id",
                _stable_id("clause", [dimension, normalize_scope_text(text), self.source]),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "clause_id": self.clause_id,
            "dimension": self.dimension,
            "text": self.text,
            "source": self.source,
        }


def clauses_from_scope(scope: str, *, source: str = "root") -> tuple[ScopeClause, ...]:
    text = str(scope or "").strip()
    if not text or text == "*":
        return ()
    return (ScopeClause(text=text, source=source),)


def scope_fingerprint(clauses: Iterable[ScopeClause]) -> str:
    return _stable_id("scope", [clause.to_dict() for clause in clauses])


@dataclass(frozen=True)
class AuthorizationRule:
    """A total-policy rule ``r_t = <Tool, Scope, Stage>``."""

    rule_id: str
    tool: str
    scope: str
    stage: str
    reason: str = ""
    valid: bool = True
    scope_clauses: tuple[ScopeClause, ...] = ()
    parent_rule_ids: tuple[str, ...] = ()
    schema: str = AUTHORIZATION_RULE_SCHEMA

    def __post_init__(self) -> None:
        if not self.tool:
            raise ValueError("authorization rule requires a public tool name")
        canonical_stage = normalize_stage(self.stage)
        object.__setattr__(self, "stage", canonical_stage)
        clauses = tuple(self.scope_clauses) or clauses_from_scope(self.scope)
        object.__setattr__(self, "scope_clauses", clauses)
        if not self.scope:
            object.__setattr__(self, "scope", " *".strip())

    @property
    def scope_clause_fingerprint(self) -> str:
        return scope_fingerprint(self.scope_clauses)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "rule_id": self.rule_id,
            "tool": self.tool,
            "scope": self.scope,
            "scope_clauses": [clause.to_dict() for clause in self.scope_clauses],
            "scope_fingerprint": self.scope_clause_fingerprint,
            "stage": self.stage,
            "reason": self.reason,
            "valid": self.valid,
            "parent_rule_ids": list(self.parent_rule_ids),
        }


def make_rule(
    *,
    tool: str,
    scope: str = "*",
    scope_clauses: Iterable[ScopeClause] = (),
    stage: str,
    reason: str = "",
    valid: bool = True,
    parent_rule_ids: Iterable[str] = (),
    id_salt: Any = None,
) -> AuthorizationRule:
    clauses = tuple(scope_clauses) or clauses_from_scope(scope)
    canonical = normalize_stage(stage)
    rendered_scope = scope if scope else (" AND ".join(c.text for c in clauses) or "*")
    rule_id = _stable_id(
        "authz",
        [tool, [c.to_dict() for c in clauses], canonical, valid, list(parent_rule_ids), id_salt],
    )
    return AuthorizationRule(
        rule_id=rule_id,
        tool=tool,
        scope=rendered_scope,
        scope_clauses=clauses,
        stage=canonical,
        reason=reason,
        valid=valid,
        parent_rule_ids=tuple(parent_rule_ids),
    )


@dataclass(frozen=True)
class AuthorizationPolicy:
    """Versioned total policy ``P_i`` over one exposed public tool surface."""

    rules: dict[str, AuthorizationRule]
    policy_id: str = ""
    parent_policy_ids: tuple[str, ...] = ()
    generation: str = CHAIN_GENERATION
    schema: str = AUTHORIZATION_POLICY_SCHEMA

    def __post_init__(self) -> None:
        copied = dict(self.rules)
        if any(name != rule.tool for name, rule in copied.items()):
            raise ValueError("policy keys must equal rule tool identifiers")
        if len(copied) != len(set(copied)):
            raise ValueError("authorization policy contains duplicate tools")
        object.__setattr__(self, "rules", copied)
        if not self.policy_id:
            object.__setattr__(
                self,
                "policy_id",
                _stable_id(
                    "policy",
                    {
                        "rules": {name: rule.to_dict() for name, rule in sorted(copied.items())},
                        "parents": list(self.parent_policy_ids),
                        "generation": self.generation,
                    },
                ),
            )

    @property
    def tool_names(self) -> tuple[str, ...]:
        return tuple(self.rules)

    def to_dict(self, *, agent_visible: bool = False) -> dict[str, Any]:
        # Rules contain no approval evidence or evaluator-only diagnostics, so the
        # same effective policy snapshot is safe for the downstream agent.
        return {
            "schema": self.schema,
            "generation": self.generation,
            "policy_id": self.policy_id,
            "parent_policy_ids": list(self.parent_policy_ids),
            "rules": {name: rule.to_dict() for name, rule in self.rules.items()},
        }


@dataclass
class AuthorizationStageState:
    version: int = 0
    approved_rule_ids: set[str] = field(default_factory=set)
    approved_scope_fingerprints: dict[str, str] = field(default_factory=dict)
    approved_lineages: dict[str, str | None] = field(default_factory=dict)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    schema: str = AUTHORIZATION_STAGE_SCHEMA

    def is_approved(self, rule: AuthorizationRule, lineage_id: str | None = None) -> bool:
        return (
            rule.rule_id in self.approved_rule_ids
            and self.approved_scope_fingerprints.get(rule.rule_id) == rule.scope_clause_fingerprint
            and self.approved_lineages.get(rule.rule_id) == lineage_id
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "version": self.version,
            "approved_rule_ids": sorted(self.approved_rule_ids),
            "approved_scope_fingerprints": dict(self.approved_scope_fingerprints),
            "evidence": [dict(row) for row in self.evidence],
        }

    def to_public_dict(self) -> dict[str, Any]:
        return {"schema": self.schema, "version": self.version}


@dataclass(frozen=True)
class AuthorizationAssessment:
    formal_verdict: str
    reason: str
    rule: AuthorizationRule | None
    tool_match: bool
    scope_result: str
    stage_satisfied: bool
    failure_dimension: str | None = None
    policy_id: str | None = None
    lineage_id: str | None = None

    @property
    def verdict(self) -> str:
        return FORMAL_TO_GATEWAY[self.formal_verdict]

    @property
    def trace_verdict(self) -> str:
        return FORMAL_TO_TRACE[self.formal_verdict]

    def audit_dict(self, stage_version: int) -> dict[str, Any]:
        rule = self.rule
        return {
            "policy_schema": AUTHORIZATION_POLICY_SCHEMA,
            "policy_id": self.policy_id,
            "lineage_id": self.lineage_id,
            "rule_id": rule.rule_id if rule else None,
            "scope_fingerprint": rule.scope_clause_fingerprint if rule else None,
            "tool_match": self.tool_match,
            "rule_scope": rule.scope if rule else None,
            "scope_clauses": [c.to_dict() for c in rule.scope_clauses] if rule else [],
            "scope_result": self.scope_result,
            "required_stage": rule.stage if rule else None,
            "stage_satisfied": self.stage_satisfied,
            "stage_version": stage_version,
            "failure_dimension": self.failure_dimension,
            "formal_verdict": self.formal_verdict,
            "gateway_verdict": self.verdict,
            "trace_verdict": self.trace_verdict,
        }


ScopeMatcher = Callable[[AuthorizationRule, dict[str, Any], dict[str, Any]], tuple[str, str]]


@dataclass(frozen=True)
class CompilationDiagnostics:
    unknown_tools: tuple[str, ...] = ()
    duplicate_tools: tuple[str, ...] = ()
    fallback_tools: tuple[str, ...] = ()
    parse_failed: bool = False


def _parse_policy_rules(raw: str) -> tuple[list[dict[str, Any]], bool]:
    text = str(raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return [], True
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            return [], True
    rows = obj.get("rules", []) if isinstance(obj, dict) else []
    return [row for row in rows if isinstance(row, dict)], False


def compile_authorization_policy(
    user_query: str,
    public_tools: list[dict[str, str]],
    extractor: Callable[[str, list[dict[str, str]]], str],
) -> tuple[AuthorizationPolicy, CompilationDiagnostics]:
    """``Compile(u, T)``: the only shared operation allowed to consume ``u``.

    The extractor receives only the original request and public tool surface.  Code
    then enforces exactly-one-rule coverage and emits conservative total fallbacks.
    """
    raw = extractor(user_query, public_tools)
    rows, parse_failed = _parse_policy_rules(raw)
    names = {str(tool.get("name", "")) for tool in public_tools}
    policy: dict[str, AuthorizationRule] = {}
    unknown: list[str] = []
    duplicates: set[str] = set()
    invalid: set[str] = set()
    seen: set[str] = set()
    for row in rows:
        name = str(row.get("tool", "")).strip()
        if name not in names:
            unknown.append(name)
            continue
        if name in seen:
            duplicates.add(name)
            invalid.add(name)
            policy.pop(name, None)
            continue
        seen.add(name)
        scope = str(row.get("scope", "")).strip()
        try:
            stage = normalize_stage(str(row.get("stage", "")))
        except ValueError:
            invalid.add(name)
            continue
        if not scope:
            invalid.add(name)
            continue
        policy[name] = make_rule(
            tool=name,
            scope=scope,
            stage=stage,
            reason=str(row.get("reason", ""))[:500],
            id_salt="root",
        )
    fallback = sorted((names - set(policy)) | invalid)
    for name in fallback:
        policy[name] = make_rule(
            tool=name,
            scope="*",
            stage="post_approval",
            reason="authorization extractor omitted or malformed this public tool",
            valid=False,
            id_salt="root_fallback",
        )
    return AuthorizationPolicy(rules=policy), CompilationDiagnostics(
        unknown_tools=tuple(unknown),
        duplicate_tools=tuple(sorted(duplicates)),
        fallback_tools=tuple(fallback),
        parse_failed=parse_failed,
    )


def _is_ask_tool(tool_spec: dict[str, Any]) -> bool:
    text = (str(tool_spec.get("name", "")) + " " + str(tool_spec.get("description", "")))
    text = text.replace("_", " ").replace("-", " ").lower()
    return "ask user" in text or "confirmation request" in text


def monitor_authorization_call(
    policy: AuthorizationPolicy,
    stage_state: AuthorizationStageState,
    tool_spec: dict[str, Any],
    args: dict[str, Any],
    scope_matcher: ScopeMatcher,
    *,
    lineage_id: str | None = None,
) -> AuthorizationAssessment:
    """The single deterministic verdict-ordering implementation for Source/Chain."""
    name = str(tool_spec.get("name", ""))
    rule = policy.rules.get(name)
    common = {"policy_id": policy.policy_id, "lineage_id": lineage_id}
    if _is_ask_tool(tool_spec):
        return AuthorizationAssessment(
            FORMAL_ALLOW, "ask-user gate remains available", rule, True, "match", True,
            **common,
        )
    if rule is None:
        return AuthorizationAssessment(
            FORMAL_CONFIRM, "no structured authorization rule covers this public tool",
            None, False, "uncertain", False, "missing_rule", **common,
        )
    if rule.stage == "never":
        return AuthorizationAssessment(
            FORMAL_DENY, f"rule never: {rule.reason}", rule, True, "not_checked", False,
            "never", **common,
        )
    if not rule.valid:
        return AuthorizationAssessment(
            FORMAL_CONFIRM, "invalid total fallback rule requires user confirmation",
            rule, True, "uncertain", False, "missing_rule", **common,
        )

    stage_ok = rule.stage == "current_request" or (
        rule.stage == "post_approval" and stage_state.is_approved(rule, lineage_id)
    )
    scope_result, scope_reason = scope_matcher(rule, tool_spec, args)
    if scope_result == "mismatch":
        return AuthorizationAssessment(
            FORMAL_DENY, f"scope mismatch: {scope_reason}", rule, True, scope_result,
            stage_ok, "scope_mismatch", **common,
        )
    if scope_result != "match":
        return AuthorizationAssessment(
            FORMAL_CONFIRM, f"scope uncertain: {scope_reason}", rule, True, "uncertain",
            stage_ok, "scope_uncertain", **common,
        )
    if not stage_ok:
        return AuthorizationAssessment(
            FORMAL_CONFIRM, f"stage mismatch: {rule.stage} requires genuine user approval",
            rule, True, scope_result, False, "stage_mismatch", **common,
        )
    return AuthorizationAssessment(
        FORMAL_ALLOW, f"tool, scope, and stage match: {rule.reason}", rule, True,
        scope_result, True, **common,
    )


def total_invalid_fallback(
    parent: AuthorizationPolicy,
    *,
    reason: str,
    parent_policy_ids: Iterable[str] | None = None,
) -> AuthorizationPolicy:
    rules: dict[str, AuthorizationRule] = {}
    for name, inherited in parent.rules.items():
        stage = max_stage(inherited.stage, "post_approval")
        rules[name] = make_rule(
            tool=name,
            scope=inherited.scope,
            scope_clauses=inherited.scope_clauses,
            stage=stage,
            reason=reason,
            valid=False,
            parent_rule_ids=(inherited.rule_id,),
            id_salt="invalid_fallback",
        )
    return AuthorizationPolicy(
        rules=rules,
        parent_policy_ids=tuple(parent_policy_ids or (parent.policy_id,)),
    )
