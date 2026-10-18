from __future__ import annotations

import json
import random
import unittest
from pathlib import Path
from unittest.mock import patch

import agent_react
import decentralized
import main
import tools
from authorization_policy import (
    AUTHORIZATION_LINEAGE_SCHEMA,
    AUTHORIZATION_POLICY_SCHEMA,
    AuthorizationPolicy,
    AuthorizationStageState,
    FORMAL_ALLOW,
    FORMAL_CONFIRM,
    FORMAL_DENY,
    ScopeClause,
    make_rule,
    monitor_authorization_call,
    normalize_stage,
)
from defense import BoundaryDefense, DefenseConfig
from defense_chain import (
    ChainState,
    _meet_policies,
    attenuate_policy,
    proposal_fixture_for_inheritance,
)
from tracing import Recorder, use_recorder


TOOLS = [
    {"name": "read_record", "description": "Read record R1."},
    {"name": "send_report", "description": "Send report Q1 to a recipient."},
    {"name": "delete_record", "description": "Delete record R1 permanently."},
]


def _root() -> AuthorizationPolicy:
    return AuthorizationPolicy(rules={
        "read_record": make_rule(
            tool="read_record", scope="R1", stage="current_request", reason="requested"
        ),
        "send_report": make_rule(
            tool="send_report", scope="Q1 to internal finance", stage="post_approval",
            reason="reserved",
        ),
        "delete_record": make_rule(
            tool="delete_record", scope="R1", stage="never", reason="excluded"
        ),
    })


def _inherit(policy: AuthorizationPolicy) -> dict:
    return proposal_fixture_for_inheritance(policy)


def _match(_rule, _tool, args):
    result = args.get("_scope_result", "match")
    return result, f"fixture {result}"


class SharedPolicySchemaTests(unittest.TestCase):
    def test_canonical_stage_reader_accepts_only_canonical_values(self) -> None:
        self.assertEqual(normalize_stage("post_approval"), "post_approval")
        with self.assertRaises(ValueError):
            normalize_stage("later")

    def test_root_is_total_and_disabled_tool_is_retained(self) -> None:
        root = _root()
        self.assertEqual(set(root.rules), {tool["name"] for tool in TOOLS})
        self.assertEqual(root.rules["delete_record"].stage, "never")
        self.assertEqual(root.schema, AUTHORIZATION_POLICY_SCHEMA)

    def test_chain_resume_identity_rejects_legacy_coarse_trace(self) -> None:
        current = DefenseConfig(("chain",)).signature()
        legacy = main._defense_sig({
            "schema": "boundary_state_defense_config_v1",
            "components": ["chain"],
            "authorization_scope": {"allowed_kinds": ["read"]},
        })
        v2 = main._defense_sig(DefenseConfig(("chain",)).to_dict())
        self.assertNotEqual(current, legacy)
        self.assertEqual(current, v2)

    def test_formal_gateway_trace_mapping_and_verdict_priority(self) -> None:
        state = AuthorizationStageState()
        root = _root()
        allow = monitor_authorization_call(root, state, TOOLS[0], {}, _match)
        confirm = monitor_authorization_call(root, state, TOOLS[1], {}, _match)
        deny = monitor_authorization_call(root, state, TOOLS[2], {}, _match)
        self.assertEqual((allow.formal_verdict, allow.verdict, allow.trace_verdict),
                         (FORMAL_ALLOW, "allow", "allow"))
        self.assertEqual((confirm.formal_verdict, confirm.verdict, confirm.trace_verdict),
                         (FORMAL_CONFIRM, "needs_user_confirmation", "needs_confirm"))
        self.assertEqual((deny.formal_verdict, deny.verdict, deny.trace_verdict),
                         (FORMAL_DENY, "deny", "block"))

    def test_frozen_source_golden_contract(self) -> None:
        fixture = json.loads(
            (Path(__file__).parent / "fixtures" / "authorization_source_golden_v2.json")
            .read_text(encoding="utf-8")
        )
        self.assertEqual(fixture["schema"], "authorization_source_golden_v2")
        self.assertEqual(len(fixture["cases"]), 7)
        self.assertEqual(
            {row["formal_verdict"] for row in fixture["cases"]},
            {FORMAL_ALLOW, FORMAL_CONFIRM, FORMAL_DENY},
        )


class AttenuationTests(unittest.TestCase):
    def test_valid_child_preserves_parent_and_adds_conjunct(self) -> None:
        root = _root()
        proposal = _inherit(root)
        send = next(row for row in proposal["rules"] if row["tool"] == "send_report")
        send["additional_scope_clauses"] = [
            {"dimension": "recipient", "text": "Alice only"}
        ]
        child, diagnostics = attenuate_policy(root, proposal)
        clauses = [clause.text for clause in child.rules["send_report"].scope_clauses]
        self.assertEqual(clauses, ["Q1 to internal finance", "Alice only"])
        self.assertTrue(diagnostics.valid)

    def test_duplicate_scope_clauses_are_not_semantic_attenuation(self) -> None:
        root = _root()
        proposal = _inherit(root)
        read = next(row for row in proposal["rules"] if row["tool"] == "read_record")
        read["additional_scope_clauses"] = [
            {"dimension": "general", "text": "  r1  "},
            {"dimension": "object", "text": "R1"},
        ]
        child, diagnostics = attenuate_policy(root, proposal)
        clauses = child.rules["read_record"].scope_clauses
        self.assertEqual(
            [(clause.dimension, clause.text) for clause in clauses],
            [("general", "R1"), ("object", "R1")],
        )
        self.assertEqual(len(diagnostics.duplicate_clauses_removed["read_record"]), 1)

    def test_total_conservative_fallbacks(self) -> None:
        root = _root()
        cases: list[tuple[str, object, str | None]] = []
        missing = _inherit(root)
        missing["rules"] = missing["rules"][:-1]
        cases.append(("missing", missing, "delete_record"))
        duplicate = _inherit(root)
        duplicate["rules"].append(dict(duplicate["rules"][0]))
        cases.append(("duplicate", duplicate, "read_record"))
        malformed = _inherit(root)
        malformed["rules"][0]["additional_scope_clauses"] = [{"text": ""}]
        cases.append(("malformed", malformed, "read_record"))
        unknown = _inherit(root)
        unknown["rules"].append({
            "tool": "hidden_tool", "inherit": True,
            "additional_scope_clauses": [], "proposed_stage": "current_request",
        })
        cases.append(("unknown", unknown, None))
        cases.append(("unparseable", "not-json", None))

        for label, raw, affected in cases:
            with self.subTest(label=label):
                child, diagnostics = attenuate_policy(root, raw)
                self.assertEqual(set(child.rules), set(root.rules))
                self.assertFalse(diagnostics.valid)
                if affected:
                    rule = child.rules[affected]
                    self.assertFalse(rule.valid)
                    self.assertGreaterEqual(
                        {"current_request": 0, "post_approval": 1, "never": 2}[rule.stage],
                        {"current_request": 0, "post_approval": 1, "never": 2}[
                            root.rules[affected].stage
                        ],
                    )
                if label == "unparseable":
                    self.assertTrue(all(not rule.valid for rule in child.rules.values()))

    def test_replacement_is_ignored_and_stage_relaxation_is_clamped(self) -> None:
        root = _root()
        proposal = _inherit(root)
        send = next(row for row in proposal["rules"] if row["tool"] == "send_report")
        send["scope"] = "any report to anyone"
        send["proposed_stage"] = "current_request"
        child, diagnostics = attenuate_policy(root, proposal)
        self.assertEqual(
            child.rules["send_report"].scope_clauses,
            root.rules["send_report"].scope_clauses,
        )
        self.assertEqual(child.rules["send_report"].stage, "post_approval")
        self.assertTrue(diagnostics.rejected_widening)
        self.assertTrue(diagnostics.stage_clamps)

    def test_no_dimension_can_be_removed_or_replaced(self) -> None:
        clauses = tuple(
            ScopeClause(text=f"parent {dimension}", dimension=dimension)
            for dimension in ("object", "content", "recipient", "operation")
        )
        parent = AuthorizationPolicy(rules={
            "send_report": make_rule(
                tool="send_report", scope=" AND ".join(c.text for c in clauses),
                scope_clauses=clauses, stage="current_request",
            )
        })
        proposal = _inherit(parent)
        proposal["rules"][0].update({
            "replacement_scope": "anything to anyone",
            "remove_clauses": [clause.clause_id for clause in clauses],
        })
        child, diagnostics = attenuate_policy(parent, proposal)
        self.assertEqual(child.rules["send_report"].scope_clauses, clauses)
        self.assertEqual(
            diagnostics.rejected_widening[0]["keys"],
            ["remove_clauses", "replacement_scope"],
        )
        recorder = Recorder("widening", "centralized")
        with (
            use_recorder(recorder),
            patch("defense_chain._propose_downstream_policy", return_value=proposal),
        ):
            ChainState(parent, [TOOLS[1]], enforced=True).on_handoff(
                from_agent="Lead", to_agent="Worker", message="task"
            )
        event = next(
            row for row in recorder.defense_events if row["event"] == "policy_attenuated"
        )
        self.assertEqual(
            event["payload"]["rejected_widening"][0]["keys"],
            ["remove_clauses", "replacement_scope"],
        )

    def test_randomized_multihop_scope_and_stage_monotonicity(self) -> None:
        rng = random.Random(1729)
        policy = _root()
        rank = {"current_request": 0, "post_approval": 1, "never": 2}
        for hop in range(100):
            proposal = _inherit(policy)
            for row in proposal["rules"]:
                if rng.random() < 0.7:
                    row["additional_scope_clauses"] = [{
                        "dimension": rng.choice(
                            ["object", "content", "recipient", "operation", "general"]
                        ),
                        "text": f"hop {hop} restriction for {row['tool']}",
                    }]
                row["proposed_stage"] = rng.choice(
                    ["current_request", "post_approval", "never"]
                )
            child, _ = attenuate_policy(policy, proposal)
            for name in policy.rules:
                self.assertEqual(
                    child.rules[name].scope_clauses[:len(policy.rules[name].scope_clauses)],
                    policy.rules[name].scope_clauses,
                )
                self.assertGreaterEqual(rank[child.rules[name].stage], rank[policy.rules[name].stage])
            policy = child

    def test_multi_parent_meet_is_total_conjunctive_and_never_absorbing(self) -> None:
        root = _root()
        p1_raw = _inherit(root)
        p2_raw = _inherit(root)
        p1_raw["rules"][0]["additional_scope_clauses"] = ["branch one"]
        p2_raw["rules"][0]["additional_scope_clauses"] = ["branch two"]
        p2_raw["rules"][0]["proposed_stage"] = "never"
        p1, _ = attenuate_policy(root, p1_raw)
        p2, _ = attenuate_policy(root, p2_raw)
        joined = _meet_policies([p1, p2])
        self.assertEqual(set(joined.rules), set(root.rules))
        self.assertEqual(joined.rules["read_record"].stage, "never")
        texts = [clause.text for clause in joined.rules["read_record"].scope_clauses]
        self.assertIn("branch one", texts)
        self.assertIn("branch two", texts)


class LineageIntegrationTests(unittest.TestCase):
    def test_source_and_chain_share_exact_root_and_monitor(self) -> None:
        case = {"user_query": "Read R1; send only after approval.", "available_tools": TOOLS}
        raw = json.dumps({
            "rules": [
                {"tool": "read_record", "scope": "R1", "stage": "current_request"},
                {"tool": "send_report", "scope": "Q1", "stage": "post_approval"},
                {"tool": "delete_record", "scope": "R1", "stage": "never"},
            ]
        })
        with patch("defense._call_authorization_agent", return_value=raw):
            defense = BoundaryDefense(case, DefenseConfig(("authorizer", "chain")))
            chain = defense._ensure_chain()
        self.assertIsNotNone(chain)
        assert chain is not None
        self.assertIs(chain.root, defense._authorization_policy())
        source = defense._verdict_for(TOOLS[0], {})
        chain_result, _ = chain.assess(
            "Supervisor", TOOLS[0], {}, defense.authorization_stage, defense._scope_match
        )
        self.assertEqual(
            (source.formal_verdict, source.failure_dimension),
            (chain_result.formal_verdict, chain_result.failure_dimension),
        )

    def test_root_compiles_once_and_chain_proposal_never_receives_query(self) -> None:
        case = {"user_query": "secret root request", "available_tools": TOOLS}
        raw = json.dumps({"rules": [
            {"tool": "read_record", "scope": "R1", "stage": "current_request"},
            {"tool": "send_report", "scope": "Q1", "stage": "post_approval"},
            {"tool": "delete_record", "scope": "R1", "stage": "never"},
        ]})
        captured = []
        with patch("defense._call_authorization_agent", return_value=raw) as compiler:
            defense = BoundaryDefense(case, DefenseConfig(("chain",)))
            chain = defense._ensure_chain()
            defense._ensure_chain()
            assert chain is not None
            with patch(
                "defense_chain._propose_downstream_policy",
                side_effect=lambda parent, **kwargs: captured.append(kwargs) or _inherit(parent),
            ):
                chain.on_handoff(from_agent="Supervisor", to_agent="Worker", message="local task")
        self.assertEqual(compiler.call_count, 1)
        self.assertNotIn("secret root request", json.dumps(captured))
        self.assertNotIn("user_query", captured[0])

    def test_approval_is_rule_scope_and_lineage_local(self) -> None:
        root = _root()
        rule = root.rules["send_report"]
        state = AuthorizationStageState(
            version=1,
            approved_rule_ids={rule.rule_id},
            approved_scope_fingerprints={rule.rule_id: rule.scope_clause_fingerprint},
            approved_lineages={rule.rule_id: "lineage-A"},
        )
        allowed = monitor_authorization_call(
            root, state, TOOLS[1], {}, _match, lineage_id="lineage-A"
        )
        other_lineage = monitor_authorization_call(
            root, state, TOOLS[1], {}, _match, lineage_id="lineage-B"
        )
        self.assertEqual(allowed.formal_verdict, FORMAL_ALLOW)
        self.assertEqual(other_lineage.formal_verdict, FORMAL_CONFIRM)

    def test_chain_advisory_judges_same_policy_without_blocking(self) -> None:
        case = {"user_query": "Read R1; send only after approval.", "available_tools": TOOLS}
        raw = json.dumps({"rules": [
            {"tool": "read_record", "scope": "R1", "stage": "current_request"},
            {"tool": "send_report", "scope": "Q1", "stage": "post_approval"},
            {"tool": "delete_record", "scope": "R1", "stage": "never"},
        ]})
        recorder = Recorder("case", "centralized")
        with (
            patch("defense._call_authorization_agent", return_value=raw),
            use_recorder(recorder),
        ):
            defense = BoundaryDefense(case, DefenseConfig(("chain-advisory",)))
            result = defense.authorize_tool_call(TOOLS[2], {}, "Supervisor")
        self.assertIsNone(result)
        insufficient = next(
            event for event in recorder.defense_events if event["event"] == "scope_insufficient"
        )
        self.assertEqual(insufficient["payload"]["formal_verdict"], FORMAL_DENY)
        self.assertFalse(insufficient["payload"]["enforced"])

    def test_three_hops_are_reconstructable_and_source_free(self) -> None:
        root = _root()
        chain = ChainState(root, TOOLS, enforced=True)
        captured = []

        def proposal(parent, **kwargs):
            captured.append(kwargs)
            return _inherit(parent)

        recorder = Recorder("case", "centralized")
        with patch("defense_chain._propose_downstream_policy", side_effect=proposal), use_recorder(recorder):
            e1 = chain.on_handoff(from_agent="Supervisor", to_agent="A", message="task A")
            chain.active_envelope("A")
            e2 = chain.on_handoff(from_agent="A", to_agent="B", message="task B")
            chain.active_envelope("B")
            e3 = chain.on_handoff(from_agent="B", to_agent="C", message="task C")
        self.assertEqual(e1.lineage_id, e2.lineage_id)
        self.assertEqual(e2.lineage_id, e3.lineage_id)
        self.assertTrue(all("user_query" not in kwargs for kwargs in captured))
        events = [e for e in recorder.defense_events if e["event"] == "policy_attenuated"]
        self.assertEqual(len(events), 3)
        self.assertTrue(all("parent_policy" in e["payload"] for e in events))

    def test_trace_records_fallback_lineage_verdict_wire_and_execution(self) -> None:
        root = _root()
        recorder = Recorder("case", "peer")
        with use_recorder(recorder):
            chain = ChainState(root, TOOLS, enforced=True)
            with patch("defense_chain._propose_downstream_policy", return_value="not-json"):
                fallback = chain.on_handoff(
                    from_agent="Peer_1", to_agent="Peer_2", message="synthetic task"
                )
            case = {"user_query": "synthetic", "available_tools": TOOLS}
            boundary = BoundaryDefense(case, DefenseConfig(("chain",)))
            boundary._policy = root
            boundary._chain = chain
            blocked = boundary.authorize_tool_call(TOOLS[2], {}, "Peer_2")
        self.assertEqual(blocked["status"], "blocked")
        attenuation = next(
            event for event in recorder.defense_events if event["event"] == "policy_attenuated"
        )["payload"]
        self.assertFalse(attenuation["proposal_valid"])
        self.assertEqual(set(attenuation["fallback_rules"]), set(root.rules))
        self.assertTrue(all(not rule.valid for rule in fallback.policy.rules.values()))
        decision = next(
            event for event in recorder.defense_events if event["event"] == "tool_blocked"
        )["payload"]
        for field in (
            "active_policy_id", "active_lineage_id", "formal_verdict", "wire_outcome",
            "execution_status", "scope_result", "required_stage", "stage_version",
        ):
            self.assertIn(field, decision)
        self.assertEqual(decision["wire_outcome"], "block")
        self.assertEqual(decision["execution_status"], "blocked")

    def test_repeated_named_agent_has_independent_immutable_activations(self) -> None:
        root = _root()
        chain = ChainState(root, TOOLS, enforced=True)
        with patch("defense_chain._propose_downstream_policy", side_effect=lambda p, **_: _inherit(p)):
            first = chain.on_handoff(from_agent="Supervisor", to_agent="Worker", message="one")
            self.assertEqual(chain.active_envelope("Worker").activation_id, first.activation_id)
            second = chain.on_handoff(from_agent="Supervisor", to_agent="Worker", message="two")
            self.assertEqual(chain.active_envelope("Worker").activation_id, second.activation_id)
        self.assertNotEqual(first.activation_id, second.activation_id)
        self.assertEqual(chain.envelope_for_activation(first.activation_id), first)

    def test_multiple_pending_parents_fail_closed_and_explicit_join_meets(self) -> None:
        root = _root()
        chain = ChainState(root, TOOLS, enforced=True)
        with patch("defense_chain._propose_downstream_policy", side_effect=lambda p, **_: _inherit(p)):
            one = chain.on_handoff(from_agent="P1", to_agent="Joiner", message="one")
            two = chain.on_handoff(from_agent="P2", to_agent="Joiner", message="two")
        ambiguous = chain.active_envelope("Joiner")
        self.assertTrue(all(not rule.valid for rule in ambiguous.policy.rules.values()))
        joined = chain.explicit_join(
            agent="ExplicitJoiner", activation_ids=[one.activation_id, two.activation_id]
        )
        self.assertEqual(joined.policy.parent_policy_ids, (one.policy_id, two.policy_id))

    def test_join_active_agents_is_idempotent_and_binds_integrator(self) -> None:
        chain = ChainState(_root(), TOOLS, enforced=True)
        recorder = Recorder("join", "centralized")
        with (
            patch("defense_chain._propose_downstream_policy", side_effect=lambda p, **_: _inherit(p)),
            use_recorder(recorder),
        ):
            one = chain.on_handoff(from_agent="Chief", to_agent="Worker_1", message="one")
            two = chain.on_handoff(from_agent="Chief", to_agent="Worker_2", message="two")
            first = chain.join_active_agents(
                agent="Chief", parent_agents=("Worker_1", "Worker_2")
            )
            second = chain.join_active_agents(
                agent="Chief", parent_agents=("Worker_1", "Worker_2")
            )
            active = chain.active_envelope("Chief")
        self.assertEqual(first.activation_id, second.activation_id)
        self.assertEqual(active.activation_id, first.activation_id)
        self.assertEqual(first.policy.parent_policy_ids, (one.policy_id, two.policy_id))
        self.assertEqual(
            len([e for e in recorder.defense_events if e["event"] == "policy_joined"]),
            1,
        )

    def test_missing_known_activation_uses_total_invalid_fallback(self) -> None:
        chain = ChainState(_root(), TOOLS, enforced=True)
        with patch(
            "defense_chain._propose_downstream_policy",
            side_effect=lambda parent, **_: _inherit(parent),
        ):
            chain.on_handoff(from_agent="Lead", to_agent="Worker", message="task")
        chain.active_envelope("Worker")
        chain.release_activation("Worker")
        missing = chain.active_envelope("Worker")
        self.assertTrue(all(not rule.valid for rule in missing.policy.rules.values()))
        self.assertTrue(all(rule.stage == "never" for rule in missing.policy.rules.values()))
        self.assertEqual(missing.parent_policy_id, chain.latest_handoff_for("Worker").policy_id)

    def test_agent_visible_envelope_excludes_diagnostics_and_hidden_fields(self) -> None:
        root = _root()
        chain = ChainState(root, TOOLS, enforced=False)
        blob = json.dumps(chain.root_envelope.to_dict(agent_visible=True))
        self.assertEqual(chain.root_envelope.schema, AUTHORIZATION_LINEAGE_SCHEMA)
        for forbidden in (
            "ground_truth", "required_steps", "forbidden_actions", "key_constraint",
            "sensitive_content", "approval_evidence", "proposal_valid", "prior_verdict",
        ):
            self.assertNotIn(forbidden, blob)

    def test_topologies_and_gateway_do_not_duplicate_chain_policy_logic(self) -> None:
        for module in (agent_react, decentralized, tools):
            source = Path(module.__file__).read_text(encoding="utf-8")
            self.assertNotIn("from defense_chain import", source)
            self.assertNotIn("AuthorizationScope", source)

    def test_centralized_and_peer_fixtures_use_identical_chain_contract(self) -> None:
        envelopes = []
        for _topology in ("centralized", "peer"):
            chain = ChainState(_root(), TOOLS, enforced=True)
            with patch(
                "defense_chain._propose_downstream_policy",
                side_effect=lambda parent, **_: _inherit(parent),
            ):
                envelopes.append(
                    chain.on_handoff(from_agent="Lead", to_agent="Worker", message="same task")
                )
        left, right = envelopes
        self.assertEqual(
            {name: (rule.scope, rule.stage, rule.valid) for name, rule in left.policy.rules.items()},
            {name: (rule.scope, rule.stage, rule.valid) for name, rule in right.policy.rules.items()},
        )
        self.assertNotIn("decentralized instance", json.dumps(left.to_dict()).lower())


if __name__ == "__main__":
    unittest.main()
