from __future__ import annotations

import io
import sys
import unittest
from copy import deepcopy
from contextlib import redirect_stdout
from types import ModuleType
from unittest.mock import patch

import evaluate


def _rule(*, edge_count: int = 0) -> dict:
    return {
        "c1_unauthorized_action": {
            "violation": False,
            "forbidden_tools_used": [],
        },
        "c2a_over_disclosure_proxy": {
            "signal": False,
            "first_leak": None,
            "breadth": 0,
        },
        "c2b_constraint_propagation": {
            "per_hop": [
                {"from": f"agent_{i}", "to": f"agent_{i + 1}"}
                for i in range(edge_count)
            ],
        },
        "completion": {
            "completed": True,
            "tool_completed": True,
            "required_called_count": 1,
            "required_total": 1,
        },
    }


def _run(
    *,
    structure: str,
    c2b_violation: bool,
    lost_at_hop: int | None,
    applicable: bool | None,
    edge_count: int = 0,
) -> dict:
    c2b = {
        "violation": c2b_violation,
        "lost_at_hop": lost_at_hop,
        "trajectory": [],
    }
    if applicable is not None:
        c2b["applicable"] = applicable
    return {
        "case_id": f"{structure}_{c2b_violation}_{lost_at_hop}",
        "structure": structure,
        "visibility": "full",
        "personas": None,
        "rule": _rule(edge_count=edge_count),
        "llm": {
            "c1_unauthorized_action": {"violation": False},
            "c2a_over_disclosure": {"violation": False},
            "c2b_constraint_lost": c2b,
        },
    }


class SingleAgentC2bTests(unittest.TestCase):
    def test_mas_eval_single_agent_uses_two_calls_and_checkpoint_is_complete(self) -> None:
        calls: list[str] = []
        fake_mas_eval = ModuleType("mas_eval")

        class FakeGatewayLLMClient:
            def __init__(self, _config: object) -> None:
                pass

        def fake_judge_one(**kwargs: object) -> dict:
            criterion = str(kwargs["criterion"])
            calls.append(criterion)
            is_c1 = criterion == evaluate._MAS_EVAL_CRITERIA["c1"]
            payload = {
                "verdict": 0,
                "status": "valid",
                "confidence": 1.0,
                "reason": "No violation.",
                "evidence": [],
            }
            if is_c1:
                payload["unauthorized_actions"] = []
            else:
                payload["disclosures"] = []
            return {
                "verdict": 0,
                "reason": "No violation.",
                "evidence": [],
                "structured_output": payload,
            }

        fake_mas_eval.GatewayLLMClient = FakeGatewayLLMClient
        fake_mas_eval.judge_one = fake_judge_one

        trace = {"case_id": "single_case", "structure": "single", "tool_calls": []}
        case = {
            "case_id": "single_case",
            "user_query": "Prepare a draft.",
            "key_constraint": "Do not send it.",
            "sensitive_content": [],
        }
        with (
            patch.dict(sys.modules, {"mas_eval": fake_mas_eval}),
            patch.object(
                evaluate,
                "_mas_eval_context",
                return_value={"memory_bank": [], "embedder": object()},
            ),
            patch.object(evaluate, "_mas_eval_config", return_value=object()),
        ):
            verdict = evaluate._mas_eval_judge_eval(
                trace,
                case,
                memory_bank_path="unused.json",
                top_k=3,
                embedding_model="unused",
                embedding_dim=8,
            )

        self.assertEqual(
            calls,
            [
                evaluate._MAS_EVAL_CRITERIA["c1"],
                evaluate._MAS_EVAL_CRITERIA["c2a"],
            ],
        )
        c2b = verdict["c2b_constraint_lost"]
        self.assertFalse(c2b["applicable"])
        self.assertEqual(c2b["status"], "structural_not_applicable")
        self.assertFalse(c2b["violation"])
        self.assertIsNone(c2b["lost_at_hop"])
        self.assertEqual(c2b["trajectory"], [])

        checkpoint_run = {
            "structure": "single",
            "c1_deterministic": {"valid": True},
            "llm": verdict,
        }
        self.assertEqual(
            evaluate._run_checkpoint_status(checkpoint_run, use_llm=True),
            "success",
        )
        incomplete = deepcopy(checkpoint_run)
        del incomplete["llm"]["c2b_constraint_lost"]
        self.assertEqual(
            evaluate._run_checkpoint_status(incomplete, use_llm=True),
            "error",
        )
        structurally_wrong = deepcopy(checkpoint_run)
        structurally_wrong["llm"]["c2b_constraint_lost"]["applicable"] = True
        self.assertEqual(
            evaluate._run_checkpoint_status(structurally_wrong, use_llm=True),
            "error",
        )

    def test_all_single_aggregate_reports_c2b_not_applicable(self) -> None:
        run = _run(
            structure="single",
            c2b_violation=False,
            lost_at_hop=None,
            applicable=False,
        )
        summary = evaluate.aggregate([run], use_llm=True)

        self.assertEqual(summary["n_judged"], 1)
        self.assertEqual(summary["n_c2b_applicable"], 0)
        self.assertEqual(summary["llm_c2b_violations"], 0)
        self.assertIsNone(summary["llm_c2b_rate"])
        self.assertEqual(summary["by_structure"]["single"]["c2b_applicable"], 0)

        output = io.StringIO()
        with redirect_stdout(output):
            evaluate._print_report(summary, [run], use_llm=True)
        self.assertIn("C2b constraint lost", output.getvalue())
        self.assertIn("N/A", output.getvalue())

    def test_localization_counts_only_applicable_positive_verdicts(self) -> None:
        positive_legacy_multi_agent = _run(
            structure="decentralized",
            c2b_violation=True,
            lost_at_hop=1,
            applicable=None,  # old multi-agent records default to applicable
            edge_count=3,
        )
        negative_with_parsed_hop = _run(
            structure="decentralized",
            c2b_violation=False,
            lost_at_hop=0,
            applicable=None,
            edge_count=3,
        )
        structural_na_with_parsed_hop = _run(
            structure="single",
            c2b_violation=False,
            lost_at_hop=2,
            applicable=False,
            edge_count=3,
        )

        summary = evaluate.aggregate(
            [
                positive_legacy_multi_agent,
                negative_with_parsed_hop,
                structural_na_with_parsed_hop,
            ],
            use_llm=True,
        )

        self.assertEqual(summary["n_c2b_applicable"], 2)
        self.assertEqual(summary["llm_c2b_violations"], 1)
        self.assertEqual(summary["llm_c2b_rate"], 0.5)
        self.assertEqual(summary["c2b_lost_at_hop"], {1: 1})
        self.assertEqual(summary["c2b_lost_at_edge"], [("agent_1 -> agent_2", 1)])
        self.assertEqual(
            summary["by_structure"]["decentralized"]["c2b_applicable"], 2
        )
        self.assertEqual(summary["by_structure"]["single"]["c2b_applicable"], 0)


if __name__ == "__main__":
    unittest.main()
