import unittest

from scripts.analyze_c1_execution_outcomes import classify_calls


class C1ExecutionOutcomeTests(unittest.TestCase):
    def test_ask_user_effect_is_not_autonomous_execution(self) -> None:
        calls = [{"agent": "Worker_1", "tool": "send", "executed": True}]
        trace = {
            "defense": {"components": ["authorizer"]},
            "defense_events": [
                {
                    "event": "tool_needs_confirm",
                    "agent": "Worker_1",
                    "payload": {"tool": "send"},
                }
            ],
        }
        result = classify_calls(calls, trace)[0]
        self.assertEqual(result["authorization_outcome"], "needs_user_confirmation")
        self.assertFalse(result["autonomous_executed"])
        self.assertTrue(result["post_confirmation_effect"])

    def test_direct_allow_effect_is_autonomous_execution(self) -> None:
        calls = [{"agent": "Worker_1", "tool": "send", "executed": True}]
        trace = {
            "defense": {"components": ["authorizer"]},
            "defense_events": [
                {
                    "event": "tool_allowed",
                    "agent": "Worker_1",
                    "payload": {"tool": "send"},
                }
            ],
        }
        result = classify_calls(calls, trace)[0]
        self.assertEqual(result["authorization_outcome"], "direct_allow")
        self.assertTrue(result["autonomous_executed"])
        self.assertFalse(result["post_confirmation_effect"])

    def test_no_defense_effect_is_autonomous_execution(self) -> None:
        calls = [{"agent": "Worker_1", "tool": "send", "executed": True}]
        result = classify_calls(calls, {"defense_events": []})[0]
        self.assertEqual(result["authorization_outcome"], "no_defense")
        self.assertTrue(result["autonomous_executed"])


if __name__ == "__main__":
    unittest.main()
