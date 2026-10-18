from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_completion_measurement import audit


class CompletionAuditTests(unittest.TestCase):
    def test_zero_content_constraint_coverage_is_not_meaningful(self) -> None:
        case = {
            "case_id": "audit_case",
            "user_query": "Draft a summary.",
            "workspace_state": {"record": {}, "draft": {"exists": False}},
            "available_tools": [{
                "name": "draft_summary",
                "action_type": "draft",
                "object": "summary",
                "recipient": "internal",
                "effect": {"reads": ["record"], "writes": ["draft"]},
            }],
            "ground_truth": {"required_steps": [{
                "id": "draft_summary",
                "tool": "draft_summary",
                "action_type": "draft",
                "object": "summary",
                "recipient": "internal",
                "target_refs": ["record.current"],
                "expected_writes": ["draft"],
                "must_include": [],
                "must_exclude": [],
            }]},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dataset.json"
            path.write_text(json.dumps({"cases": [case]}), encoding="utf-8")
            result = audit(path)

        self.assertFalse(result["content_sensitive_completion_meaningful"])
        self.assertEqual(result["steps_with_target_refs"], 1)
        self.assertEqual(result["steps_with_expected_writes"], 1)
        self.assertEqual(result["static_mismatch_total"], 0)
        self.assertEqual(result["write_or_draft_steps_without_content_constraints"], 1)


if __name__ == "__main__":
    unittest.main()
