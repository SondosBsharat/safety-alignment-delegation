from __future__ import annotations

import unittest

from tools import invoke_tool, tool_parameter_schema
from tracing import Recorder, use_recorder


def _case() -> dict:
    return {
        "case_id": "draft_content_case",
        "available_tools": [
            {
                "name": "draft_summary",
                "action_type": "draft",
                "object": "summary",
                "recipient": "internal",
                "description": "Draft a summary for review.",
                "effect": {"reads": ["record"], "writes": ["summary_draft"]},
            }
        ],
        "workspace_state": {
            "record": {"content": "Current record"},
            "summary_draft": {"exists": False, "content": None},
        },
    }


class ToolGatewayTests(unittest.TestCase):
    def test_draft_tool_schema_requires_content(self) -> None:
        schema = tool_parameter_schema(_case()["available_tools"][0])

        self.assertEqual(schema["required"], ["content"])
        self.assertIn("content", schema["properties"])

    def test_empty_draft_call_is_error_and_does_not_write(self) -> None:
        case = _case()
        result = invoke_tool(case, "draft_summary", {}, agent="tester")

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["code"], "missing_draft_content")
        self.assertFalse(case["workspace_state"]["summary_draft"]["exists"])

    def test_draft_with_content_writes_artifact(self) -> None:
        case = _case()
        result = invoke_tool(
            case,
            "draft_summary",
            {"content": "Summary based on current record."},
            agent="tester",
        )

        self.assertEqual(result["status"], "ok")
        self.assertTrue(case["workspace_state"]["summary_draft"]["exists"])
        self.assertEqual(
            case["workspace_state"]["summary_draft"]["content"],
            "Summary based on current record.",
        )

    def test_trace_records_actual_read_and_persisted_write_objects(self) -> None:
        case = _case()
        recorder = Recorder(case["case_id"], "single")
        with use_recorder(recorder):
            invoke_tool(
                case,
                "draft_summary",
                {"content": "Summary based on current record."},
                agent="tester",
            )

        self.assertEqual(
            recorder.tool_calls[0]["observed_effect"],
            {"reads": ["record"], "writes": ["summary_draft"]},
        )


if __name__ == "__main__":
    unittest.main()
