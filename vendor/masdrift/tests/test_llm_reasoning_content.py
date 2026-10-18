from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

from llm import (
    _strip_openai_response_state_ids,
    _strip_reasoning_content,
    _with_thinking_disabled_if_needed,
)


class LlmReasoningContentTests(unittest.TestCase):
    def test_auto_disables_deepseek_thinking_mode(self) -> None:
        with patch.dict(os.environ, {"MAS_LLM_DISABLE_THINKING": "auto"}):
            kwargs = _with_thinking_disabled_if_needed(
                {},
                "deepseek-v4-flash",
                "https://api.deepseek.com",
            )

        self.assertEqual(kwargs["extra_body"]["thinking"]["type"], "disabled")

    def test_auto_does_not_add_thinking_body_for_plain_openai(self) -> None:
        with patch.dict(os.environ, {"MAS_LLM_DISABLE_THINKING": "auto"}):
            kwargs = _with_thinking_disabled_if_needed(
                {},
                "gpt-4o-mini",
                None,
            )

        self.assertNotIn("extra_body", kwargs)

    def test_env_false_allows_explicit_thinking_mode(self) -> None:
        with patch.dict(os.environ, {"MAS_LLM_DISABLE_THINKING": "0"}):
            kwargs = _with_thinking_disabled_if_needed(
                {"extra_body": {"thinking": {"type": "enabled"}}},
                "deepseek-v4-flash",
                "https://api.deepseek.com",
            )

        self.assertEqual(kwargs["extra_body"]["thinking"]["type"], "enabled")

    def test_strip_reasoning_content_keeps_visible_message_and_tool_call(self) -> None:
        messages = [
            HumanMessage(content="visible request"),
            AIMessage(
                content=[
                    {"type": "reasoning", "reasoning_content": "hidden chain"},
                    {"type": "text", "text": "visible answer"},
                ],
                additional_kwargs={
                    "reasoning_content": "hidden chain",
                    "tool_calls": [{"id": "call-1"}],
                },
                tool_calls=[
                    {
                        "name": "read_workspace",
                        "args": {},
                        "id": "call-1",
                    }
                ],
            ),
        ]

        cleaned = _strip_reasoning_content(messages)

        self.assertIs(cleaned[0], messages[0])
        self.assertNotIn("reasoning_content", cleaned[1].additional_kwargs)
        self.assertEqual(cleaned[1].additional_kwargs["tool_calls"], [{"id": "call-1"}])
        self.assertEqual(cleaned[1].content, [{"type": "text", "text": "visible answer"}])
        self.assertEqual(cleaned[1].tool_calls[0]["name"], "read_workspace")

    def test_strip_openai_response_state_keeps_call_pairing(self) -> None:
        original = AIMessage(
            id="msg_azure_a",
            content=[
                {"type": "reasoning", "id": "rs_azure_a", "summary": []},
                {"type": "text", "id": "msg_azure_a", "text": "handoff"},
                {
                    "type": "function_call",
                    "id": "fc_azure_a",
                    "call_id": "call-1",
                    "name": "read_workspace",
                    "arguments": "{}",
                },
            ],
            additional_kwargs={
                "reasoning": {"type": "reasoning", "id": "rs_azure_a"},
                "__openai_function_call_ids__": {"call-1": "fc_azure_a"},
            },
            response_metadata={"id": "resp_azure_a", "model": "gpt-5.4"},
            tool_calls=[{"name": "read_workspace", "args": {}, "id": "call-1"}],
        )

        cleaned = _strip_openai_response_state_ids([original])[0]

        self.assertIsNone(cleaned.id)
        self.assertNotIn("id", cleaned.response_metadata)
        self.assertNotIn("reasoning", cleaned.additional_kwargs)
        self.assertNotIn("__openai_function_call_ids__", cleaned.additional_kwargs)
        self.assertEqual(cleaned.tool_calls[0]["id"], "call-1")
        function_block = next(
            block for block in cleaned.content if block["type"] == "function_call"
        )
        self.assertNotIn("id", function_block)
        self.assertEqual(function_block["call_id"], "call-1")


if __name__ == "__main__":
    unittest.main()
