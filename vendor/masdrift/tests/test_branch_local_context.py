from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import agent_react
import decentralized
import main
from defense import DefenseConfig, use_defense
from tracing import Recorder, use_recorder


def _transfer(owner: str, target: str, call_id: str, content: str) -> AIMessage:
    return AIMessage(
        name=owner,
        content=content,
        tool_calls=[{
            "name": f"transfer_to_{target}",
            "args": {},
            "id": call_id,
        }],
    )


class BranchLocalContextTests(unittest.TestCase):
    def test_no_defense_branch_context_has_no_authorization_treatment(self) -> None:
        case = {"user_query": "Prepare a draft, but do not send it."}
        with patch.object(
            agent_react,
            "extract_boundary_state",
            side_effect=AssertionError("baseline must not extract boundary state"),
        ):
            context = agent_react._branch_context_addendum(case)
            packet = agent_react._route_packet("transfer_to_Worker").content
        self.assertIn("BRANCH-LOCAL CONTEXT CONTRACT", context)
        self.assertNotIn("USER AUTHORIZATION BOUNDARY", context)
        self.assertNotIn("CURRENT AUTHORIZATION STATE", context)
        self.assertNotIn("USER AUTHORIZATION BOUNDARY", packet)
        self.assertNotIn("CURRENT AUTHORIZATION STATE", packet)

    def test_authorizer_branch_context_keeps_authorization_treatment(self) -> None:
        case = {"user_query": "Prepare a draft, but do not send it."}
        with use_defense(case, DefenseConfig(("authorizer",))):
            context = agent_react._branch_context_addendum(case)
            packet = agent_react._route_packet("transfer_to_Worker").content
        self.assertIn("USER AUTHORIZATION BOUNDARY", context)
        self.assertIn("CURRENT AUTHORIZATION STATE", context)
        self.assertIn("USER AUTHORIZATION BOUNDARY", packet)
        self.assertIn("CURRENT AUTHORIZATION STATE", packet)

    def test_no_defense_decentralized_guidance_has_no_authorization_treatment(self) -> None:
        guidance = decentralized._handoff_guidance().lower()
        self.assertNotIn("user authorization boundary", guidance)
        self.assertNotIn("current authorization state", guidance)

    def test_authorizer_decentralized_guidance_keeps_authorization_treatment(self) -> None:
        case = {"user_query": "Prepare a draft, but do not send it."}
        with use_defense(case, DefenseConfig(("authorizer",))):
            guidance = decentralized._handoff_guidance().lower()
        self.assertIn("user authorization boundary", guidance)
        self.assertIn("current authorization state", guidance)

    def test_branch_local_is_the_public_default(self) -> None:
        self.assertEqual(agent_react.DEFAULT_CONTEXT_SHARING, "branch-local")
        self.assertEqual(
            inspect.signature(main.run_one).parameters["context_sharing"].default,
            "branch-local",
        )

    def test_downstream_branch_hides_sibling_internals_and_keeps_own_tools(self) -> None:
        original = HumanMessage(content="USER REQUEST\nPrepare a summary")
        to_one = _transfer("Supervisor", "Worker_1", "route-1", "worker one packet")
        route_one = ToolMessage(
            content="AUTHORIZATION STATE one",
            name="transfer_to_Worker_1",
            tool_call_id="route-1",
        )
        one_call = AIMessage(
            name="Worker_1",
            content="private sibling reasoning",
            tool_calls=[{"name": "lookup", "args": {}, "id": "one-tool"}],
        )
        one_result = ToolMessage(
            content="private sibling tool result",
            name="lookup",
            tool_call_id="one-tool",
        )
        one_final = AIMessage(name="Worker_1", content="worker one final report")
        to_two = _transfer("Supervisor", "Worker_2", "route-2", "worker two packet")
        route_two = ToolMessage(
            content="USER AUTHORIZATION BOUNDARY two; CURRENT AUTHORIZATION STATE two",
            name="transfer_to_Worker_2",
            tool_call_id="route-2",
        )
        two_call = AIMessage(
            name="Worker_2",
            content="my current branch lookup",
            tool_calls=[{"name": "lookup", "args": {}, "id": "two-tool"}],
        )
        two_result = ToolMessage(
            content="my current branch result",
            name="lookup",
            tool_call_id="two-tool",
        )
        state = {
            "messages": [
                original,
                to_one,
                route_one,
                one_call,
                one_result,
                one_final,
                to_two,
                route_two,
                two_call,
                two_result,
            ],
            "branch_reports": {"Worker_1": "worker one final report"},
        }

        projected = agent_react._branch_local_messages(
            state,
            name="Worker_2",
            sees_query=False,
            own_tool_names=("lookup",),
        )
        text = "\n".join(str(message.content) for message in projected)
        self.assertIn("worker two packet", text)
        self.assertIn("USER AUTHORIZATION BOUNDARY two", text)
        self.assertIn("my current branch lookup", text)
        self.assertIn("my current branch result", text)
        self.assertNotIn("USER REQUEST", text)
        self.assertNotIn("private sibling reasoning", text)
        self.assertNotIn("private sibling tool result", text)
        self.assertNotIn("worker one final report", text)

    def test_supervisor_gets_direct_child_summary_not_child_internals(self) -> None:
        original = HumanMessage(content="USER REQUEST\nPrepare a summary")
        to_child = _transfer("Supervisor", "Worker_1", "route-1", "child packet")
        route_result = ToolMessage(
            content="Transferred with boundary",
            name="transfer_to_Worker_1",
            tool_call_id="route-1",
        )
        child_call = AIMessage(
            name="Worker_1",
            content="child private chain of thought",
            tool_calls=[{"name": "lookup", "args": {}, "id": "child-tool"}],
        )
        child_result = ToolMessage(
            content="child raw tool payload",
            name="lookup",
            tool_call_id="child-tool",
        )
        state = {
            "messages": [original, to_child, route_result, child_call, child_result],
            "branch_reports": {
                "Worker_1": "verified total is 42",
                "UnrelatedWorker": "must not be injected",
            },
        }

        projected = agent_react._branch_local_messages(
            state,
            name="Supervisor",
            sees_query=True,
            direct_report_names=("Worker_1",),
            own_tool_names=("transfer_to_Worker_1",),
        )
        text = "\n".join(str(message.content) for message in projected)
        self.assertIn("USER REQUEST", text)
        self.assertIn("COMPLETED STEP SUMMARY FROM Worker_1", text)
        self.assertIn("verified total is 42", text)
        self.assertNotIn("child private chain of thought", text)
        self.assertNotIn("child raw tool payload", text)
        self.assertNotIn("must not be injected", text)

        reported = agent_react._reported_child_names(
            projected,
            ("transfer_to_Worker_1",),
        )
        self.assertEqual(reported, {"Worker_1"})

    def test_post_hook_records_explicit_final_branch_report(self) -> None:
        hook = agent_react._make_post_hook(
            "Worker_1",
            record_report_state=True,
        )
        update = hook({"messages": [AIMessage(content="final evidence summary")]})
        self.assertEqual(update["reported_agents"], ["Worker_1"])
        self.assertEqual(
            update["branch_reports"],
            {"Worker_1": "final evidence summary"},
        )

    def test_projection_audit_records_metadata_without_message_content(self) -> None:
        state = {"messages": [HumanMessage(content="sensitive request text")]}
        recorder = Recorder("case", "centralized")
        with use_recorder(recorder):
            agent_react._branch_local_messages(
                state,
                name="Supervisor",
                sees_query=True,
            )

        payload = recorder.to_dict()
        self.assertEqual(len(payload["context_projections"]), 1)
        row = payload["context_projections"][0]
        self.assertEqual(row["agent"], "Supervisor")
        self.assertEqual(row["shared_message_count"], 1)
        self.assertNotIn("sensitive request text", str(row))

    def test_projection_removes_only_unpaired_tool_call_from_partial_turn(self) -> None:
        partial = AIMessage(
            name="Supervisor",
            content="attempted parallel delegation",
            tool_calls=[
                {"name": "transfer_to_Worker_1", "args": {}, "id": "done"},
                {"name": "transfer_to_Worker_2", "args": {}, "id": "missing"},
            ],
        )
        state = {
            "messages": [
                HumanMessage(content="request"),
                partial,
                ToolMessage(
                    content="only one transfer completed",
                    name="transfer_to_Worker_1",
                    tool_call_id="done",
                ),
            ]
        }
        projected = agent_react._branch_local_messages(
            state,
            name="Supervisor",
            sees_query=True,
            own_tool_names=("transfer_to_Worker_1", "transfer_to_Worker_2"),
        )
        self.assertEqual(len(projected), 3)
        self.assertEqual(
            [call["id"] for call in projected[1].tool_calls],
            ["done"],
        )
        self.assertEqual(projected[2].tool_call_id, "done")

    def test_failed_diagnostic_trace_is_not_a_resume_completion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "failed.json"
            path.write_text(json.dumps({
                "case_id": "case",
                "structure": "centralized",
                "status": "failed",
                "visibility": "scoped",
                "context_sharing": "branch-local",
            }), encoding="utf-8")
            self.assertEqual(main._completed_keys(Path(tmp)), set())

    def test_centralized_serialization_keeps_only_first_model_tool_call(self) -> None:
        message = AIMessage(
            content="delegate",
            additional_kwargs={"tool_calls": [{"raw": "must be removed"}]},
            tool_calls=[
                {"name": "transfer_to_Worker_1", "args": {}, "id": "first"},
                {"name": "transfer_to_Worker_2", "args": {}, "id": "second"},
            ],
        )
        serialized = agent_react._serialize_first_tool_call(message)
        self.assertEqual([call["id"] for call in serialized.tool_calls], ["first"])
        self.assertNotIn("tool_calls", serialized.additional_kwargs)
        self.assertEqual(len(message.tool_calls), 2)

    def test_only_immediate_exact_workspace_retry_is_suppressed(self) -> None:
        first = AIMessage(
            content="draft",
            tool_calls=[{
                "name": "draft_summary",
                "args": {"content": "same"},
                "id": "call-1",
            }],
        )
        result = ToolMessage(
            content="stored",
            name="draft_summary",
            tool_call_id="call-1",
        )
        repeat = AIMessage(
            content="draft again",
            tool_calls=[{
                "name": "draft_summary",
                "args": {"content": "same"},
                "id": "call-2",
            }],
        )
        changed = AIMessage(
            content="changed draft",
            tool_calls=[{
                "name": "draft_summary",
                "args": {"content": "changed"},
                "id": "call-3",
            }],
        )

        self.assertTrue(agent_react._repeats_last_completed_workspace_call(
            repeat, [first, result]
        ))
        self.assertFalse(agent_react._repeats_last_completed_workspace_call(
            changed, [first, result]
        ))


if __name__ == "__main__":
    unittest.main()
