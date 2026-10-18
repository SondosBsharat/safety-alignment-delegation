from __future__ import annotations

import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END

import agent_react as agent_react_module
import centralized as centralized_module
from agent_react import (
    CENTRALIZED_TASK_BRIEF_SECTIONS,
    GatedChatModel,
    GatewayTool,
    _centralized_handoff_tool_content,
    _coerce_missing_child_reports_to_transfer,
    _coerce_pending_delegation_to_transfer,
    _reported_child_names,
    _system_prompt,
)
from centralized import (
    MID_ROLE,
    SUPERVISOR_ROLE,
    TEAM_ROLE,
    TOP_ROLE,
    UNIFORM_MGR_ROLE,
    UNIFORM_SUPERVISOR_ROLE,
    _SUPERVISOR_INTEGRATION_GUIDANCE,
    _SUPERVISOR_ROUTING_GUIDANCE,
    _WORKER_GUIDANCE,
    _structure_brief,
)
from decentralized import _ENTRY_GUIDANCE, _HANDOFF_GUIDANCE


FORBIDDEN_PROMPT_TERMS = (
    "ground_truth",
    "required_steps",
    "forbidden_actions",
    "evaluator",
    "answer key",
    "answer-key",
    "off-screen scoring",
)


class _DummyTool:
    def __init__(self, name: str) -> None:
        self.name = name


class _DummyModel:
    model_name = "unit-test-model"

    def bind_tools(self, tools, **kwargs):
        self.bound_tools = tools
        self.bound_kwargs = kwargs
        return self


def _sample_case() -> dict:
    return {
        "case_id": "unit_case",
        "user_query": "Prepare an internal summary for my review.",
        "available_tools": [
            {
                "name": "read_case_notes",
                "description": "Read the case notes.",
                "action_type": "read",
                "effect": {"reads": ["case_notes"]},
            },
            {
                "name": "draft_summary",
                "description": "Draft an internal summary.",
                "action_type": "draft",
                "effect": {"writes": ["summary_draft"]},
            },
        ],
        "workspace_state": {
            "case_notes": {"content": "Relevant facts for review."},
        },
    }


def _capture_centralized_nodes(**kwargs):
    def _node(state):
        return {}

    _capture_centralized_nodes.calls.setdefault(kwargs["name"], []).append(kwargs)
    return _node


_capture_centralized_nodes.calls = {}


class CentralizedRoutingTests(unittest.TestCase):
    def test_flat_centralized_supervisor_routes_then_integrates_with_tools(self) -> None:
        _capture_centralized_nodes.calls = {}
        with patch.object(
            centralized_module,
            "make_agent_node",
            side_effect=_capture_centralized_nodes,
        ):
            centralized_module.build_graph(
                _sample_case(),
                visibility="full",
                layers=1,
                width=2,
                personas="generic",
            )

        router, integrator = _capture_centralized_nodes.calls["Supervisor"]
        self.assertFalse(router["include_case_tools"])
        self.assertEqual(
            [target for target, _ in router["handoff_targets"]],
            ["Worker_1", "Worker_2"],
        )
        self.assertTrue(router["serialize_tools"])
        self.assertTrue(router["centralized_routing_guard"])
        self.assertTrue(router["require_all_child_reports"])
        self.assertFalse(router["record_final_handoff"])
        self.assertTrue(router["track_report_state"])

        self.assertTrue(integrator["include_case_tools"])
        self.assertEqual(integrator["handoff_targets"], ())
        self.assertIn("visible completion self-check", integrator["role_instruction"])
        self.assertTrue(integrator["track_report_state"])
        self.assertTrue(integrator["record_report_state"])
        self.assertTrue(integrator["join_chain_direct_reports"])

        for worker_name in ("Worker_1", "Worker_2"):
            worker = _capture_centralized_nodes.calls[worker_name][0]
            self.assertTrue(worker.get("include_case_tools", True))
            self.assertEqual(worker["handoff_targets"], ())
            self.assertTrue(worker["track_report_state"])
            self.assertTrue(worker["record_report_state"])

    def test_nested_centralized_supervisors_route_then_integrate_with_tools(self) -> None:
        _capture_centralized_nodes.calls = {}
        with patch.object(
            centralized_module,
            "make_agent_node",
            side_effect=_capture_centralized_nodes,
        ):
            centralized_module.build_graph(
                _sample_case(),
                visibility="full",
                layers=3,
                width=1,
                personas="generic",
            )

        for supervisor_name in ("ChiefSupervisor", "MidSupervisor_1", "TeamSupervisor_1_1"):
            router, integrator = _capture_centralized_nodes.calls[supervisor_name]
            self.assertFalse(router["include_case_tools"])
            self.assertEqual(len(router["handoff_targets"]), 1)
            self.assertTrue(router["serialize_tools"])
            self.assertTrue(router["centralized_routing_guard"])
            self.assertTrue(router["require_all_child_reports"])
            self.assertFalse(router["record_final_handoff"])
            self.assertTrue(router["track_report_state"])
            self.assertTrue(integrator["include_case_tools"])
            self.assertEqual(integrator["handoff_targets"], ())
            self.assertTrue(integrator["track_report_state"])
            self.assertTrue(integrator["record_report_state"])
            self.assertTrue(integrator["join_chain_direct_reports"])

        workers = [
            name for name in _capture_centralized_nodes.calls
            if name.startswith("Worker")
        ]
        self.assertEqual(workers, ["Worker_1_1_1"])
        worker = _capture_centralized_nodes.calls["Worker_1_1_1"][0]
        self.assertTrue(worker.get("include_case_tools", True))
        self.assertEqual(worker["handoff_targets"], ())

    def test_supervisor_router_waits_for_all_reported_children_before_integration(self) -> None:
        route = centralized_module._route_after_supervisor(
            supervisor_name="ChiefSupervisor",
            integrator_name="ChiefSupervisor__integrator",
            child_names=["TeamSupervisor_1", "TeamSupervisor_2"],
        )

        self.assertEqual(
            route({
                "reported_agents": ["TeamSupervisor_1", "TeamSupervisor_2"],
                "messages": [],
            }),
            "ChiefSupervisor__integrator",
        )
        self.assertEqual(
            route({
                "reported_agents": ["TeamSupervisor_1"],
                "messages": [
                    ToolMessage(
                        content="handoff",
                        name="transfer_to_TeamSupervisor_2",
                        tool_call_id="route-2",
                    )
                ],
            }),
            END,
        )
        self.assertEqual(
            route({
                "reported_agents": ["TeamSupervisor_1"],
                "messages": [AIMessage(content="I am ready to summarize.")],
            }),
            "ChiefSupervisor",
        )

    def test_make_agent_node_supports_workspace_and_transfer_tools_serialized(self) -> None:
        captured = {}

        def fake_create_react_agent(**kwargs):
            captured.update(kwargs)
            return kwargs

        with (
            patch.object(agent_react_module, "get_llm", return_value=_DummyModel()),
            patch.object(
                agent_react_module,
                "create_react_agent",
                side_effect=fake_create_react_agent,
            ),
        ):
            agent_react_module.make_agent_node(
                name="Supervisor",
                role_instruction="Route and integrate.",
                structure_brief="centralized unit test",
                case=_sample_case(),
                handoff_targets=(("Worker_1", "Transfer work to Worker_1."),),
                include_case_tools=True,
                serialize_tools=True,
                centralized_routing_guard=True,
            )

        tool_names = [tool.name for tool in captured["tools"]]
        self.assertIn("read_case_notes", tool_names)
        self.assertIn("draft_summary", tool_names)
        self.assertIn("transfer_to_Worker_1", tool_names)
        self.assertTrue(captured["model"].serialize_tools)
        self.assertTrue(captured["model"].centralized_routing_guard)

    def test_supervisor_workspace_tool_calls_keep_supervisor_agent_name(self) -> None:
        case = _sample_case()
        tool = GatewayTool(
            name="read_case_notes",
            description="Read the case notes.",
            case=case,
            agent_name="Supervisor",
            tool_name="read_case_notes",
        )

        with patch.object(
            agent_react_module,
            "invoke_tool",
            return_value={"status": "ok"},
        ) as invoke:
            result = tool._run(topic="summary")

        self.assertEqual(result, {"status": "ok"})
        invoke.assert_called_once_with(
            case,
            "read_case_notes",
            {"topic": "summary"},
            agent="Supervisor",
        )

    def test_centralized_handoff_content_names_active_target(self) -> None:
        content = _centralized_handoff_tool_content(
            "TeamSupervisor_1_1",
            "Transferred to TeamSupervisor_1_1.",
        )

        self.assertIn("CENTRALIZED ROUTING INSTRUCTION", content)
        self.assertIn("You are now the active agent: TeamSupervisor_1_1", content)
        self.assertIn("call exactly one transfer tool", content)
        self.assertIn("Every direct child must receive a complete packet", content)
        self.assertIn("final integration step", content)
        for section in CENTRALIZED_TASK_BRIEF_SECTIONS:
            if section in {
                "USER AUTHORIZATION BOUNDARY",
                "CURRENT AUTHORIZATION STATE",
            }:
                self.assertNotIn(section, content)
                continue
            self.assertIn(section, content)

    def test_coerced_waiting_final_carries_complete_packet(self) -> None:
        ai = AIMessage(content="I delegated the work and am waiting.")

        routed = _coerce_pending_delegation_to_transfer(
            ai,
            messages=[],
            route_tool_names=("transfer_to_ResearchWorker",),
        )

        self.assertIsNot(routed, ai)
        self.assertEqual(routed.tool_calls[0]["name"], "transfer_to_ResearchWorker")
        for section in CENTRALIZED_TASK_BRIEF_SECTIONS:
            if section in {
                "USER AUTHORIZATION BOUNDARY",
                "CURRENT AUTHORIZATION STATE",
            }:
                self.assertNotIn(section, routed.content)
                continue
            self.assertIn(section, routed.content)
        self.assertIn("complete the packet end-to-end", routed.content)
        self.assertNotIn("one-tool", routed.content.lower())

    def test_waiting_final_is_coerced_to_matching_transfer(self) -> None:
        ai = AIMessage(
            content=(
                "I delegated the work to TeamSupervisor_1_2 and am waiting for "
                "them to report back."
            )
        )

        routed = _coerce_pending_delegation_to_transfer(
            ai,
            messages=[],
            route_tool_names=(
                "transfer_to_TeamSupervisor_1_1",
                "transfer_to_TeamSupervisor_1_2",
            ),
        )

        self.assertIsNot(routed, ai)
        self.assertEqual(
            routed.tool_calls[0]["name"],
            "transfer_to_TeamSupervisor_1_2",
        )

    def test_worker_tool_result_allows_supervisor_finalization(self) -> None:
        ai = AIMessage(content="Waiting is over; here is the consolidated result.")
        messages = [
            ToolMessage(
                content="record content",
                name="read_patient_record",
                tool_call_id="tool-1",
            )
        ]

        routed = _coerce_pending_delegation_to_transfer(
            ai,
            messages=messages,
            route_tool_names=("transfer_to_TeamSupervisor_1_1",),
        )

        self.assertIs(routed, ai)

    def test_all_child_report_guard_routes_missing_child_before_integration(self) -> None:
        messages = [
            AIMessage(
                content="Routing to worker 1",
                tool_calls=[{
                    "name": "transfer_to_Worker_1",
                    "args": {},
                    "id": "route-1",
                }],
            ),
            AIMessage(content="Worker 1 report", name="Worker_1"),
        ]

        routed = _coerce_missing_child_reports_to_transfer(
            AIMessage(content="Ready to integrate."),
            messages,
            ("transfer_to_Worker_1", "transfer_to_Worker_2"),
        )

        self.assertEqual(routed.tool_calls[0]["name"], "transfer_to_Worker_2")

    def test_all_child_report_guard_allows_finalize_after_all_children_report(self) -> None:
        messages = [
            AIMessage(
                content="Routing to worker 1",
                tool_calls=[{
                    "name": "transfer_to_Worker_1",
                    "args": {},
                    "id": "route-1",
                }],
            ),
            AIMessage(content="Worker 1 report", name="Worker_1"),
            AIMessage(
                content="Routing to worker 2",
                tool_calls=[{
                    "name": "transfer_to_Worker_2",
                    "args": {},
                    "id": "route-2",
                }],
            ),
            AIMessage(content="Worker 2 report", name="Worker_2"),
        ]
        ai = AIMessage(content="All children reported; integration can begin.")

        routed = _coerce_missing_child_reports_to_transfer(
            ai,
            messages,
            ("transfer_to_Worker_1", "transfer_to_Worker_2"),
        )

        self.assertIs(routed, ai)
        self.assertEqual(
            _reported_child_names(
                messages,
                ("transfer_to_Worker_1", "transfer_to_Worker_2"),
            ),
            {"Worker_1", "Worker_2"},
        )

    def test_all_child_report_guard_suppresses_redundant_transfer(self) -> None:
        messages = [
            AIMessage(
                content="Routing to worker 1",
                tool_calls=[{
                    "name": "transfer_to_Worker_1",
                    "args": {},
                    "id": "route-1",
                }],
            ),
            AIMessage(content="Worker 1 report", name="Worker_1"),
            AIMessage(
                content="Routing to worker 2",
                tool_calls=[{
                    "name": "transfer_to_Worker_2",
                    "args": {},
                    "id": "route-2",
                }],
            ),
            AIMessage(content="Worker 2 report", name="Worker_2"),
        ]
        redundant = AIMessage(
            content="Delegating again",
            tool_calls=[{
                "name": "transfer_to_Worker_1",
                "args": {},
                "id": "repeat",
            }],
        )

        routed = _coerce_missing_child_reports_to_transfer(
            redundant,
            messages,
            ("transfer_to_Worker_1", "transfer_to_Worker_2"),
        )

        self.assertEqual(routed.tool_calls, [])
        self.assertIn("Routing is complete", routed.content)

    def test_decentralized_default_model_has_no_routing_guard(self) -> None:
        model = GatedChatModel(inner=_DummyModel(), centralized_routing_guard=False)
        bound = model.bind_tools([_DummyTool("transfer_to_PeerSkeptic")])

        self.assertFalse(bound.centralized_routing_guard)
        self.assertEqual(bound.route_tool_names, ("transfer_to_PeerSkeptic",))

    def test_direct_worker_supervisor_guidance_requires_architecture_plan(self) -> None:
        text = _system_prompt(
            "Supervisor",
            SUPERVISOR_ROLE + _SUPERVISOR_ROUTING_GUIDANCE,
            _structure_brief(1, [2]),
        )

        self.assertIn("architecture-aware plan", text)
        self.assertIn("visible request text", text)
        self.assertIn("child names and roles", text)
        self.assertIn("personas", text)
        self.assertIn("centralized layers=1, width=2", text)
        for section in CENTRALIZED_TASK_BRIEF_SECTIONS:
            self.assertIn(section, text)

    def test_nested_supervisor_guidance_preserves_packets(self) -> None:
        text = _system_prompt(
            "ChiefSupervisor",
            TOP_ROLE + _SUPERVISOR_ROUTING_GUIDANCE,
            _structure_brief(3, [2, 2, 2]),
        )
        nested_text = MID_ROLE + TEAM_ROLE + UNIFORM_MGR_ROLE + _SUPERVISOR_ROUTING_GUIDANCE

        self.assertIn("supervisor tree", text)
        self.assertIn("centralized layers=3, width=2,2,2", text)
        self.assertIn("preserve the relevant visible task goal", nested_text)
        self.assertIn("completion contract", nested_text)
        self.assertIn("Do not reduce a broad workstream to a single lookup", nested_text)

    def test_generic_flat_supervisor_gets_packet_contract(self) -> None:
        text = UNIFORM_SUPERVISOR_ROLE + _SUPERVISOR_ROUTING_GUIDANCE

        self.assertIn("complete task-brief packets", text)
        self.assertIn("actually call exactly one transfer tool", text)
        self.assertIn("Route before finalizing", text)
        self.assertIn("Every direct child must receive a complete packet", text)
        self.assertIn("final integration step", text)
        for section in CENTRALIZED_TASK_BRIEF_SECTIONS:
            self.assertIn(section, text)

    def test_supervisor_guidance_requires_integration_tool_use_after_reports(self) -> None:
        text = _system_prompt(
            "Supervisor",
            SUPERVISOR_ROLE + _SUPERVISOR_INTEGRATION_GUIDANCE,
            _structure_brief(1, [2]),
        )

        self.assertIn("All direct child branches have already reported", text)
        self.assertIn("visible completion integrator", text)
        self.assertIn("worker reports", text)
        self.assertIn("prior visible tool outputs", text)
        self.assertIn("visible tool names and descriptions", text)
        self.assertIn("call that tool before finalizing", text)
        self.assertIn("publicly described tool behavior", text)
        self.assertNotIn("answer key", text.lower())
        self.assertNotIn("off-screen scoring", text.lower())

    def test_worker_guidance_requires_workspace_tool_completion(self) -> None:
        self.assertIn("Complete the delegated task-brief packet end-to-end", _WORKER_GUIDANCE)
        self.assertIn("all relevant workspace tools", _WORKER_GUIDANCE)
        self.assertIn("do not stop after one partial lookup", _WORKER_GUIDANCE)
        self.assertIn("Stay inside the packet scope", _WORKER_GUIDANCE)

    def test_centralized_prompt_and_handoff_avoid_answer_key_terms(self) -> None:
        routed = _coerce_pending_delegation_to_transfer(
            AIMessage(content="I delegated the work and am waiting."),
            messages=[],
            route_tool_names=("transfer_to_Worker_1",),
        )
        texts = [
            _system_prompt(
                "Supervisor",
                SUPERVISOR_ROLE + _SUPERVISOR_ROUTING_GUIDANCE,
                _structure_brief(1, [2]),
            ),
            _system_prompt(
                "ChiefSupervisor",
                TOP_ROLE + TEAM_ROLE + _SUPERVISOR_ROUTING_GUIDANCE + _SUPERVISOR_INTEGRATION_GUIDANCE,
                _structure_brief(2, [2, 2]),
            ),
            _WORKER_GUIDANCE,
            _centralized_handoff_tool_content("Worker_1", "Transferred to Worker_1."),
            str(routed.content),
        ]
        combined = "\n".join(texts).lower()

        for term in FORBIDDEN_PROMPT_TERMS:
            self.assertNotIn(term, combined)

    def test_decentralized_guidance_has_no_centralized_packet_contract(self) -> None:
        text = _HANDOFF_GUIDANCE + _ENTRY_GUIDANCE

        self.assertNotIn("task-brief packet", text)
        self.assertNotIn("architecture-aware plan", text)
        self.assertNotIn("visible completion self-check", text)
        self.assertNotIn("visible completion integrator", text)
        for section in CENTRALIZED_TASK_BRIEF_SECTIONS:
            self.assertNotIn(section, text)


if __name__ == "__main__":
    unittest.main()
