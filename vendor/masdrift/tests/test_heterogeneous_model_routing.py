from __future__ import annotations

import os
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock, patch

import centralized
import decentralized
from llm import (
    apply_role_profile,
    get_agent_model_routing_metadata,
    use_agent_model_routes,
)
from main import _configure_worker_llm
from tracing import Recorder, record_token_usage, use_model_role, use_recorder


def _args(**overrides):
    values = {
        "llm_provider": None,
        "llm_model": None,
        "llm_base_url": None,
        "llm_api_keys_env": None,
    }
    for role in ("lead", "delegate"):
        for field in (
            "provider", "model", "base_url", "api_keys_env", "reasoning_effort"
        ):
            values[f"{role}_llm_{field}"] = None
    values.update(overrides)
    return SimpleNamespace(**values)


def _case():
    return {
        "case_id": "routing_case",
        "user_query": "Prepare a review.",
        "available_tools": [],
        "workspace_state": {},
    }


class HeterogeneousModelRoutingTests(unittest.TestCase):
    def test_role_tokens_are_additive_to_worker_total(self):
        recorder = Recorder("case", "decentralized")
        with use_recorder(recorder):
            with use_model_role("lead_agent"):
                record_token_usage(
                    input_tokens=10, output_tokens=2, total_tokens=12
                )
            with use_model_role("delegated_agent"):
                record_token_usage(
                    input_tokens=7, output_tokens=3, total_tokens=10
                )
        trace = recorder.to_dict()
        self.assertEqual(trace["token_stats"]["worker"]["total_tokens"], 22)
        self.assertEqual(
            trace["agent_role_token_stats"]["lead_agent"]["total_tokens"], 12
        )
        self.assertEqual(
            trace["agent_role_token_stats"]["delegated_agent"]["total_tokens"], 10
        )

    def test_concurrent_role_routes_do_not_cross_runs(self):
        barrier = __import__("threading").Barrier(2)

        def observe(label):
            routes = {
                "lead_agent": {"model": f"lead-{label}", "api_key": f"secret-{label}"},
                "delegated_agent": {"model": f"delegate-{label}", "api_key": f"secret-{label}"},
            }
            with use_agent_model_routes(routes):
                barrier.wait()
                return get_agent_model_routing_metadata()

        with patch.dict(os.environ, {"MAS_LLM_PROVIDER": "openai"}, clear=True):
            with ThreadPoolExecutor(max_workers=2) as pool:
                left, right = list(pool.map(observe, ("a", "b")))
        self.assertEqual(left["profiles"]["lead_agent"]["model"], "lead-a")
        self.assertEqual(right["profiles"]["lead_agent"]["model"], "lead-b")
        self.assertNotIn("secret-", repr((left, right)))

    def test_role_cli_inherits_transport_and_keeps_models_separate(self):
        env = {"RELAY_KEYS": "key-a,key-b"}
        _configure_worker_llm(
            _args(
                llm_provider="openai",
                llm_base_url="https://relay.example.com",
                llm_api_keys_env="RELAY_KEYS",
                lead_llm_model="gpt-5.6-sol",
                delegate_llm_model="gpt-5.4-nano",
            ),
            env,
        )
        self.assertEqual(env["OPENAI_BASE_URL"], "https://relay.example.com/v1")
        self.assertEqual(env["MAS_LEAD_LLM_MODEL"], "gpt-5.6-sol")
        self.assertEqual(env["MAS_DELEGATE_LLM_MODEL"], "gpt-5.4-nano")
        self.assertNotIn("MAS_LEAD_LLM_API_KEYS", env)

    def test_active_route_metadata_is_secret_free(self):
        with patch.dict(
            os.environ,
            {
                "MAS_LLM_PROVIDER": "openai",
                "OPENAI_MODEL": "uniform",
                "OPENAI_BASE_URL": "https://relay.example.com/v1",
                "MAS_LEAD_LLM_MODEL": "gpt-5.6-sol",
                "MAS_DELEGATE_LLM_MODEL": "gpt-5.4-nano",
            },
            clear=True,
        ):
            routes = {
                "lead_agent": apply_role_profile({"api_key": "secret-a"}, "lead_agent"),
                "delegated_agent": apply_role_profile(
                    {"api_key": "secret-b"}, "delegated_agent"
                ),
            }
            with use_agent_model_routes(routes):
                metadata = get_agent_model_routing_metadata()
        rendered = repr(metadata)
        self.assertEqual(metadata["profiles"]["lead_agent"]["model"], "gpt-5.6-sol")
        self.assertEqual(
            metadata["profiles"]["delegated_agent"]["model"], "gpt-5.4-nano"
        )
        self.assertNotIn("secret-a", rendered)
        self.assertNotIn("secret-b", rendered)

    def test_decentralized_entry_peer_is_lead(self):
        calls = []

        def capture(**kwargs):
            calls.append(kwargs)
            return lambda state: {}

        with patch.object(decentralized, "make_agent_node", side_effect=capture):
            decentralized.build_graph(
                _case(), width=4, personas="generic", visibility="full"
            )
        roles = {call["name"]: call["model_role"] for call in calls}
        self.assertEqual(roles["Peer1"], "lead_agent")
        self.assertEqual(
            {roles[f"Peer{i}"] for i in (2, 3, 4)}, {"delegated_agent"}
        )

    def test_typed_entry_peer_is_lead(self):
        calls = []

        def capture(**kwargs):
            calls.append(kwargs)
            return lambda state: {}

        with patch.object(decentralized, "make_agent_node", side_effect=capture):
            decentralized.build_graph(
                _case(), width=4, personas="typed", visibility="full"
            )
        roles = {call["name"]: call["model_role"] for call in calls}
        self.assertEqual(roles["PeerAnalyst"], "lead_agent")
        self.assertTrue(
            all(role == "delegated_agent" for name, role in roles.items()
                if name != "PeerAnalyst")
        )

    def test_only_root_c3_supervisor_is_lead(self):
        calls = []

        def capture(**kwargs):
            calls.append(kwargs)
            return lambda state: {}

        with patch.object(centralized, "make_agent_node", side_effect=capture):
            centralized.build_graph(
                _case(), layers=3, width=2, personas="generic", visibility="full"
            )
        by_name = {}
        for call in calls:
            by_name.setdefault(call["name"], []).append(call["model_role"])
        self.assertEqual(set(by_name["ChiefSupervisor"]), {"lead_agent"})
        for name, roles in by_name.items():
            if name != "ChiefSupervisor":
                self.assertEqual(set(roles), {"delegated_agent"}, name)

if __name__ == "__main__":
    unittest.main()
