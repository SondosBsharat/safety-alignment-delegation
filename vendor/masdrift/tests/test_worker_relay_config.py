import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from llm import get_endpoint_pool, get_llm, use_endpoint
from main import (
    _active_worker_llm_metadata,
    _apply_worker_pool_overrides,
    _configure_worker_llm,
    _explicit_worker_llm_resume_key,
    _normalize_openai_base_url,
)


def _args(**overrides):
    values = {
        "llm_provider": None,
        "llm_model": None,
        "llm_base_url": None,
        "llm_api_keys_env": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class WorkerRelayConfigurationTests(unittest.TestCase):
    def test_normalizes_origin_and_preserves_v1_root(self):
        self.assertEqual(
            _normalize_openai_base_url("https://relay.example.com"),
            "https://relay.example.com/v1",
        )
        self.assertEqual(
            _normalize_openai_base_url("https://relay.example.com/v1/"),
            "https://relay.example.com/v1",
        )

    def test_rejects_full_chat_completions_endpoint(self):
        with self.assertRaisesRegex(ValueError, "API root"):
            _normalize_openai_base_url("https://relay.example.com/v1/chat/completions")

    def test_omitted_flags_preserve_existing_environment(self):
        env = {
            "MAS_LLM_PROVIDER": "openai",
            "OPENAI_MODEL": "existing-model",
            "OPENAI_BASE_URL": "https://existing.example/v1",
            "MAS_LLM_POOL": "[]",
            "MAS_EVAL_MODEL": "deepseek-v4-pro",
        }
        before = dict(env)
        _configure_worker_llm(_args(), env)
        self.assertEqual(env, before)

    def test_explicit_worker_settings_do_not_modify_judge_settings(self):
        env = {
            "RELAY_GPT_KEYS": "secret-a,secret-b",
            "MAS_EVAL_PROVIDER": "openai",
            "MAS_EVAL_MODEL": "deepseek-v4-pro",
            "MAS_EVAL_BASE_URL": "https://judge.example/v1",
            "MAS_EVAL_API_KEYS": "judge-secret",
        }
        judge_before = {k: v for k, v in env.items() if k.startswith("MAS_EVAL_")}
        _configure_worker_llm(
            _args(
                llm_provider="openai",
                llm_model="gpt-5.4-mini",
                llm_base_url="https://relay.example.com",
                llm_api_keys_env="RELAY_GPT_KEYS",
            ),
            env,
        )
        self.assertEqual(env["MAS_LLM_PROVIDER"], "openai")
        self.assertEqual(env["OPENAI_MODEL"], "gpt-5.4-mini")
        self.assertEqual(env["OPENAI_BASE_URL"], "https://relay.example.com/v1")
        self.assertEqual(env["OPENAI_API_KEYS"], "secret-a,secret-b")
        self.assertEqual(
            {k: v for k, v in env.items() if k.startswith("MAS_EVAL_")},
            judge_before,
        )

    def test_missing_named_secret_fails_without_a_key_value(self):
        with self.assertRaisesRegex(ValueError, "MISSING_RELAY_KEYS"):
            _configure_worker_llm(
                _args(llm_provider="openai", llm_api_keys_env="MISSING_RELAY_KEYS"),
                {},
            )

    def test_named_key_pool_uses_existing_round_robin_contract(self):
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEYS": "key-a,key-b",
                "OPENAI_BASE_URL": "https://relay.example/v1",
                "OPENAI_MODEL": "old-model",
            },
            clear=True,
        ):
            pool = get_endpoint_pool()
        self.assertEqual([item["api_key"] for item in pool], ["key-a", "key-b"])
        overridden = _apply_worker_pool_overrides(
            pool, model="new-model", base_url="https://relay.example.com"
        )
        self.assertEqual(
            [item["model"] for item in overridden], ["new-model", "new-model"]
        )
        self.assertEqual(
            [item["base_url"] for item in overridden],
            ["https://relay.example.com/v1", "https://relay.example.com/v1"],
        )

    def test_official_google_model_and_key_pool_mapping(self):
        env = {"OFFICIAL_KEYS": "secret-a, secret-b"}
        _configure_worker_llm(
            _args(
                llm_provider="google",
                llm_model="official-model",
                llm_api_keys_env="OFFICIAL_KEYS",
            ),
            env,
        )
        self.assertEqual(env["MAS_LLM_PROVIDER"], "google")
        self.assertEqual(env["GEMINI_MODEL"], "official-model")
        self.assertNotIn("OPENAI_API_KEYS", env)
        pool = __import__("json").loads(env["MAS_LLM_POOL"])
        self.assertEqual(
            [item["api_key"] for item in pool], ["secret-a", "secret-b"]
        )

    def test_native_provider_rejects_openai_base_url(self):
        with self.assertRaisesRegex(ValueError, "native SDK endpoint"):
            _configure_worker_llm(
                _args(
                    llm_provider="google",
                    llm_base_url="https://relay.example.com/v1",
                ),
                {},
            )

    def test_native_clients_consume_pinned_endpoint_key_and_model(self):
        class FakeChatModel:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        fake_module = SimpleNamespace(ChatGoogleGenerativeAI=FakeChatModel)
        with (
            patch.dict(sys.modules, {"langchain_google_genai": fake_module}),
            patch.dict(os.environ, {"MAS_LLM_PROVIDER": "google"}, clear=True),
            use_endpoint({"api_key": "pinned-secret", "model": "pinned-model"}),
        ):
            client = get_llm()
        self.assertEqual(client.kwargs["google_api_key"], "pinned-secret")
        self.assertEqual(client.kwargs["model"], "pinned-model")

    def test_google_official_single_key_uses_gemini_api_key(self):
        class FakeGoogleModel:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        fake_module = SimpleNamespace(ChatGoogleGenerativeAI=FakeGoogleModel)
        with (
            patch.dict(sys.modules, {"langchain_google_genai": fake_module}),
            patch.dict(
                os.environ,
                {
                    "MAS_LLM_PROVIDER": "google",
                    "GEMINI_API_KEY": "gemini-secret",
                    "GEMINI_MODEL": "gemini-model",
                },
                clear=True,
            ),
        ):
            client = get_llm()
        self.assertEqual(client.kwargs["google_api_key"], "gemini-secret")
        self.assertEqual(client.kwargs["model"], "gemini-model")

    def test_trace_metadata_and_resume_key_exclude_secrets(self):
        args = _args(
            llm_provider="openai",
            llm_model="gpt-5.4-mini",
            llm_base_url="https://relay.example.com",
            llm_api_keys_env="RELAY_KEYS",
        )
        with patch.dict(
            os.environ,
            {
                "MAS_LLM_PROVIDER": "openai",
                "OPENAI_MODEL": "gpt-5.4-mini",
                "OPENAI_BASE_URL": "https://relay.example.com/v1",
                "OPENAI_API_KEYS": "secret-a,secret-b",
            },
            clear=True,
        ):
            metadata = _active_worker_llm_metadata()
            resume_key = _explicit_worker_llm_resume_key(args)
        self.assertEqual(
            metadata,
            {
                "provider": "openai",
                "model": "gpt-5.4-mini",
                "base_url": "https://relay.example.com/v1",
            },
        )
        self.assertNotIn("secret", resume_key)

    def test_no_cli_flags_preserve_legacy_resume_key(self):
        self.assertIsNone(_explicit_worker_llm_resume_key(_args()))


if __name__ == "__main__":
    unittest.main()
