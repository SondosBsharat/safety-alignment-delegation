from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_model_defense_matrix.py"
SPEC = importlib.util.spec_from_file_location("run_model_defense_matrix", MODULE_PATH)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


class ModelDefenseMatrixRunnerTests(unittest.TestCase):
    def test_condition_registry_matches_canonical_nine(self) -> None:
        self.assertEqual(
            [item.name for item in runner.CONDITIONS],
            ["S1", "N2", "N4", "N8", "C1", "C2", "C3", "N4-t", "C2-t"],
        )

    def test_official_openai_env_clears_ambient_relay(self) -> None:
        args = runner.parse_args([
            "--mode", "smoke",
            "--model", "gpt-5.4-nano",
            "--key-env", "TEST_GPT_KEY",
        ])
        old = os.environ.copy()
        try:
            os.environ["TEST_GPT_KEY"] = "secret-value"
            os.environ["OPENAI_BASE_URL"] = "https://api.deepseek.com"
            env = runner.subprocess_env(args)
        finally:
            os.environ.clear()
            os.environ.update(old)
        self.assertEqual(env["OPENAI_BASE_URL"], "")
        self.assertEqual(env["OPENAI_MODEL"], "gpt-5.4-nano")
        self.assertNotIn("secret-value", " ".join(runner.build_command(
            args,
            condition=runner.CONDITIONS[4],
            defense="none",
            out_dir=Path(tempfile.gettempdir()) / "model-matrix-test",
            workers=8,
        )))

    def test_qwen_base_url_is_explicit_in_command(self) -> None:
        args = runner.parse_args([
            "--mode", "smoke",
            "--model", "qwen-test",
            "--key-env", "TEST_QWEN_KEY",
            "--base-url", "https://example.invalid/v1",
        ])
        command = runner.build_command(
            args,
            condition=runner.CONDITIONS[1],
            defense="authorizer",
            out_dir=Path(tempfile.gettempdir()) / "model-matrix-test",
            workers=4,
        )
        self.assertIn("--llm-base-url", command)
        self.assertIn("https://example.invalid/v1", command)
        self.assertEqual(runner.endpoint_origin(args.base_url), "https://example.invalid")

    def test_dataset_task_id_maps_to_trace_case_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "dataset.json"
            dataset.write_text(
                json.dumps({"tasks": [{"task_id": "case-a"}, {"task_id": "case-b"}]}),
                encoding="utf-8",
            )
            self.assertEqual(runner.expected_ids(dataset), {"case-a", "case-b"})

    def test_c3_command_can_use_sampled_dataset(self) -> None:
        args = runner.parse_args([
            "--mode", "full", "--model", "test-model", "--key-env", "TEST_KEY",
        ])
        sample = Path(tempfile.gettempdir()) / "c3-sample.json"
        command = runner.build_command(
            args,
            condition=next(item for item in runner.CONDITIONS if item.name == "C3"),
            defense="none",
            out_dir=Path(tempfile.gettempdir()) / "model-matrix-test",
            workers=8,
            dataset=sample,
        )
        self.assertEqual(command[command.index("--dataset") + 1], str(sample))

    def test_key_index_selects_one_secret_without_entering_command(self) -> None:
        args = runner.parse_args([
            "--mode", "smoke",
            "--model", "qwen3.5-plus",
            "--key-env", "TEST_QWEN_POOL",
            "--key-index", "1",
            "--base-url", "https://relay.example.com/v1",
        ])
        old = os.environ.copy()
        try:
            os.environ["TEST_QWEN_POOL"] = "secret-zero,secret-one,secret-two"
            env = runner.subprocess_env(args)
        finally:
            os.environ.clear()
            os.environ.update(old)
        self.assertEqual(env["TEST_QWEN_POOL"], "secret-one")
        command = runner.build_command(
            args,
            condition=runner.CONDITIONS[4],
            defense="none",
            out_dir=Path(tempfile.gettempdir()) / "model-matrix-test",
            workers=1,
        )
        self.assertNotIn("secret-one", " ".join(command))


if __name__ == "__main__":
    unittest.main()
