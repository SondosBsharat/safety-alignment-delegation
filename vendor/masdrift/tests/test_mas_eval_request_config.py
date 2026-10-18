from __future__ import annotations

import sys
import threading
import time
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path("mas_eval").resolve()))
from mas_eval.simple_judge import GatewayLLMClient, LLMConfig, TextEmbedder  # noqa: E402


class _Response:
    status_code = 200
    text = ""

    @staticmethod
    def json():
        return {"choices": [{"message": {"content": '{"verdict":0}'}}]}


class MasEvalRequestConfigTests(unittest.TestCase):
    def test_deepseek_request_is_non_thinking_deterministic_json(self) -> None:
        captured = {}

        def post(url, **kwargs):
            captured["url"] = url
            captured.update(kwargs)
            return _Response()

        fake_requests = types.SimpleNamespace(post=post)
        config = LLMConfig(
            api_base="https://api.deepseek.com",
            api_endpoint="/v1",
            provider="deepseek",
            model="deepseek-v4-flash",
            api_key="test-only",
        )
        with patch.dict(sys.modules, {"requests": fake_requests}):
            output = GatewayLLMClient(config).chat(
                [{"role": "user", "content": "Return JSON."}]
            )

        self.assertEqual(output, '{"verdict":0}')
        body = captured["json"]
        self.assertEqual(body["thinking"], {"type": "disabled"})
        self.assertEqual(body["temperature"], 0.0)
        self.assertEqual(body["response_format"], {"type": "json_object"})
        self.assertNotIn("extension", body)

    def test_embedder_serializes_shared_model_encode_calls(self) -> None:
        import torch

        class DetectConcurrentModel:
            def __init__(self) -> None:
                self.active = 0
                self.guard = threading.Lock()

            def encode(self, texts, **kwargs):
                with self.guard:
                    self.active += 1
                    if self.active > 1:
                        raise RuntimeError("concurrent encode detected")
                try:
                    time.sleep(0.01)
                    return torch.ones((len(texts), 4), dtype=torch.float32)
                finally:
                    with self.guard:
                        self.active -= 1

        embedder = object.__new__(TextEmbedder)
        embedder.model_name = "fake"
        embedder.dim = 4
        embedder._torch = torch
        embedder._embed_lock = threading.Lock()
        embedder._model = DetectConcurrentModel()
        with ThreadPoolExecutor(max_workers=8) as executor:
            vectors = list(executor.map(lambda i: embedder.embed([str(i)]), range(16)))

        self.assertEqual(len(vectors), 16)
        self.assertTrue(all(vector.shape == (1, 4) for vector in vectors))


if __name__ == "__main__":
    unittest.main()
