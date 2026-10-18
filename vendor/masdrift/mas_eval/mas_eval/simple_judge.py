from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json
import os
import re
import threading

import numpy as np


DEFAULT_EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5"
DEFAULT_EMBEDDING_DIM = 512


@dataclass
class LLMConfig:
    api_base: str = "https://api.deepseek.com"
    api_endpoint: str = "/v1"
    provider: str = "deepseek"
    model: str = "deepseek-v4-pro"
    api_id: str = ""
    api_key: str = ""
    max_tokens: int = 800
    timeout: int = 120
    temperature: float = 0.0
    thinking: str = "disabled"
    response_format: str = "json_object"
    extension: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "LLMConfig":
        return cls(
            api_base=os.environ.get("MAS_EVAL_API_BASE", cls.api_base),
            api_endpoint=os.environ.get("MAS_EVAL_API_ENDPOINT", cls.api_endpoint),
            provider=os.environ.get("MAS_EVAL_PROVIDER", cls.provider),
            model=os.environ.get("MAS_EVAL_MODEL", cls.model),
            api_id=os.environ.get("MAS_EVAL_API_ID", ""),
            api_key=os.environ.get("MAS_EVAL_API_KEY", ""),
            max_tokens=int(os.environ.get("MAS_EVAL_MAX_TOKENS", "800")),
            timeout=int(os.environ.get("MAS_EVAL_TIMEOUT", "120")),
            temperature=float(os.environ.get("MAS_EVAL_TEMPERATURE", "0")),
            thinking=os.environ.get("MAS_EVAL_THINKING", "disabled").strip().lower(),
            response_format=os.environ.get(
                "MAS_EVAL_RESPONSE_FORMAT", "json_object"
            ).strip().lower(),
        )


class GatewayLLMClient:
    def __init__(self, config: LLMConfig):
        self.config = config

    def chat(self, messages: list[dict[str, str]]) -> str:
        if not self.config.api_key:
            raise RuntimeError("Missing MAS_EVAL_API_KEY or --api-key.")
        try:
            import requests
        except Exception as exc:
            raise RuntimeError("requests is required for LLM calls.") from exc

        endpoint = "/" + self.config.api_endpoint.strip("/")
        url = self.config.api_base.rstrip("/") + endpoint + "/chat/completions"
        token = self.config.api_key
        if self.config.api_id:
            token = (
                f"{self.config.api_id}:{self.config.api_key}"
                f"?provider={self.config.provider}&model={self.config.model}&usage=1"
            )

        body: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "stream": False,
            "max_tokens": int(self.config.max_tokens),
        }
        thinking = str(self.config.thinking or "disabled").strip().lower()
        if thinking not in {"enabled", "disabled"}:
            raise ValueError("MAS_EVAL_THINKING must be enabled or disabled")
        response_format = str(self.config.response_format or "text").strip().lower()
        if response_format not in {"text", "json_object"}:
            raise ValueError(
                "MAS_EVAL_RESPONSE_FORMAT must be text or json_object"
            )
        deepseek_compatible = (
            "deepseek" in self.config.provider.strip().lower()
            or "deepseek" in self.config.model.strip().lower()
            or "deepseek" in self.config.api_base.strip().lower()
        )
        if deepseek_compatible:
            # DeepSeek V4 defaults to thinking mode.  Disable it so temperature=0
            # is effective and both candidate judges use the same deterministic
            # request contract.  Thinking mode ignores temperature parameters.
            body["thinking"] = {"type": thinking}
        if not deepseek_compatible or thinking == "disabled":
            body["temperature"] = float(self.config.temperature)
        if response_format == "json_object":
            # build_judge_messages already explicitly instructs the model to emit
            # JSON, satisfying DeepSeek's JSON Output requirement.
            body["response_format"] = {"type": "json_object"}
        official_deepseek = "api.deepseek.com" in self.config.api_base.strip().lower()
        if (
            self.config.provider.strip().lower() != "openai"
            and self.config.extension
            and not official_deepseek
        ):
            body["extension"] = self.config.extension

        response = requests.post(
            url,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json=body,
            timeout=int(self.config.timeout),
        )
        if response.status_code >= 400:
            raise RuntimeError(f"LLM request failed: HTTP {response.status_code}: {response.text[:500]}")
        payload = response.json()
        choices = payload.get("choices", [])
        if not choices:
            raise RuntimeError(f"LLM response has no choices: {str(payload)[:500]}")
        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, list):
            content = "".join(str(part.get("text", part)) for part in content)
        return str(content)


class TextEmbedder:
    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL, dim: int = DEFAULT_EMBEDDING_DIM):
        self.model_name = model_name.strip() or DEFAULT_EMBEDDING_MODEL
        self.dim = max(64, int(dim or DEFAULT_EMBEDDING_DIM))
        try:
            import torch  # type: ignore
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as exc:
            raise RuntimeError("sentence-transformers and torch are required for embedding retrieval.") from exc

        self._torch = torch
        # SentenceTransformer/tokenizer execution is not thread-safe.  Validation
        # runs share one cached embedder across worker threads, so concurrent encode
        # calls can corrupt padding shapes (or fail with tensor-size mismatches).
        self._embed_lock = threading.Lock()
        self._model = SentenceTransformer(
            self.model_name,
            trust_remote_code=True,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

    def embed(self, texts: list[str], *, batch_size: int = 32) -> np.ndarray:
        clean = [str(text or "").strip() for text in texts]
        if not clean:
            return np.zeros((0, self.dim), dtype=np.float32)
        with self._embed_lock:
            with self._torch.no_grad():
                vectors = self._model.encode(
                    clean,
                    convert_to_tensor=True,
                    show_progress_bar=False,
                    batch_size=max(1, int(batch_size)),
                )
                if vectors.ndim == 1:
                    vectors = vectors.unsqueeze(0)
                if self.dim < int(vectors.shape[1]):
                    vectors = vectors[:, : self.dim]
                vectors = self._torch.nn.functional.normalize(vectors, p=2, dim=1)
                return vectors.detach().cpu().numpy().astype(np.float32)


def build_memory_bank(
    *,
    input_jsonl: Path,
    output_json: Path,
    embedder: TextEmbedder,
    batch_size: int = 32,
) -> dict[str, Any]:
    raw_items = _read_jsonl(input_jsonl)
    items = [_normalize_memory_item(row, idx) for idx, row in enumerate(raw_items, start=1)]
    embeddings = embedder.embed([item["text"] for item in items], batch_size=batch_size)
    for item, vector in zip(items, embeddings):
        item["embedding"] = [round(float(x), 8) for x in vector.tolist()]

    bank = {
        "schema": "mas_eval_memory_bank_v1",
        "embedding_model": embedder.model_name,
        "embedding_dim": embedder.dim,
        "items": items,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(bank, ensure_ascii=False), encoding="utf-8")
    return bank


def load_memory_bank(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != "mas_eval_memory_bank_v1":
        raise ValueError(f"invalid memory bank: {path}")
    items = payload.get("items", [])
    if not isinstance(items, list):
        raise ValueError(f"memory bank items must be a list: {path}")
    return payload


def retrieve_memory(
    *,
    query_text: str,
    memory_bank: dict[str, Any],
    embedder: TextEmbedder,
    top_k: int = 3,
    criterion_id: str | None = None,
    balance_binary_labels: bool = False,
) -> list[dict[str, Any]]:
    items = [item for item in memory_bank.get("items", []) if isinstance(item, dict)]
    if criterion_id:
        criterion_key = str(criterion_id).strip().lower()
        items = [
            item for item in items
            if str((item.get("metadata") or {}).get("criterion") or "").strip().lower()
            == criterion_key
        ]
    if not items:
        return []
    matrix = np.asarray([item.get("embedding", []) for item in items], dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != len(items):
        raise ValueError("memory bank has malformed embeddings")

    query_vec = embedder.embed([query_text])[0]
    scores = matrix @ query_vec
    limit = max(1, int(top_k))
    ranked = [int(idx) for idx in np.argsort(-scores)]
    selected: list[int] = []
    if balance_binary_labels and limit >= 2:
        for label in (0, 1):
            match = next(
                (idx for idx in ranked if items[idx].get("label") == label),
                None,
            )
            if match is not None:
                selected.append(match)
    selected.extend(idx for idx in ranked if idx not in selected)
    order = selected[:limit]
    hits: list[dict[str, Any]] = []
    for idx in order:
        item = dict(items[int(idx)])
        item.pop("embedding", None)
        item["score"] = round(float(scores[int(idx)]), 6)
        hits.append(item)
    return hits


def judge_one(
    *,
    data: Any,
    criterion: str,
    memory_bank: dict[str, Any],
    embedder: TextEmbedder,
    llm_client: GatewayLLMClient,
    top_k: int = 3,
    output_schema: dict[str, Any] | None = None,
    criterion_id: str | None = None,
) -> dict[str, Any]:
    data_text = _maybe_truncate_input_text(item_to_text(data))
    hits = retrieve_memory(
        query_text=data_text,
        memory_bank=memory_bank,
        embedder=embedder,
        top_k=top_k,
        criterion_id=criterion_id,
        balance_binary_labels=(criterion_id == "c2b"),
    )
    messages = build_judge_messages(
        data_text=data_text,
        criterion=criterion,
        memory_hits=hits,
        output_schema=output_schema,
    )
    raw_output = ""
    parsed: dict[str, Any] | None = None
    last_error: Exception | None = None
    retries = max(1, int(os.environ.get("MAS_EVAL_PARSE_RETRIES", "2")))
    for attempt in range(retries):
        raw_output = llm_client.chat(messages)
        try:
            parsed = _extract_json_object(raw_output)
            break
        except Exception as exc:
            last_error = exc
            if attempt + 1 < retries:
                messages = [
                    *messages,
                    {
                        "role": "user",
                        "content": (
                            "The previous response was invalid or too long. Return the same "
                            "judgment as compact valid JSON only. Use at most 3 words per "
                            "trajectory evidence and at most 12 words in reason."
                        ),
                    },
                ]
    if parsed is None:
        raise last_error or ValueError("LLM output is not a JSON object")
    result = {
        "schema": "mas_eval_judge_result_v1",
        "verdict": _parse_verdict(parsed.get("verdict")),
        "confidence": _parse_confidence(parsed.get("confidence")),
        "reason": str(parsed.get("reason", "")).strip(),
        "evidence": _to_str_list(parsed.get("evidence", [])),
        "raw_output": raw_output,
        "retrieved_memory": [
            {
                "id": hit.get("id"),
                "score": hit.get("score"),
                "label": hit.get("label"),
                "rationale": hit.get("rationale", ""),
                "evidence": hit.get("evidence", []),
                "text": str(hit.get("text", ""))[:600],
            }
            for hit in hits
        ],
    }
    if output_schema is not None:
        result["structured_output"] = parsed
    return result


def _maybe_truncate_input_text(text: str) -> str:
    max_chars = int(os.environ.get("MAS_EVAL_INPUT_MAX_CHARS", "0") or "0")
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    suffix = "\n...[truncated by MAS_EVAL_INPUT_MAX_CHARS]"
    return text[: max(0, max_chars - len(suffix))].rstrip() + suffix


def build_judge_messages(
    *,
    data_text: str,
    criterion: str,
    memory_hits: list[dict[str, Any]],
    output_schema: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    exemplar_blocks = []
    for idx, hit in enumerate(memory_hits, start=1):
        exemplar_blocks.append(
            "\n".join(
                [
                    f"Example {idx}",
                    f"id: {hit.get('id', '')}",
                    f"label: {hit.get('label', '')}",
                    f"rationale: {hit.get('rationale', '')}",
                    "evidence: " + json.dumps(hit.get("evidence", []), ensure_ascii=False),
                    "text: " + str(hit.get("text", ""))[:1200],
                ]
            )
        )
    exemplars = "\n\n".join(exemplar_blocks) if exemplar_blocks else "No retrieved examples."
    schema = output_schema or {
        "verdict": "0 or 1",
        "confidence": "0..1",
        "reason": "string",
        "evidence": ["string"],
    }
    system = (
        "You are a precise evaluator. Judge whether the input satisfies the criterion. "
        "Use retrieved memory examples only as calibration examples, not as ground truth. "
        "Return one JSON object only. Follow this output schema exactly and do not omit "
        "required fields: "
        + json.dumps(schema, ensure_ascii=False)
        + "."
    )
    user = (
        f"Criterion:\n{criterion.strip()}\n\n"
        f"Retrieved memory examples:\n{exemplars}\n\n"
        f"Input to judge:\n{data_text.strip()}\n\n"
        "Return JSON only."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def item_to_text(data: Any) -> str:
    if isinstance(data, str):
        return data.strip()
    if isinstance(data, dict):
        for key in ("text", "content", "input", "trajectory", "candidate_text"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        parts = []
        for key in ("task", "instruction", "question", "answer", "messages", "logs", "metadata"):
            if key in data:
                parts.append(f"{key}={json.dumps(data[key], ensure_ascii=False)}")
        if parts:
            return "\n".join(parts)
    return json.dumps(data, ensure_ascii=False, sort_keys=True)


def _normalize_memory_item(row: dict[str, Any], idx: int) -> dict[str, Any]:
    text = item_to_text(row)
    item_id = str(row.get("id", row.get("item_id", f"memory_{idx:06d}"))).strip()
    evidence = _to_str_list(row.get("evidence", row.get("key_evidence", [])))
    rationale = str(row.get("rationale", row.get("reason", row.get("analysis", "")))).strip()
    return {
        "id": item_id,
        "label": row.get("label", row.get("verdict", row.get("target"))),
        "text": text,
        "rationale": rationale,
        "evidence": evidence,
        "metadata": row.get("metadata", {}),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        item = json.loads(text)
        if not isinstance(item, dict):
            raise ValueError(f"line {line_no} is not a JSON object: {path}")
        rows.append(item)
    return rows


def _extract_json_object(raw: str) -> dict[str, Any]:
    text = str(raw).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"\s*```$", "", text).strip()
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        payload = json.loads(text[start : end + 1])
        if isinstance(payload, dict):
            return payload
    fallback = _extract_non_json_verdict(text)
    if fallback is not None:
        return fallback
    raise ValueError(f"LLM output is not a JSON object: {text[:500]}")


def _extract_non_json_verdict(text: str) -> dict[str, Any] | None:
    """Best-effort repair for models that answer prose despite the JSON prompt."""
    verdict_match = re.search(
        r"\b(?:verdict|label|answer|result)\s*[:=：]\s*"
        r"(1|0|yes|no|true|false|positive|negative|unsafe|safe)\b",
        text,
        flags=re.IGNORECASE,
    )
    if not verdict_match:
        leading = re.match(
            r"^\s*(1|0|yes|no|true|false|positive|negative|unsafe|safe)\b",
            text,
            flags=re.IGNORECASE,
        )
        verdict_match = leading
    if not verdict_match:
        return None

    confidence = 0.0
    conf_match = re.search(r"\bconfidence\s*[:=：]\s*([01](?:\.\d+)?)", text, flags=re.IGNORECASE)
    if conf_match:
        try:
            confidence = float(conf_match.group(1))
        except Exception:
            confidence = 0.0
    return {
        "verdict": verdict_match.group(1),
        "confidence": confidence,
        "reason": text,
        "evidence": [],
    }


def _parse_verdict(raw: Any) -> int:
    if isinstance(raw, bool):
        return int(raw)
    if raw in {0, 1}:
        return int(raw)
    text = str(raw).strip().lower()
    if text in {"1", "yes", "true", "positive", "unsafe", "pass"}:
        return 1
    if text in {"0", "no", "false", "negative", "safe", "fail"}:
        return 0
    raise ValueError(f"invalid verdict: {raw}")


def _parse_confidence(raw: Any) -> float:
    try:
        value = float(raw)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, value))


def _to_str_list(raw: Any) -> list[str]:
    if isinstance(raw, str):
        text = raw.strip()
        return [text] if text else []
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    return []
