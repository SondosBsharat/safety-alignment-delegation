"""Unified LLM provider factory.

默认 OpenAI；也支持 DeepSeek 和 Gemini provider。
"""
from __future__ import annotations

import contextvars
import copy
import json
import os
import re
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Literal

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

# 从项目根的 .env 读取 API key 等（不覆盖已存在的环境变量，
# 所以交互式终端里 .zshrc 的值仍优先）。锚定到本文件所在目录，
# 与当前工作目录无关。
load_dotenv(Path(__file__).resolve().parent / ".env")

ProviderName = Literal["openai", "deepseek", "google"]
AgentModelRole = Literal["lead_agent", "delegated_agent"]

_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off"}

def _is_deepseek_compatible(model_name: str | None, base_url: str | None) -> bool:
    model = (model_name or "").lower()
    base = (base_url or "").lower()
    return "deepseek" in model or "deepseek" in base


def _should_disable_thinking(model_name: str | None, base_url: str | None) -> bool:
    """Disable DeepSeek thinking mode for tool-using MAS agents by default.

    DeepSeek V4 thinking mode returns `reasoning_content` and, when tool calls are
    involved, expects that field to be replayed in later requests. This benchmark
    treats each agent activation as a visible-message contract; hidden reasoning
    must not become shared MAS state. `auto` disables thinking only for
    DeepSeek-compatible OpenAI endpoints. Set MAS_LLM_DISABLE_THINKING=0 to opt out.
    """
    raw = os.environ.get("MAS_LLM_DISABLE_THINKING", "auto").strip().lower()
    if raw in _FALSY:
        return False
    if raw in _TRUTHY:
        return True
    return _is_deepseek_compatible(model_name, base_url)


def _with_thinking_disabled_if_needed(
    kwargs: dict[str, Any],
    model_name: str | None,
    base_url: str | None,
) -> dict[str, Any]:
    kwargs = dict(kwargs)
    if not _should_disable_thinking(model_name, base_url):
        return kwargs
    extra_body = dict(kwargs.get("extra_body") or {})
    thinking = dict(extra_body.get("thinking") or {})
    thinking["type"] = "disabled"
    extra_body["thinking"] = thinking
    kwargs["extra_body"] = extra_body
    return kwargs


def _drop_reasoning_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            k: _drop_reasoning_payload(v)
            for k, v in value.items()
            if k not in {"reasoning_content", "reasoning"}
        }
    if isinstance(value, list):
        out = []
        for item in value:
            if isinstance(item, dict) and item.get("type") in {"reasoning", "thinking"}:
                continue
            out.append(_drop_reasoning_payload(item))
        return out
    return value


def _bound_model_name(llm: Any) -> str:
    """Best-effort model lookup through LangChain RunnableBinding wrappers."""
    current = llm
    seen: set[int] = set()
    for _ in range(6):
        if current is None or id(current) in seen:
            break
        seen.add(id(current))
        name = getattr(current, "model_name", None) or getattr(current, "model", None)
        if isinstance(name, str) and name:
            return name
        current = getattr(current, "bound", None) or getattr(current, "first", None)
    return ""


def _drop_openai_response_item_ids(value: Any) -> Any:
    """Remove server-owned Responses item IDs while preserving function call IDs.

    A relay may route two requests made with the same key to different Azure OpenAI
    resources. Replaying ``msg_*``, ``rs_*`` or ``fc_*`` item IDs then fails with
    "item was created under a different ... resource". ``call_id`` remains intact,
    so ToolMessage/function-call pairing is unchanged.
    """
    if isinstance(value, dict):
        block_type = str(value.get("type") or "")
        if block_type in {"reasoning", "thinking"}:
            return None
        return {
            k: cleaned
            for k, v in value.items()
            if k != "id"
            for cleaned in [_drop_openai_response_item_ids(v)]
            if cleaned is not None
        }
    if isinstance(value, list):
        return [
            cleaned
            for item in value
            for cleaned in [_drop_openai_response_item_ids(item)]
            if cleaned is not None
        ]
    return value


def _strip_openai_response_state_ids(messages):
    """Make GPT Responses history portable across stateless relay backends.

    Model providers and OpenAI-compatible relays can return different JSON
    shapes, metadata keys, and identifier prefixes. Adjust this normalization
    to match the provider and response format used by your deployment.
    """
    cleaned = []
    for message in messages or []:
        if not isinstance(message, AIMessage):
            cleaned.append(message)
            continue
        updates: dict[str, Any] = {}
        additional = dict(getattr(message, "additional_kwargs", None) or {})
        for key in ("__openai_function_call_ids__", "reasoning", "tool_outputs"):
            additional.pop(key, None)
        if additional != (getattr(message, "additional_kwargs", None) or {}):
            updates["additional_kwargs"] = additional
        content = getattr(message, "content", None)
        new_content = _drop_openai_response_item_ids(content)
        if new_content != content:
            updates["content"] = new_content
        message_id = getattr(message, "id", None)
        if isinstance(message_id, str) and message_id.startswith(("msg_", "resp_")):
            updates["id"] = None
        metadata = dict(getattr(message, "response_metadata", None) or {})
        if isinstance(metadata.get("id"), str) and metadata["id"].startswith("resp_"):
            metadata.pop("id", None)
            updates["response_metadata"] = metadata
        if updates:
            if hasattr(message, "model_copy"):
                cleaned.append(message.model_copy(deep=True, update=updates))
            else:
                clone = copy.copy(message)
                for key, value in updates.items():
                    setattr(clone, key, value)
                cleaned.append(clone)
        else:
            cleaned.append(message)
    return cleaned


def _strip_reasoning_content(messages):
    """Return message copies with provider-only thinking payloads removed.

    This keeps chain-of-thought out of agent state and avoids replaying a partial
    DeepSeek thinking-mode transcript through LangGraph's shared message channel.
    """
    cleaned = []
    for m in messages or []:
        if not isinstance(m, AIMessage):
            cleaned.append(m)
            continue
        updates: dict[str, Any] = {}
        additional = getattr(m, "additional_kwargs", None) or {}
        new_additional = _drop_reasoning_payload(additional)
        if new_additional != additional:
            updates["additional_kwargs"] = new_additional
        content = getattr(m, "content", None)
        new_content = _drop_reasoning_payload(content)
        if new_content != content:
            updates["content"] = new_content
        if updates:
            if hasattr(m, "model_copy"):
                cleaned.append(m.model_copy(deep=True, update=updates))
            else:
                clone = copy.copy(m)
                for key, value in updates.items():
                    setattr(clone, key, value)
                cleaned.append(clone)
        else:
            cleaned.append(m)
    return cleaned


# === Endpoint pool: parallel runs round-robin across several keys / relays =====
# For concurrent batch runs (main.py --workers N) each run can be pinned to one
# endpoint (api_key [+ base_url + model]) so several keys / relays share the load.
# An "endpoint" is a dict {api_key, base_url?, model?}; missing fields fall back to
# the selected provider's env defaults. The active endpoint for the current run lives in a
# ContextVar (the same out-of-band pattern as the tracer), so it is thread-isolated:
# each worker thread sets its own once per run and every get_llm() call in that run
# picks it up. With no pool configured the ContextVar stays None and behaviour is
# identical to before (single OPENAI_* endpoint).
#
# Configure either form in .env / the environment:
#   MAS_LLM_POOL  — JSON list, most expressive (per-entry key + relay + model), e.g.
#     MAS_LLM_POOL='[{"api_key":"sk-a","base_url":"https://relay-a/v1"},
#                    {"api_key":"sk-b","base_url":"https://relay-b/v1","model":"deepseek-v4-flash"}]'
#   OPENAI_API_KEYS — comma-separated keys sharing OPENAI_BASE_URL / OPENAI_MODEL.

_endpoint: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "mas_llm_endpoint", default=None
)
_agent_model_routes: contextvars.ContextVar[dict[str, dict[str, Any]] | None] = (
    contextvars.ContextVar("mas_agent_model_routes", default=None)
)


def _role_prefix(role: AgentModelRole) -> str:
    return "MAS_LEAD_LLM" if role == "lead_agent" else "MAS_DELEGATE_LLM"


def _role_env(role: AgentModelRole, field: str) -> str | None:
    return os.environ.get(f"{_role_prefix(role)}_{field}")


def role_profiles_configured() -> bool:
    return any(
        os.environ.get(f"{prefix}_{field}")
        for prefix in ("MAS_LEAD_LLM", "MAS_DELEGATE_LLM")
        for field in ("PROVIDER", "MODEL", "BASE_URL", "API_KEYS", "REASONING_EFFORT")
    )


def get_role_endpoint_pool(role: AgentModelRole) -> list[dict[str, Any]]:
    """Return the optional secret-bearing endpoint pool for one logical model role."""
    keys = _role_env(role, "API_KEYS")
    if not keys:
        return []
    return [
        {
            "api_key": key.strip(),
            "provider": _role_env(role, "PROVIDER"),
            "model": _role_env(role, "MODEL"),
            "base_url": _role_env(role, "BASE_URL"),
            "reasoning_effort": _role_env(role, "REASONING_EFFORT"),
        }
        for key in keys.split(",") if key.strip()
    ]


def apply_role_profile(
    endpoint: dict[str, Any] | None, role: AgentModelRole
) -> dict[str, Any]:
    """Overlay a role's non-secret config on a per-run credential endpoint."""
    item = dict(endpoint or {})
    for field, env_field in (
        ("provider", "PROVIDER"),
        ("model", "MODEL"),
        ("base_url", "BASE_URL"),
        ("reasoning_effort", "REASONING_EFFORT"),
    ):
        value = _role_env(role, env_field)
        if value:
            item[field] = value
    return item


@contextmanager
def use_agent_model_routes(
    routes: dict[str, dict[str, Any]] | None,
) -> Iterator[None]:
    """Pin lead/delegate model endpoints for one concurrent benchmark run."""
    tok = _agent_model_routes.set(routes)
    try:
        yield
    finally:
        _agent_model_routes.reset(tok)


def get_endpoint_pool() -> list[dict[str, Any]]:
    """Provider-neutral endpoints to round-robin over (empty => one env endpoint)."""
    raw = os.environ.get("MAS_LLM_POOL")
    if raw:
        pool = json.loads(raw)
        if not isinstance(pool, list):
            raise ValueError("MAS_LLM_POOL must be a JSON list of endpoint objects")
        return pool
    keys = os.environ.get("OPENAI_API_KEYS")
    if keys:
        base = os.environ.get("OPENAI_BASE_URL")
        model = os.environ.get("OPENAI_MODEL")
        return [
            {"api_key": k.strip(), "base_url": base, "model": model}
            for k in keys.split(",") if k.strip()
        ]
    return []


@contextmanager
def use_endpoint(endpoint: dict[str, Any] | None) -> Iterator[None]:
    """Pin every get_llm() call in this context to `endpoint` (None => env default)."""
    tok = _endpoint.set(endpoint)
    try:
        yield
    finally:
        _endpoint.reset(tok)


def get_active_endpoint(model_role: AgentModelRole | None = None) -> dict[str, Any]:
    """Return the endpoint currently pinned by use_endpoint(), if any."""
    routes = _agent_model_routes.get() or {}
    if model_role and model_role in routes:
        return routes[model_role]
    return _endpoint.get() or {}


def get_agent_model_routing_metadata() -> dict[str, Any] | None:
    """Secret-free configured/active role provenance for traces and resume keys."""
    if not role_profiles_configured() and not _agent_model_routes.get():
        return None
    uniform_provider = os.environ.get("MAS_LLM_PROVIDER", "openai")
    uniform_model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    uniform_base = os.environ.get("OPENAI_BASE_URL")
    profiles: dict[str, dict[str, Any]] = {}
    for role in ("lead_agent", "delegated_agent"):
        ep = get_active_endpoint(role) or {}
        profiles[role] = {
            "provider": ep.get("provider") or _role_env(role, "PROVIDER") or uniform_provider,
            "model": ep.get("model") or _role_env(role, "MODEL") or uniform_model,
            "base_url": ep.get("base_url") or _role_env(role, "BASE_URL") or uniform_base,
            "reasoning_effort": (
                ep.get("reasoning_effort") or _role_env(role, "REASONING_EFFORT")
                or os.environ.get("OPENAI_REASONING_EFFORT")
            ),
        }
    return {
        "schema": "lead_delegate_model_routing_v1",
        "assignment_policy": "stable_logical_identity",
        "profiles": profiles,
    }


def get_eval_endpoint_pool() -> list[dict[str, Any]]:
    """Endpoints the LLM judge (evaluate.py) round-robins over — configured INDEPENDENTLY
    from the worker pool so evaluation can run on a different (often stronger) API.

    Empty => no eval-specific pool; the judge uses the single MAS_EVAL_* / OPENAI_*
    endpoint resolved by get_eval_llm(). Configure either form in .env / the env:
      MAS_EVAL_POOL     — JSON list of {api_key, base_url?, model?} (most expressive)
      MAS_EVAL_API_KEYS — comma-separated keys sharing MAS_EVAL_BASE_URL / MAS_EVAL_MODEL
    """
    raw = os.environ.get("MAS_EVAL_POOL")
    if raw:
        pool = json.loads(raw)
        if not isinstance(pool, list):
            raise ValueError("MAS_EVAL_POOL must be a JSON list of endpoint objects")
        return pool
    keys = os.environ.get("MAS_EVAL_API_KEYS")
    if keys:
        base = os.environ.get("MAS_EVAL_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
        model = os.environ.get("MAS_EVAL_MODEL") or os.environ.get("OPENAI_MODEL")
        return [
            {"api_key": k.strip(), "base_url": base, "model": model}
            for k in keys.split(",") if k.strip()
        ]
    return []


def get_llm(
    provider: ProviderName | None = None,
    model: str | None = None,
    model_role: AgentModelRole | None = None,
    temperature: float = 0.7,
    **kwargs,
):
    routes = _agent_model_routes.get() or {}
    ep = routes.get(model_role or "") or _endpoint.get() or {}
    provider = provider or ep.get("provider") or os.environ.get("MAS_LLM_PROVIDER", "openai")

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        # Per-run endpoint override (round-robin pool for --workers N); falls back to
        # the OPENAI_* env vars when no pool is active.
        api_key = ep.get("api_key") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        # RELAY-WORKAROUND (中转 API): optional OpenAI-compatible relay/proxy (e.g.
        # relay.example.com). Set OPENAI_BASE_URL to the base that ends in /v1 (the SDK appends
        # /chat/completions). Leave unset to hit api.openai.com directly.
        base_url = ep.get("base_url") or os.environ.get("OPENAI_BASE_URL") or None
        model_name = model or ep.get("model") or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        # Reasoning models (gpt-5*, o-series) reject any non-default temperature.
        # gpt-5-chat* are NOT reasoning models — they take temperature like
        # gpt-4o — so exclude them.
        is_reasoning = (
            (model_name.startswith("gpt-5") and not model_name.startswith("gpt-5-chat"))
            or model_name.startswith(("o1", "o3", "o4"))
        )
        if not is_reasoning:
            kwargs = _with_thinking_disabled_if_needed(kwargs, model_name, base_url)
            return ChatOpenAI(
                model=model_name, temperature=temperature, api_key=api_key,
                base_url=base_url, **kwargs
            )

        # Reasoning model. Optionally lower/disable reasoning to cut cost via
        # OPENAI_REASONING_EFFORT (e.g. "none" / "minimal" / "low" / "medium" /
        # "high"). "none" fully disables reasoning on gpt-5.1+ (gpt-5.4/5.5);
        # "minimal" is the floor on the original gpt-5/o-series.
        #
        # Temperature: these models accept only the default (1). langchain-openai
        # is model-aware and silently drops any other value, so we just pass 1.
        # Custom temperature control is NOT available on reasoning models, even
        # with reasoning off — that is an API/library limit, not ours.
        #
        # use_responses_api: function tools + reasoning_effort are rejected on
        # /v1/chat/completions; the Responses API is required. So whenever an
        # effort is set we route through it.
        effort = ep.get("reasoning_effort") or os.environ.get("OPENAI_REASONING_EFFORT")
        extra: dict[str, Any] = {"temperature": 1}
        if effort:
            extra["reasoning_effort"] = effort
            extra["use_responses_api"] = True
        kwargs = _with_thinking_disabled_if_needed(kwargs, model_name, base_url)
        return ChatOpenAI(model=model_name, api_key=api_key, base_url=base_url, **extra, **kwargs)

    if provider == "deepseek":
        # DeepSeek exposes an OpenAI-compatible API, so we reuse ChatOpenAI with a
        # custom base_url. Models: "deepseek-chat" (V3, supports tool calling) /
        # "deepseek-reasoner" (R1). Key: DEEPSEEK_API_KEY.
        from langchain_openai import ChatOpenAI

        api_key = ep.get("api_key") or os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY not set")
        base_url = ep.get("base_url") or os.environ.get(
            "DEEPSEEK_BASE_URL", "https://api.deepseek.com"
        )
        model_name = model or ep.get("model") or os.environ.get(
            "DEEPSEEK_MODEL", "deepseek-chat"
        )
        kwargs = _with_thinking_disabled_if_needed(kwargs, model_name, base_url)
        # deepseek-reasoner (R1) rejects a custom temperature, like OpenAI's
        # reasoning models; deepseek-chat takes it normally.
        if model_name == "deepseek-reasoner":
            return ChatOpenAI(
                model=model_name, api_key=api_key, base_url=base_url, **kwargs
            )
        return ChatOpenAI(
            model=model_name, temperature=temperature, api_key=api_key,
            base_url=base_url, **kwargs,
        )

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        api_key = (ep.get("api_key") or os.environ.get("GEMINI_API_KEY")
                   or os.environ.get("GOOGLE_API_KEY"))
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY not set")
        return ChatGoogleGenerativeAI(
            model=(model or ep.get("model")
                   or os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")),
            temperature=temperature,
            google_api_key=api_key,
            **kwargs,
        )

    raise ValueError(f"Unknown provider: {provider!r}")


# === Eval (LLM-judge) LLM: configured INDEPENDENTLY from the worker LLM ========
# evaluate.py's LLM judge can run on a different (usually stronger) API than the
# agents in main.py. Every MAS_EVAL_* var falls back to its worker-LLM counterpart
# when unset, so leaving them ALL unset reproduces today's behaviour (judge uses the
# same config as the agents). To use a better judge, set any subset:
#   MAS_EVAL_PROVIDER          (fallback MAS_LLM_PROVIDER / "openai")
#   MAS_EVAL_MODEL             (fallback OPENAI_MODEL / the provider default)
#   MAS_EVAL_API_KEY           (fallback OPENAI_API_KEY / the provider key)
#   MAS_EVAL_BASE_URL          (fallback OPENAI_BASE_URL; openai-compatible relays)
#   MAS_EVAL_REASONING_EFFORT  (fallback OPENAI_REASONING_EFFORT; gpt-5*/o-series)
# Round-robin over several judge keys via MAS_EVAL_POOL / MAS_EVAL_API_KEYS (see
# get_eval_endpoint_pool). The active endpoint (use_endpoint) is honored just like
# get_llm, so the eval pool round-robin works the same way.
def get_eval_llm(model: str | None = None, temperature: float = 0.0, **kwargs):
    provider = os.environ.get("MAS_EVAL_PROVIDER") or os.environ.get("MAS_LLM_PROVIDER", "openai")

    if provider in ("openai", "deepseek"):
        from langchain_openai import ChatOpenAI

        ep = _endpoint.get() or {}
        if provider == "openai":
            api_key = (ep.get("api_key") or os.environ.get("MAS_EVAL_API_KEY")
                       or os.environ.get("OPENAI_API_KEY"))
            base_url = (ep.get("base_url") or os.environ.get("MAS_EVAL_BASE_URL")
                        or os.environ.get("OPENAI_BASE_URL") or None)
            model_name = (model or ep.get("model") or os.environ.get("MAS_EVAL_MODEL")
                          or os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
            if not api_key:
                raise RuntimeError("eval key not set (MAS_EVAL_API_KEY or OPENAI_API_KEY)")
        else:  # deepseek (OpenAI-compatible)
            api_key = (ep.get("api_key") or os.environ.get("MAS_EVAL_API_KEY")
                       or os.environ.get("DEEPSEEK_API_KEY"))
            base_url = (ep.get("base_url") or os.environ.get("MAS_EVAL_BASE_URL")
                        or os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
            model_name = (model or ep.get("model") or os.environ.get("MAS_EVAL_MODEL")
                          or os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"))
            if not api_key:
                raise RuntimeError("eval key not set (MAS_EVAL_API_KEY or DEEPSEEK_API_KEY)")

        is_reasoning = (
            (model_name.startswith("gpt-5") and not model_name.startswith("gpt-5-chat"))
            or model_name.startswith(("o1", "o3", "o4"))
        )
        if not is_reasoning:
            kwargs = _with_thinking_disabled_if_needed(kwargs, model_name, base_url)
            return ChatOpenAI(model=model_name, temperature=temperature, api_key=api_key,
                              base_url=base_url, **kwargs)
        effort = (os.environ.get("MAS_EVAL_REASONING_EFFORT")
                  or os.environ.get("OPENAI_REASONING_EFFORT"))
        extra: dict[str, Any] = {"temperature": 1}
        if effort:
            extra["reasoning_effort"] = effort
            extra["use_responses_api"] = True
        kwargs = _with_thinking_disabled_if_needed(kwargs, model_name, base_url)
        return ChatOpenAI(model=model_name, api_key=api_key, base_url=base_url, **extra, **kwargs)

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        api_key = (os.environ.get("MAS_EVAL_API_KEY")
                   or os.environ.get("GEMINI_API_KEY")
                   or os.environ.get("GOOGLE_API_KEY"))
        if not api_key:
            raise RuntimeError(
                "eval key not set (MAS_EVAL_API_KEY, GEMINI_API_KEY, or GOOGLE_API_KEY)"
            )
        return ChatGoogleGenerativeAI(
            model=(model or os.environ.get("MAS_EVAL_MODEL")
                   or os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")),
            temperature=temperature, google_api_key=api_key, **kwargs,
        )

    raise ValueError(f"Unknown eval provider: {provider!r}")


# === Robust invocation: optional serialization + retry ======================
# Two independent knobs, both env-controlled:
#   LLM_SERIALIZE (default "0"/off): when ON, a process-wide lock serializes every
#     LLM call. This is the old RELAY-WORKAROUND for flaky OpenAI-compatible relays
#     that return spurious 401 "无效的令牌" when one token is hit concurrently. With a
#     reliable DIRECT API (e.g. api.deepseek.com / api.openai.com) leave it OFF so the
#     parallel branches (equi peers / hierarchical followers / nested subteam /
#     dynamic role fan-out) actually run concurrently instead of one-at-a-time.
#   LLM_MAX_ATTEMPTS (default 5): bounded exponential-backoff retry, kept ON either
#     way — it just absorbs transient 401/429/5xx and is harmless on a healthy API.
_LLM_LOCK = threading.Lock()
_LLM_MAX_ATTEMPTS = int(os.environ.get("LLM_MAX_ATTEMPTS", "5"))
_LLM_SERIALIZE = os.environ.get("LLM_SERIALIZE", "0").strip().lower() in (
    "1", "true", "yes", "on",
)


def invoke_llm(llm, messages):
    """Invoke a chat model with bounded backoff retry; serialize only if asked.

    - When LLM_SERIALIZE is on, the lock serializes concurrent branches so a flaky
      relay never sees the same token hit in parallel. When off (default), calls run
      concurrently — required for the parallelism the four structures express.
    - Exponential backoff absorbs intermittent transient errors the OpenAI client
      does not retry on its own. The sleep is OUTSIDE any lock, so a backing-off call
      never blocks other branches.

    Raises the last exception if every attempt fails (a genuinely dead key still ends
    up raising — the caller/main.py decides whether to skip that run).
    """
    last_exc = None
    messages = _strip_reasoning_content(messages)
    if _bound_model_name(llm).lower().startswith("gpt-5"):
        messages = _strip_openai_response_state_ids(messages)
    for attempt in range(_LLM_MAX_ATTEMPTS):
        try:
            if _LLM_SERIALIZE:
                with _LLM_LOCK:
                    result = llm.invoke(messages)
            else:
                result = llm.invoke(messages)
            _record_token_usage(messages, result)
            return result
        except Exception as exc:  # treat opaque relay/transport errors as transient
            last_exc = exc
            if attempt < _LLM_MAX_ATTEMPTS - 1:
                time.sleep(min(2 ** attempt, 20))
    raise last_exc


def _estimate_tokens(text: str) -> int:
    """Cheap fallback when a relay omits usage: ~4 chars/token."""
    return max(1, len(text or "") // 4)


def _msg_text(m) -> str:
    c = getattr(m, "content", m)
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return " ".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in c)
    return str(c)


def _record_token_usage(messages, result) -> None:
    """Record this call's token usage (out-of-band) for cost accounting.

    Prefers the provider's real usage_metadata; falls back to a flagged estimate when
    a relay does not return it. Never raises into the caller. The stage is read from
    the tracing ContextVar (worker by default; defense/eval code sets its own).
    """
    try:
        from tracing import record_token_usage
        inp = out = tot = 0
        estimated = False
        um = getattr(result, "usage_metadata", None)
        if isinstance(um, dict) and um.get("total_tokens"):
            inp = um.get("input_tokens", 0) or 0
            out = um.get("output_tokens", 0) or 0
            tot = um.get("total_tokens", 0) or (inp + out)
        else:
            rm = getattr(result, "response_metadata", None) or {}
            tu = rm.get("token_usage") or rm.get("usage") or {}
            if tu.get("total_tokens"):
                inp = tu.get("prompt_tokens", 0) or 0
                out = tu.get("completion_tokens", 0) or 0
                tot = tu.get("total_tokens", 0) or (inp + out)
            else:
                estimated = True
                inp = sum(_estimate_tokens(_msg_text(m)) for m in (messages or []))
                out = _estimate_tokens(_msg_text(result))
                tot = inp + out
        record_token_usage(input_tokens=inp, output_tokens=out,
                            total_tokens=tot, estimated=estimated)
    except Exception:
        pass  # accounting must never break a run
