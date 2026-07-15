"""Shared LLM/endpoint helpers used by the agent and the retrieval tool."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Callable

from src.config import (
    FREE_LLM_API_KEY,
    GROQ_API_KEY,
    LITELLM_MASTER_KEY,
    LLM_REQUEST_TIMEOUT_S,
    OPENROUTER_API_KEY,
    OPENROUTER_PROVIDER_PIN,
)


def _openrouter_provider_extra_body(api_base: str) -> dict | None:
    """extra_body pinning OpenRouter to one backend provider, or None.

    OpenRouter can route "the same" model call to different actual backends
    per request (observed live: qwen/qwen3-32b served by Nebius on one call,
    DeepInfra on another) -- different engines/quantizations behind one model
    name is a plausible source of answer-to-answer variance beyond ordinary
    temp=0 sampling noise. Only applies when OPENROUTER_PROVIDER_PIN is set
    and api_base is actually OpenRouter.
    """
    if not OPENROUTER_PROVIDER_PIN or "openrouter.ai" not in api_base.lower():
        return None
    return {"provider": {"order": [OPENROUTER_PROVIDER_PIN], "allow_fallbacks": False}}


@lru_cache(maxsize=16)
def _get_openai_client(base_url: str, api_key: str):
    """Return a cached OpenAI client per (base_url, api_key) pair.

    _llm_call is invoked many times per question (split decision, repair
    check, grounding check — once per sub-question when a question is
    multi-part), and each openai.OpenAI() instantiation opens its own
    connection pool. Without caching, high call volume (e.g. eval runs)
    accumulates unclosed clients/threads over a run instead of reusing one.
    """
    import openai

    return openai.OpenAI(
        base_url=base_url, api_key=api_key, timeout=LLM_REQUEST_TIMEOUT_S
    )


def _make_groq_llm_fn() -> Callable[[str], str]:
    """Return a Groq LLM callable for excel_cleaner.process_file."""
    import openai

    # Build the Groq OpenAI-compatible client once and close over it.
    client = openai.OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=GROQ_API_KEY,
    )

    def _call(prompt: str) -> str:
        """Send one prompt to the Groq model and return the text completion."""
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return resp.choices[0].message.content

    return _call


def _is_thinking_model(model_name: str) -> bool:
    """Return True for models that emit <think> blocks and accept /no_think."""
    name = model_name.lower()
    return any(k in name for k in ("qwen", "qwq", "deepseek-r", "r1"))


def _to_openai_base(api_base: str) -> str:
    """Normalize an API base URL to end in /v1, as the OpenAI client expects."""
    base = api_base.rstrip("/")
    return base if base.endswith("/v1") else f"{base}/v1"


def _llm_call(
    prompt: str,
    api_base: str,
    model_name: str,
    api_key: str = "",
    max_tokens: int = 128,
    temperature: float = 0.0,
) -> str:
    """Send one prompt to an OpenAI-compatible endpoint; return the reply with <think> blocks stripped."""
    # Resolve key: explicit arg wins; otherwise pick by the actual host being
    # called — LITELLM_MASTER_KEY only authenticates the LiteLLM proxy itself,
    # so it must not be sent to a real provider being hit directly.
    _base = api_base.lower()
    if "localhost:4000" in _base or "127.0.0.1:4000" in _base:
        _fallback_key = LITELLM_MASTER_KEY
    elif "localhost:3011" in _base or "127.0.0.1:3011" in _base:
        _fallback_key = FREE_LLM_API_KEY
    elif "openrouter.ai" in _base:
        _fallback_key = OPENROUTER_API_KEY
    elif "groq.com" in _base:
        _fallback_key = GROQ_API_KEY
    else:
        _fallback_key = GROQ_API_KEY or LITELLM_MASTER_KEY
    key = api_key or _fallback_key or "EMPTY"
    client = _get_openai_client(api_base, key)
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
        extra_body=_openrouter_provider_extra_body(api_base),
    )
    raw = resp.choices[0].message.content
    # Strip <think> blocks emitted by reasoning models (Qwen3, QwQ, DeepSeek-R1, etc.)
    return re.sub(r"(?s)<think>.*?</think>", "", raw).strip()
