"""Shared LLM/endpoint helpers used by the agent and the retrieval tool."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Callable

from vault_rag.config import (
    EXCEL_AGENT_API_BASE,
    EXCEL_AGENT_API_KEY,
    EXCEL_AGENT_MODEL,
    LLM_REQUEST_TIMEOUT_S,
    OPENROUTER_PROVIDER_PIN,
)
from vault_rag.llm_credentials import key_for_base


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


def _make_excel_cleaner_llm_fn() -> Callable[[str], str]:
    """Return an LLM callable for excel_cleaner.process_file.

    Uses the same EXCEL_AGENT_* endpoint as query_excel's own SQL generation
    (src/tools/excel.py), not a hardcoded Groq client. This used to be
    hardcoded to Groq directly with no config override -- reproduced live
    (2026-07-15) that Groq's org is billing-restricted (the same restriction
    already documented in TODO.md for GENERATION_API_BASE, but this path was
    never migrated when that one was), so every Excel/CSV ingest was silently
    skipping the LLM-cleaning step and falling through to "Skipping DuckDB
    load" -- a newly uploaded spreadsheet would get a Qdrant summary but no
    queryable rows, with no error surfaced to the user.
    """
    import openai

    client = openai.OpenAI(
        base_url=_to_openai_base(EXCEL_AGENT_API_BASE),
        api_key=EXCEL_AGENT_API_KEY,
    )

    def _call(prompt: str) -> str:
        """Send one prompt to the excel-agent model and return the text completion."""
        resp = client.chat.completions.create(
            model=EXCEL_AGENT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            extra_body=_openrouter_provider_extra_body(EXCEL_AGENT_API_BASE),
        )
        raw = resp.choices[0].message.content
        return raw if raw is not None else ""

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
    # Resolve key: explicit arg wins; otherwise the shared resolver (BYOK
    # store override, then env-key-by-host) -- see key_for_base's docstring
    # for why this must be the same function build_rag_agent uses, not its
    # own copy of the matching.
    key = api_key or key_for_base(api_base)
    client = _get_openai_client(api_base, key)
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
        extra_body=_openrouter_provider_extra_body(api_base),
    )
    raw = resp.choices[0].message.content
    # None when a reasoning model (e.g. gpt-oss) spends the whole max_tokens
    # budget on hidden chain-of-thought before emitting any visible content
    # (finish_reason="length", content=None) -- degrade to "" rather than
    # crash the re.sub below, so a starved call fails soft like any other
    # empty answer instead of raising out of the whole pipeline.
    if raw is None:
        return ""
    # Strip <think> blocks emitted by reasoning models (Qwen3, QwQ, DeepSeek-R1, etc.)
    return re.sub(r"(?s)<think>.*?</think>", "", raw).strip()
