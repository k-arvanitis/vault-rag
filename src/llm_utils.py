"""Shared LLM/endpoint helpers used by the agent and the retrieval tool."""
from __future__ import annotations

import os
import re

from src.config import LITELLM_MASTER_KEY


def _is_thinking_model(model_name: str) -> bool:
    """Return True for models that emit <think> blocks and accept /no_think."""
    name = model_name.lower()
    return any(k in name for k in ("qwen", "qwq", "deepseek-r", "r1"))



def _to_openai_base(api_base: str) -> str:
    """Normalize an API base URL to end in /v1, as the OpenAI client expects."""
    base = api_base.rstrip("/")
    return base if base.endswith("/v1") else f"{base}/v1"


def _llm_call(prompt: str, api_base: str, model_name: str, api_key: str = "", max_tokens: int = 128, temperature: float = 0.0) -> str:
    """Send one prompt to an OpenAI-compatible endpoint; return the reply with <think> blocks stripped."""
    import openai
    # Resolve key: explicit arg → LiteLLM master key → Groq → OpenAI → dummy
    key = (
        api_key
        or LITELLM_MASTER_KEY
        or os.getenv("GROQ_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or "EMPTY"
    )
    client = openai.OpenAI(base_url=api_base, api_key=key)
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    raw = resp.choices[0].message.content
    # Strip <think> blocks emitted by reasoning models (Qwen3, QwQ, DeepSeek-R1, etc.)
    return re.sub(r"(?s)<think>.*?</think>", "", raw).strip()
