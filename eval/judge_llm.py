"""Custom DeepEval judge LLM backed by any OpenAI-compatible endpoint.

Local (default):  Qwen3-32B-AWQ via vLLM at GENERATION_API_BASE
CI/CD:            Set JUDGE_API_BASE + JUDGE_API_KEY + JUDGE_MODEL to any
                  OpenAI-compatible service (e.g. Groq, Together, Fireworks).

Example Groq setup (free tier):
    JUDGE_API_BASE=https://api.groq.com/openai/v1
    JUDGE_API_KEY=gsk_...
    JUDGE_MODEL=llama-3.3-70b-versatile
"""
from __future__ import annotations

import os
import re
from typing import Optional

from openai import OpenAI
from deepeval.models.base_model import DeepEvalBaseLLM


class OpenAICompatibleJudge(DeepEvalBaseLLM):
    """DeepEval judge that talks to any OpenAI-compatible API."""

    def __init__(self) -> None:
        api_base = os.getenv(
            "JUDGE_API_BASE",
            os.getenv("GENERATION_API_BASE", "http://127.0.0.1:8003"),
        )
        # Normalise: ensure path ends with /v1
        api_base = api_base.rstrip("/")
        if not api_base.endswith("/v1"):
            api_base = f"{api_base}/v1"

        self._model_name = os.getenv(
            "JUDGE_MODEL",
            os.getenv("GENERATION_MODEL", "Qwen/Qwen3-32B-AWQ"),
        )
        self._api_key = os.getenv("JUDGE_API_KEY", "not-needed")
        self._client = OpenAI(base_url=api_base, api_key=self._api_key)

    # ------------------------------------------------------------------
    # DeepEvalBaseLLM interface
    # ------------------------------------------------------------------

    def load_model(self) -> OpenAI:
        return self._client

    def generate(self, prompt: str) -> str:
        """Generate a response from the judge LLM.

        No schema kwarg — DeepEval's _generate_schema catches the TypeError and
        uses its own trimAndLoadJson + schema(**data) fallback, which is more
        robust than us trying to parse Pydantic models ourselves.
        """
        # Inject numerical-equivalence instruction into Faithfulness verdict prompts.
        # DeepEval's verdict step checks each claim against context — without this hint
        # Qwen3 marks "limit is 10 mg/L" as unsupported when context says "max: 10 mg/L".
        _NUMERICAL_HINT = (
            "\n\nIMPORTANT: Numerical values are considered supported if the same number "
            "and unit appear anywhere in the retrieval context, regardless of phrasing "
            "differences. For example, 'limit is 10 mg/L' and 'max concentration: 10 mg/L' "
            "refer to the same claim and MUST be marked as supported.\n"
        )
        is_verdict_prompt = (
            "verdict" in prompt.lower()
            and ("claim" in prompt.lower() or "statement" in prompt.lower())
            and "context" in prompt.lower()
        )
        if is_verdict_prompt:
            prompt = _NUMERICAL_HINT + prompt

        # /no_think is Qwen3-specific — skip for OpenAI models
        is_qwen = "qwen" in self._model_name.lower()
        clean_prompt = f"/no_think\n{prompt}" if is_qwen else prompt
        # Only force json_object for DeepEval *scoring* prompts.
        # These always ask for both "score" and "reason" as top-level JSON keys.
        # Synthesizer evolution/generation prompts do NOT have both — avoid breaking them.
        wants_json = '"score"' in prompt and '"reason"' in prompt
        kwargs: dict = dict(
            model=self._model_name,
            messages=[{"role": "user", "content": clean_prompt}],
            temperature=0.0,
            max_tokens=2048,
        )
        if wants_json:
            kwargs["response_format"] = {"type": "json_object"}
        response = self._client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or ""
        # Strip any residual <think>...</think> blocks — Qwen3 occasionally emits
        # these even with /no_think, which breaks DeepEval's JSON parsing.
        content = re.sub(r"(?is)<think>.*?</think>\s*", "", content).strip()
        return content

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return self._model_name
