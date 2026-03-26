"""Language detection and translation utilities.

Used to normalise non-English queries to English before hitting the RAG agent,
and to translate English answers back to the original language afterwards.

Translation rules (encoded in the prompts):
- Codes, legal references, and acronyms (ΚΥΑ, ΦΕΚ, ΕΣΔΕΑ …) are kept verbatim.
- Numerical values and units are kept verbatim.
- All natural-language text is translated faithfully.
"""
from __future__ import annotations

import re

_GREEK_RANGE = (("\u0370", "\u03FF"), ("\u1F00", "\u1FFF"))

_TO_EN_SYSTEM = """\
You are a translator specialising in environmental and energy policy documents.
Translate the user message from Greek to English.

Rules — preserve EXACTLY as written (do NOT translate):
- Legal codes and references (ΚΥΑ, ΦΕΚ, ΕΣΔΕΑ, and any alphanumeric code such as ΚΥΑ 8668/2007)
- Greek acronyms used as technical labels or identifiers (ΕΣΔΕΑ, ΕΠΠΕΡΑΑ, etc.)
- Proper nouns that are internationally recognised
- All numerical values and measurement units

Translate all other text naturally and accurately.
Output ONLY the translated text — no explanations, no preamble."""

_TO_EL_SYSTEM = """\
You are a translator specialising in environmental and energy policy documents.
Translate the user message from English to Greek.

Rules — preserve EXACTLY as written (do NOT translate):
- Legal codes and references (ΚΥΑ, ΦΕΚ, ΕΣΔΕΑ, and any alphanumeric code)
- Acronyms used as technical labels or identifiers
- All numerical values and measurement units

Translate all other text naturally and accurately.
Output ONLY the translated text — no explanations, no preamble."""


def is_greek(text: str) -> bool:
    """Return True if >20 % of the characters are Greek Unicode."""
    if not text:
        return False
    greek = sum(
        1
        for c in text
        if any(lo <= c <= hi for lo, hi in _GREEK_RANGE)
    )
    return greek / len(text) > 0.20


def _llm_translate(system: str, text: str, api_base: str, model: str) -> str:
    from openai import OpenAI

    client = OpenAI(base_url=api_base.rstrip("/") + "/v1" if not api_base.endswith("/v1") else api_base, api_key="na")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"/no_think\n{text}"},
        ],
        temperature=0,
        max_tokens=1024,
    )
    content = (response.choices[0].message.content or text).strip()
    # Strip residual <think> blocks that Qwen3 may emit despite /no_think
    content = re.sub(r"(?is)<think>.*?</think>\s*", "", content).strip()
    return content or text


def translate_to_english(text: str, api_base: str, model: str) -> str:
    """Translate Greek text to English, preserving codes and labels."""
    return _llm_translate(_TO_EN_SYSTEM, text, api_base, model)


def translate_to_greek(text: str, api_base: str, model: str) -> str:
    """Translate English text to Greek, preserving codes and labels."""
    return _llm_translate(_TO_EL_SYSTEM, text, api_base, model)
