"""VLM caller for figure descriptions in text-layer PDF pages."""

from __future__ import annotations

import base64
import hashlib
import json
import logging
from pathlib import Path

from openai import OpenAI

from vault_rag.config import GROQ_API_KEY, OPENROUTER_API_KEY, VLM_MODEL, VLM_PROVIDER

logger = logging.getLogger(__name__)

_VLM_PROMPT = (
    "Describe all information in this figure, chart, diagram, or image. "
    "Be specific about numbers, labels, trends, and values. "
    "If this is a table, transcribe it fully. "
    "If this is a logo, brand mark, or purely decorative graphic (a divider "
    "line, header/footer ornament, border) with no numbers, labels, or other "
    "real data in it, reply with ONLY a short name for it and nothing else — "
    "no color, font, or layout description. Examples: 'LACERA logo', "
    "'decorative header graphic'. "
    "Otherwise, be concise — maximum 3 sentences."
)

# Content-addressed cache so re-ingesting an unchanged PDF doesn't re-pay for a
# VLM call on a figure it already described (same image bytes -> same hash).
_CACHE_PATH = Path(__file__).resolve().parents[2] / "data" / "output" / "vlm_cache.json"
_cache: dict[str, str] | None = None


def _load_cache() -> dict[str, str]:
    global _cache
    if _cache is None:
        _cache = (
            json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
            if _CACHE_PATH.exists()
            else {}
        )
    return _cache


def call_vlm_description(image_bytes: bytes) -> str:
    """Send PNG image bytes to the configured VLM and return a text description.

    Successful descriptions are cached to disk by image hash — a re-ingest of the
    same PDF reuses them instead of re-calling the VLM. Failures are never cached,
    so a later run retries once the VLM is reachable again.

    Args:
        image_bytes: Raw PNG bytes of the figure or image to describe.

    Returns:
        A concise text description, or "description unavailable" on any failure.
    """
    key = hashlib.sha256(image_bytes).hexdigest()
    cache = _load_cache()
    if key in cache:
        return cache[key]

    description = _call_vlm(image_bytes)
    if description != "description unavailable":
        cache[key] = description
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_PATH.write_text(json.dumps(cache), encoding="utf-8")
    return description


def _call_vlm(image_bytes: bytes) -> str:
    try:
        if VLM_PROVIDER in ("groq", "openrouter"):
            base_url, api_key = (
                ("https://api.groq.com/openai/v1", GROQ_API_KEY or "no-key")
                if VLM_PROVIDER == "groq"
                else ("https://openrouter.ai/api/v1", OPENROUTER_API_KEY or "no-key")
            )
            client = OpenAI(base_url=base_url, api_key=api_key)
            b64 = base64.b64encode(image_bytes).decode()
            response = client.chat.completions.create(
                model=VLM_MODEL,
                max_tokens=300,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": _VLM_PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{b64}"},
                            },
                        ],
                    }
                ],
            )
            return response.choices[0].message.content.strip()

        logger.warning("Unknown VLM_PROVIDER %r — skipping description", VLM_PROVIDER)
        return "description unavailable"

    except Exception:
        logger.exception("VLM description call failed")
        return "description unavailable"
