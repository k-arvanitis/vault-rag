"""VLM caller for figure descriptions in text-layer PDF pages."""
from __future__ import annotations

import base64
import logging

from openai import OpenAI

from src.config import GROQ_API_KEY, VLM_MODEL, VLM_PROVIDER

logger = logging.getLogger(__name__)

_VLM_PROMPT = (
    "Describe all information in this figure, chart, diagram, or image. "
    "Be specific about numbers, labels, trends, and values. "
    "If this is a table, transcribe it fully. "
    "Be concise — maximum 3 sentences."
)


def call_vlm_description(image_bytes: bytes) -> str:
    """Send PNG image bytes to the configured VLM and return a text description.

    Args:
        image_bytes: Raw PNG bytes of the figure or image to describe.

    Returns:
        A concise text description, or "description unavailable" on any failure.
    """
    try:
        if VLM_PROVIDER == "groq":
            client = OpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=GROQ_API_KEY or "no-key",
            )
            b64 = base64.b64encode(image_bytes).decode()
            response = client.chat.completions.create(
                model=VLM_MODEL,
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
