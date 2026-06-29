"""Thin wrapper that sends a fitz Pixmap to the LightOn OCR endpoint."""

from __future__ import annotations

import base64
import io
import logging

import requests
from PIL import Image

from src.config import OCR_API_BASE, OCR_MODEL

logger = logging.getLogger(__name__)

_OCR_PROMPT = (
    "Transcribe this page into Markdown. If there are images, figures, or diagrams, "
    "provide a brief description of what they contain within the text flow."
)


def call_lighton_ocr(pix) -> str:
    """Send a fitz Pixmap to the LightOn OCR endpoint and return Markdown text.

    Converts the pixmap to a JPEG data URL and posts it to the OpenAI-compatible
    chat completions endpoint served by the local LightOn vLLM server.

    Args:
        pix: A fitz.Pixmap rendered from a scanned PDF page.

    Returns:
        Markdown string from OCR, or an empty string on failure.
    """
    mode = "RGBA" if pix.alpha else "RGB"
    img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
    if pix.alpha:
        img = img.convert("RGB")

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()
    data_url = f"data:image/jpeg;base64,{b64}"

    endpoint = f"{OCR_API_BASE.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": OCR_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": _OCR_PROMPT},
                ],
            }
        ],
        "max_tokens": 4096,
        "temperature": 0.0,
    }

    try:
        response = requests.post(endpoint, json=payload, timeout=120)
        response.raise_for_status()
        return str(response.json()["choices"][0]["message"]["content"])
    except Exception:
        logger.exception("LightOn OCR call failed")
        return ""
