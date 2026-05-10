"""CPU OCR fallback using `unstructured` + tesseract.

Used when PDF_PARSER=cpu so the demo can run on CPU-only infrastructure
(e.g. Render) without the LightOn OCR vLLM server. Roughly 10x slower per
page than the GPU path.
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def call_unstructured_ocr(pix) -> str:
    """OCR a fitz Pixmap with unstructured + tesseract and return Markdown text.

    Renders the pixmap to a single-page PDF on disk, then runs unstructured's
    `partition_pdf` with `strategy="hi_res"` so layout-aware parsing + tesseract
    OCR run together. Output elements are joined into a Markdown-ish string
    (tables get fenced as HTML, everything else as paragraphs).

    Args:
        pix: A fitz.Pixmap rendered from a scanned PDF page.

    Returns:
        Markdown string from OCR, or an empty string on failure.
    """
    try:
        from unstructured.partition.pdf import partition_pdf
    except ImportError:
        logger.error("unstructured is not installed; install with `uv add unstructured[pdf]`")
        return ""

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        pix.pdfocr_save(str(tmp_path))
    except Exception:
        try:
            import fitz
            single = fitz.open()
            single.new_page(width=pix.width, height=pix.height).insert_image(
                fitz.Rect(0, 0, pix.width, pix.height), pixmap=pix
            )
            single.save(str(tmp_path))
            single.close()
        except Exception:
            logger.exception("Could not render pixmap to PDF for unstructured OCR")
            tmp_path.unlink(missing_ok=True)
            return ""

    try:
        elements = partition_pdf(filename=str(tmp_path), strategy="hi_res")
    except Exception:
        logger.exception("unstructured partition_pdf failed")
        tmp_path.unlink(missing_ok=True)
        return ""
    finally:
        tmp_path.unlink(missing_ok=True)

    parts: list[str] = []
    for el in elements:
        category = getattr(el, "category", "") or ""
        if category == "Table":
            html = (getattr(el, "metadata", None) and el.metadata.text_as_html) or str(el)
            parts.append(html)
        else:
            text = str(el).strip()
            if text:
                parts.append(text)

    return "\n\n".join(parts)
