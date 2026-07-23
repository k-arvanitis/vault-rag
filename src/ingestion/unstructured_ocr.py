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


def call_unstructured_ocr(pix, dpi: int = 300) -> str:
    """OCR a fitz Pixmap with unstructured + tesseract and return Markdown text.

    Renders the pixmap to a single-page PDF on disk, then runs unstructured's
    `partition_pdf` with `strategy="hi_res"` so layout-aware parsing + tesseract
    OCR run together. Output elements are joined into a Markdown-ish string
    (tables get fenced as HTML, everything else as paragraphs), each preceded
    by an `<!-- ocr_bbox:[x0,y0,x1,y1] -->` comment giving its location in PDF
    points on the original page, when unstructured reports coordinates for it.

    Args:
        pix: A fitz.Pixmap rendered from a scanned PDF page.
        dpi: The DPI `pix` was rendered at (must match the caller's
            `page.get_pixmap(dpi=...)` call) -- needed to scale bbox
            coordinates back to PDF points.

    Returns:
        Markdown string from OCR, or an empty string on failure.
    """
    try:
        from unstructured.partition.pdf import partition_pdf
    except ImportError:
        logger.error(
            "unstructured is not installed; install with `uv add unstructured[pdf]`"
        )
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

    # unstructured's hi_res layout detection re-renders our single-page PDF at
    # its own internal resolution (system.width/height), not the pix.width/
    # pix.height we requested -- so a coordinate is only meaningful relative to
    # its own system size. Scale through that, then from pixmap-space (rendered
    # at `dpi`) down to PDF points (72/inch) to land in the same coordinate
    # space fitz.search_for/get_pixmap(clip=...) already use for born-digital
    # pages. Both systems use a top-left origin, so no y-flip is needed.
    parts: list[str] = []
    for el in elements:
        category = getattr(el, "category", "") or ""
        coords = getattr(el.metadata, "coordinates", None) if el.metadata else None
        bbox_comment = ""
        if coords and coords.points and coords.system:
            scale_x = pix.width / coords.system.width * 72 / dpi
            scale_y = pix.height / coords.system.height * 72 / dpi
            # unstructured's points are numpy float64 -- left as-is, list's
            # repr renders "np.float64(1.2)" instead of "1.2" and silently
            # fails the plain-digit ocr_bbox regex downstream (chunker.py,
            # answer_pipeline.py), dropping every bbox with no error.
            xs = [float(p[0]) * scale_x for p in coords.points]
            ys = [float(p[1]) * scale_y for p in coords.points]
            bbox = [min(xs), min(ys), max(xs), max(ys)]
            bbox_comment = f"<!-- ocr_bbox:{bbox} -->\n"
        if category == "Table":
            html = (getattr(el, "metadata", None) and el.metadata.text_as_html) or str(
                el
            )
            parts.append(bbox_comment + html)
        else:
            text = str(el).strip()
            if text:
                parts.append(bbox_comment + text)

    return "\n\n".join(parts)
