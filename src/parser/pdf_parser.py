"""Two-path PDF ingestion parser.

Routes each page to either:
- pymupdf4llm  — pages with a text layer (len(text) >= 50 chars)
- LightOn OCR  — scanned pages with no usable text layer
"""
from __future__ import annotations

import logging
import re
import tempfile
from pathlib import Path

import fitz
import pymupdf4llm

from src.config import IMAGE_SIZE_LIMIT, VLM_ENABLED
from src.ingestion.ocr import call_lighton_ocr
from src.ingestion.vlm import call_vlm_description

logger = logging.getLogger(__name__)

# Raster image written to disk by pymupdf4llm
_IMG_REF_RE = re.compile(r'!\[\]\(([^)]+\.png)\)')

# Vector graphic whose text was extracted instead of rasterised
_PICTURE_TEXT_RE = re.compile(
    r'-{3,}\s*Start of picture text\s*-{3,}.*?-{3,}\s*End of picture text\s*-{3,}',
    re.DOTALL | re.IGNORECASE,
)

_LABEL_TEXT = "pymupdf4llm"
_LABEL_OCR = "LightOn OCR"


def parse_pdf(path: str) -> list[tuple[str, str]]:
    """Parse a PDF and return one (markdown, pipeline_label) tuple per page.

    Each page is routed independently:
    - Pages with a text layer (>=50 chars) go through pymupdf4llm → label "pymupdf4llm".
    - Scanned pages (no text layer) are rendered and sent to LightOn OCR → label "LightOn OCR".

    Args:
        path: Absolute or relative path to the PDF file.

    Returns:
        List of (page_markdown, pipeline_label) tuples, one per page.
    """
    results: list[tuple[str, str]] = []
    doc = fitz.open(path)
    n_pages = len(doc)

    with tempfile.TemporaryDirectory() as tmp_dir:
        for page_number, page in enumerate(doc):
            text = page.get_text().strip()

            if len(text) < 50:
                print(f"[INGEST] Page {page_number + 1}/{n_pages} → {_LABEL_OCR} (scanned)")
                pix = page.get_pixmap(dpi=300)
                page_string = call_lighton_ocr(pix)
                label = _LABEL_OCR
            else:
                print(f"[INGEST] Page {page_number + 1}/{n_pages} → {_LABEL_TEXT} (text-layer)")
                page_string = _parse_text_layer_page(path, page_number, tmp_dir)
                label = _LABEL_TEXT

            results.append((page_string, label))

    return results


def _parse_text_layer_page(path: str, page_number: int, image_dir: str) -> str:
    """Extract and enrich one text-layer page using pymupdf4llm.

    Handles two kinds of visual content:
    - Raster images written to disk → identified by ![](path.png) references.
    - Vector graphics whose text was extracted → identified by
      "--- Start of picture text ---" blocks; these are rendered via fitz
      and sent to the VLM for a proper description.

    Args:
        path: Path to the PDF file.
        page_number: Zero-indexed page number to extract.
        image_dir: Directory where pymupdf4llm writes extracted image files.

    Returns:
        Markdown string for the page, with all image markers replaced when VLM is enabled.
    """
    result = pymupdf4llm.to_markdown(
        doc=path,
        pages=[page_number],
        page_chunks=True,
        write_images=True,
        image_path=image_dir,
        image_format="png",
        image_size_limit=IMAGE_SIZE_LIMIT,
        dpi=150,
        table_strategy="lines_strict",
    )

    chunk = result[0]
    page_markdown: str = chunk["text"]
    images: list = chunk.get("images", [])

    # Collect all image markers in document order (both types)
    all_matches: list[tuple[str, re.Match]] = [
        ("img_ref", m) for m in _IMG_REF_RE.finditer(page_markdown)
    ] + [
        ("picture_text", m) for m in _PICTURE_TEXT_RE.finditer(page_markdown)
    ]
    all_matches.sort(key=lambda x: x[1].start())

    if not all_matches:
        return page_markdown

    pdf_stem = Path(path).stem
    replacements: list[tuple[re.Match, str]] = []

    for img_idx, (match_type, match) in enumerate(all_matches):
        if match_type == "img_ref":
            if not VLM_ENABLED:
                continue
            embedded_path = Path(match.group(1))
            if not embedded_path.is_absolute():
                embedded_path = Path(image_dir) / embedded_path
            if not embedded_path.exists():
                embedded_path = Path(image_dir) / f"{pdf_stem}-{page_number}-{img_idx}.png"

            if embedded_path.exists():
                try:
                    description = call_vlm_description(embedded_path.read_bytes())
                except Exception:
                    logger.exception("VLM call failed for page %d index %d", page_number, img_idx)
                    description = "description unavailable"
            else:
                logger.warning("Image file not found for page %d index %d: %s", page_number, img_idx, embedded_path)
                description = "description unavailable"
            replacements.append((match, f"[Figure: {description}]"))

        elif match_type == "picture_text":
            if not VLM_ENABLED:
                continue
            try:
                doc = fitz.open(path)
                fitz_page = doc[page_number]
                # Crop to the image bbox when available; fall back to full page
                bbox = images[img_idx]["bbox"] if img_idx < len(images) else None
                if bbox:
                    pix = fitz_page.get_pixmap(dpi=150, clip=fitz.Rect(bbox))
                else:
                    pix = fitz_page.get_pixmap(dpi=150)
                description = call_vlm_description(pix.tobytes("png"))
            except Exception:
                logger.exception("VLM render failed for picture text block page %d index %d", page_number, img_idx)
                description = "description unavailable"
            replacements.append((match, f"[Figure: {description}]"))

    # Apply replacements from end to start to preserve string positions
    for match, replacement in sorted(replacements, key=lambda x: x[0].start(), reverse=True):
        page_markdown = page_markdown[:match.start()] + replacement + page_markdown[match.end():]

    return page_markdown
