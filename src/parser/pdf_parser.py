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

from src.config import IMAGE_SIZE_LIMIT, PDF_PARSER, VLM_ENABLED
from src.ingestion.ocr import call_lighton_ocr
from src.ingestion.vlm import call_vlm_description

logger = logging.getLogger(__name__)

# Raster image written to disk by pymupdf4llm
_IMG_REF_RE = re.compile(r"!\[\]\(([^)]+\.png)\)")

# Vector graphic whose text was extracted instead of rasterised
_PICTURE_TEXT_RE = re.compile(
    r"-{3,}\s*Start of picture text\s*-{3,}.*?-{3,}\s*End of picture text\s*-{3,}",
    re.DOTALL | re.IGNORECASE,
)

_LABEL_TEXT = "pymupdf4llm"
_LABEL_OCR = "LightOn OCR"
_LABEL_OCR_CPU = "unstructured (CPU)"

# Matches the bold "**Figure 3: ...**" caption heading pymupdf4llm extracts as real
# page text next to a figure — deliberately requires the leading "**" so it doesn't
# match a plain-prose mention like "as shown in figure 3" or a table-of-contents row.
_FIGURE_CAPTION_RE = re.compile(r"\*\*Fig(?:ure)?\.?\s*(\d+)\s*:", re.IGNORECASE)


def _nearby_figure_label(page_markdown: str, position: int, window: int = 800) -> str:
    """Return "Figure N: " if a figure caption heading precedes this position, else "".

    Without this, a figure's own VLM-described chunk never contains its own figure
    number — so a query naming a specific figure ("what does Figure 3 show") can't be
    disambiguated from a neighboring figure with similar content (verified: two
    financial-benefits figures a few pages apart were confused this way). Searches
    backward for the nearest preceding caption rather than using a small fixed
    window — real PDFs interpose a paragraph or two (e.g. a "Note:" aside) between
    the caption and the image.
    """
    snippet = page_markdown[max(0, position - window) : position]
    matches = list(_FIGURE_CAPTION_RE.finditer(snippet))
    return f"Figure {matches[-1].group(1)}: " if matches else ""


def _ocr_page(pix) -> tuple[str, str]:
    """Run the configured OCR backend on a page pixmap.

    Returns the OCR markdown plus a label identifying which backend was used.
    Selection is controlled by PDF_PARSER: "cpu" uses the unstructured fallback,
    anything else uses the LightOn OCR vLLM server.
    """
    if PDF_PARSER == "cpu":
        from src.ingestion.unstructured_ocr import call_unstructured_ocr

        return call_unstructured_ocr(pix), _LABEL_OCR_CPU
    return call_lighton_ocr(pix), _LABEL_OCR


def parse_pdf(path: str, force_pipeline: str | None = None) -> list[tuple[str, str]]:
    """Parse a PDF and return one (markdown, pipeline_label) tuple per page.

    Each page is routed independently:
    - Pages with a text layer (>=50 chars) go through pymupdf4llm → label "pymupdf4llm".
    - Scanned pages (no text layer) are rendered and sent to LightOn OCR → label "LightOn OCR".

    Args:
        path: Absolute or relative path to the PDF file.
        force_pipeline: Override auto-routing. "ocr" forces every page through LightOn OCR;
            "text" forces every page through pymupdf4llm. None uses auto-routing.

    Returns:
        List of (page_markdown, pipeline_label) tuples, one per page.
    """
    results: list[tuple[str, str]] = []
    doc = fitz.open(path)
    n_pages = len(doc)

    with tempfile.TemporaryDirectory() as tmp_dir:
        for page_number, page in enumerate(doc):
            text = page.get_text().strip()

            use_ocr = force_pipeline == "ocr" or (
                force_pipeline != "text" and len(text) < 50
            )

            if use_ocr:
                pix = page.get_pixmap(dpi=300)
                page_string, label = _ocr_page(pix)
                print(
                    f"[INGEST] Page {page_number + 1}/{n_pages} → {label} ({'forced' if force_pipeline == 'ocr' else 'scanned'})"
                )
            else:
                print(
                    f"[INGEST] Page {page_number + 1}/{n_pages} → {_LABEL_TEXT} ({'forced' if force_pipeline == 'text' else 'text-layer'})"
                )
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
    ] + [("picture_text", m) for m in _PICTURE_TEXT_RE.finditer(page_markdown)]
    all_matches.sort(key=lambda x: x[1].start())

    # pymupdf4llm's internal OCR path silently skips image extraction.
    # Fall back: use fitz directly to extract any embedded raster images that
    # pymupdf4llm missed (i.e. not already referenced in page_markdown).
    if VLM_ENABLED and not all_matches:
        fitz_doc = fitz.open(path)
        fitz_page = fitz_doc[page_number]
        fitz_images = fitz_page.get_images(full=True)
        if fitz_images:
            page_rect = fitz_page.rect
            image_info = fitz_page.get_image_info(hashes=False, xrefs=True)
            # Build xref → bbox lookup
            xref_to_bbox: dict[int, fitz.Rect] = {
                info["xref"]: fitz.Rect(info["bbox"])
                for info in image_info
                if "xref" in info
            }
            original_len = len(page_markdown)
            insert_offset = 0
            for img_info in fitz_images:
                xref = img_info[0]
                try:
                    base_image = fitz_doc.extract_image(xref)
                    img_bytes = base_image["image"]
                    description = call_vlm_description(img_bytes)
                except Exception:
                    logger.exception(
                        "Fallback VLM extraction failed for page %d xref %d",
                        page_number,
                        xref,
                    )
                    description = "description unavailable"
                bbox = xref_to_bbox.get(xref)
                if bbox and page_rect.height > 0:
                    # y-fraction applied to original length, then shifted by prior insertions
                    y_frac = bbox.y1 / page_rect.height
                    target_char = int(y_frac * original_len) + insert_offset
                    # Snap to the nearest paragraph break at or after target_char
                    para_break = page_markdown.find("\n\n", target_char)
                    if para_break == -1:
                        para_break = len(page_markdown)
                    pos = para_break
                else:
                    pos = len(page_markdown)
                label = _nearby_figure_label(page_markdown, pos)
                bbox_comment = f"<!-- bbox:{list(bbox)} -->\n" if bbox else ""
                marker = f"[FIGURE_START]\n{bbox_comment}{label}{description}\n[FIGURE_END]\n\n"
                page_markdown = (
                    page_markdown[:pos] + "\n\n" + marker + page_markdown[pos:]
                )
                insert_offset += len(marker) + 2
        return page_markdown

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
                embedded_path = (
                    Path(image_dir) / f"{pdf_stem}-{page_number}-{img_idx}.png"
                )

            if embedded_path.exists():
                try:
                    description = call_vlm_description(embedded_path.read_bytes())
                except Exception:
                    logger.exception(
                        "VLM call failed for page %d index %d", page_number, img_idx
                    )
                    description = "description unavailable"
            else:
                logger.warning(
                    "Image file not found for page %d index %d: %s",
                    page_number,
                    img_idx,
                    embedded_path,
                )
                description = "description unavailable"
            label = _nearby_figure_label(page_markdown, match.start())
            bbox = images[img_idx].get("bbox") if img_idx < len(images) else None
            bbox_comment = f"<!-- bbox:{list(bbox)} -->\n" if bbox else ""
            replacements.append(
                (
                    match,
                    f"[FIGURE_START]\n{bbox_comment}{label}{description}\n[FIGURE_END]",
                )
            )

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
                logger.exception(
                    "VLM render failed for picture text block page %d index %d",
                    page_number,
                    img_idx,
                )
                description = "description unavailable"
            label = _nearby_figure_label(page_markdown, match.start())
            bbox_comment = f"<!-- bbox:{list(bbox)} -->\n" if bbox else ""
            replacements.append(
                (
                    match,
                    f"[FIGURE_START]\n{bbox_comment}{label}{description}\n[FIGURE_END]",
                )
            )

    # Apply replacements from end to start to preserve string positions
    for match, replacement in sorted(
        replacements, key=lambda x: x[0].start(), reverse=True
    ):
        page_markdown = (
            page_markdown[: match.start()] + replacement + page_markdown[match.end() :]
        )

    return page_markdown
