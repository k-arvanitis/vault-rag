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
_LABEL_OCR_CPU = "unstructured OCR"

# Matches the bold "**Figure 3: ...**" caption heading pymupdf4llm extracts as real
# page text next to a figure — deliberately requires the leading "**" so it doesn't
# match a plain-prose mention like "as shown in figure 3" or a table-of-contents row.
_FIGURE_CAPTION_RE = re.compile(r"\*\*Fig(?:ure)?\.?\s*(\d+)\s*:", re.IGNORECASE)

# Points from the top of the page below which a figure is treated as a
# repeating page-header banner rather than in-content -- see
# _parse_text_layer_page's reordering step.
_HEADER_BANNER_Y_THRESHOLD = 100

_TABLE_ROW_RE = re.compile(r"^\|(?P<cells>.+)\|[ \t]*$")
_TABLE_SEPARATOR_ROW_RE = re.compile(r"^\|[ \t]*:?-+:?[ \t]*(\|[ \t]*:?-+:?[ \t]*)+\|?[ \t]*$")
_MIN_DUPLICATE_CELL_LEN = 10


def _dedupe_duplicate_table_cells(page_markdown: str) -> str:
    """Blank a table cell whose text is a near-duplicate of an earlier cell
    in the same row.

    pymupdf4llm's table_strategy sometimes spills a term/heading's text into
    the adjacent column instead of stopping at the real column boundary --
    reproduced on doc_001 page 3's headerless "IV. Definitions" table (both
    header cells came out as the paragraph sentence above the table, since
    that's what pymupdf4llm mistook for the header) and on doc_002 page 1's
    "Interpretation" table (an ordinary body row, "1.1 In these terms and
    conditions:" repeated verbatim in both the term and definition columns,
    and a wrapped "Key Personnel" row duplicated into an identical pair).
    Detected by substring containment between normalized cell texts within
    a row -- not by comparing to surrounding prose, and not hardcoded to
    either document's wording -- so it generalizes across both cases and any
    other row with the same mis-parse.
    """
    lines = page_markdown.split("\n")
    for i, line in enumerate(lines):
        if _TABLE_SEPARATOR_ROW_RE.match(line):
            continue
        row_match = _TABLE_ROW_RE.match(line)
        if not row_match:
            continue
        cells = row_match.group("cells").split("|")
        if len(cells) < 2:
            continue
        normalized = []
        for cell in cells:
            n = re.sub(r"<br\s*/?>", " ", cell)
            n = re.sub(r"\*\*|_", "", n).strip().lower()
            normalized.append(n)
        changed = False
        for a in range(len(cells)):
            if not normalized[a]:
                continue
            for b in range(a + 1, len(cells)):
                if len(normalized[b]) >= _MIN_DUPLICATE_CELL_LEN and normalized[b] in normalized[a]:
                    cells[b] = ""
                    changed = True
        if changed:
            lines[i] = "|" + "|".join(cells) + "|"
    return "\n".join(lines)


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


_OCR_DPI = 300


def _ocr_page(pix) -> tuple[str, str]:
    """Run the configured OCR backend on a page pixmap.

    Returns the OCR markdown plus a label identifying which backend was used.
    Selection is controlled by PDF_PARSER: "cpu" uses the unstructured fallback,
    anything else uses the LightOn OCR vLLM server.
    """
    if PDF_PARSER == "cpu":
        from src.ingestion.unstructured_ocr import call_unstructured_ocr

        return call_unstructured_ocr(pix, dpi=_OCR_DPI), _LABEL_OCR_CPU
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
                pix = page.get_pixmap(dpi=_OCR_DPI)
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
    page_markdown: str = _dedupe_duplicate_table_cells(chunk["text"])
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
                # A repeating header banner sits at the top of the page visually
                # but the proportional-position estimate below can still snap it
                # past the first paragraph (the next "\n\n" at-or-after its target
                # char is always the END of whichever paragraph it landed inside) --
                # same class of bug as _parse_text_layer_page's main reorder step.
                # Anything in the top margin goes straight to the front instead.
                if bbox and bbox.y0 < _HEADER_BANNER_Y_THRESHOLD:
                    pos = 0
                elif bbox and page_rect.height > 0:
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
                    bbox,
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
                    bbox,
                )
            )

    # A repeating header banner (logo, divider) sits at the top of the page
    # visually (small bbox y0), but pymupdf4llm's linear text order can place
    # its marker after the first paragraph -- reproduced on doc_001 page 3,
    # where the header logo landed between the intro sentence and the
    # definitions table instead of before either. Pull anything whose bbox
    # top falls within the page's top margin out of its extracted position
    # and prepend it, so reading order matches what's actually on the page.
    header_markers: list[str] = []
    for match, replacement, bbox in sorted(
        replacements, key=lambda x: x[0].start(), reverse=True
    ):
        is_header = bbox is not None and bbox[1] < _HEADER_BANNER_Y_THRESHOLD
        page_markdown = (
            page_markdown[: match.start()]
            + ("" if is_header else replacement)
            + page_markdown[match.end() :]
        )
        if is_header:
            header_markers.append(replacement)

    if header_markers:
        page_markdown = "\n\n".join(reversed(header_markers)) + "\n\n" + page_markdown

    return page_markdown
