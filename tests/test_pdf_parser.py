"""Unit tests for src/parser/pdf_parser.py.

All external calls (fitz, pymupdf4llm, OCR, VLM) are mocked.
No real files or API calls are made.
"""
from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_page(text: str) -> MagicMock:
    """Return a mock fitz page whose get_text() returns *text*."""
    page = MagicMock()
    page.get_text.return_value = text
    return page


def _make_doc(pages: list[MagicMock]) -> MagicMock:
    """Return a mock fitz document that iterates over *pages*."""
    doc = MagicMock()
    doc.__iter__ = MagicMock(return_value=iter(pages))
    doc.__len__ = MagicMock(return_value=len(pages))
    return doc


def _mock_pymupdf_chunk(text: str, images: list | None = None) -> list[dict]:
    return [{"text": text, "images": images or []}]


# ---------------------------------------------------------------------------
# Test 1 — text layer, no images: pymupdf4llm called, OCR NOT called
# ---------------------------------------------------------------------------

def test_text_layer_no_images_uses_pymupdf(tmp_path):
    page_text = "A" * 60  # >= 50 chars → text layer path
    expected_md = "# Heading\n\nSome content"

    with (
        patch("src.parser.pdf_parser.fitz") as mock_fitz,
        patch("src.parser.pdf_parser.pymupdf4llm") as mock_pymupdf,
        patch("src.parser.pdf_parser.call_lighton_ocr") as mock_ocr,
        patch("src.parser.pdf_parser.call_vlm_description") as mock_vlm,
        patch("src.parser.pdf_parser.VLM_ENABLED", True),
    ):
        mock_fitz.open.return_value = _make_doc([_make_page(page_text)])
        mock_pymupdf.to_markdown.return_value = _mock_pymupdf_chunk(expected_md, images=[])

        from src.parser.pdf_parser import parse_pdf

        result = parse_pdf("fake.pdf")

    assert result == [(expected_md, "pymupdf4llm")]
    mock_pymupdf.to_markdown.assert_called_once()
    mock_ocr.assert_not_called()
    mock_vlm.assert_not_called()


# ---------------------------------------------------------------------------
# Test 2 — no text layer: LightOn OCR called, pymupdf4llm NOT called
# ---------------------------------------------------------------------------

def test_scanned_page_uses_lighton_ocr():
    page_text = ""  # < 50 chars → OCR path
    ocr_result = "Scanned page markdown"

    with (
        patch("src.parser.pdf_parser.fitz") as mock_fitz,
        patch("src.parser.pdf_parser.pymupdf4llm") as mock_pymupdf,
        patch("src.parser.pdf_parser.call_lighton_ocr") as mock_ocr,
    ):
        mock_page = _make_page(page_text)
        mock_pixmap = MagicMock()
        mock_page.get_pixmap.return_value = mock_pixmap
        mock_fitz.open.return_value = _make_doc([mock_page])
        mock_ocr.return_value = ocr_result

        from src.parser.pdf_parser import parse_pdf

        result = parse_pdf("fake.pdf")

    assert result == [(ocr_result, "LightOn OCR")]
    mock_ocr.assert_called_once_with(mock_pixmap)
    mock_page.get_pixmap.assert_called_once_with(dpi=300)
    mock_pymupdf.to_markdown.assert_not_called()


# ---------------------------------------------------------------------------
# Test 3 — text layer + image + VLM_ENABLED=True: VLM called, ref replaced
# ---------------------------------------------------------------------------

def test_text_layer_with_image_calls_vlm(tmp_path):
    img_file = tmp_path / "doc-0-0.png"
    img_file.write_bytes(b"fakepng")

    page_text = "B" * 60
    page_md = f"# Title\n\n![](doc-0-0.png)\n\nMore text"
    vlm_desc = "A bar chart showing quarterly revenue."

    with (
        patch("src.parser.pdf_parser.fitz") as mock_fitz,
        patch("src.parser.pdf_parser.pymupdf4llm") as mock_pymupdf,
        patch("src.parser.pdf_parser.call_lighton_ocr"),
        patch("src.parser.pdf_parser.call_vlm_description") as mock_vlm,
        patch("src.parser.pdf_parser.VLM_ENABLED", True),
        patch("src.parser.pdf_parser.tempfile.TemporaryDirectory") as mock_tmpdir,
    ):
        mock_tmpdir.return_value.__enter__ = MagicMock(return_value=str(tmp_path))
        mock_tmpdir.return_value.__exit__ = MagicMock(return_value=False)
        mock_fitz.open.return_value = _make_doc([_make_page(page_text)])
        mock_pymupdf.to_markdown.return_value = _mock_pymupdf_chunk(
            page_md, images=[{"bbox": [0, 0, 100, 100]}]
        )
        mock_vlm.return_value = vlm_desc

        from src.parser.pdf_parser import parse_pdf

        result = parse_pdf("doc.pdf")

    assert result[0][0] == f"# Title\n\n[Figure: {vlm_desc}]\n\nMore text"
    assert result[0][1] == "pymupdf4llm"
    mock_vlm.assert_called_once_with(b"fakepng")


# ---------------------------------------------------------------------------
# Test 4 — text layer + image + VLM_ENABLED=False: ref left as-is
# ---------------------------------------------------------------------------

def test_vlm_disabled_leaves_image_refs_unchanged(tmp_path):
    page_text = "C" * 60
    page_md = "# Title\n\n![](doc-0-0.png)\n\nText"

    with (
        patch("src.parser.pdf_parser.fitz") as mock_fitz,
        patch("src.parser.pdf_parser.pymupdf4llm") as mock_pymupdf,
        patch("src.parser.pdf_parser.call_lighton_ocr"),
        patch("src.parser.pdf_parser.call_vlm_description") as mock_vlm,
        patch("src.parser.pdf_parser.VLM_ENABLED", False),
        patch("src.parser.pdf_parser.tempfile.TemporaryDirectory") as mock_tmpdir,
    ):
        mock_tmpdir.return_value.__enter__ = MagicMock(return_value=str(tmp_path))
        mock_tmpdir.return_value.__exit__ = MagicMock(return_value=False)
        mock_fitz.open.return_value = _make_doc([_make_page(page_text)])
        mock_pymupdf.to_markdown.return_value = _mock_pymupdf_chunk(
            page_md, images=[{"bbox": [0, 0, 100, 100]}]
        )

        from src.parser.pdf_parser import parse_pdf

        result = parse_pdf("doc.pdf")

    assert result[0][0] == page_md
    mock_vlm.assert_not_called()


# ---------------------------------------------------------------------------
# Test 5 — VLM raises exception: fallback inserted, no propagation
# ---------------------------------------------------------------------------

def test_vlm_exception_inserts_fallback(tmp_path):
    img_file = tmp_path / "doc-0-0.png"
    img_file.write_bytes(b"fakepng")

    page_text = "D" * 60
    page_md = f"# Title\n\n![](doc-0-0.png)\n\nText"

    with (
        patch("src.parser.pdf_parser.fitz") as mock_fitz,
        patch("src.parser.pdf_parser.pymupdf4llm") as mock_pymupdf,
        patch("src.parser.pdf_parser.call_lighton_ocr"),
        patch("src.parser.pdf_parser.call_vlm_description") as mock_vlm,
        patch("src.parser.pdf_parser.VLM_ENABLED", True),
        patch("src.parser.pdf_parser.tempfile.TemporaryDirectory") as mock_tmpdir,
    ):
        mock_tmpdir.return_value.__enter__ = MagicMock(return_value=str(tmp_path))
        mock_tmpdir.return_value.__exit__ = MagicMock(return_value=False)
        mock_fitz.open.return_value = _make_doc([_make_page(page_text)])
        mock_pymupdf.to_markdown.return_value = _mock_pymupdf_chunk(
            page_md, images=[{"bbox": [0, 0, 100, 100]}]
        )
        mock_vlm.side_effect = RuntimeError("API timeout")

        from src.parser.pdf_parser import parse_pdf

        # Must not raise
        result = parse_pdf("doc.pdf")

    # vlm raises, but call_vlm_description itself is supposed to catch — simulate
    # the real vlm.py catching it and returning "description unavailable"
    # Here we test that pdf_parser also survives an uncaught exception from vlm.
    # The replacement should be the fallback string.
    assert "[Figure:" in result[0][0]


# ---------------------------------------------------------------------------
# Test 6 — mixed document: page 0 text layer, page 1 scanned
# ---------------------------------------------------------------------------

def test_mixed_document_routes_pages_correctly(tmp_path):
    text_page = _make_page("E" * 60)   # text layer
    scan_page = _make_page("")          # scanned
    scan_pixmap = MagicMock()
    scan_page.get_pixmap.return_value = scan_pixmap

    pymupdf_md = "# Text page content"
    ocr_md = "Scanned page OCR result"

    with (
        patch("src.parser.pdf_parser.fitz") as mock_fitz,
        patch("src.parser.pdf_parser.pymupdf4llm") as mock_pymupdf,
        patch("src.parser.pdf_parser.call_lighton_ocr") as mock_ocr,
        patch("src.parser.pdf_parser.call_vlm_description"),
        patch("src.parser.pdf_parser.VLM_ENABLED", True),
        patch("src.parser.pdf_parser.tempfile.TemporaryDirectory") as mock_tmpdir,
    ):
        mock_tmpdir.return_value.__enter__ = MagicMock(return_value=str(tmp_path))
        mock_tmpdir.return_value.__exit__ = MagicMock(return_value=False)
        mock_fitz.open.return_value = _make_doc([text_page, scan_page])
        mock_pymupdf.to_markdown.return_value = _mock_pymupdf_chunk(pymupdf_md, images=[])
        mock_ocr.return_value = ocr_md

        from src.parser.pdf_parser import parse_pdf

        result = parse_pdf("mixed.pdf")

    assert len(result) == 2
    assert result[0] == (pymupdf_md, "pymupdf4llm")
    assert result[1] == (ocr_md, "LightOn OCR")
    mock_pymupdf.to_markdown.assert_called_once()
    mock_ocr.assert_called_once_with(scan_pixmap)
