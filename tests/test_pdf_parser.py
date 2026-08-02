"""Unit tests for src/parser/pdf_parser.py.

All external calls (fitz, pymupdf4llm, OCR, VLM) are mocked.
No real files or API calls are made.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

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
        patch("vault_rag.parser.pdf_parser.fitz") as mock_fitz,
        patch("vault_rag.parser.pdf_parser.pymupdf4llm") as mock_pymupdf,
        patch("vault_rag.parser.pdf_parser.call_lighton_ocr") as mock_ocr,
        patch("vault_rag.parser.pdf_parser.call_vlm_description") as mock_vlm,
        patch("vault_rag.parser.pdf_parser.VLM_ENABLED", True),
    ):
        mock_fitz.open.return_value = _make_doc([_make_page(page_text)])
        mock_pymupdf.to_markdown.return_value = _mock_pymupdf_chunk(
            expected_md, images=[]
        )

        from vault_rag.parser.pdf_parser import parse_pdf

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
        patch("vault_rag.parser.pdf_parser.fitz") as mock_fitz,
        patch("vault_rag.parser.pdf_parser.pymupdf4llm") as mock_pymupdf,
        patch("vault_rag.parser.pdf_parser.call_lighton_ocr") as mock_ocr,
    ):
        mock_page = _make_page(page_text)
        mock_pixmap = MagicMock()
        mock_page.get_pixmap.return_value = mock_pixmap
        mock_fitz.open.return_value = _make_doc([mock_page])
        mock_ocr.return_value = ocr_result

        from vault_rag.parser.pdf_parser import parse_pdf

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
    page_md = "# Title\n\n![](doc-0-0.png)\n\nMore text"
    vlm_desc = "A bar chart showing quarterly revenue."

    with (
        patch("vault_rag.parser.pdf_parser.fitz") as mock_fitz,
        patch("vault_rag.parser.pdf_parser.pymupdf4llm") as mock_pymupdf,
        patch("vault_rag.parser.pdf_parser.call_lighton_ocr"),
        patch("vault_rag.parser.pdf_parser.call_vlm_description") as mock_vlm,
        patch("vault_rag.parser.pdf_parser.VLM_ENABLED", True),
        patch("vault_rag.parser.pdf_parser.tempfile.TemporaryDirectory") as mock_tmpdir,
    ):
        mock_tmpdir.return_value.__enter__ = MagicMock(return_value=str(tmp_path))
        mock_tmpdir.return_value.__exit__ = MagicMock(return_value=False)
        mock_fitz.open.return_value = _make_doc([_make_page(page_text)])
        mock_pymupdf.to_markdown.return_value = _mock_pymupdf_chunk(
            page_md, images=[{"bbox": [0, 300, 100, 400]}]
        )
        mock_vlm.return_value = vlm_desc

        from vault_rag.parser.pdf_parser import parse_pdf

        result = parse_pdf("doc.pdf")

    assert (
        result[0][0]
        == f"# Title\n\n[FIGURE_START]\n<!-- bbox:[0, 300, 100, 400] -->\n{vlm_desc}\n[FIGURE_END]\n\nMore text"
    )
    assert result[0][1] == "pymupdf4llm"
    mock_vlm.assert_called_once_with(b"fakepng")


# ---------------------------------------------------------------------------
# Test 4 — text layer + image + VLM_ENABLED=False: ref left as-is
# ---------------------------------------------------------------------------


def test_vlm_disabled_leaves_image_refs_unchanged(tmp_path):
    page_text = "C" * 60
    page_md = "# Title\n\n![](doc-0-0.png)\n\nText"

    with (
        patch("vault_rag.parser.pdf_parser.fitz") as mock_fitz,
        patch("vault_rag.parser.pdf_parser.pymupdf4llm") as mock_pymupdf,
        patch("vault_rag.parser.pdf_parser.call_lighton_ocr"),
        patch("vault_rag.parser.pdf_parser.call_vlm_description") as mock_vlm,
        patch("vault_rag.parser.pdf_parser.VLM_ENABLED", False),
        patch("vault_rag.parser.pdf_parser.tempfile.TemporaryDirectory") as mock_tmpdir,
    ):
        mock_tmpdir.return_value.__enter__ = MagicMock(return_value=str(tmp_path))
        mock_tmpdir.return_value.__exit__ = MagicMock(return_value=False)
        mock_fitz.open.return_value = _make_doc([_make_page(page_text)])
        mock_pymupdf.to_markdown.return_value = _mock_pymupdf_chunk(
            page_md, images=[{"bbox": [0, 0, 100, 100]}]
        )

        from vault_rag.parser.pdf_parser import parse_pdf

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
    page_md = "# Title\n\n![](doc-0-0.png)\n\nText"

    with (
        patch("vault_rag.parser.pdf_parser.fitz") as mock_fitz,
        patch("vault_rag.parser.pdf_parser.pymupdf4llm") as mock_pymupdf,
        patch("vault_rag.parser.pdf_parser.call_lighton_ocr"),
        patch("vault_rag.parser.pdf_parser.call_vlm_description") as mock_vlm,
        patch("vault_rag.parser.pdf_parser.VLM_ENABLED", True),
        patch("vault_rag.parser.pdf_parser.tempfile.TemporaryDirectory") as mock_tmpdir,
    ):
        mock_tmpdir.return_value.__enter__ = MagicMock(return_value=str(tmp_path))
        mock_tmpdir.return_value.__exit__ = MagicMock(return_value=False)
        mock_fitz.open.return_value = _make_doc([_make_page(page_text)])
        mock_pymupdf.to_markdown.return_value = _mock_pymupdf_chunk(
            page_md, images=[{"bbox": [0, 0, 100, 100]}]
        )
        mock_vlm.side_effect = RuntimeError("API timeout")

        from vault_rag.parser.pdf_parser import parse_pdf

        # Must not raise
        result = parse_pdf("doc.pdf")

    assert "description unavailable\n[FIGURE_END]" in result[0][0]


# ---------------------------------------------------------------------------
# Test 6 — mixed document: page 0 text layer, page 1 scanned
# ---------------------------------------------------------------------------


def test_mixed_document_routes_pages_correctly(tmp_path):
    text_page = _make_page("E" * 60)  # text layer
    scan_page = _make_page("")  # scanned
    scan_pixmap = MagicMock()
    scan_page.get_pixmap.return_value = scan_pixmap

    pymupdf_md = "# Text page content"
    ocr_md = "Scanned page OCR result"

    with (
        patch("vault_rag.parser.pdf_parser.fitz") as mock_fitz,
        patch("vault_rag.parser.pdf_parser.pymupdf4llm") as mock_pymupdf,
        patch("vault_rag.parser.pdf_parser.call_lighton_ocr") as mock_ocr,
        patch("vault_rag.parser.pdf_parser.call_vlm_description"),
        patch("vault_rag.parser.pdf_parser.VLM_ENABLED", True),
        patch("vault_rag.parser.pdf_parser.tempfile.TemporaryDirectory") as mock_tmpdir,
    ):
        mock_tmpdir.return_value.__enter__ = MagicMock(return_value=str(tmp_path))
        mock_tmpdir.return_value.__exit__ = MagicMock(return_value=False)
        mock_fitz.open.return_value = _make_doc([text_page, scan_page])
        mock_pymupdf.to_markdown.return_value = _mock_pymupdf_chunk(
            pymupdf_md, images=[]
        )
        mock_ocr.return_value = ocr_md

        from vault_rag.parser.pdf_parser import parse_pdf

        result = parse_pdf("mixed.pdf")

    assert len(result) == 2
    assert result[0] == (pymupdf_md, "pymupdf4llm")
    assert result[1] == (ocr_md, "LightOn OCR")
    mock_pymupdf.to_markdown.assert_called_once()
    mock_ocr.assert_called_once_with(scan_pixmap)


# ---------------------------------------------------------------------------
# Test 7 — page-header banner reordering (OCR-fallback fitz-extraction path)
# ---------------------------------------------------------------------------


def test_header_banner_moved_to_start_in_ocr_fallback_path(tmp_path):
    """pymupdf4llm's own OCR path sometimes skips image extraction entirely
    (no ![]() / picture-text markers at all), so pdf_parser falls back to
    pulling images straight out of fitz -- see the `if VLM_ENABLED and not
    all_matches` branch. That branch estimates insertion position from the
    image's proportional y-position rather than reusing the main path's
    replacements list, so it needed its own header-margin check."""
    page_text = "D" * 60
    # No image markers in the markdown at all -- forces the fitz fallback.
    page_md = "First paragraph of body text.\n\nMore text below."
    vlm_desc = "LACERA logo."

    fitz_page = MagicMock()
    fitz_page.rect = MagicMock(height=792.0)
    fitz_page.get_images.return_value = [(7, 0, 0, 0, 0, 0, 0, 0, 0, 0)]
    fitz_page.get_image_info.return_value = [
        {"xref": 7, "bbox": (54.0, 36.0, 559.0, 64.8)}
    ]

    fitz_doc = MagicMock()
    fitz_doc.__getitem__ = MagicMock(return_value=fitz_page)
    fitz_doc.extract_image.return_value = {"image": b"fakepng"}

    with (
        patch("vault_rag.parser.pdf_parser.fitz") as mock_fitz,
        patch("vault_rag.parser.pdf_parser.pymupdf4llm") as mock_pymupdf,
        patch("vault_rag.parser.pdf_parser.call_lighton_ocr"),
        patch("vault_rag.parser.pdf_parser.call_vlm_description") as mock_vlm,
        patch("vault_rag.parser.pdf_parser.VLM_ENABLED", True),
        patch("vault_rag.parser.pdf_parser.tempfile.TemporaryDirectory") as mock_tmpdir,
    ):
        mock_tmpdir.return_value.__enter__ = MagicMock(return_value=str(tmp_path))
        mock_tmpdir.return_value.__exit__ = MagicMock(return_value=False)
        class _FakeRect:
            def __init__(self, b):
                self.x0, self.y0, self.x1, self.y1 = b

            def __iter__(self):
                return iter((self.x0, self.y0, self.x1, self.y1))

        mock_fitz.Rect = _FakeRect
        mock_fitz.open.side_effect = [
            _make_doc([_make_page(page_text)]),  # outer page-routing pass
            fitz_doc,  # inner fallback extraction pass
        ]
        mock_pymupdf.to_markdown.return_value = _mock_pymupdf_chunk(page_md, images=None)
        mock_vlm.return_value = vlm_desc

        from vault_rag.parser.pdf_parser import parse_pdf

        result = parse_pdf("doc.pdf")

    page_out = result[0][0]
    assert page_out.startswith("\n\n[FIGURE_START]") or page_out.startswith("[FIGURE_START]")
    assert page_out.index("LACERA logo.") < page_out.index("First paragraph")


# ---------------------------------------------------------------------------
# Test 8 — page-header banner reordering (main replacements-list path)
# ---------------------------------------------------------------------------


def test_header_banner_figure_moved_to_start_of_page(tmp_path):
    img_file = tmp_path / "doc-0-0.png"
    img_file.write_bytes(b"fakepng")

    page_text = "C" * 60
    # pymupdf4llm placed the header logo's marker after the paragraph, even
    # though its bbox (y0=36) is visually at the top of the page.
    page_md = "First paragraph of body text.\n\n![](doc-0-0.png)\n\nMore text"
    vlm_desc = "LACERA logo."

    with (
        patch("vault_rag.parser.pdf_parser.fitz") as mock_fitz,
        patch("vault_rag.parser.pdf_parser.pymupdf4llm") as mock_pymupdf,
        patch("vault_rag.parser.pdf_parser.call_lighton_ocr"),
        patch("vault_rag.parser.pdf_parser.call_vlm_description") as mock_vlm,
        patch("vault_rag.parser.pdf_parser.VLM_ENABLED", True),
        patch("vault_rag.parser.pdf_parser.tempfile.TemporaryDirectory") as mock_tmpdir,
    ):
        mock_tmpdir.return_value.__enter__ = MagicMock(return_value=str(tmp_path))
        mock_tmpdir.return_value.__exit__ = MagicMock(return_value=False)
        mock_fitz.open.return_value = _make_doc([_make_page(page_text)])
        mock_pymupdf.to_markdown.return_value = _mock_pymupdf_chunk(
            page_md, images=[{"bbox": [54.0, 36.0, 559.0, 64.8]}]
        )
        mock_vlm.return_value = vlm_desc

        from vault_rag.parser.pdf_parser import parse_pdf

        result = parse_pdf("doc.pdf")

    page_out = result[0][0]
    assert page_out.startswith("[FIGURE_START]")
    assert page_out.index("LACERA logo.") < page_out.index("First paragraph")


# ---------------------------------------------------------------------------
# Test 9 — _dedupe_duplicate_table_cells
# ---------------------------------------------------------------------------


def test_dedupe_duplicate_table_cells_blanks_duplicated_header():
    from vault_rag.parser.pdf_parser import _dedupe_duplicate_table_cells

    md = (
        "Whenever the following words appear in this Policy, they will be "
        "construed to have the following meaning:\n\n"
        "|Whenever the following words appear in this Policy, they will be "
        "construed to have the following meaning:|Whenever the following words appear|\n"
        "|---|---|\n"
        "|**Amendment:**|An agreed addition to, deletion from, or modification.|\n"
    )

    fixed = _dedupe_duplicate_table_cells(md)

    assert "construed to have the following meaning:|Whenever" not in fixed
    assert "|**Amendment:**|An agreed addition to, deletion from, or modification.|" in fixed


def test_dedupe_duplicate_table_cells_leaves_real_headers_untouched():
    from vault_rag.parser.pdf_parser import _dedupe_duplicate_table_cells

    md = (
        "Some unrelated paragraph text that has nothing to do with the table below.\n\n"
        "|Term|Definition|\n"
        "|---|---|\n"
        "|**Amendment:**|An agreed addition to, deletion from, or modification.|\n"
    )

    fixed = _dedupe_duplicate_table_cells(md)

    assert fixed == md


def test_dedupe_duplicate_table_cells_blanks_duplicated_body_row():
    from vault_rag.parser.pdf_parser import _dedupe_duplicate_table_cells

    # Reproduced on doc_002 page 1's "Interpretation" table: an ordinary body
    # row (not the header) where pymupdf4llm spilled the lead-in text into
    # both columns.
    md = (
        "|**1**<br>**Interpretation**|**Interpretation**|\n"
        "|---|---|\n"
        "|1.1<br>In these terms and conditions:|In these terms and conditions:|\n"
        '|"Agreement"|means the contract between the Customer and the Supplier;|\n'
    )

    fixed = _dedupe_duplicate_table_cells(md)
    lines = fixed.splitlines()

    assert lines[0] == "|**1**<br>**Interpretation**||"
    assert lines[2] == "|1.1<br>In these terms and conditions:||"
    assert (
        lines[3]
        == '|"Agreement"|means the contract between the Customer and the Supplier;|'
    )


def test_dedupe_duplicate_table_cells_blanks_fully_identical_row():
    from vault_rag.parser.pdf_parser import _dedupe_duplicate_table_cells

    # Reproduced on doc_002 page 1's wrapped "Key Personnel" row: pymupdf4llm
    # duplicated the entire cell text into both columns verbatim.
    md = (
        "|Term|Definition|\n"
        "|---|---|\n"
        '|"Key Personnel" means any persons specified as such in the Award Letter|'
        '"Key Personnel" means any persons specified as such in the Award Letter|\n'
    )

    fixed = _dedupe_duplicate_table_cells(md)

    assert (
        '|"Key Personnel" means any persons specified as such in the Award Letter|' + "|"
        in fixed
    )
    assert fixed.count('"Key Personnel"') == 1
