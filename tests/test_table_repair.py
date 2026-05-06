"""Tests for the table-repair logic in src/ingest.py.

Both functions are private (underscore-prefixed) but imported directly because
they represent the most non-trivial engineering in the pipeline and need
explicit coverage. All tests are self-contained — no live services required.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.ingest import (
    _derive_column_names,
    _fill_empty_last_column_from_text,
    _fix_html_multirow_header,
)


# ---------------------------------------------------------------------------
# _derive_column_names
# ---------------------------------------------------------------------------

class TestDeriveColumnNames:
    """_derive_column_names converts a flat header string from a two-row-header
    table (all group names first, then all sub-column tokens) into a list of
    fully-qualified column names like "ModelA R@1", "ModelA R@5", etc."""

    def test_single_group_three_sub_cols(self):
        """One model group with three retrieval metrics."""
        header = "Tokens ModelA R@1 R@5 R@10"
        # First data row: one label + three decimals → 4 total matches len(cols)
        data = "Dataset1 0.123 0.456 0.789"
        cols = _derive_column_names(header, data)
        assert cols == ["Tokens", "ModelA R@1", "ModelA R@5", "ModelA R@10"]

    def test_two_groups_three_sub_cols_each(self):
        """Two model groups, each with R@1 R@5 R@10."""
        header = "Tokens ModelA ModelB R@1 R@5 R@10 R@1 R@5 R@10"
        data = "Dataset1 0.11 0.22 0.33 0.44 0.55 0.66"
        cols = _derive_column_names(header, data)
        assert len(cols) == 7  # Tokens + 3 per group × 2 groups
        assert cols[0] == "Tokens"
        assert "ModelA R@1" in cols
        assert "ModelB R@10" in cols

    def test_connector_glued_to_group(self):
        """Group names joined with '+' should be treated as one group name."""
        header = "Dataset ModelA + ModelB R@1 R@5 R@1 R@5"
        data = "NQ 0.55 0.72 0.60 0.78"
        cols = _derive_column_names(header, data)
        assert any("ModelA + ModelB" in c for c in cols)

    def test_no_sub_col_tokens_returns_empty(self):
        """If there are no R@K / NDCG@K tokens, return empty list."""
        header = "Model Accuracy Precision Recall"
        data = "GPT4 0.91 0.88 0.85"
        assert _derive_column_names(header, data) == []

    def test_empty_header_returns_empty(self):
        assert _derive_column_names("", "some data 0.1 0.2") == []

    def test_column_count_mismatch_returns_empty(self):
        """If derived column count doesn't match the data row, return empty."""
        # 3 groups × 3 sub-cols = 9 + 1 standalone = 10, but data row has only 2 nums
        header = "Tokens A B C R@1 R@5 R@10 R@1 R@5 R@10 R@1 R@5 R@10"
        data = "NQ 0.1 0.2"
        assert _derive_column_names(header, data) == []


# ---------------------------------------------------------------------------
# _fix_html_multirow_header
# ---------------------------------------------------------------------------

class TestFixHtmlMultirowHeader:
    """_fix_html_multirow_header converts an HTML table whose first <tbody> row
    is a sub-column header row (R@1, R@5 …) into a flat markdown pipe table."""

    _SAMPLE_HTML = """<table>
<thead><tr><th>Tokens</th><th>ModelA</th><th></th><th></th></tr></thead>
<tbody>
<tr><td></td><td>R@1</td><td>R@5</td><td>R@10</td></tr>
<tr><td>NQ</td><td>0.55</td><td>0.72</td><td>0.81</td></tr>
<tr><td>HotpotQA</td><td>0.48</td><td>0.65</td><td>0.74</td></tr>
</tbody>
</table>"""

    def test_returns_markdown_string(self):
        result = _fix_html_multirow_header(self._SAMPLE_HTML)
        assert result is not None
        assert "|" in result  # markdown pipe table

    def test_no_tbody_returns_none(self):
        html = "<table><thead><tr><th>A</th></tr></thead></table>"
        assert _fix_html_multirow_header(html) is None

    def test_first_body_row_not_sub_col_returns_none(self):
        """If the first body row contains no R@K tokens, it's not a two-row header."""
        html = """<table>
<thead><tr><th>Model</th><th>Score</th></tr></thead>
<tbody>
<tr><td>GPT4</td><td>0.91</td></tr>
</tbody>
</table>"""
        assert _fix_html_multirow_header(html) is None

    def test_output_does_not_contain_sub_col_row_as_data(self):
        """The R@1 / R@5 / R@10 header row must not appear as a data row."""
        result = _fix_html_multirow_header(self._SAMPLE_HTML)
        assert result is not None
        lines = [ln for ln in result.splitlines() if ln.startswith("|")]
        # Data rows should only contain numeric values or dataset names, not bare "R@1"
        data_rows = lines[2:]  # skip header and separator
        for row in data_rows:
            assert "R@1" not in row or "ModelA R@1" in row


# ---------------------------------------------------------------------------
# _fill_empty_last_column_from_text
# ---------------------------------------------------------------------------

class TestFillEmptyLastColumnFromText:
    """_fill_empty_last_column_from_text recovers missing rightmost-column cells
    in pipe tables by scanning the PDF text layer. Tests mock pypdf so no real
    PDF is required."""

    _MARKDOWN_WITH_EMPTY_LAST = """\
<!-- PAGE 1 -->

| Method | R@1 | R@5 | R@10 |
|--------|-----|-----|------|
| BM25 | 0.450 | 0.620 | 0.710 |  |
| DPR | 0.510 | 0.680 | 0.750 |  |
"""

    _PDF_TEXT = "BM25 0.450 0.620 0.710 0.830 DPR 0.510 0.680 0.750 0.890"

    def _mock_reader(self, text: str):
        page = MagicMock()
        page.extract_text.return_value = text
        reader = MagicMock()
        reader.pages = [page]
        return reader

    def test_fills_empty_last_column(self):
        """Values present in the PDF text layer should be recovered."""
        from pathlib import Path
        reader = self._mock_reader(self._PDF_TEXT)
        with patch("pypdf.PdfReader", return_value=reader):
            result = _fill_empty_last_column_from_text(
                self._MARKDOWN_WITH_EMPTY_LAST,
                Path("fake.pdf"),
                verbose=False,
            )
        assert "0.830" in result
        assert "0.890" in result

    def test_no_empty_last_column_unchanged(self):
        """If no cells are empty, the markdown must be returned unchanged."""
        from pathlib import Path
        complete = """\
<!-- PAGE 1 -->

| Method | R@1 | R@5 |
|--------|-----|-----|
| BM25 | 0.450 | 0.620 |
| DPR | 0.510 | 0.680 |
"""
        reader = self._mock_reader("BM25 0.450 0.620 DPR 0.510 0.680")
        with patch("pypdf.PdfReader", return_value=reader):
            result = _fill_empty_last_column_from_text(
                complete, Path("fake.pdf"), verbose=False
            )
        assert result == complete

    def test_pdf_extraction_failure_leaves_markdown_intact(self):
        """If pypdf raises, the original markdown must be returned unchanged."""
        from pathlib import Path
        with patch("src.ingest.PdfReader", side_effect=Exception("corrupt PDF")):
            result = _fill_empty_last_column_from_text(
                self._MARKDOWN_WITH_EMPTY_LAST,
                Path("fake.pdf"),
                verbose=False,
            )
        # Empty cells should still be empty — no crash, no corruption
        assert "|  |" in result or "| |" in result
