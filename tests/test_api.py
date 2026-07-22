from unittest.mock import patch

import api
from api import (
    _find_matched_row_index,
    _payloads_to_docs,
    _truncate_markdown_table,
    _validate_startup_env,
    _window_rows_around_match,
)


class TestTruncateMarkdownTable:
    def test_caps_data_rows_and_notes_omission(self):
        header = "| A | B |\n| --- | --- |\n"
        rows = "\n".join(f"| r{i} | v{i} |" for i in range(100))
        md = header + rows
        result = _truncate_markdown_table(md, max_rows=10)
        assert result.count("| r") == 10
        assert "90 more rows omitted" in result

    def test_preserves_non_table_header_lines(self):
        md = "[File: x.xlsx | Sheet: S]\nSheet summary: 5 rows.\n\n| A |\n| --- |\n| 1 |\n| 2 |"
        result = _truncate_markdown_table(md, max_rows=1)
        assert result.startswith("[File: x.xlsx | Sheet: S]")
        assert "1 more rows omitted" in result

    def test_returns_unchanged_when_no_table_found(self):
        md = "just plain text, no pipes here"
        assert _truncate_markdown_table(md) == md

    def test_no_truncation_note_when_under_limit(self):
        md = "| A |\n| --- |\n| 1 |"
        result = _truncate_markdown_table(md, max_rows=60)
        assert "omitted" not in result


class TestFindMatchedRowIndex:
    """Mirrors frontend/lib/tableMatch.ts's findMatchedRowIndex -- kept in
    sync deliberately so server-side search and client-side display agree
    on which row is "the" match."""

    def test_finds_row_whose_cell_appears_in_quote(self):
        rows = [["a", "b"], ["PLAY EQUIPMENT TEAM", "317.50"]]
        assert _find_matched_row_index(rows, "...PLAY EQUIPMENT TEAM 317.50") == 1

    def test_no_match_returns_minus_one(self):
        rows = [["a", "b"], ["c", "d"]]
        assert _find_matched_row_index(rows, "nothing here matches") == -1

    def test_empty_quote_returns_minus_one(self):
        rows = [["a", "b"]]
        assert _find_matched_row_index(rows, "") == -1
        assert _find_matched_row_index(rows, None) == -1

    def test_short_cells_ignored_to_avoid_false_positives(self):
        # A 2-char cell like "of" would match almost any prose quote.
        rows = [["of", "to"]]
        assert _find_matched_row_index(rows, "the summary of expenditure") == -1

    def test_prefers_row_with_more_matching_cells_over_first_match(self):
        # Reproduced live: "PLAY EQUIPMENT TEAM" is a category repeated
        # across many unrelated rows -- the row that ALSO matches the
        # quote's exact total ("317.50") is the real one, not whichever
        # same-category row happens to come first in the sheet.
        rows = [
            ["03/04/2025", "PLAY EQUIPMENT TEAM", "420.00"],  # wrong row, earlier
            ["03/04/2025", "PLAY EQUIPMENT TEAM", "317.50"],  # true match
        ]
        quote = "PLAY EQUIPMENT TEAM 317.50"
        assert _find_matched_row_index(rows, quote) == 1

    def test_tie_breaks_to_earliest_row(self):
        rows = [["MATCH", "x"], ["MATCH", "y"]]
        assert _find_matched_row_index(rows, "MATCH") == 0


class TestWindowRowsAroundMatch:
    """Server-side row search + windowing -- reproduced live 2026-07-22: a
    citation's real row (transaction 6089041, doc_007) fell past the
    endpoint's first-60-rows cap on a larger CSV, so no amount of
    client-side matching could ever find or highlight it (it was never in
    the data sent to the browser). This searches the FULL sheet and returns
    a window around the real match instead."""

    def test_returns_none_when_no_match(self):
        header = [["H1", "H2"]]
        body = [["a", "b"]] * 5
        assert _window_rows_around_match(header, body, "no match here", 60) is None

    def test_finds_match_beyond_first_60_rows(self):
        header = [["H1", "H2"]]
        body = [["filler", str(i)] for i in range(100)]
        body[85] = ["PLAY EQUIPMENT TEAM", "317.50"]
        result = _window_rows_around_match(
            header, body, "...PLAY EQUIPMENT TEAM 317.50...", max_rows=60
        )
        assert result is not None
        assert result[0] == header[0]
        assert ["PLAY EQUIPMENT TEAM", "317.50"] in result[1:]

    def test_window_stays_within_max_rows_bound(self):
        header = [["H1"]]
        body = [["filler"]] * 500
        body[250] = ["TARGET"]
        result = _window_rows_around_match(header, body, "TARGET", max_rows=60)
        assert result is not None
        # header + up to max_rows body rows.
        assert len(result) <= 61

    def test_match_near_start_does_not_go_negative(self):
        header = [["H1"]]
        body = [["TARGET"]] + [["filler"]] * 100
        result = _window_rows_around_match(header, body, "TARGET", max_rows=60)
        assert result is not None
        assert result[1] == ["TARGET"]

    def test_preserves_trailing_notes_appended_after_the_table(self):
        """Notes extracted during ingestion (excel_cleaner.SheetMetadata.notes)
        are appended after the table in the .md file -- truncation must not
        eat into them once the table itself is large enough to overflow
        max_rows, since they live past table_end, not inside it."""
        header = "| A | B |\n| --- | --- |\n"
        rows = "\n".join(f"| r{i} | v{i} |" for i in range(100))
        notes = "\n\n**Notes:**\n- Figures exclude VAT.\n- Source: finance team."
        md = header + rows + notes
        result = _truncate_markdown_table(md, max_rows=10)
        assert result.count("| r") == 10
        assert "90 more rows omitted" in result
        assert "**Notes:**" in result
        assert "Figures exclude VAT." in result


class TestPayloadsToDocs:
    def test_pdf_gets_page_count_from_max_page_metadata(self):
        payloads = [
            {"metadata": {"source_file": "a.pdf", "page": 1}},
            {"metadata": {"source_file": "a.pdf", "page": 3}},
        ]
        docs = _payloads_to_docs(payloads)
        assert docs[0]["page_count"] == 3
        assert docs[0]["sheet_count"] is None

    def test_excel_gets_sheet_count_and_row_count(self):
        payloads = [
            {"metadata": {"source_file": "b.xlsx", "sheet_name": "Sheet1", "chunk_type": "sheet_summary"}},
            {"metadata": {"source_file": "b.xlsx", "sheet_name": "Sheet2", "chunk_type": "sheet_summary"}},
            {
                "metadata": {"source_file": "b.xlsx", "chunk_type": "document_summary"},
                "content": "Sheet summary: 100 rows.\n...\nSheet summary: 50 rows.",
            },
        ]
        docs = _payloads_to_docs(payloads)
        assert docs[0]["sheet_count"] == 2
        assert docs[0]["row_count"] == 150
        assert docs[0]["page_count"] is None

    def test_sheet_metadata_on_a_pdf_is_ignored(self):
        """A PDF's own embedded-table extraction can carry sheet_name/table_N
        metadata -- found live on a stray duplicate doc entry. Must not show
        as spreadsheet sheet_count on what's actually a PDF."""
        payloads = [
            {"metadata": {"source_file": "c.pdf", "sheet_name": "table_1", "chunk_type": "sheet_summary"}},
        ]
        docs = _payloads_to_docs(payloads)
        assert docs[0]["sheet_count"] is None

    def test_display_title_extracted_from_document_summary(self):
        payloads = [
            {
                "metadata": {"source_file": "doc_001_procurement_policy.pdf", "chunk_type": "document_summary"},
                "content": "Document ID: doc_001\nTitle: Procurement Approval Policy\n\nSummary text.",
            },
        ]
        docs = _payloads_to_docs(payloads)
        assert docs[0]["display_title"] == "Procurement Approval Policy"

    def test_display_title_none_when_no_title_line(self):
        payloads = [{"metadata": {"source_file": "d.pdf", "page": 1}}]
        docs = _payloads_to_docs(payloads)
        assert docs[0]["display_title"] is None


class TestValidateStartupEnv:
    def test_logs_error_when_openrouter_base_missing_key(self, caplog):
        with (
            patch.object(api, "GENERATION_API_BASE", "https://openrouter.ai/api/v1"),
            patch.object(api, "OPENROUTER_API_KEY", ""),
        ):
            with caplog.at_level("ERROR"):
                _validate_startup_env()
        assert any("OPENROUTER_API_KEY" in r.message for r in caplog.records)

    def test_no_error_when_openrouter_key_present(self, caplog):
        with (
            patch.object(api, "GENERATION_API_BASE", "https://openrouter.ai/api/v1"),
            patch.object(api, "OPENROUTER_API_KEY", "sk-present"),
        ):
            with caplog.at_level("ERROR"):
                _validate_startup_env()
        assert not any(r.levelname == "ERROR" for r in caplog.records)

    def test_logs_error_when_groq_base_missing_key(self, caplog):
        with (
            patch.object(api, "GENERATION_API_BASE", "https://api.groq.com/openai/v1"),
            patch.object(api, "GROQ_API_KEY", ""),
        ):
            with caplog.at_level("ERROR"):
                _validate_startup_env()
        assert any("GROQ_API_KEY" in r.message for r in caplog.records)

    def test_warns_not_errors_for_local_litellm_proxy_without_master_key(self, caplog):
        with (
            patch.object(api, "GENERATION_API_BASE", "http://localhost:4000/v1"),
            patch.object(api, "LITELLM_MASTER_KEY", ""),
        ):
            with caplog.at_level("WARNING"):
                _validate_startup_env()
        assert any(r.levelname == "WARNING" for r in caplog.records)
        assert not any(r.levelname == "ERROR" for r in caplog.records)
