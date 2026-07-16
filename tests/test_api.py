from api import _payloads_to_docs, _truncate_markdown_table


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
