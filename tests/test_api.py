from api import _truncate_markdown_table


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
