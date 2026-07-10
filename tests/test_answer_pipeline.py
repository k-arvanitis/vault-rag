"""Tests for the title-question shortcut in src/answer_pipeline.py."""

from __future__ import annotations

from unittest.mock import patch

from src.answer_pipeline import _TITLE_QUESTION_RE, _title_shortcut_answer, answer_query


def _summary_hit(title: str, file_name: str = "doc_001_procurement_policy.md") -> dict:
    content = (
        "## Document Summary\n\nDocument ID: doc_001\n"
        f"Title: {title}\nFile: {file_name}\n\nSome paraphrased summary text."
    )
    return {"content": content, "metadata": {"file_name": file_name}, "score": 0.9}


class TestTitleQuestionDetection:
    def test_matches_title_of_phrasing(self):
        assert _TITLE_QUESTION_RE.search(
            "What is the title of the LACERA procurement policy document?"
        )

    def test_matches_document_title_phrasing(self):
        assert _TITLE_QUESTION_RE.search("What is the document title shown at the top?")

    def test_does_not_match_unrelated_question(self):
        assert not _TITLE_QUESTION_RE.search(
            "What is the total amount for transaction 123?"
        )


class TestTitleShortcutAnswer:
    def test_returns_none_for_non_title_question(self):
        assert (
            _title_shortcut_answer("What is the total amount for transaction 123?")
            is None
        )

    def test_extracts_title_line_from_document_summary_hit(self):
        with patch(
            "src.answer_pipeline.retrieve",
            return_value=[
                _summary_hit("POLICY FOR THE PROCUREMENT OF GOODS AND SERVICES (PGS)")
            ],
        ):
            result = _title_shortcut_answer("What is the title of this document?")
        assert result is not None
        title, collected = result
        assert title == "POLICY FOR THE PROCUREMENT OF GOODS AND SERVICES (PGS)"
        assert len(collected) == 1
        assert "file=doc_001_procurement_policy.md" in collected[0]

    def test_returns_none_when_no_hit_has_title_line(self):
        no_title_hit = {
            "content": "## Document Summary\n\nDocument ID: doc_002\nFile: doc_002.md\n\nNo title line here.",
            "metadata": {"file_name": "doc_002.md"},
            "score": 0.9,
        }
        with patch("src.answer_pipeline.retrieve", return_value=[no_title_hit]):
            assert _title_shortcut_answer("What is the title of this document?") is None

    def test_answer_query_bypasses_the_agent_entirely(self):
        """agent=None would crash any code path that touches the agent -- if this
        returns cleanly, the shortcut never reached the normal answer flow."""
        with patch(
            "src.answer_pipeline.retrieve",
            return_value=[
                _summary_hit("CUSTOMER INVOICE", "doc_005_fueling_records_invoice.md")
            ],
        ):
            result = answer_query(None, "What is the title of this document?")
        assert result["answer"] == "CUSTOMER INVOICE"
        assert result["sql"] == []
        assert result["sources"][0]["filename"].startswith("doc_005")
