"""Tests for the multi-part question split in src/answer_quality.py."""

from __future__ import annotations

from src.answer_quality import (
    _is_bare_filename_answer,
    _is_multi_part_query,
    _split_multi_part_query,
)


class TestMultiPartSplitBoundaryCoverage:
    """_is_multi_part_query's "\\?\\s+and\\s+for\\b" detection pattern must have a
    matching split rule in _split_multi_part_query, or a detected multi-part
    question silently falls through to "return unsplit" -- reproduced live via
    eval qa_id doc_006_doc_007_cross_document_qa__qa_2, whose "? And for ..."
    second half was silently dropped from the final answer entirely."""

    def test_question_mark_and_for_boundary_is_detected_and_split(self):
        q = (
            "For Supplier Name 'Screwfix Direct' on 2025-04-03 in PLACE / STREET "
            "SCENE what is the Purchase of Expenditure and NET Amount? And for "
            "Transaction Number 6089041 what is the Summary of Purpose of "
            "Expenditure and Total?"
        )
        assert _is_multi_part_query(q)
        parts = _split_multi_part_query(q)
        assert len(parts) == 2
        assert parts[0].endswith("NET Amount?")
        assert parts[1].startswith("And for Transaction Number 6089041")


class TestBareFilenameGuard:
    """_is_bare_filename_answer must catch a filename NAMED with no extracted
    value -- reproduced live: unanswerable_qa__qa_5 answered
    "doc_005_fueling_records_invoice" (no extension), which the extension-
    only regex missed entirely."""

    def test_extensionless_stem_is_flagged(self):
        q = "Which document gives the GPS coordinates of Llano Airport fuel transactions?"
        assert _is_bare_filename_answer(q, "doc_005_fueling_records_invoice")

    def test_full_filename_still_flagged(self):
        q = "Which document gives the GPS coordinates of Llano Airport fuel transactions?"
        assert _is_bare_filename_answer(q, "doc_005_fueling_records_invoice.pdf")

    def test_bare_doc_id_in_real_comparison_answer_not_flagged(self):
        q = "Which document defines employment rules, and which tracks financial data?"
        answer = "doc_009 defines the employment rules; doc_013 tracks financial data"
        assert not _is_bare_filename_answer(q, answer)

    def test_filename_with_real_extracted_value_not_flagged(self):
        q = "What is the notice period?"
        answer = "14 days, per doc_009_hr_policy.pdf"
        assert not _is_bare_filename_answer(q, answer)
