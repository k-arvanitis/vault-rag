"""Unit tests for the pure post-processing helpers in eval/run_eval.py."""
from __future__ import annotations

from eval.run_eval import _accuracy_by_type, _render_failures_md


def test_accuracy_by_type_averages_per_question_type():
    rows = [
        {"question_type": "single_doc_factoid", "correctness": 1.0},
        {"question_type": "single_doc_factoid", "correctness": 0.5},
        {"question_type": "unanswerable", "correctness": 1.0},
        {"question_type": "cross_document_compare", "correctness": None},
    ]
    out = _accuracy_by_type(rows)
    assert out["single_doc_factoid"] == {"count": 2, "correctness": 0.75}
    assert out["unanswerable"] == {"count": 1, "correctness": 1.0}
    assert "cross_document_compare" not in out


def test_render_failures_md_lists_only_low_scoring_rows():
    rows = [
        {
            "qa_id": "qa_1",
            "question_type": "numeric_lookup",
            "correctness": 0.2,
            "question": "How much?",
            "gold_answer": "$5",
            "predicted_answer": "$50",
        },
        {
            "qa_id": "qa_2",
            "question_type": "single_doc_factoid",
            "correctness": 1.0,
            "question": "Who signed it?",
            "gold_answer": "Jane",
            "predicted_answer": "Jane",
        },
    ]
    out = _render_failures_md(rows, threshold=0.5)
    assert "qa_1" in out
    assert "qa_2" not in out
    assert "numeric extraction" in out
    assert "1 of 2 questions" in out
