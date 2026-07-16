from eval.generate_summary_doc import render


def _fake_summary() -> dict:
    return {
        "benchmark_date": "2026-01-01",
        "answer_model": "test-model",
        "judge_model": "test-judge",
        "document_count": 5,
        "question_count": 10,
        "vector_retrieval_metrics": {
            "question_count": 7,
            "hit_at_5": 0.9,
            "hit_at_10": 0.95,
            "mrr": 0.8,
            "evidence_recall_at_10": 0.85,
            "evidence_recall_at_20": 0.9,
        },
        "structured_retrieval_metrics": {"question_count": 2, "answer_accuracy": 0.75},
        "unanswerable_metrics": {"question_count": 1, "correct_refusal_rate": 1.0},
        "agent_answer_metrics": {
            "correctness": 0.8,
            "faithfulness": 0.85,
            "answer_relevancy": 0.9,
        },
        "judge_breakdown": {"custom_llm_judge": 10},
        "correctness_by_question_type": {
            "table_lookup": {"count": 3, "correctness": 1.0},
            "ocr_extraction": {"count": 7, "correctness": 0.5},
        },
    }


class TestRenderSummaryDoc:
    def test_renders_canonical_metadata_fields(self):
        out = render(_fake_summary())
        assert "2026-01-01" in out
        assert "test-model" in out
        assert "test-judge" in out
        assert "**Documents:** 5" in out
        assert "**Questions:** 10" in out

    def test_renders_headline_percentages(self):
        out = render(_fake_summary())
        assert "80.0%" in out  # correctness
        assert "85.0%" in out  # faithfulness
        assert "90.0%" in out  # relevancy or evidence_recall_at_20 (both 0.9)

    def test_sorts_question_types_by_correctness_descending(self):
        out = render(_fake_summary())
        table_idx = out.index("table_lookup")
        ocr_idx = out.index("ocr_extraction")
        assert table_idx < ocr_idx

    def test_does_not_fabricate_a_coverage_metric(self):
        """Multi-document evidence coverage isn't measured by this run --
        the doc must say so, not invent a number for it."""
        out = render(_fake_summary())
        assert "Known gap" in out
        assert "not yet a formal benchmark metric" in out
