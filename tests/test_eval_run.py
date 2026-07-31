"""Unit tests for the pure post-processing helpers in eval/run_eval.py."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from eval.run_eval import (
    _accuracy_by_type,
    _citation_evidence_metrics,
    _custom_judge_answer,
    _exact_match_score,
    _render_failures_md,
)


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


class TestExactMatchDateEquivalence:
    """DD/MM/YYYY and ISO dates are the same fact in different formats --
    ingestion normalizes all spreadsheet dates to ISO (src/duckdb_store.py's
    _normalize_dates), so a SQL-answered date can never literally match a
    DD/MM/YYYY gold string; _canonicalize_dates must treat them as equal
    before token comparison, not just via substring match."""

    def test_slash_dmy_matches_iso_prose_answer(self):
        # The exact reproduced case: doc_007_published_spend_report_april_25_qa__qa_2.
        assert (
            _exact_match_score(
                "2025-04-29, the recorded date for transaction number 6123276",
                "29/04/2025",
            )
            == 1.0
        )

    def test_slash_dmy_matches_iso_bare_value(self):
        assert _exact_match_score("2005-12-15", "15/12/2005") == 1.0

    def test_unrelated_dates_do_not_match(self):
        assert _exact_match_score("2025-01-01", "31/12/1999") == 0.0


class TestJudgePromptSkipFaithfulness:
    """Excel/SQL-answered and unanswerable questions have no retrieved TEXT
    context (query_excel returns a value, not passages). Reproduced live
    2026-07-22: sending the full faithfulness-scoring rules alongside an
    empty "RETRIEVED CONTEXT: No context retrieved." section made gpt-4o-mini
    score CORRECTNESS 0.0 on demonstrably correct Excel comparison answers --
    the faithfulness rules bled into correctness judgment. skip_faithfulness
    must drop both the faithfulness rules and the RETRIEVED CONTEXT section
    from the prompt entirely, not just null the returned score afterward."""

    def _mock_client(self, response_json: dict):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(response_json)
        mock_client.chat.completions.create.return_value = mock_response
        return mock_client

    @patch("eval.run_eval._judge_config", return_value=("gpt-4o-mini", "http://x", "key"))
    def test_skip_faithfulness_omits_context_section_from_prompt(self, _cfg):
        mock_client = self._mock_client(
            {"correctness": 1.0, "answer_relevancy": 1.0, "reason": "ok"}
        )
        with patch("openai.OpenAI", return_value=mock_client):
            _custom_judge_answer(
                {"question": "What is X?"},
                "the answer",
                "the answer",
                ["some retrieved chunk that should not appear"],
                skip_faithfulness=True,
            )
        sent_prompt = mock_client.chat.completions.create.call_args.kwargs["messages"][
            1
        ]["content"]
        assert "RETRIEVED CONTEXT" not in sent_prompt
        assert "HEDGED/INFERRED CLAIMS" not in sent_prompt
        assert "some retrieved chunk that should not appear" not in sent_prompt

    @patch("eval.run_eval._judge_config", return_value=("gpt-4o-mini", "http://x", "key"))
    def test_default_still_includes_faithfulness_rules_and_context(self, _cfg):
        mock_client = self._mock_client(
            {"correctness": 1.0, "faithfulness": 1.0, "answer_relevancy": 1.0, "reason": "ok"}
        )
        with patch("openai.OpenAI", return_value=mock_client):
            _custom_judge_answer(
                {"question": "What is X?"},
                "the answer",
                "the answer",
                ["a real retrieved chunk"],
            )
        sent_prompt = mock_client.chat.completions.create.call_args.kwargs["messages"][
            1
        ]["content"]
        assert "RETRIEVED CONTEXT" in sent_prompt
        assert "HEDGED/INFERRED CLAIMS" in sent_prompt
        assert "a real retrieved chunk" in sent_prompt


class TestCitationEvidenceMetrics:
    """M-4 (MITIGATION_PLAN.md FM-4): offline citation-evidence metric."""

    def _row(self, **overrides):
        row = {
            "predicted_answer": "The limit is $5,000 [1].",
            "gold_evidence": [{"doc_id": "doc_001", "quote": "the limit is five thousand dollars"}],
            "sources": [
                {"document_id": "doc_001", "filename": "a.pdf", "excerpt": "The limit is five thousand dollars total."}
            ],
        }
        row.update(overrides)
        return row

    def test_coverage_and_precision_on_a_correct_citation(self):
        out = _citation_evidence_metrics([self._row()])
        assert out["question_count"] == 1
        assert out["citation_coverage"] == 1.0
        assert out["citation_precision"] == 1.0
        assert out["uncited_answer_rate"] == 0.0

    def test_cited_source_from_wrong_document_scores_zero_coverage(self):
        row = self._row(
            sources=[{"document_id": "doc_002", "filename": "b.pdf", "excerpt": "unrelated text"}]
        )
        out = _citation_evidence_metrics([row])
        assert out["citation_coverage"] == 0.0
        assert out["citation_precision"] == 0.0

    def test_uncited_answer_counted_when_not_unsupported(self):
        row = self._row(predicted_answer="The limit is $5,000.")  # no [N] marker
        out = _citation_evidence_metrics([row])
        assert out["uncited_answer_question_count"] == 1
        assert out["uncited_answer_rate"] == 1.0
        # No markers -> no cited sources -> excluded from coverage/precision, not scored 0.
        assert out["question_count"] == 0

    def test_unsupported_answer_excluded_from_uncited_rate_and_coverage(self):
        row = self._row(predicted_answer="Unsupported")
        out = _citation_evidence_metrics([row])
        assert out["uncited_answer_question_count"] == 0
        assert out["uncited_answer_rate"] is None
        assert out["question_count"] == 0

    def test_row_without_persisted_sources_excluded_not_zero(self):
        row = self._row(sources=None)
        out = _citation_evidence_metrics([row])
        assert out["question_count"] == 0
        assert out["citation_coverage"] is None
        assert out["citation_precision"] is None
        # Uncited-answer rate needs only predicted_answer, so it's still computed.
        assert out["uncited_answer_rate"] == 0.0

    def test_empty_input_returns_none_not_zero_division(self):
        out = _citation_evidence_metrics([])
        assert out == {
            "question_count": 0,
            "citation_coverage": None,
            "citation_precision": None,
            "uncited_answer_question_count": 0,
            "uncited_answer_rate": None,
        }


class TestLenientQuoteMatcherReuse:
    """_contains (imported from eval/gold_rank_at_1.py, not reimplemented) backs
    citation precision -- exercise its three documented behaviours in that role."""

    def test_exact_normalized_substring_matches(self):
        from eval.gold_rank_at_1 import _contains

        assert _contains(
            "Excerpt text: the limit is five thousand dollars, per policy.",
            "the limit is five thousand dollars",
        )

    def test_eight_word_window_tolerates_markdown_ocr_drift(self):
        from eval.gold_rank_at_1 import _contains

        # Same 8+ content words, but reflowed with markdown/OCR noise in between.
        excerpt = "**The** limit is  five\nthousand   dollars **per fiscal year** policy."
        quote = "The limit is five thousand dollars per fiscal"
        assert _contains(excerpt, quote)

    def test_genuine_non_match_returns_false(self):
        from eval.gold_rank_at_1 import _contains

        assert not _contains(
            "This chunk discusses vendor onboarding timelines entirely.",
            "the limit is five thousand dollars per fiscal year",
        )
