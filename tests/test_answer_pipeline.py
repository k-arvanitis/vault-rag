"""Tests for the title-question shortcut in src/answer_pipeline.py."""

from __future__ import annotations

from unittest.mock import patch

from src.answer_pipeline import (
    _TITLE_QUESTION_RE,
    _title_shortcut_answer,
    answer_one,
    answer_query,
    parse_sources,
)


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


class TestParseSourcesContract:
    def test_source_dict_carries_page_and_doc_fields(self):
        header = (
            "[1] file=doc_001_procurement_policy.pdf chunk=4 page=7 "
            "score=0.9123 doc_id=doc_001 title=Procurement%20Policy"
        )
        body = "## Some Section\n\nThe actual chunk text goes here."
        sources = parse_sources([f"{header}\n{body}"])
        assert len(sources) == 1
        source = sources[0]
        assert source["page"] == 7
        assert source["document_id"] == "doc_001"
        assert source["document_title"] == "Procurement Policy"
        assert source["sheet"] is None
        assert source["chunk_id"] is not None
        assert source["score"] == 0.9123

    def test_source_dict_page_absent_when_not_provided(self):
        header = "[1] file=doc_002.xlsx sheet=Sheet1 score=0.8"
        body = "Sheet summary: some table content."
        sources = parse_sources([f"{header}\n{body}"])
        assert len(sources) == 1
        source = sources[0]
        assert source["page"] is None
        assert source["sheet"] == "Sheet1"

    def test_quote_strips_prev_next_chunk_neighbor_wrapper(self):
        """[prev chunk]/[next chunk] context injected by _format_hits for the LLM's
        benefit must not leak into the citation's quote — it isn't real PDF text,
        so leaving it in would break fitz.search_for-based highlighting."""
        header = "[1] file=doc_001.pdf chunk=4 page=12 score=4.6484"
        body = (
            "[prev chunk]\nSome earlier paragraph.\n\n"
            "[this chunk]\nThe actual retrieved passage that matches the PDF.\n\n"
            "[next chunk]\nSome later paragraph."
        )
        sources = parse_sources([f"{header}\n{body}"])
        assert len(sources) == 1
        assert (
            sources[0]["quote"] == "The actual retrieved passage that matches the PDF."
        )

    def test_quote_strips_figure_block_synthetic_text(self):
        """A [FIGURE_START]/[FIGURE_END] block holds a VLM-generated description,
        not real PDF text — leaving it in the quote breaks fitz.search_for-based
        highlighting since that text was never on the actual page."""
        header = "[1] file=doc_001.pdf chunk=4 page=12 score=1.0"
        body = (
            "Contracts are used for complex Goods and/or Services. Whenever "
            "possible, the use of Contracts is preferred.\n\n"
            "[FIGURE_START]\nThe image displays a logo.\n[FIGURE_END]"
        )
        sources = parse_sources([f"{header}\n{body}"])
        assert len(sources) == 1
        quote = sources[0]["quote"]
        assert "FIGURE" not in quote
        assert "logo" not in quote
        assert quote.startswith("Contracts are used for complex Goods")

    def test_eight_cap_preserves_one_slot_per_distinct_file(self):
        """A redundant re-query of a document that already has plenty of chunks
        must not crowd a different document's genuinely retrieved chunk out of
        the capped list entirely -- reproduced in a cross-document comparison
        where a wasted re-query of the already-answered document filled all 8
        slots before the other document's chunk was ever considered."""
        many_doc_a_chunks = [
            f"[1] file=doc_a.pdf chunk={i} page={i}\nChunk body text number {i}."
            for i in range(9)
        ]
        one_doc_b_chunk = "[1] file=doc_b.pdf chunk=0 page=1\nThe only doc_b chunk."
        sources = parse_sources(many_doc_a_chunks + [one_doc_b_chunk])
        filenames = {s["filename"] for s in sources}
        assert len(sources) == 8
        assert "doc_b.pdf" in filenames


class TestForcedDocScope:
    """The UI's source-scope control (Ask across: one document) forces routing
    to a specific doc_id instead of the semantic auto-detection route_question
    normally does — verifies the routing directive names the right tool/doc and
    that route_question (the auto-detect path) is never called when bypassed."""

    def test_forced_pdf_doc_id_routes_to_search_knowledge_base(self):
        with (
            patch("src.answer_pipeline.route_question") as mock_route,
            patch(
                "src.answer_pipeline.run_once",
                return_value=("An answer.", [], {}),
            ) as mock_run_once,
        ):
            answer_one(
                agent=object(),
                question="What does this say?",
                forced_doc_id="doc_001_procurement_policy.pdf",
            )
        mock_route.assert_not_called()
        directed_question = mock_run_once.call_args[0][1]
        assert "search_knowledge_base" in directed_question
        assert "doc_001_procurement_policy.pdf" in directed_question

    def test_forced_spreadsheet_doc_id_routes_to_query_excel(self):
        with (
            patch("src.answer_pipeline.route_question") as mock_route,
            patch(
                "src.answer_pipeline.run_once",
                return_value=("An answer.", [], {}),
            ) as mock_run_once,
        ):
            answer_one(
                agent=object(),
                question="What's the total?",
                forced_doc_id="doc_006_purchase_card_transactions.xlsx",
            )
        mock_route.assert_not_called()
        directed_question = mock_run_once.call_args[0][1]
        assert "query_excel" in directed_question

    def test_answer_query_skips_title_shortcut_when_doc_forced(self):
        """The title shortcut's own retrieve() call isn't scoped to a document,
        so it must not short-circuit a scoped query."""
        with (
            patch("src.answer_pipeline._title_shortcut_answer") as mock_shortcut,
            patch(
                "src.answer_pipeline.answer_one",
                return_value=("An answer.", [], {}),
            ),
        ):
            answer_query(
                agent=object(),
                question="What is the title of this document?",
                forced_doc_id="doc_001_procurement_policy.pdf",
            )
        mock_shortcut.assert_not_called()


def _src(filename: str, document_id: str | None = None) -> dict:
    return {
        "filename": filename,
        "document_id": document_id,
        "page": 1,
        "quote": "x",
        "section": None,
    }


class TestComparisonMissingMentionedDocRetry:
    """Reproduced live: the model wrote off a comparison's missing side with
    different wording each run ("Unsupported", "No details ... are provided"),
    so a keyword check on the answer text is unreliable. The doc_id-coverage
    check doesn't depend on wording at all -- it just checks whether every
    doc_id named in the question actually has a matching source."""

    def test_missing_named_doc_triggers_retry_regardless_of_answer_wording(self):
        with (
            patch("src.answer_pipeline.parse_sources") as mock_parse,
            patch(
                "src.answer_pipeline.run_once",
                side_effect=[
                    (
                        "doc_010 says leave is unpaid. No details about leave "
                        "policies are provided in the retrieved content from doc_009.",
                        [],
                        {},
                    ),
                    (
                        "doc_009: compassionate leave is paid up to 5 days.\n"
                        "doc_010: leave is unpaid.",
                        [],
                        {},
                    ),
                ],
            ) as mock_run_once,
        ):
            mock_parse.side_effect = [
                [_src("doc_010_handbook.pdf", "doc_010")],
                [
                    _src("doc_009_hr.pdf", "doc_009"),
                    _src("doc_010_handbook.pdf", "doc_010"),
                ],
            ]
            ans, _coll, _tr = answer_one(
                agent=object(),
                question="Compare the leave policies in doc_009 and doc_010",
            )
        assert mock_run_once.call_count == 2
        retry_question = mock_run_once.call_args_list[1][0][1]
        assert "doc_009" in retry_question
        assert "no evidence" in retry_question
        assert "compassionate leave" in ans

    def test_no_retry_when_both_named_docs_covered(self):
        with (
            patch("src.answer_pipeline.parse_sources") as mock_parse,
            patch(
                "src.answer_pipeline.run_once",
                return_value=("doc_009: a. doc_010: b.", [], {}),
            ) as mock_run_once,
        ):
            mock_parse.return_value = [
                _src("doc_009_hr.pdf", "doc_009"),
                _src("doc_010_handbook.pdf", "doc_010"),
            ]
            answer_one(
                agent=object(),
                question="Compare the leave policies in doc_009 and doc_010",
            )
        mock_run_once.assert_called_once()


class TestComparisonPartialUnsupportedRetry:
    """A comparison answer can name both documents (n_sources >= 2) yet still
    write off one side as "Unsupported" in the synthesized text -- reproduced
    live: both documents were actually retrieved, but the model gave up on one
    side because its content spanned several related sub-topics. The retry
    trigger must catch this even though the retrieved-source-count check alone
    doesn't see a problem."""

    def test_partial_unsupported_with_two_sources_triggers_retry(self):
        with (
            patch("src.answer_pipeline.parse_sources") as mock_parse,
            patch(
                "src.answer_pipeline.run_once",
                side_effect=[
                    ("doc_009: Unsupported\ndoc_010: real answer.", [], {}),
                    ("doc_009: a real answer.\ndoc_010: real answer.", [], {}),
                ],
            ) as mock_run_once,
        ):
            mock_parse.return_value = [_src("doc_009_hr.pdf"), _src("doc_010_handbook.pdf")]
            ans, _coll, _tr = answer_one(
                agent=object(),
                question="Compare the leave policies in doc_009 and doc_010",
            )
        assert mock_run_once.call_count == 2
        retry_question = mock_run_once.call_args_list[1][0][1]
        assert "already retrieved relevant content for both documents" in retry_question
        assert ans == "doc_009: a real answer.\ndoc_010: real answer."

    def test_no_retry_when_no_unsupported_fragment(self):
        with (
            patch("src.answer_pipeline.parse_sources") as mock_parse,
            patch(
                "src.answer_pipeline.run_once",
                return_value=("doc_009: a. doc_010: b.", [], {}),
            ) as mock_run_once,
        ):
            mock_parse.return_value = [_src("doc_009_hr.pdf"), _src("doc_010_handbook.pdf")]
            answer_one(
                agent=object(),
                question="Compare the leave policies in doc_009 and doc_010",
            )
        mock_run_once.assert_called_once()

    def test_retry_keeps_original_when_still_partial(self):
        """If the retry doesn't actually fix it, keep the first answer rather
        than silently swapping in an equally-broken retry."""
        with (
            patch("src.answer_pipeline.parse_sources") as mock_parse,
            patch(
                "src.answer_pipeline.run_once",
                side_effect=[
                    ("doc_009: Unsupported\ndoc_010: real answer.", [], {}),
                    ("doc_009: Unsupported\ndoc_010: real answer.", [], {}),
                ],
            ),
        ):
            mock_parse.return_value = [_src("doc_009_hr.pdf"), _src("doc_010_handbook.pdf")]
            ans, _coll, _tr = answer_one(
                agent=object(),
                question="Compare the leave policies in doc_009 and doc_010",
            )
        assert ans == "doc_009: Unsupported\ndoc_010: real answer."
