"""Tests for the title-question shortcut in src/answer_pipeline.py."""

from __future__ import annotations

from unittest.mock import patch

from src.answer_pipeline import (
    _TITLE_QUESTION_RE,
    _excel_citations_to_sources,
    _title_shortcut_answer,
    answer_one,
    answer_query,
    build_citation_map,
    parse_sources,
    stream_answer,
    strip_leaked_headers,
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


class TestLeakedReasoningChannelMarker:
    """gpt-oss's harmony response format names its hidden chain-of-thought
    channel "analysis" and the real answer channel "final" -- reproduced live
    via OpenRouter streaming that this boundary isn't always cleanly stripped,
    leaking the bare channel-name word glued directly onto the real answer
    with no space ("1. final5239.0"). Must not strip legitimate prose use of
    either word, which always has a space after it."""

    def test_strips_leaked_final_marker_glued_to_digit(self):
        assert strip_leaked_headers("1. final5239.0") == "1. 5239.0"

    def test_strips_leaked_final_marker_glued_to_word(self):
        assert (
            strip_leaked_headers("finalJuly 1 2024 – September 30 2024")
            == "July 1 2024 – September 30 2024"
        )

    def test_strips_leaked_analysis_marker(self):
        assert (
            strip_leaked_headers("2. analysisThe query_excel returned X")
            == "2. The query_excel returned X"
        )

    def test_does_not_strip_legitimate_prose_use_of_final(self):
        assert strip_leaked_headers("The final amount due is 500.00.") == (
            "The final amount due is 500.00."
        )

    def test_does_not_strip_legitimate_prose_use_of_analysis(self):
        assert strip_leaked_headers(
            "Further analysis shows the lease expired in 2024."
        ) == "Further analysis shows the lease expired in 2024."


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


class TestInlineCitationRenumbering:
    """The model's [N] markers number the last tool call's raw hits, which
    don't match the final deduped/reordered sources list -- build_citation_map
    resolves each to its real position so a true inline [N] can be shown
    instead of stripped."""

    def _one_call(self, *headers_and_bodies: tuple[str, str]) -> list[str]:
        return [f"{h}\n{b}" for h, b in headers_and_bodies]

    def test_renumbers_marker_to_final_source_position(self):
        collected = self._one_call(
            ("[1] file=doc_a.pdf chunk=3 page=2 score=0.5", "First chunk text."),
            ("[2] file=doc_b.pdf chunk=1 page=1 score=0.9", "Second chunk text."),
        )
        sources = parse_sources(collected)
        # parse_sources iterates calls in reverse and there's only one call here,
        # so order matches insertion: doc_a then doc_b (positions 1, 2).
        citation_map = build_citation_map(collected, sources)
        assert citation_map == {1: 1, 2: 2}
        answer = strip_leaked_headers("The value is X [1] and Y [2].", citation_map)
        assert answer == "The value is X [1] and Y [2]."

    def test_renumbers_marker_when_diversity_cap_reorders_positions(self):
        """parse_sources moves a file's second chunk after every other distinct
        file's first chunk (the 8-cap diversity guarantee) -- the model's raw
        marker order and the final sources order can genuinely diverge."""
        collected = self._one_call(
            ("[1] file=doc_a.pdf chunk=1 page=1 score=0.9", "Doc A first chunk."),
            ("[2] file=doc_a.pdf chunk=2 page=2 score=0.8", "Doc A second chunk."),
            ("[3] file=doc_b.pdf chunk=1 page=1 score=0.7", "Doc B first chunk."),
        )
        sources = parse_sources(collected)
        filenames_in_order = [s["filename"] for s in sources]
        assert filenames_in_order == ["doc_a.pdf", "doc_b.pdf", "doc_a.pdf"]
        citation_map = build_citation_map(collected, sources)
        assert citation_map == {1: 1, 2: 3, 3: 2}

    def test_unresolvable_marker_in_range_still_stripped(self):
        """A marker the model emitted that doesn't correspond to any chunk in
        the last call (hallucinated, or referencing an earlier call) falls
        back to the old strip-on-sight behavior rather than leaking a wrong
        number."""
        answer = strip_leaked_headers("The value is X [1].", citation_map={})
        assert answer == "The value is X."

    def test_marker_beyond_tool_result_range_left_alone(self):
        """A bracketed year like [2024] is never a citation marker -- must not
        be touched even with an empty citation_map."""
        assert strip_leaked_headers("Filed in [2024].", citation_map={}) == "Filed in [2024]."


class TestStreamAnswer:
    """stream_answer streams tokens live for the common single-part case, but
    falls back to the full non-streaming answer_query pipeline (as one lump
    token) for comparison/multi-part questions -- those need the complete
    answer text before their retry/merge logic can run."""

    def test_streams_tokens_live_for_single_part_question(self):
        def fake_stream_agent(agent, query, **kwargs):
            kwargs["tool_calls"].append("search_knowledge_base")
            yield "Hello "
            yield "world."

        with (
            patch("src.answer_pipeline.route_question", return_value={}),
            patch("src.answer_pipeline.stream_agent", side_effect=fake_stream_agent),
        ):
            events = list(stream_answer(agent=object(), question="What is X?"))

        token_events = [e for e in events if "token" in e]
        assert [e["token"] for e in token_events] == ["Hello ", "world."]
        final = events[-1]
        assert final["done"] is True
        assert final["answer"] == "Hello world."
        assert final["tools"] == ["search_knowledge_base"]

    def test_retries_first_attempt_unsupported_like_answer_one(self):
        """answer_one retries a bare "Unsupported" first attempt (real
        Groq/temp=0 nondeterminism, see its docstring) -- the single-part
        streaming path must keep that safety net, not silently drop it just
        because it can't be done live token-by-token."""

        def fake_stream_agent(agent, query, **kwargs):
            yield "Unsupported"

        with (
            patch("src.answer_pipeline.route_question", return_value={}),
            patch("src.answer_pipeline.stream_agent", side_effect=fake_stream_agent),
            patch(
                "src.answer_pipeline.run_once",
                return_value=("The real answer.", [], {"sql": [], "tools": ["search_knowledge_base"], "rejected": []}),
            ) as mock_run_once,
        ):
            events = list(stream_answer(agent=object(), question="What is X?"))

        mock_run_once.assert_called_once()
        assert mock_run_once.call_args.kwargs["attempt"] == "unsupported-retry"
        final = events[-1]
        assert final["done"] is True
        assert final["answer"] == "The real answer."

    def test_comparison_question_falls_back_to_full_pipeline(self):
        canned = {
            "answer": "Doc A allows longer.",
            "sources": [],
            "sql": [],
            "tools": ["search_knowledge_base"],
            "rejected_sources": [],
        }
        with patch("src.answer_pipeline.answer_query", return_value=canned) as mock_aq:
            events = list(
                stream_answer(
                    agent=object(),
                    question="Comparing doc_a and doc_b, which allows a longer extension?",
                )
            )
        mock_aq.assert_called_once()
        assert events[0] == {"token": "Doc A allows longer."}
        assert events[-1]["done"] is True
        assert events[-1]["answer"] == "Doc A allows longer."

    def test_multi_part_question_falls_back_to_full_pipeline(self):
        canned = {
            "answer": "1. First part.\n\n2. Second part.",
            "sources": [],
            "sql": [],
            "tools": [],
            "rejected_sources": [],
        }
        with patch("src.answer_pipeline.answer_query", return_value=canned) as mock_aq:
            events = list(
                stream_answer(
                    agent=object(),
                    question="What is the title, and what is the effective date?",
                )
            )
        mock_aq.assert_called_once()
        assert events[-1]["answer"] == canned["answer"]


class TestExcelCitationsToSources:
    """query_excel never emits retrieval chunks (its result is SQL, not
    chunks — see run_once), so a SQL-answered question used to get sources: []
    every time, and SpreadsheetEvidence had nothing to render."""

    def test_converts_citation_to_source_card(self):
        citation = {
            "source_file": "doc_006_purchase_card_transactions_q1_2025_26.xlsx",
            "sheet_name": "DataAnalysis",
            "quote": "Screwfix Direct  2025-04-03  39.54",
        }
        sources = _excel_citations_to_sources([citation], existing=[])
        assert len(sources) == 1
        source = sources[0]
        assert source["filename"] == citation["source_file"]
        assert source["sheet"] == "DataAnalysis"
        assert source["quote"] == citation["quote"]
        assert source["page"] is None

    def test_skips_citation_missing_source_file_or_sheet(self):
        assert _excel_citations_to_sources([{"sheet_name": "Sheet1"}], existing=[]) == []
        assert _excel_citations_to_sources([{"source_file": "x.xlsx"}], existing=[]) == []

    def test_dedupes_against_already_parsed_source(self):
        existing = [{"filename": "doc_006.xlsx", "sheet": "DataAnalysis"}]
        citation = {"source_file": "doc_006.xlsx", "sheet_name": "DataAnalysis"}
        assert _excel_citations_to_sources([citation], existing=existing) == []


class TestComparisonSkipsGroundingCheck:
    """Reproduced live with gpt-oss-120b: the post-generation grounding check
    flagged correct comparative answers as ungrounded and downgraded them to
    Unsupported ~1/3 of the time, even with both named documents present in
    the retrieved context. Comparison questions already get their own
    doc-coverage retry (see _comparison_incompleteness), so the grounding
    check is redundant there and actively harmful -- must be skipped."""

    def test_comparison_question_skips_grounding_check(self):
        with patch(
            "src.answer_pipeline.run_once", return_value=("An answer.", [], {})
        ) as mock_run_once:
            answer_one(agent=object(), question="Compare doc_009 and doc_010")
        assert mock_run_once.call_args.kwargs["skip_grounding_check"] is True

    def test_non_comparison_question_runs_grounding_check(self):
        with patch(
            "src.answer_pipeline.run_once", return_value=("An answer.", [], {})
        ) as mock_run_once:
            answer_one(agent=object(), question="What is the total invoice amount?")
        assert mock_run_once.call_args.kwargs["skip_grounding_check"] is False


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
