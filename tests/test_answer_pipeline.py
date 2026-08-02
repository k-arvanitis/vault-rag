"""Tests for the title-question shortcut in src/answer_pipeline.py."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from vault_rag.answer_pipeline import (
    _TITLE_QUESTION_RE,
    _condense_followup_question,
    _excel_citations_to_sources,
    _is_malformed_generation,
    _narrow_quotes_to_answer,
    _resolve_comparison_doc_ids,
    _resolve_comparison_doc_ids_llm,
    _title_shortcut_answer,
    answer_comparison_deterministic,
    answer_one,
    answer_query,
    build_citation_map,
    parse_sources,
    stream_answer,
    strip_leaked_headers,
)
from vault_rag.rag_agent import FinalCorrection


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
            "vault_rag.answer_pipeline.retrieve",
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
        with patch("vault_rag.answer_pipeline.retrieve", return_value=[no_title_hit]):
            assert _title_shortcut_answer("What is the title of this document?") is None

    def test_answer_query_bypasses_the_agent_entirely(self):
        """agent=None would crash any code path that touches the agent -- if this
        returns cleanly, the shortcut never reached the normal answer flow."""
        with patch(
            "vault_rag.answer_pipeline.retrieve",
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
        assert (
            strip_leaked_headers("Further analysis shows the lease expired in 2024.")
            == "Further analysis shows the lease expired in 2024."
        )


class TestMalformedGeneration:
    """gpt-oss occasionally never separates hidden reasoning / a raw tool-call
    payload from the real answer at all, returning it whole as the "answer" --
    reproduced live in the 2026-07-21 eval run (n=2/109): a full raw harmony
    reasoning dump, and a bare tool-call JSON object. Both were gold=Unsupported
    refusal questions that scored 0.0 because the leaked garbage wasn't
    recognized as a failed generation."""

    def test_detects_raw_harmony_channel_dump(self):
        # Reproduced live: maturity_dataset_2025_qa__qa_2
        assert _is_malformed_generation(
            "<|channel|>commentary<|message|>We need to query the sheet that "
            "contains ranking. Possibly a sheet named score includes total scores"
        )

    def test_detects_bare_tool_call_json(self):
        # Reproduced live: transactions_q1_2025_26_qa__qa_9
        assert _is_malformed_generation(
            '{\n  "action": "search_knowledge_base",\n  "arguments": {\n'
            '    "query": "fueling receipts"\n  }\n}'
        )

    def test_detects_bracket_tool_call_leak(self):
        # Reproduced live 2026-07-21, doc_011 spain maturity question
        assert _is_malformed_generation(
            '[search_knowledge_base: topic="Spain open data maturity questionnaire"]'
        )
        assert _is_malformed_generation(
            '[search_knowledge_base query="fueling records invoice"]'
        )
        assert _is_malformed_generation('[query_excel query="SELECT * FROM t"]')

    def test_detects_tool_key_json_leak(self):
        # Reproduced live 2026-07-22: doc_015_food_sop_manual_qa__qa_13 --
        # a different key name ("tool" not "action") than the already-caught
        # bare tool-call JSON shape, same underlying failure.
        assert _is_malformed_generation(
            '{\n  "tool": "search_knowledge_base",\n  "parameters": {\n'
            '    "query": "penalty amount for failing to submit SOPs",\n'
            '    "doc_id": "doc_015"\n  }\n}'
        )

    def test_detects_json_leak_after_narrated_prose(self):
        # Reproduced live tonight forcing the excel-modality hard block
        # (FORCED_MODALITY): the model narrated its next move in prose
        # before the leaked JSON, so an anchored "^" pattern (which only
        # caught a leak that WAS the whole generation) missed it.
        assert _is_malformed_generation(
            "We must call search_knowledge_base with topic words alone to find "
            'relevant doc summary chunk and get doc_id. Topic: "total NET Amount '
            'spent on MATERIALS".Let\'s call.{\n  "action": "search_knowledge_base",'
            '\n  "arguments": {\n    "query": "total NET Amount spent on MATERIALS"'
            "\n  }\n}"
        )

    def test_detects_narrated_reasoning_with_no_structured_artifact(self):
        # Reproduced live tonight forcing the excel-modality hard block: the
        # model narrates its own next tool call in plain prose with no
        # JSON/bracket leftover at all -- none of the structured-shape
        # patterns above catch this, only the bare tool-name mention does.
        assert _is_malformed_generation(
            "We must call search_knowledge_base with topic words alone to "
            'identify relevant doc summary chunk. The topic maybe "materials '
            'net amount spent".'
        )

    def test_does_not_flag_a_real_answer(self):
        assert not _is_malformed_generation("The total is $297 billion.")

    def test_does_not_flag_prose_mentioning_action_word(self):
        assert not _is_malformed_generation(
            "The board approved the action plan on March 1."
        )


class TestStripsInlineLeakedSourceHeader:
    """The model sometimes echoes a raw chunk header fragment mid-sentence
    instead of on its own line ("$297 billion. Source: file=doc_003.pdf") --
    the whole-line-anchored _LEAKED_HEADER_RE never matches that."""

    def test_strips_inline_source_file_leak(self):
        assert (
            strip_leaked_headers(
                "$297 billion. Source: file=doc_003_fed_annual_report_2024.pdf"
            )
            == "$297 billion."
        )

    def test_strips_inline_source_file_leak_with_chunk_and_page(self):
        assert (
            strip_leaked_headers(
                "The answer is 42. Sources: file=doc_008.pdf chunk=1 page=1"
            )
            == "The answer is 42."
        )

    def test_strips_dangling_source_label_with_no_file(self):
        """Model trails off before ever writing "file=..." -- a bare label
        glued mid-sentence at end of line, reproduced live."""
        assert (
            strip_leaked_headers("1. $297 billion. Source:\n\n2. 29 [1]")
            == "1. $297 billion.\n\n2. 29"
        )

    def test_does_not_strip_prose_source_with_content_on_same_line(self):
        assert (
            strip_leaked_headers("See the primary source: the annual report.")
            == "See the primary source: the annual report."
        )

    def test_strips_dagger_glued_file_citation(self):
        """gpt-oss emits [N†file=doc_X.pdf] citations; _INLINE_CITATION_RE's
        \\[(\\d+)\\] never matches (a dagger follows the digit), so the raw
        filename leaked into answers -- reproduced live 2026-07-21."""
        assert (
            strip_leaked_headers(
                "The agreement is governed by English law"
                "[22†file=doc_002_services_contract_terms.pdf]."
            )
            == "The agreement is governed by English law."
        )

    def test_resolves_dagger_citation_to_real_source_position(self):
        """A dagger citation whose N is a real, resolvable citation index
        (per citation_map) must renumber to that position, not get stripped
        -- reproduced live 2026-07-21: gpt-oss's ONLY citation in the answer
        was this dagger form, so blind stripping left zero [N] markers and
        the UI fell back to showing every retrieved candidate as cited."""
        assert (
            strip_leaked_headers(
                "Sole Source Procurements must be approved by the "
                "CEO【1†file=doc_001_procurement_policy.pdf】",
                citation_map={1: 1},
            )
            == "Sole Source Procurements must be approved by the CEO[1]"
        )

    def test_strips_parenthetical_bare_file_leak(self):
        assert (
            strip_leaked_headers(
                "Vacation policy (file=doc_010_rosemont_employee_handbook_2024.pdf) applies."
            )
            == "Vacation policy applies."
        )

    def test_strips_bare_file_after_stripped_marker(self):
        assert (
            strip_leaked_headers(
                "30 days to complete a harassment investigation. "
                "[6] file=doc_010_rosemont_employee_handbook_2024.pdf"
            )
            == "30 days to complete a harassment investigation."
        )

    def test_strips_dangling_source_label_glued_to_unresolved_citation(self):
        """ "Source: [1]" where [1] doesn't resolve to a real source (no
        citation_map) is stripped as a whole -- reproduced live in a
        multi-part answer's second part."""
        assert (
            strip_leaked_headers("1. $297 billion. Source:\n\n2. 29. Source: [1]")
            == "1. $297 billion.\n\n2. 29."
        )

    def test_keeps_source_label_when_citation_resolves(self):
        """ "Source: [N]" is the model's own legitimate citation style when
        [N] resolves via citation_map -- must not be treated as a leak."""
        assert (
            strip_leaked_headers("The value is 42. Source: [1].", citation_map={1: 2})
            == "The value is 42. Source: [2]."
        )


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

    def test_unformatted_tool_text_is_skipped_not_fabricated_as_unknown(self):
        """Reproduced live 2026-07-21: when a tool call returns plain text
        with no real "[N] file=..." header (e.g. a "No relevant results
        found" message), it used to become a source with filename="unknown"
        -- a fake citation for content that was never actually retrieved.
        Must be skipped entirely instead."""
        assert parse_sources(["No relevant results found for this query."]) == []

    def test_figure_bbox_nulled_when_answer_is_about_unrelated_text_in_same_chunk(self):
        """Reproduced live: a short chunk pairing real answer text with a
        verbose logo description passed the "figure is >40% of the chunk"
        gate even though the answer has nothing to do with the logo --
        _narrow_quotes_to_answer must null the bbox out once the answer
        text is known, since it has nothing to do with the figure."""
        header = "[1] file=doc_001_procurement_policy.pdf chunk=4 page=2 score=0.79"
        body = (
            "## Authorizing Manager: Ricki Contreras, Administrative Services Division\n"
            "**Original Issue Date: December 15, 2005**\n"
            "[FIGURE_START]\n"
            "<!-- bbox:[54.0, 36.0, 559.0, 64.8] -->\n"
            "The image displays a logo for LACERA in large light blue block letters "
            "with a thick black background border and a thin blue line at the bottom.\n"
            "[FIGURE_END]\n"
            "**Mandatory Review: September 2027**"
        )
        sources = parse_sources([f"{header}\n{body}"])
        assert sources[0]["figure_bbox"] is not None  # present pre-narrowing
        _narrow_quotes_to_answer(
            sources, "Ricki Contreras, Administrative Services Division"
        )
        assert sources[0]["figure_bbox"] is None
        assert "_figure_text" not in sources[0]

    def test_figure_bbox_kept_when_answer_is_about_the_figure(self):
        """A genuinely figure-grounded answer (the figure IS what the
        citation supports) must keep its bbox."""
        header = "[1] file=doc_008.pdf chunk=17 page=16 score=1.2"
        body = (
            "[FIGURE_START]\n"
            "<!-- bbox:[10.0, 20.0, 300.0, 150.0] -->\n"
            "Figure 4: Defense mission achieved the largest financial benefits, "
            "with a Budget of $197 billion identified in 2024.\n"
            "[FIGURE_END]"
        )
        sources = parse_sources([f"{header}\n{body}"])
        _narrow_quotes_to_answer(
            sources, "Defense, with $197 billion in identified financial benefits"
        )
        assert sources[0]["figure_bbox"] == [10.0, 20.0, 300.0, 150.0]

    def test_ocr_bbox_retargeted_to_cited_element_not_chunk_first_element(self):
        """Reproduced live: doc_016a page 2's chunk opens with the page title
        and RECITALS heading, but the actual cited sentence ("four-year term
        ... ended on June 30, 2019") is a later element -- parse_sources'
        default (first bbox in the chunk) pointed the crop at the title, not
        the cited text. _narrow_quotes_to_answer must re-point it once the
        real answer is known."""
        header = "[1] file=doc_016a_original_lease.pdf chunk=1 page=2 score=0.9"
        body = (
            "<!-- ocr_bbox:[100.0, 50.0, 300.0, 70.0] -->\n"
            "LEASE AGREEMENT\n\n"
            "<!-- ocr_bbox:[80.0, 90.0, 320.0, 105.0] -->\n"
            "RECITALS\n\n"
            "<!-- ocr_bbox:[70.0, 120.0, 530.0, 170.0] -->\n"
            "The four-year term of the current lease agreement between Tenant "
            "and Third-Party Beneficiary ended on June 30, 2019."
        )
        sources = parse_sources([f"{header}\n{body}"])
        assert sources[0]["ocr_bbox"] == [100.0, 50.0, 300.0, 70.0]  # default: first
        _narrow_quotes_to_answer(sources, "The four-year term ended on June 30, 2019")
        assert sources[0]["ocr_bbox"] == [70.0, 120.0, 530.0, 170.0]
        assert "_ocr_segments" not in sources[0]

    def test_excerpt_strips_page_boundary_marker(self):
        """<!-- PAGE N pymupdf4llm --> is pipeline bookkeeping, not text on the
        page -- it must not leak into the Evidence panel's quote."""
        header = "[1] file=doc_008.pdf chunk=1 page=1 score=3.38"
        body = (
            "<!-- PAGE 1 pymupdf4llm --> UNITED STATES GOVERNMENT ACCOUNTABILITY "
            "OFFICE <!-- PAGE 2 pymupdf4llm --> Highlights of the report."
        )
        sources = parse_sources([f"{header}\n{body}"])
        assert "PAGE" not in sources[0]["excerpt"]
        assert "pymupdf4llm" not in sources[0]["excerpt"]
        assert "UNITED STATES GOVERNMENT ACCOUNTABILITY OFFICE" in sources[0]["excerpt"]

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

    def test_quote_strips_orphaned_figure_end_tag_with_no_matching_start(self):
        """A chunk boundary can split a figure block so only [FIGURE_END] (no
        [FIGURE_START]) lands in this chunk's body -- reproduced live: the
        paired regex doesn't match an orphan tag, leaking it as literal
        visible text ahead of the real content (a markdown table, in the
        live case)."""
        header = "[1] file=doc_002.pdf chunk=9 page=3 score=0.8"
        body = (
            "[FIGURE_END] |Amendment:|An agreed addition to, deletion from, "
            "correction, or modification of a Contract signed by all authorized "
            "parties.|"
        )
        sources = parse_sources([f"{header}\n{body}"])
        assert len(sources) == 1
        quote = sources[0]["quote"]
        assert "FIGURE" not in quote
        assert "|" not in quote
        assert quote.startswith("Amendment:")

    def test_quote_strips_markdown_table_syntax(self):
        """A chunk can itself be a raw markdown table (pymupdf4llm's own
        rendering of a PDF table), not wrapped in TABLE_START/END markers --
        reproduced live: pipe characters and separator rows leaked straight
        into the quote, along with a literal <br> tag from a wrapped cell."""
        header = "[1] file=doc_002.pdf chunk=9 page=3 score=0.8"
        body = (
            "|Amendment:|An agreed addition to, deletion from, correction, or<br>"
            "modification of a Contract signed by all authorized parties.|\n"
            "|---|---|\n"
            "|Contract:|The agreement between the parties.|"
        )
        sources = parse_sources([f"{header}\n{body}"])
        assert len(sources) == 1
        quote = sources[0]["quote"]
        assert "|" not in quote
        assert "<br>" not in quote
        assert "---" not in quote
        assert "Amendment:" in quote

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

    def test_eight_cap_round_robins_two_equal_files_instead_of_starving_second(self):
        """M-1: a two-document comparison emitting 12 chunk blocks per document
        used to keep 7 of doc_a's chunks and only doc_a's own single guaranteed
        slot for doc_b (the 'one slot per file, then fill in list order'
        scheme) -- doc_b's real rank-2 evidence was silently discarded even
        though it was genuinely retrieved. The cap must instead round-robin by
        filename: each file's rank-1, then each file's rank-2, ... This test
        fails under the old (diverse + rest)[:8] tail and passes under
        round-robin."""
        doc_a_chunks = [
            f"[1] file=doc_a.pdf chunk={i} page={i}\nDoc A chunk number {i}."
            for i in range(12)
        ]
        doc_b_chunks = [
            f"[1] file=doc_b.pdf chunk={i} page={i}\nDoc B chunk number {i}."
            for i in range(12)
        ]
        sources = parse_sources(doc_a_chunks + doc_b_chunks)
        assert len(sources) == 8
        counts = {"doc_a.pdf": 0, "doc_b.pdf": 0}
        for s in sources:
            counts[s["filename"]] += 1
        assert counts == {"doc_a.pdf": 4, "doc_b.pdf": 4}
        # doc_b's rank-2 chunk (its second chunk, chunk=1) specifically --
        # this is the slot the old scheme dropped.
        assert any(
            s["filename"] == "doc_b.pdf" and s["location"] == "chunk 1" for s in sources
        )

    def test_single_file_source_order_unchanged_by_round_robin(self):
        """Round-robin over a single filename group must produce the exact
        same order as before -- the blast radius of M-1 is bounded to the
        multi-document case."""
        collected = [
            f"[1] file=doc_a.pdf chunk={i} page={i}\nDoc A chunk number {i}."
            for i in range(5)
        ]
        sources = parse_sources(collected)
        assert [s["location"] for s in sources] == [
            "chunk 0",
            "chunk 1",
            "chunk 2",
            "chunk 3",
            "chunk 4",
        ]


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

    def test_two_calls_both_resolve_distinctly(self):
        """Two tool calls in one answer are renumbered to be globally unique
        before the model ever reads them (src/rag_agent.py's
        _renumber_tool_markers, tested separately in test_rag_agent.py) --
        the second call's first chunk is marker [2], not a colliding [1].
        Before this fix, build_citation_map only mapped the LAST call, so
        the first call's marker was silently dropped even though it was
        never ambiguous."""
        collected = [
            "[1] file=doc_a.pdf chunk=1 page=1 score=0.9\nDoc A fact.",
            "---CALL_BOUNDARY---",
            "[2] file=doc_b.pdf chunk=1 page=1 score=0.9\nDoc B fact.",
        ]
        sources = parse_sources(collected)
        citation_map = build_citation_map(collected, sources)
        assert len(citation_map) == 2
        assert citation_map[1] != citation_map[2]

    def test_single_call_marker_map_unchanged(self):
        """A single-call answer is unaffected by mapping every call instead
        of just the last -- there's only one call either way."""
        collected = self._one_call(
            ("[1] file=doc_a.pdf chunk=1 page=1 score=0.9", "Doc A first chunk."),
            ("[2] file=doc_a.pdf chunk=2 page=2 score=0.8", "Doc A second chunk."),
            ("[3] file=doc_b.pdf chunk=1 page=1 score=0.7", "Doc B first chunk."),
        )
        sources = parse_sources(collected)
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
        assert (
            strip_leaked_headers("Filed in [2024].", citation_map={})
            == "Filed in [2024]."
        )


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
            patch("vault_rag.answer_pipeline.route_question", return_value={}),
            patch("vault_rag.answer_pipeline.stream_agent", side_effect=fake_stream_agent),
        ):
            events = list(stream_answer(agent=object(), question="What is X?"))

        token_events = [e for e in events if "token" in e]
        assert [e["token"] for e in token_events] == ["Hello ", "world."]
        final = events[-1]
        assert final["done"] is True
        assert final["answer"] == "Hello world."
        assert final["tools"] == ["search_knowledge_base"]

    def test_final_correction_replaces_not_appends_streamed_text(self):
        """When stream_agent's repair pass / grounding check changes the
        text after live-streaming the raw version, it yields a
        FinalCorrection -- stream_answer must use it as the answer, not
        concatenate it onto what was already streamed as token events."""

        def fake_stream_agent(agent, query, **kwargs):
            yield "Draft answer"
            yield FinalCorrection("The corrected, complete answer.")

        with (
            patch("vault_rag.answer_pipeline.route_question", return_value={}),
            patch("vault_rag.answer_pipeline.stream_agent", side_effect=fake_stream_agent),
        ):
            events = list(stream_answer(agent=object(), question="What is X?"))

        token_events = [e for e in events if "token" in e]
        # Only the raw draft streamed as token events -- the correction is
        # not sent as its own token event (see stream_answer's docstring).
        assert [e["token"] for e in token_events] == ["Draft answer"]
        final = events[-1]
        assert final["done"] is True
        assert final["answer"] == "The corrected, complete answer."

    def test_calls_stream_agent_with_live_tokens_true(self):
        captured = {}

        def fake_stream_agent(agent, query, **kwargs):
            captured.update(kwargs)
            yield "An answer."

        with (
            patch("vault_rag.answer_pipeline.route_question", return_value={}),
            patch("vault_rag.answer_pipeline.stream_agent", side_effect=fake_stream_agent),
        ):
            list(stream_answer(agent=object(), question="What is X?"))

        assert captured["live_tokens"] is True

    def test_retries_first_attempt_unsupported_like_answer_one(self):
        """answer_one retries a bare "Unsupported" first attempt (real
        Groq/temp=0 nondeterminism, see its docstring) -- the single-part
        streaming path must keep that safety net, not silently drop it just
        because it can't be done live token-by-token."""

        def fake_stream_agent(agent, query, **kwargs):
            yield "Unsupported"

        with (
            patch("vault_rag.answer_pipeline.route_question", return_value={}),
            patch("vault_rag.answer_pipeline.stream_agent", side_effect=fake_stream_agent),
            patch(
                "vault_rag.answer_pipeline.run_once",
                return_value=(
                    "The real answer.",
                    [],
                    {"sql": [], "tools": ["search_knowledge_base"], "rejected": []},
                ),
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
        with patch("vault_rag.answer_pipeline.answer_query", return_value=canned) as mock_aq:
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
        with patch("vault_rag.answer_pipeline.answer_query", return_value=canned) as mock_aq:
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
        assert (
            _excel_citations_to_sources([{"sheet_name": "Sheet1"}], existing=[]) == []
        )
        assert (
            _excel_citations_to_sources([{"source_file": "x.xlsx"}], existing=[]) == []
        )

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
            "vault_rag.answer_pipeline.run_once", return_value=("An answer.", [], {})
        ) as mock_run_once:
            answer_one(agent=object(), question="Compare doc_009 and doc_010")
        assert mock_run_once.call_args.kwargs["skip_grounding_check"] is True

    def test_non_comparison_question_runs_grounding_check(self):
        with patch(
            "vault_rag.answer_pipeline.run_once", return_value=("An answer.", [], {})
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
            patch("vault_rag.answer_pipeline.route_question") as mock_route,
            patch(
                "vault_rag.answer_pipeline.run_once",
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
            patch("vault_rag.answer_pipeline.route_question") as mock_route,
            patch(
                "vault_rag.answer_pipeline.run_once",
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

    def test_auto_routed_excel_modality_hard_blocks_search_knowledge_base(self):
        """route_question's own auto-detection (not the UI's forced_doc_id) can
        also resolve excel modality -- verified live: the routing directive
        alone didn't stop the agent from calling search_knowledge_base anyway.
        FORCED_MODALITY must be set for the run_once call in this case too."""
        from vault_rag.tools.retrieval_tool import FORCED_MODALITY

        seen_modality = {}

        def fake_run_once(agent, q, **kwargs):
            seen_modality["value"] = FORCED_MODALITY.get()
            return "An answer.", [], {}

        with (
            patch(
                "vault_rag.answer_pipeline.route_question",
                return_value={"modality": "excel", "source_file": "doc_006.xlsx"},
            ),
            patch("vault_rag.answer_pipeline.run_once", side_effect=fake_run_once),
        ):
            answer_one(agent=object(), question="What's the total spend?")
        assert seen_modality["value"] == "excel"
        assert FORCED_MODALITY.get() is None  # reset after the call

    def test_answer_query_skips_title_shortcut_when_doc_forced(self):
        """The title shortcut's own retrieve() call isn't scoped to a document,
        so it must not short-circuit a scoped query."""
        with (
            patch("vault_rag.answer_pipeline._title_shortcut_answer") as mock_shortcut,
            patch(
                "vault_rag.answer_pipeline.answer_one",
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
            patch("vault_rag.answer_pipeline.parse_sources") as mock_parse,
            patch(
                "vault_rag.answer_pipeline.run_once",
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
            patch("vault_rag.answer_pipeline.parse_sources") as mock_parse,
            patch(
                "vault_rag.answer_pipeline.run_once",
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

    def test_resolved_doc_ids_trigger_retry_on_a_natural_language_comparison(self):
        """M-7: the question below names no literal doc_XXX id at all, so
        the old (unwidened) _missing_mentioned_docs check would see fewer
        than two "mentioned" docs and return [] -- both it and the
        grounding check (skipped for comparisons) were off at once,
        leaving only the coarse n_sources<2 fallback. answer_one must use
        the ids M-6's resolver already found (passed in as resolved_doc_ids)
        to see that doc_010 has no evidence and retry -- this fails on the
        pre-M-7 code, which ignores resolved_doc_ids entirely."""
        with (
            patch("vault_rag.answer_pipeline.parse_sources") as mock_parse,
            patch(
                "vault_rag.answer_pipeline.run_once",
                side_effect=[
                    (
                        "The HR manual covers paid leave in detail.",
                        [],
                        {},
                    ),
                    (
                        "The HR manual covers paid leave; the budget tracker "
                        "has no leave policy content.",
                        [],
                        {},
                    ),
                ],
            ) as mock_run_once,
        ):
            mock_parse.side_effect = [
                [_src("doc_009_hr.pdf", "doc_009")],
                [
                    _src("doc_009_hr.pdf", "doc_009"),
                    _src("doc_010_budget.pdf", "doc_010"),
                ],
            ]
            ans, _coll, _tr = answer_one(
                agent=object(),
                question=(
                    "Between the Rosemont HR policy manual and the OSSE AFE "
                    "budget tracker, which one covers leave policies?"
                ),
                resolved_doc_ids=["doc_009", "doc_010"],
            )
        assert mock_run_once.call_count == 2
        retry_question = mock_run_once.call_args_list[1][0][1]
        assert "doc_010" in retry_question
        assert "no evidence" in retry_question


class TestComparisonPartialUnsupportedRetry:
    """A comparison answer can name both documents (n_sources >= 2) yet still
    write off one side as "Unsupported" in the synthesized text -- reproduced
    live: both documents were actually retrieved, but the model gave up on one
    side because its content spanned several related sub-topics. The retry
    trigger must catch this even though the retrieved-source-count check alone
    doesn't see a problem."""

    def test_partial_unsupported_with_two_sources_triggers_retry(self):
        with (
            patch("vault_rag.answer_pipeline.parse_sources") as mock_parse,
            patch(
                "vault_rag.answer_pipeline.run_once",
                side_effect=[
                    ("doc_009: Unsupported\ndoc_010: real answer.", [], {}),
                    ("doc_009: a real answer.\ndoc_010: real answer.", [], {}),
                ],
            ) as mock_run_once,
        ):
            mock_parse.return_value = [
                _src("doc_009_hr.pdf"),
                _src("doc_010_handbook.pdf"),
            ]
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
            patch("vault_rag.answer_pipeline.parse_sources") as mock_parse,
            patch(
                "vault_rag.answer_pipeline.run_once",
                return_value=("doc_009: a. doc_010: b.", [], {}),
            ) as mock_run_once,
        ):
            mock_parse.return_value = [
                _src("doc_009_hr.pdf"),
                _src("doc_010_handbook.pdf"),
            ]
            answer_one(
                agent=object(),
                question="Compare the leave policies in doc_009 and doc_010",
            )
        mock_run_once.assert_called_once()

    def test_retry_keeps_original_when_still_partial(self):
        """If the retry doesn't actually fix it, keep the first answer rather
        than silently swapping in an equally-broken retry."""
        with (
            patch("vault_rag.answer_pipeline.parse_sources") as mock_parse,
            patch(
                "vault_rag.answer_pipeline.run_once",
                side_effect=[
                    ("doc_009: Unsupported\ndoc_010: real answer.", [], {}),
                    ("doc_009: Unsupported\ndoc_010: real answer.", [], {}),
                ],
            ),
        ):
            mock_parse.return_value = [
                _src("doc_009_hr.pdf"),
                _src("doc_010_handbook.pdf"),
            ]
            ans, _coll, _tr = answer_one(
                agent=object(),
                question="Compare the leave policies in doc_009 and doc_010",
            )
        assert ans == "doc_009: Unsupported\ndoc_010: real answer."


def _chunk(
    marker: int, filename: str, body: str = "Some retrieved passage text."
) -> str:
    return f"[{marker}] file={filename} chunk=1 page=1 score=0.9\n{body}"


class _FakeTool:
    """Stands in for the search_knowledge_base StructuredTool -- .func mirrors
    the real (content, artifact) tuple contract, keyed by doc_id so each test
    controls exactly what each document "has"."""

    def __init__(self, responses: dict[str, str]):
        self.responses = responses
        self.calls: list[tuple[str, str]] = []

    def func(self, question: str, doc_id: str = "") -> tuple[str, dict]:
        self.calls.append((question, doc_id))
        content = self.responses.get(doc_id, "No relevant information found.")
        return content, {"rejected": []}


class _FakeLLMResponse:
    def __init__(self, content: str):
        self.content = content


class _FakeLLM:
    def __init__(self, answer: str):
        self.answer = answer
        self.invoked_with: list = []

    def invoke(self, messages):
        self.invoked_with.append(messages)
        return _FakeLLMResponse(self.answer)


def _fake_agent(
    tool: _FakeTool, llm: _FakeLLM, doc_registry: dict[str, str]
) -> SimpleNamespace:
    return SimpleNamespace(
        _tools_by_name={"search_knowledge_base": tool},
        _llm=llm,
        _doc_registry=doc_registry,
    )


class TestResolveComparisonDocIds:
    def test_two_named_doc_ids_in_question(self):
        result = _resolve_comparison_doc_ids(
            "Compare the leave policies in doc_009 and doc_010", None, {}
        )
        assert result == ["doc_009", "doc_010"]

    def test_more_than_two_named_doc_ids(self):
        result = _resolve_comparison_doc_ids(
            "Compare doc_001, doc_002 and doc_003", None, {}
        )
        assert result is not None
        assert set(result) == {"doc_001", "doc_002", "doc_003"}

    def test_two_docs_via_source_scope(self):
        registry = {
            "doc_006_purchase_card_transactions_q1_2025_26": "doc_006",
            "doc_007_published_spend_report_april_25": "doc_007",
        }
        result = _resolve_comparison_doc_ids(
            "Compare these two",
            [
                "doc_006_purchase_card_transactions_q1_2025_26.xlsx",
                "doc_007_published_spend_report_april_25.csv",
            ],
            registry,
        )
        assert result == ["doc_006", "doc_007"]

    def test_similar_filenames_resolve_to_distinct_ids(self):
        """doc_016b/doc_016c share almost their entire stem -- substring
        matching must not collapse them into the same document."""
        registry = {
            "doc_016a_original_lease": "doc_016a",
            "doc_016b_first_amendment": "doc_016b",
            "doc_016c_second_amendment": "doc_016c",
        }
        result = _resolve_comparison_doc_ids(
            "Compare these",
            ["doc_016b_first_amendment.pdf", "doc_016c_second_amendment.pdf"],
            registry,
        )
        assert result == ["doc_016b", "doc_016c"]

    def test_ambiguous_question_returns_none(self):
        """No named doc_ids, no source scope, no registry match -- must fall
        back to the agent-based path rather than guessing."""
        result = _resolve_comparison_doc_ids(
            "Compare the two most recent versions of the contract", None, {}
        )
        assert result is None

    def test_llm_fallback_not_consulted_without_catalogue(self):
        """catalogue/api_base/model_name all default to None -- the LLM step
        must not be reachable unless a caller explicitly wires it in."""
        with patch("vault_rag.answer_pipeline._llm_call") as mock_call:
            result = _resolve_comparison_doc_ids(
                "Which document identifies 42 new topic areas?", None, {}
            )
        mock_call.assert_not_called()
        assert result is None


_CATALOGUE = {
    "doc_001": "Annual Procurement Policy (LACERA)",
    "doc_002": "Q1 Purchase Card Transactions (Village of Bensenville)",
}


class TestResolveComparisonDocIdsLlm:
    def test_genuine_pair_with_verified_phrases_passes(self):
        reply = (
            "The question contrasts a procurement policy and a spending report.\n"
            'ANSWER: doc_001 ("Annual Procurement Policy"), '
            'doc_002 ("Q1 Purchase Card Transactions")'
        )
        with patch("vault_rag.answer_pipeline._llm_call", return_value=reply):
            result = _resolve_comparison_doc_ids_llm(
                "Compare the procurement policy and the card transactions report",
                _CATALOGUE,
                "http://fake-llm/v1",
                "fake-model",
            )
        assert result == ["doc_001", "doc_002"]

    def test_hallucinated_phrase_degrades_to_none(self):
        """A pick whose quoted phrase doesn't actually occur in that
        document's own catalogue entry must not be trusted -- this is the
        gate that stops a hallucinated id from producing a confidently wrong
        pair (the failure mode that sank the first attempt at this)."""
        reply = (
            'ANSWER: doc_001 ("42 new topic areas"), '
            'doc_002 ("Llano Airport transactions")'
        )
        with patch("vault_rag.answer_pipeline._llm_call", return_value=reply):
            result = _resolve_comparison_doc_ids_llm(
                "Which document identifies 42 new topic areas, and which "
                "covers Llano Airport transactions?",
                _CATALOGUE,
                "http://fake-llm/v1",
                "fake-model",
            )
        assert result is None

    def test_explicit_none_falls_back_cleanly(self):
        with patch("vault_rag.answer_pipeline._llm_call", return_value="ANSWER: NONE"):
            result = _resolve_comparison_doc_ids_llm(
                "Compare the two most recent versions of the contract",
                _CATALOGUE,
                "http://fake-llm/v1",
                "fake-model",
            )
        assert result is None

    def test_llm_failure_falls_back_cleanly(self):
        with patch(
            "vault_rag.answer_pipeline._llm_call", side_effect=RuntimeError("timeout")
        ):
            result = _resolve_comparison_doc_ids_llm(
                "Compare doc_001 and doc_002", _CATALOGUE, "http://fake-llm/v1", "fake-model"
            )
        assert result is None

    def test_reachable_as_last_resort_through_resolve_comparison_doc_ids(self):
        """Both production call sites pass catalogue/api_base/model_name --
        this is the shape they use to reach the LLM step."""
        reply = (
            'ANSWER: doc_001 ("Annual Procurement Policy"), '
            'doc_002 ("Q1 Purchase Card Transactions")'
        )
        with patch("vault_rag.answer_pipeline._llm_call", return_value=reply):
            result = _resolve_comparison_doc_ids(
                "Compare the procurement policy and the card transactions report",
                None,
                {},
                catalogue=_CATALOGUE,
                api_base="http://fake-llm/v1",
                model_name="fake-model",
            )
        assert result == ["doc_001", "doc_002"]


class TestAnswerComparisonDeterministic:
    def test_two_named_documents_both_covered(self):
        tool = _FakeTool(
            {
                "doc_009": _chunk(1, "doc_009_hr.pdf"),
                "doc_010": _chunk(1, "doc_010_handbook.pdf"),
            }
        )
        llm = _FakeLLM("Doc 009 says X [1]. Doc 010 says Y [2].")
        agent = _fake_agent(tool, llm, {})
        result = answer_comparison_deterministic(
            agent, "Compare the leave policies in doc_009 and doc_010"
        )
        assert result is not None
        filenames = {s["filename"] for s in result["sources"]}
        assert filenames == {"doc_009_hr.pdf", "doc_010_handbook.pdf"}
        assert "No relevant evidence" not in result["answer"]

    def test_two_documents_selected_through_source_scope(self):
        registry = {"doc_006_data": "doc_006", "doc_007_data": "doc_007"}
        tool = _FakeTool(
            {
                "doc_006": _chunk(1, "doc_006_data.xlsx"),
                "doc_007": _chunk(1, "doc_007_data.csv"),
            }
        )
        llm = _FakeLLM("Comparison answer.")
        agent = _fake_agent(tool, llm, registry)
        result = answer_comparison_deterministic(
            agent,
            "Compare these two spreadsheets",
            forced_doc_id=["doc_006_data.xlsx", "doc_007_data.csv"],
        )
        assert result is not None
        assert {s["filename"] for s in result["sources"]} == {
            "doc_006_data.xlsx",
            "doc_007_data.csv",
        }

    def test_one_document_missing_evidence_notes_it_without_fabricating(self):
        tool = _FakeTool(
            {"doc_009": _chunk(1, "doc_009_hr.pdf")}
        )  # doc_010 absent -> "No relevant information found."
        llm = _FakeLLM("Doc 009 says X [1].")
        agent = _fake_agent(tool, llm, {})
        result = answer_comparison_deterministic(
            agent, "Compare the leave policies in doc_009 and doc_010"
        )
        assert result is not None
        assert {s["filename"] for s in result["sources"]} == {"doc_009_hr.pdf"}
        assert "No relevant evidence was found for doc_010" in result["answer"]

    def test_more_than_two_compared_documents_all_retrieved(self):
        tool = _FakeTool(
            {
                "doc_001": _chunk(1, "doc_001_a.pdf"),
                "doc_002": _chunk(1, "doc_002_b.pdf"),
                "doc_003": _chunk(1, "doc_003_c.pdf"),
            }
        )
        llm = _FakeLLM("Three-way comparison.")
        agent = _fake_agent(tool, llm, {})
        result = answer_comparison_deterministic(
            agent, "Compare doc_001, doc_002 and doc_003"
        )
        assert result is not None
        assert {s["filename"] for s in result["sources"]} == {
            "doc_001_a.pdf",
            "doc_002_b.pdf",
            "doc_003_c.pdf",
        }
        assert len(tool.calls) == 3

    def test_ambiguous_question_returns_none_for_caller_fallback(self):
        tool = _FakeTool({})
        llm = _FakeLLM("should not be reached")
        agent = _fake_agent(tool, llm, {})
        result = answer_comparison_deterministic(
            agent, "Compare the two most recent versions of the contract"
        )
        assert result is None
        assert tool.calls == []

    def test_all_documents_missing_evidence_returns_none(self):
        tool = _FakeTool({})  # both doc_ids come back empty
        llm = _FakeLLM("should not be reached")
        agent = _fake_agent(tool, llm, {})
        result = answer_comparison_deterministic(
            agent, "Compare the leave policies in doc_009 and doc_010"
        )
        assert result is None

    def test_high_marker_beyond_max_tool_results_still_resolves_or_strips_cleanly(self):
        """Reproduced live 2026-07-21 (b4): the two per-document retrieval
        calls are renumbered into ONE combined marker sequence that can
        legitimately exceed MAX_TOOL_RESULTS (12) -- e.g. 9 chunks from one
        doc + 8 from the other = markers 1..17. _strip_inline_citation used to
        assume any marker > MAX_TOOL_RESULTS was a literal number in prose
        (like a year) and left it raw, so a genuine-but-uncited-in-the-final-
        capped-list marker like [17] leaked visibly into the answer. Must
        strip cleanly (empty), never survive as a bare "[17]"."""
        doc_a_chunks = "\n\n".join(
            _chunk(i, "doc_001_a.pdf", f"Doc A passage {i}.") for i in range(1, 10)
        )
        doc_b_chunks = "\n\n".join(
            _chunk(i, "doc_002_b.pdf", f"Doc B passage {i}.") for i in range(1, 9)
        )
        tool = _FakeTool({"doc_001": doc_a_chunks, "doc_002": doc_b_chunks})
        # Marker 17 is the LAST of the 17 combined chunks -- real, but capped
        # out of the final 8-source list (parse_sources keeps the earliest
        # per-file slots first), so it must resolve to nothing, not leak raw.
        llm = _FakeLLM("Doc A says X [1]. Doc B says Y [17].")
        agent = _fake_agent(tool, llm, {})
        result = answer_comparison_deterministic(agent, "Compare doc_001 and doc_002")
        assert result is not None
        assert "[17]" not in result["answer"]


class TestAnswerQueryComparisonRouting:
    def test_non_comparison_multi_doc_question_skips_deterministic_path(self):
        """A question that just happens to mention two doc_ids without any
        comparison language must not be forced through the deterministic
        comparison path -- it isn't asking for a comparison at all."""
        with (
            patch("vault_rag.answer_pipeline.answer_comparison_deterministic") as mock_det,
            patch(
                "vault_rag.answer_pipeline.answer_one", return_value=("An answer.", [], {})
            ),
        ):
            answer_query(
                agent=object(),
                question="List everything mentioned about doc_009 and doc_010.",
            )
        mock_det.assert_not_called()

    def test_comparison_question_tries_deterministic_path_first(self):
        with (
            patch(
                "vault_rag.answer_pipeline.answer_comparison_deterministic",
                return_value={
                    "answer": "det answer",
                    "sources": [],
                    "sql": [],
                    "tools": [],
                    "rejected_sources": [],
                    "collected": [],
                },
            ) as mock_det,
            patch("vault_rag.answer_pipeline.answer_one") as mock_answer_one,
        ):
            result = answer_query(
                agent=object(),
                question="Compare the leave policies in doc_009 and doc_010",
            )
        mock_det.assert_called_once()
        mock_answer_one.assert_not_called()
        assert result["answer"] == "det answer"

    def test_comparison_question_falls_back_when_deterministic_path_declines(self):
        chunk = "[1] file=doc_001.pdf chunk=0 score=0.9\nSome content."
        with (
            patch(
                "vault_rag.answer_pipeline.answer_comparison_deterministic", return_value=None
            ),
            patch(
                "vault_rag.answer_pipeline.answer_one",
                return_value=("fallback answer", [chunk], {}),
            ) as mock_answer_one,
        ):
            result = answer_query(
                agent=object(),
                question="Compare the two most recent versions of the contract",
            )
        mock_answer_one.assert_called_once()
        assert result["answer"] == "fallback answer"


class TestSinglePartExcelAnswerGetsMarker:
    """A single, non-split question answered purely by query_excel also never
    gets an [N] marker (same root cause as the multi-part case: SQL output
    never enters the bracketed chunk stream). Reproduced live: "Sources used
    · 1" looked fine only by coincidence (a lone Excel citation is often the
    only candidate at all) -- with any other retrieved candidate present it
    would silently read as "used" without ever being individually cited."""

    def test_single_part_excel_answer_gets_deterministic_marker(self):
        def fake_answer_one(agent, part, trace=None, forced_doc_id=None, usage=None, **kwargs):
            return (
                "12976.92, the total NET Amount spent on MATERIALS.",
                [],
                {
                    "excel_citations": [
                        {
                            "source_file": "doc_006_purchase_card_transactions_q1_2025_26.xlsx",
                            "sheet_name": "DataAnalysis",
                            "quote": "12976.92",
                        }
                    ]
                },
            )

        with patch("vault_rag.answer_pipeline.answer_one", side_effect=fake_answer_one):
            result = answer_query(
                agent=object(),
                question="What is the total NET Amount spent on MATERIALS?",
            )

        assert "[1]" in result["answer"]
        assert result["sources"][0]["filename"] == (
            "doc_006_purchase_card_transactions_q1_2025_26.xlsx"
        )


class TestMultiPartAnswerCitations:
    """A multi-part question runs each part as its own agent call, so a
    citation_map built from the merged chunks (which only resolves the LAST
    call's numbering, see build_citation_map) can't map either part's [N]
    markers -- they used to be silently stripped, leaving the merged answer
    with zero real citations and the UI falling back to "show every
    retrieved candidate" for a multi-part cross-document answer. Each part's
    own [N] is now resolved separately against the final merged sources."""

    def test_each_parts_citation_survives_and_resolves_to_its_own_source(self):
        part_a_chunk = "[1] file=doc_003_fed_annual_report_2024.pdf chunk=16 page=2\nThe Fed reduced holdings by $297 billion."
        part_b_chunk = "[1] file=doc_008_gao_24_106915.pdf chunk=1 page=1\n42 new topic areas were identified."

        def fake_answer_one(agent, part, trace=None, forced_doc_id=None, usage=None, **kwargs):
            if "Federal Reserve" in part:
                return "$297 billion [1]", [part_a_chunk], {}
            return "42 [1]", [part_b_chunk], {}

        with patch("vault_rag.answer_pipeline.answer_one", side_effect=fake_answer_one):
            result = answer_query(
                agent=object(),
                question=(
                    "According to the Federal Reserve report, how much did holdings "
                    "drop? According to the GAO report, how many new topic areas?"
                ),
            )

        assert "[1]" in result["answer"]
        assert "[2]" in result["answer"]
        assert [s["filename"] for s in result["sources"][:2]] == [
            "doc_003_fed_annual_report_2024.pdf",
            "doc_008_gao_24_106915.pdf",
        ]

    def test_excel_part_gets_a_deterministic_marker_not_a_retrieval_chunk(self):
        """Reproduced live: a part answered from query_excel ("12892.0, the
        largest...") carried no [N] marker at all -- query_excel's result
        never enters the bracketed chunk stream search_knowledge_base's does,
        so the model has nothing to cite. Must attach a marker pointing at
        the real excel_citations source, not the part's top retrieved chunk
        -- reproduced live, that chunk can be unrelated noise from the agent
        searching the wrong document while still answering correctly via SQL."""
        part_a_chunk = "[1] file=doc_016a_original_lease.pdf chunk=1 page=2\nRent for the first year is $31,052.08."
        # Wrong-document noise the agent's own search_knowledge_base call
        # returned in the SQL-answered part -- must NOT be cited.
        part_b_chunk = (
            "[1] file=doc_001_procurement_policy.pdf chunk=9 page=3\nIV. Definitions."
        )

        def fake_answer_one(agent, part, trace=None, forced_doc_id=None, usage=None, **kwargs):
            if "rent" in part.lower():
                return (
                    "$31,052.08, the annual rent for the first year [1]",
                    [part_a_chunk],
                    {},
                )
            return (
                "12892.0, the largest single purchase card transaction amount.",
                [part_b_chunk],
                {
                    "excel_citations": [
                        {
                            "source_file": "doc_006_purchase_card_transactions_q1_2025_26.xlsx",
                            "sheet_name": "DataAnalysis",
                            "quote": "12892.0",
                        }
                    ]
                },
            )

        with patch("vault_rag.answer_pipeline.answer_one", side_effect=fake_answer_one):
            result = answer_query(
                agent=object(),
                question=(
                    "What is the annual rent for the first year of the lease, "
                    "and what is the largest single purchase card transaction amount?"
                ),
            )

        assert "[1]" in result["answer"]
        assert "[2]" in result["answer"]
        cited = result["sources"][:2]
        assert {s["filename"] for s in cited} == {
            "doc_016a_original_lease.pdf",
            "doc_006_purchase_card_transactions_q1_2025_26.xlsx",
        }
        # The wrong-document noise chunk must not be cited as evidence.
        excel_source = next(s for s in cited if s["filename"].endswith(".xlsx"))
        assert excel_source["sheet"] == "DataAnalysis"


class TestNoEvidenceForcesUnsupported:
    """Reproduced live 2026-07-21: with zero chunks retrieved, the agent
    answered from its own tool-call arguments (a doc_id string containing a
    year) instead of real content, and the citation-renumbering fallback
    invented a fake "unknown"-filename source card for it. A "verify every
    answer" product must never show a source with nothing behind it."""

    def test_ungrounded_answer_with_no_chunks_becomes_unsupported(self):
        with patch(
            "vault_rag.answer_pipeline.answer_one",
            return_value=("2025", [], {}),
        ):
            result = answer_query(
                agent=object(), question="What year is it titled for?"
            )
        assert result["answer"] == "Unsupported"

    def test_clarifying_question_with_no_chunks_is_kept_not_replaced(self):
        """Reproduced live 2026-07-24: a broad "summarize across all
        documents" question made no tool call and got a real clarifying
        question back -- this must survive, not be silently swapped for a
        bare, unhelpful "Unsupported"."""
        clarifying = (
            "Clarify: which specific policy areas—procurement, data privacy, "
            "employee conduct, travel—should be summarized?"
        )
        with patch(
            "vault_rag.answer_pipeline.answer_one",
            return_value=(clarifying, [], {}),
        ):
            result = answer_query(
                agent=object(),
                question="Summarize the main policies across all documents.",
            )
        assert result["answer"] == clarifying
        assert result["sources"] == []
        assert result["sources"] == []

    def test_ungrounded_answer_with_junk_collected_text_becomes_unsupported(self):
        """Reproduced live 2026-07-21: `collected` held a tool's own plain
        "no results" text (not a real [N] file=... chunk) -- non-empty
        `collected`, but zero real sources once parsed. Must still refuse,
        not just when `collected` is literally []."""
        with patch(
            "vault_rag.answer_pipeline.answer_one",
            return_value=("2025", ["No relevant results found."], {}),
        ):
            result = answer_query(
                agent=object(), question="What year is it titled for?"
            )
        assert result["answer"] == "Unsupported"
        assert result["sources"] == []

    def test_grounded_answer_with_real_chunks_is_unaffected(self):
        chunk = "[1] file=doc_011.xlsx chunk=0 score=0.9\nTitle: 2025 Questionnaire"
        with patch(
            "vault_rag.answer_pipeline.answer_one",
            return_value=("2025 [1]", [chunk], {}),
        ):
            result = answer_query(
                agent=object(), question="What year is it titled for?"
            )
        assert result["answer"] != "Unsupported"
        assert len(result["sources"]) == 1

    def test_sql_only_answer_with_no_chunks_is_unaffected(self):
        """query_excel answers via SQL, not retrieved chunks -- zero
        `collected` chunks is the normal, expected shape for it, not a sign
        of a groundless answer."""
        with patch(
            "vault_rag.answer_pipeline.answer_one",
            return_value=("42", [], {"sql": ["SELECT 42"]}),
        ):
            result = answer_query(agent=object(), question="What is the total?")
        assert result["answer"] == "42"

    def test_already_unsupported_with_no_chunks_stays_unsupported(self):
        with patch(
            "vault_rag.answer_pipeline.answer_one",
            return_value=("Unsupported", [], {}),
        ):
            result = answer_query(
                agent=object(), question="What year is it titled for?"
            )
        assert result["answer"] == "Unsupported"
        assert result["sources"] == []


_FAKE_AGENT = SimpleNamespace(
    _generation_api_base="http://fake-llm/v1", _generation_model="fake-model"
)


class TestCondenseFollowupQuestion:
    def test_no_history_is_a_noop(self):
        """eval/run_eval.py never passes history -- must not call the LLM at all."""
        with patch("vault_rag.answer_pipeline._llm_call") as mock_call:
            result = _condense_followup_question(
                "Who must that notice be given to?", None, _FAKE_AGENT
            )
        mock_call.assert_not_called()
        assert result == "Who must that notice be given to?"

    def test_no_agent_generation_config_is_a_noop(self):
        """No agent (or one missing _generation_api_base/_generation_model,
        e.g. a bare `object()` used in other tests) must not call the LLM."""
        history = [{"question": "q", "answer": "a"}]
        with patch("vault_rag.answer_pipeline._llm_call") as mock_call:
            result = _condense_followup_question(
                "Who must that notice be given to?", history, object()
            )
        mock_call.assert_not_called()
        assert result == "Who must that notice be given to?"

    def test_rewrites_followup_using_history(self):
        history = [
            {
                "question": "What is the notice period required to terminate the lease?",
                "answer": "180 days [1]",
            }
        ]
        with patch(
            "vault_rag.answer_pipeline._llm_call",
            return_value="Who must the 180-day lease termination notice be given to?",
        ):
            result = _condense_followup_question(
                "Who must that notice be given to?", history, _FAKE_AGENT
            )
        assert result == "Who must the 180-day lease termination notice be given to?"

    def test_falls_back_to_raw_question_on_llm_failure(self):
        with patch(
            "vault_rag.answer_pipeline._llm_call",
            side_effect=RuntimeError("connection refused"),
        ):
            result = _condense_followup_question(
                "Who must that notice be given to?",
                [{"question": "q", "answer": "a"}],
                _FAKE_AGENT,
            )
        assert result == "Who must that notice be given to?"

    def test_falls_back_to_raw_question_on_malformed_rewrite(self):
        with patch(
            "vault_rag.answer_pipeline._llm_call",
            return_value='{"tool": "search_knowledge_base"}',
        ):
            result = _condense_followup_question(
                "Who must that notice be given to?",
                [{"question": "q", "answer": "a"}],
                _FAKE_AGENT,
            )
        assert result == "Who must that notice be given to?"
