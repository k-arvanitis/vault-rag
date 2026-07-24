"""Characterization tests for the search_knowledge_base retrieval tool.

These pin the current observable behavior of _make_unified_tool / _fetch_docs so
the function can be refactored safely. retrieve(), the reranker, HyDE, and the
neighbour-chunk fetch are all mocked — no live services.
"""
from __future__ import annotations

from unittest.mock import patch

from src.tools import retrieval_tool
from src.tools.retrieval_tool import FORCED_DOC_ID, FORCED_MODALITY, _make_unified_tool


class FakeRanker:
    """Reranker stub: preserves input order with descending synthetic scores."""

    def rerank(self, query, docs, top_n):
        ranked = [{"index": i, "score": 1.0 - i * 0.01} for i in range(len(docs))]
        return ranked[:top_n]


def _hit(content, *, chunk_type="page_content", source_file="doc_005_report.pdf",
         chunk_index=0, doc_id="", sheet_name=None, score=0.9):
    """Build one retrieval hit dict shaped like retrieve() output."""
    meta = {
        "chunk_type": chunk_type,
        "source_file": source_file,
        "file_name": source_file,
        "chunk_index": chunk_index,
        "score": score,
    }
    if doc_id:
        meta["doc_id"] = doc_id
    if sheet_name:
        meta["sheet_name"] = sheet_name
    return {"content": content, "metadata": meta, "score": score}


def _make_retrieve(*, doc_summary=None, sheet_summary=None, content=None, spy=None):
    """Return a retrieve() side_effect that yields canned hits keyed on chunk type.

    spy, if given, receives each call's kwargs.
    """
    doc_summary = doc_summary or []
    sheet_summary = sheet_summary or []
    content = content or []

    def _retrieve(**kwargs):
        if spy is not None:
            spy(kwargs)
        forced = kwargs.get("force_chunk_types")
        if forced == ["document_summary"]:
            return list(doc_summary)
        if forced == ["sheet_summary"]:
            return list(sheet_summary)
        return list(content)

    return _retrieve


def _build_tool(*, doc_registry=None, use_hyde=False):
    """Build the search_knowledge_base tool with a fake reranker."""
    tool, _limits = _make_unified_tool(
        qdrant_url="http://qdrant.invalid",
        collection="test",
        retrieval_top_k=10,
        rerank_top_n=8,
        ranker=FakeRanker(),
        generation_api_base="http://llm.invalid",
        generation_model="test-model",
        use_hyde=use_hyde,
        doc_registry=doc_registry or {},
    )
    return tool


class TestRetrievalToolBehavior:
    def test_basic_retrieval_formats_numbered_hits(self):
        retrieve_fn = _make_retrieve(content=[
            _hit("Alpha content about budgets.", chunk_index=0),
            _hit("Beta content about budgets.", chunk_index=1),
        ])
        with patch.object(retrieval_tool, "retrieve", side_effect=retrieve_fn), \
             patch.object(retrieval_tool, "_fetch_neighbor_chunks", return_value={}):
            out, _artifact = _build_tool().func("What is the budget?")
        assert "[1] file=doc_005_report.pdf chunk=0" in out
        assert "[2] file=doc_005_report.pdf chunk=1" in out
        assert "Alpha content about budgets." in out
        assert out.strip().endswith("a different missing fact is required.")

    def test_no_hits_returns_no_information_message(self):
        with patch.object(retrieval_tool, "retrieve", side_effect=_make_retrieve()), \
             patch.object(retrieval_tool, "_fetch_neighbor_chunks", return_value={}):
            out, _artifact = _build_tool().func("anything at all")
        assert out == "No relevant information found."

    def test_doc_id_argument_scopes_retrieval(self):
        scopes: list = []
        retrieve_fn = _make_retrieve(
            content=[_hit("scoped content", doc_id="doc_001")],
            spy=lambda kw: scopes.append(kw.get("scope_doc_id")),
        )
        with patch.object(retrieval_tool, "retrieve", side_effect=retrieve_fn), \
             patch.object(retrieval_tool, "_fetch_neighbor_chunks", return_value={}):
            _build_tool().func("a question", doc_id="doc_001")
        assert "doc_001" in scopes

    def test_inline_doc_id_in_query_scopes_retrieval(self):
        scopes: list = []
        retrieve_fn = _make_retrieve(
            content=[_hit("content", doc_id="doc_007")],
            spy=lambda kw: scopes.append(kw.get("scope_doc_id")),
        )
        with patch.object(retrieval_tool, "retrieve", side_effect=retrieve_fn), \
             patch.object(retrieval_tool, "_fetch_neighbor_chunks", return_value={}):
            _build_tool().func("what does doc_007 say about leave?")
        assert "doc_007" in scopes

    def test_two_doc_ids_trigger_parallel_scoped_retrieval(self):
        scopes: list = []
        retrieve_fn = _make_retrieve(
            content=[_hit("content", doc_id="doc_001")],
            spy=lambda kw: scopes.append(kw.get("scope_doc_id")),
        )
        with patch.object(retrieval_tool, "retrieve", side_effect=retrieve_fn), \
             patch.object(retrieval_tool, "_fetch_neighbor_chunks", return_value={}):
            _build_tool().func("compare doc_001 and doc_002 on payment terms")
        assert "doc_001" in scopes
        assert "doc_002" in scopes

    def test_forced_doc_id_list_ors_scope_across_docs(self):
        """FORCED_DOC_ID set to a list (UI multi-select) bypasses single-doc
        inference and scopes retrieval to exactly those doc_ids, OR'd."""
        scopes: list = []
        retrieve_fn = _make_retrieve(
            content=[_hit("content", doc_id="doc_001")],
            spy=lambda kw: scopes.append(kw.get("scope_doc_id")),
        )
        token = FORCED_DOC_ID.set(["doc_001", "doc_002"])
        try:
            with patch.object(retrieval_tool, "retrieve", side_effect=retrieve_fn), \
                 patch.object(retrieval_tool, "_fetch_neighbor_chunks", return_value={}):
                _build_tool().func("unrelated question text")
        finally:
            FORCED_DOC_ID.reset(token)
        assert ["doc_001", "doc_002"] in scopes

    def test_forced_modality_excel_refuses_instead_of_searching(self):
        """route_question's routing directive is only a prompt nudge -- the
        agent sometimes calls search_knowledge_base anyway on a question that
        needs query_excel. FORCED_MODALITY hard-blocks that: no retrieve()
        call happens at all, the tool just names the right one."""
        retrieve_fn = _make_retrieve(content=[_hit("content")])
        token = FORCED_MODALITY.set("excel")
        try:
            with patch.object(
                retrieval_tool, "retrieve", side_effect=retrieve_fn
            ) as mock_retrieve:
                content, _artifact = _build_tool().func("what's the total spend?")
        finally:
            FORCED_MODALITY.reset(token)
        mock_retrieve.assert_not_called()
        assert "query_excel" in content

    def test_hyde_expansion_issues_extra_retrieval(self):
        queries: list = []
        retrieve_fn = _make_retrieve(
            content=[_hit("content", chunk_index=0)],
            spy=lambda kw: queries.append(kw.get("query")),
        )
        with patch.object(retrieval_tool, "retrieve", side_effect=retrieve_fn), \
             patch.object(retrieval_tool, "_fetch_neighbor_chunks", return_value={}), \
             patch.object(retrieval_tool, "_hyde", return_value="a hypothetical answer passage"):
            _build_tool(use_hyde=True).func("a real question")
        assert "a hypothetical answer passage" in queries

    def test_sheet_summary_with_column_overlap_is_included(self):
        sheet = _hit(
            "Sheet summary.\ncolumns: supplier, transaction, amount",
            chunk_type="sheet_summary",
            source_file="doc_009_spend.xlsx",
            sheet_name="Q1",
        )
        retrieve_fn = _make_retrieve(
            sheet_summary=[sheet],
            content=[_hit("text content", chunk_index=0)],
        )
        with patch.object(retrieval_tool, "retrieve", side_effect=retrieve_fn), \
             patch.object(retrieval_tool, "_fetch_neighbor_chunks", return_value={}):
            out, _artifact = _build_tool().func("which supplier had the largest transaction amount?")
        assert "doc_009_spend.xlsx" in out

    def test_pure_spreadsheet_doc_falls_back_to_sheet_summary_without_column_overlap(self):
        """A question about the FILE ITSELF ("what year is this titled for")
        mentions no real column name, so the col_overlap>=2 gate rejects every
        sheet_summary hit -- reproduced live: a pure-.xlsx corpus doc (only
        sheet_summary chunks exist, no page_content) returned "No relevant
        information found." on every call despite a highly-relevant
        sheet_summary hit existing. Must fall back to it when nothing else
        survived, not return empty."""
        sheet = _hit(
            "[File: doc_011_spain_maturity_2025.xlsx | Sheet: Portal]\ncolumns: id, question, answer",
            chunk_type="sheet_summary",
            source_file="doc_011_spain_maturity_2025.xlsx",
            sheet_name="Portal",
            score=0.83,
        )
        retrieve_fn = _make_retrieve(sheet_summary=[sheet], content=[])
        with patch.object(retrieval_tool, "retrieve", side_effect=retrieve_fn), \
             patch.object(retrieval_tool, "_fetch_neighbor_chunks", return_value={}):
            out, _artifact = _build_tool().func(
                "What year is the Spain maturity questionnaire titled for?"
            )
        assert out is not None
        assert "doc_011_spain_maturity_2025.xlsx" in out

    def test_sheet_summary_fallback_does_not_fire_when_content_hits_exist(self):
        """The fallback must only cover the "nothing else survived" case -- it
        must not paper over a genuinely irrelevant sheet_summary when real
        page_content hits already answer the question."""
        sheet = _hit(
            "Sheet summary.\ncolumns: unrelated, columns, here",
            chunk_type="sheet_summary",
            source_file="doc_020_other.xlsx",
            sheet_name="X",
        )
        retrieve_fn = _make_retrieve(
            sheet_summary=[sheet],
            content=[_hit("the real answer text", chunk_index=0, source_file="doc_009_report.pdf")],
        )
        with patch.object(retrieval_tool, "retrieve", side_effect=retrieve_fn), \
             patch.object(retrieval_tool, "_fetch_neighbor_chunks", return_value={}):
            out, _artifact = _build_tool().func("what does the report say?")
        assert "doc_020_other.xlsx" not in out
        assert "the real answer text" in out

    def test_stem_token_overlap_boosts_doc_from_registry(self):
        injected: list = []
        retrieve_fn = _make_retrieve(
            content=[_hit("policy text", doc_id="doc_001",
                          source_file="doc_001_procurement_policy.pdf")],
            spy=lambda kw: injected.append(kw.get("scope_doc_id")),
        )
        registry = {"doc_001_procurement_policy": "doc_001"}
        with patch.object(retrieval_tool, "retrieve", side_effect=retrieve_fn), \
             patch.object(retrieval_tool, "_fetch_neighbor_chunks", return_value={}):
            out, _artifact = _build_tool(doc_registry=registry).func("procurement policy approval rules")
        assert "doc_001_procurement_policy.pdf" in out

    def test_filter_token_merges_unfiltered_pool(self):
        """A hard content filter_token (e.g. "LACERA") must not be the only
        thing deciding recall — reproduced live: the answer chunk's own text
        read "L/CERA" (OCR'd logo), never the literal token, and was silently
        excluded until the unfiltered pool was merged in."""
        filtered_hit = _hit("filtered-only content", chunk_index=0)
        unfiltered_hit = _hit("unfiltered-only content", chunk_index=1)

        def retrieve_fn(**kwargs):
            forced = kwargs.get("force_chunk_types")
            if forced == ["document_summary"]:
                return []
            if forced == ["sheet_summary"]:
                return []
            if kwargs.get("filter_token"):
                return [filtered_hit]
            return [unfiltered_hit]

        with patch.object(retrieval_tool, "retrieve", side_effect=retrieve_fn), \
             patch.object(retrieval_tool, "_fetch_neighbor_chunks", return_value={}):
            out, _artifact = _build_tool().func("Who is the manager in the LACERA document?")
        assert "filtered-only content" in out
        assert "unfiltered-only content" in out

    def test_no_filter_token_no_extra_call(self):
        """A query with no extractable filter_token must not trigger the
        extra unfiltered merge pass — only the normal retrieval calls fire."""
        calls: list = []
        retrieve_fn = _make_retrieve(
            content=[_hit("plain content", chunk_index=0)],
            spy=lambda kw: calls.append(kw),
        )
        with patch.object(retrieval_tool, "retrieve", side_effect=retrieve_fn), \
             patch.object(retrieval_tool, "_fetch_neighbor_chunks", return_value={}):
            _build_tool().func("what time is it")
        content_calls = [
            c for c in calls
            if c.get("force_chunk_types") not in (["document_summary"], ["sheet_summary"])
        ]
        assert all(c.get("filter_token") is None for c in content_calls)

    def test_filtered_hits_rank_first_pre_rerank(self):
        """Merged pool keeps filtered hits ahead of unfiltered ones before
        reranking — preserves the exact-match boost for real ID lookups."""
        filtered_hit = _hit("filtered content", chunk_index=0, source_file="doc_005_report.pdf")
        unfiltered_hit = _hit("unfiltered content", chunk_index=2, source_file="doc_005_report.pdf")

        def retrieve_fn(**kwargs):
            forced = kwargs.get("force_chunk_types")
            if forced == ["document_summary"]:
                return []
            if forced == ["sheet_summary"]:
                return []
            if kwargs.get("filter_token"):
                return [filtered_hit]
            return [unfiltered_hit]

        with patch.object(retrieval_tool, "retrieve", side_effect=retrieve_fn), \
             patch.object(retrieval_tool, "_fetch_neighbor_chunks", return_value={}):
            out, _artifact = _build_tool().func("Who is the manager in the LACERA document?")
        assert out.index("filtered content") < out.index("unfiltered content")
