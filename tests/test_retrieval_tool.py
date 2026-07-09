"""Characterization tests for the search_knowledge_base retrieval tool.

These pin the current observable behavior of _make_unified_tool / _fetch_docs so
the function can be refactored safely. retrieve(), the reranker, HyDE, and the
neighbour-chunk fetch are all mocked — no live services.
"""
from __future__ import annotations

from unittest.mock import patch

from src.tools import retrieval_tool
from src.tools.retrieval_tool import _make_unified_tool


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
