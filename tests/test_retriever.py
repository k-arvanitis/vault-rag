"""Tests for src/retriever.py.

All network calls (Ollama, Qdrant) are mocked — no live services required.
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from vault_rag.retriever import (
    _cosine_similarity,
    _metadata_filter,
    _text_filter,
    infer_query_chunk_types,
    retrieve,
)

# ---------------------------------------------------------------------------
# _cosine_similarity
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def test_identical_vectors_return_one(self):
        v = [1.0, 0.0, 0.0]
        assert math.isclose(_cosine_similarity(v, v), 1.0)

    def test_orthogonal_vectors_return_zero(self):
        assert math.isclose(_cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0)

    def test_opposite_vectors_return_minus_one(self):
        assert math.isclose(_cosine_similarity([1.0, 0.0], [-1.0, 0.0]), -1.0)

    def test_zero_vector_returns_zero(self):
        assert _cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0

    def test_dimension_mismatch_raises(self):
        import pytest

        with pytest.raises(ValueError, match="dimension mismatch"):
            _cosine_similarity([1.0, 2.0], [1.0])


# ---------------------------------------------------------------------------
# _text_filter
# ---------------------------------------------------------------------------


class TestMetadataFilterScopeDocId:
    """scope_doc_id accepts a single id (str) or multiple (list[str]) — a list
    ORs across documents (the UI's multi-select source scope), it doesn't AND,
    since one chunk only ever belongs to one document."""

    def test_single_string_scope_unchanged(self):
        f = _metadata_filter(scope_doc_id="doc_001")
        should = f["must"][0]["should"]
        assert {"key": "metadata.doc_id", "match": {"value": "doc_001"}} in should
        assert len(should) == 3

    def test_list_scope_ors_across_all_ids(self):
        f = _metadata_filter(scope_doc_id=["doc_001", "doc_002"])
        should = f["must"][0]["should"]
        assert {"key": "metadata.doc_id", "match": {"value": "doc_001"}} in should
        assert {"key": "metadata.doc_id", "match": {"value": "doc_002"}} in should
        assert len(should) == 6

    def test_empty_list_is_no_filter(self):
        assert _metadata_filter(scope_doc_id=[]) is None

    def test_none_is_no_filter(self):
        assert _metadata_filter(scope_doc_id=None) is None


class TestTextFilter:
    def test_returns_must_clause(self):
        f = _text_filter("ABC123")
        assert "must" in f
        assert f["must"][0]["key"] == "content"

    def test_token_embedded_in_filter(self):
        f = _text_filter("XYZ")
        assert f["must"][0]["match"]["text"] == "XYZ"


# ---------------------------------------------------------------------------
# query routing
# ---------------------------------------------------------------------------


class TestInferQueryChunkTypes:
    def test_always_returns_none_none(self):
        # Routing was removed in favour of column-overlap reranking; the
        # function now searches all chunk types unconditionally.
        for query in (
            "How long must the Supplier keep records after the Agreement ends?",
            "In the spreadsheet row for transaction 12345, what is the amount?",
            "What total cost is shown for the 25 transactions?",
        ):
            include, exclude = infer_query_chunk_types(query)
            assert include is None
            assert exclude is None


# ---------------------------------------------------------------------------
# retrieve — local JSON path (no Qdrant)
# ---------------------------------------------------------------------------


class TestRetrieveFromJson:
    """retrieve() with use_qdrant=False, mocking only the Ollama embedding call."""

    _FAKE_EMBED = [1.0, 0.0, 0.0]

    def _make_json(self, tmp: str, rows: list[dict]) -> Path:
        p = Path(tmp) / "embeddings.json"
        p.write_text(json.dumps(rows), encoding="utf-8")
        return p

    @patch("vault_rag.retriever._ollama_embed_query")
    def test_returns_top_k_results(self, mock_embed):
        mock_embed.return_value = self._FAKE_EMBED
        rows = [
            {
                "id": str(i),
                "embedding": [float(i % 2), 0.0, 0.0],
                "content": f"doc{i}",
                "metadata": {},
            }
            for i in range(10)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = self._make_json(tmp, rows)
            results = retrieve("q", embeddings_path=path, top_k=3, use_qdrant=False)
        assert len(results) == 3

    @patch("vault_rag.retriever._ollama_embed_query")
    def test_results_sorted_by_score_descending(self, mock_embed):
        mock_embed.return_value = [1.0, 0.0]
        rows = [
            {
                "id": "low",
                "embedding": [0.0, 1.0],
                "content": "irrelevant",
                "metadata": {},
            },
            {
                "id": "high",
                "embedding": [1.0, 0.0],
                "content": "relevant",
                "metadata": {},
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = self._make_json(tmp, rows)
            results = retrieve("q", embeddings_path=path, top_k=2, use_qdrant=False)
        assert results[0]["id"] == "high"

    @patch("vault_rag.retriever._ollama_embed_query")
    def test_empty_json_returns_empty_list(self, mock_embed):
        mock_embed.return_value = self._FAKE_EMBED
        with tempfile.TemporaryDirectory() as tmp:
            path = self._make_json(tmp, [])
            results = retrieve("q", embeddings_path=path, top_k=5, use_qdrant=False)
        assert results == []

    @patch("vault_rag.retriever._ollama_embed_query")
    def test_rows_without_embedding_skipped(self, mock_embed):
        mock_embed.return_value = self._FAKE_EMBED
        rows = [
            {"id": "1", "content": "no embedding here", "metadata": {}},
            {
                "id": "2",
                "embedding": [1.0, 0.0, 0.0],
                "content": "has embedding",
                "metadata": {},
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = self._make_json(tmp, rows)
            results = retrieve("q", embeddings_path=path, top_k=5, use_qdrant=False)
        assert len(results) == 1
        assert results[0]["id"] == "2"

    @patch("vault_rag.retriever._ollama_embed_query")
    def test_result_has_required_keys(self, mock_embed):
        mock_embed.return_value = [1.0, 0.0]
        rows = [
            {
                "id": "x",
                "embedding": [1.0, 0.0],
                "content": "hello",
                "metadata": {"k": "v"},
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = self._make_json(tmp, rows)
            results = retrieve("q", embeddings_path=path, top_k=1, use_qdrant=False)
        assert set(results[0].keys()) >= {"score", "id", "content", "metadata"}


# ---------------------------------------------------------------------------
# retrieve — Qdrant dense path
# ---------------------------------------------------------------------------


class TestRetrieveFromQdrant:
    """retrieve() with use_qdrant=True — mocks Ollama, Qdrant, and sparse embedder."""

    _FAKE_EMBED = [0.1, 0.2, 0.3]
    _QDRANT_POINTS = [
        {
            "id": "pt1",
            "score": 0.9,
            "payload": {
                "source_id": "chunk-1",
                "content": "first chunk",
                "vector_text": "ctx first",
                "metadata": {"file_name": "a.pdf"},
            },
        },
        {
            "id": "pt2",
            "score": 0.7,
            "payload": {
                "source_id": "chunk-2",
                "content": "second chunk",
                "vector_text": "",
                "metadata": {"file_name": "b.pdf"},
            },
        },
    ]

    @patch("vault_rag.retriever._collection_has_sparse", return_value=False)
    @patch("vault_rag.retriever._qdrant_search")
    @patch("vault_rag.retriever._ollama_embed_query")
    def test_dense_search_returns_mapped_hits(self, mock_embed, mock_search, _):
        mock_embed.return_value = self._FAKE_EMBED
        mock_search.return_value = self._QDRANT_POINTS
        results = retrieve("query", use_qdrant=True, top_k=5)
        assert len(results) == 2
        assert results[0]["id"] == "chunk-1"
        assert results[0]["content"] == "first chunk"
        assert results[0]["score"] == 0.9

    @patch("vault_rag.retriever._collection_has_sparse", return_value=True)
    @patch("vault_rag.retriever._qdrant_hybrid_search")
    @patch("vault_rag.retriever._ollama_embed_query")
    def test_hybrid_search_used_when_sparse_present(self, mock_embed, mock_hybrid, _):
        mock_embed.return_value = self._FAKE_EMBED
        mock_hybrid.return_value = self._QDRANT_POINTS
        with patch("vault_rag.sparse_embedder.get_sparse_embedder") as mock_sparse_cls:
            mock_sparse = MagicMock()
            mock_sparse.embed.return_value = ([0, 1], [0.5, 0.5])
            mock_sparse_cls.return_value = mock_sparse
            results = retrieve("query", use_qdrant=True, top_k=5)
        mock_hybrid.assert_called_once()
        assert len(results) == 2

    @patch("vault_rag.retriever._collection_has_sparse", return_value=False)
    @patch("vault_rag.retriever._qdrant_search")
    @patch("vault_rag.retriever._ollama_embed_query")
    def test_scoped_retrieval_with_results_issues_one_search(
        self, mock_embed, mock_search, _
    ):
        """scope_doc_key is a dead parameter (_metadata_filter never reads
        it) -- re-running the same scoped query under three different
        scope_doc_key values used to issue three byte-identical, wasted
        Qdrant round-trips whenever the primary scoped search already found
        results."""
        mock_embed.return_value = self._FAKE_EMBED
        mock_search.return_value = self._QDRANT_POINTS
        results = retrieve(
            "query", use_qdrant=True, top_k=5, scope_doc_id="doc_001"
        )
        assert len(results) == 2
        mock_search.assert_called_once()

    @patch("vault_rag.retriever._qdrant_scroll_filter", return_value=[])
    @patch("vault_rag.retriever._collection_has_sparse", return_value=False)
    @patch("vault_rag.retriever._qdrant_search")
    @patch("vault_rag.retriever._ollama_embed_query")
    def test_scoped_retrieval_empty_with_filter_token_retries_once(
        self, mock_embed, mock_search, _, __
    ):
        """An empty scoped result with a filter_token set retries exactly
        once with the token dropped -- the only genuinely different query
        in this path -- not the old three-scope_doc_key fan-out."""
        mock_embed.return_value = self._FAKE_EMBED
        mock_search.side_effect = [[], self._QDRANT_POINTS]
        results = retrieve(
            "query",
            use_qdrant=True,
            top_k=5,
            scope_doc_id="doc_001",
            filter_token="LACERA",
        )
        assert len(results) == 2
        assert mock_search.call_count == 2


# ---------------------------------------------------------------------------
# M-3: term-count sort must only fire for genuine structured lookups
# ---------------------------------------------------------------------------


class TestTermCountSortGate:
    """The final sort in retrieve() reorders hits by literal query-term count.
    _extract_table_filter_terms fires on any prose query, so the sort must be
    gated on an ID-shaped filter_token or sheet_row/sheet_table chunk types
    in the candidate pool -- not on filter_terms alone."""

    # High-fusion-score hit with no term overlap, low-fusion hit with term
    # overlap -- if the term-count sort fires, order flips; if not, score
    # order (A first) is preserved.
    _POINTS = [
        {
            "id": "ptA",
            "score": 0.9,
            "payload": {
                "source_id": "chunk-A",
                "content": "irrelevant wording with no overlap",
                "metadata": {"chunk_type": "prose"},
            },
        },
        {
            "id": "ptB",
            "score": 0.1,
            "payload": {
                "source_id": "chunk-B",
                "content": "supplier invoice transaction reference details",
                "metadata": {"chunk_type": "prose"},
            },
        },
    ]

    @patch("vault_rag.retriever._collection_has_sparse", return_value=False)
    @patch("vault_rag.retriever._qdrant_search")
    @patch("vault_rag.retriever._ollama_embed_query")
    def test_prose_query_keeps_fusion_order(self, mock_embed, mock_search, _):
        mock_embed.return_value = [0.1, 0.2]
        mock_search.return_value = self._POINTS
        results = retrieve(
            "supplier invoice transaction reference details",
            use_qdrant=True,
            top_k=5,
        )
        assert [r["id"] for r in results] == ["chunk-A", "chunk-B"]
        # over-fetch still fires even though the sort is gated off
        assert mock_search.call_args.kwargs["top_k"] == 100

    @patch("vault_rag.retriever._qdrant_scroll_filter", return_value=[])
    @patch("vault_rag.retriever._collection_has_sparse", return_value=False)
    @patch("vault_rag.retriever._qdrant_search")
    @patch("vault_rag.retriever._ollama_embed_query")
    def test_id_shaped_query_triggers_term_sort(
        self, mock_embed, mock_search, _, __
    ):
        mock_embed.return_value = [0.1, 0.2]
        mock_search.return_value = self._POINTS
        results = retrieve(
            "supplier invoice transaction reference INV58204",
            use_qdrant=True,
            top_k=5,
            filter_token="INV58204",
        )
        assert [r["id"] for r in results] == ["chunk-B", "chunk-A"]
        assert mock_search.call_args.kwargs["top_k"] == 100

    @patch("vault_rag.retriever._collection_has_sparse", return_value=False)
    @patch("vault_rag.retriever._qdrant_search")
    @patch("vault_rag.retriever._ollama_embed_query")
    def test_sheet_chunk_pool_triggers_term_sort_without_id_token(
        self, mock_embed, mock_search, _
    ):
        points = [
            {
                "id": "ptA",
                "score": 0.9,
                "payload": {
                    "source_id": "chunk-A",
                    "content": "irrelevant wording with no overlap",
                    "metadata": {"chunk_type": "sheet_row"},
                },
            },
            {
                "id": "ptB",
                "score": 0.1,
                "payload": {
                    "source_id": "chunk-B",
                    "content": "supplier invoice transaction reference details",
                    "metadata": {"chunk_type": "sheet_row"},
                },
            },
        ]
        mock_embed.return_value = [0.1, 0.2]
        mock_search.return_value = points
        results = retrieve(
            "supplier invoice transaction reference details",
            use_qdrant=True,
            top_k=5,
        )
        assert [r["id"] for r in results] == ["chunk-B", "chunk-A"]
