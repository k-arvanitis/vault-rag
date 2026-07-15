"""Tests for src/vector_store.py's point construction, mocking Qdrant HTTP calls."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.vector_store import get_source_by_duckdb_table, ingest_embeddings


def test_ingest_embeddings_sends_dense_and_sparse_under_one_vector_field(tmp_path: Path):
    """Qdrant requires named vectors (dense "" + sparse) under a single "vector"
    dict per point -- a prior bug put sparse under a separate top-level
    "sparse_vectors" key, which Qdrant silently ignores (no error, no sparse
    index), leaving hybrid search running dense-only across the whole corpus.
    """
    rows = [
        {
            "id": "0",
            "content": "hello world",
            "vector_text": "hello world",
            "embedding": [0.1, 0.2, 0.3],
            "metadata": {"file_name": "doc.md", "chunk_index": 0},
        }
    ]
    input_path = tmp_path / "embeddings.json"
    input_path.write_text(json.dumps(rows))

    sent_bodies = []

    def fake_request(method, url, body=None):
        sent_bodies.append((method, url, body))
        if method == "GET":
            raise RuntimeError("no collection")
        return {"result": "ok"}

    fake_sparse = MagicMock()
    fake_sparse.embed.return_value = ([1, 5, 9], [0.5, 0.3, 0.2])

    with (
        patch("src.vector_store._request", side_effect=fake_request),
        patch("src.sparse_embedder.get_sparse_embedder", return_value=fake_sparse),
    ):
        ingest_embeddings(input_path, url="http://qdrant", collection="test", verbose=False)

    upsert_body = next(b for m, u, b in sent_bodies if m == "PUT" and "/points" in u)
    point = upsert_body["points"][0]
    assert point["vector"][""] == [0.1, 0.2, 0.3]
    assert point["vector"]["sparse"] == {"indices": [1, 5, 9], "values": [0.5, 0.3, 0.2]}
    assert "sparse_vectors" not in point


def test_get_source_by_duckdb_table_returns_file_and_sheet():
    fake_points = {
        "result": {
            "points": [
                {
                    "payload": {
                        "metadata": {
                            "source_file": "expenses_2024.xlsx",
                            "sheet_name": "Sheet1",
                            "duckdb_table": "expenses_2024__Sheet1",
                        }
                    }
                }
            ]
        }
    }
    with patch("src.vector_store._request", return_value=fake_points) as mock_req:
        result = get_source_by_duckdb_table(
            "http://qdrant", "test", "expenses_2024__Sheet1"
        )
    assert result == {"source_file": "expenses_2024.xlsx", "sheet_name": "Sheet1"}
    body = mock_req.call_args[0][2]
    assert body["filter"]["must"][0]["match"]["value"] == "expenses_2024__Sheet1"


def test_get_source_by_duckdb_table_no_match_returns_none():
    with patch(
        "src.vector_store._request", return_value={"result": {"points": []}}
    ):
        assert get_source_by_duckdb_table("http://qdrant", "test", "missing__Sheet1") is None
