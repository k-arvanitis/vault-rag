import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv

from src.config import OLLAMA_EMBED_MODEL


def _ollama_embed_query(api_base: str, model_name: str, query: str) -> list[float]:
    """Embed a single query string via the Ollama /api/embed endpoint."""
    url = f"{api_base.rstrip('/')}/api/embed"
    payload = json.dumps({"model": model_name, "input": [query], "options": {"num_gpu": 0}}).encode("utf-8")
    req = Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(req, timeout=300) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama embed request failed ({exc.code}): {detail}") from exc
    except URLError as exc:
        raise RuntimeError(
            f"Could not connect to Ollama at {api_base}. Ensure `ollama serve` is running."
        ) from exc

    embeddings = body.get("embeddings")
    if not isinstance(embeddings, list) or not embeddings:
        raise RuntimeError(f"Unexpected Ollama response: {body}")
    return embeddings[0]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two equal-length embedding vectors."""
    if len(a) != len(b):
        raise ValueError(f"Embedding dimension mismatch: query={len(a)} doc={len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _text_filter(token: str) -> dict:
    """Build a Qdrant payload filter that requires token to appear in the content field."""
    return {"must": [{"key": "content", "match": {"text": token}}]}


_TABLE_CHUNK_TYPES = ["sheet_summary"]
_TABLE_SCHEMA_CHUNK_TYPES = ["document_summary", "sheet_summary"]


def _metadata_filter(
    *,
    chunk_types: list[str] | None = None,
    exclude_chunk_types: list[str] | None = None,
    filter_token: str | None = None,
    scope_doc_id: str | None = None,
    scope_doc_key: str = "metadata.doc_id",
) -> dict | None:
    """Build a Qdrant filter for chunk-type routing and optional text filtering.

    scope_doc_id restricts results to a single document via a should-OR across
    metadata.doc_id, metadata.source_file, and metadata.file_name — covering
    both old ingestions (source_file only) and new ones (doc_id set).
    scope_doc_key is kept for backward compat but is no longer used.
    """
    must: list[dict] = []
    must_not: list[dict] = []
    if chunk_types:
        must.append({"key": "metadata.chunk_type", "match": {"any": chunk_types}})
    if exclude_chunk_types:
        must_not.append({"key": "metadata.chunk_type", "match": {"any": exclude_chunk_types}})
    if filter_token:
        must.append({"key": "content", "match": {"text": filter_token}})
    if scope_doc_id:
        # OR across all three doc-id fields — older ingestions may only have
        # source_file/file_name; newer ones set metadata.doc_id explicitly.
        must.append({
            "should": [
                {"key": "metadata.doc_id", "match": {"value": scope_doc_id}},
                {"key": "metadata.source_file", "match": {"text": scope_doc_id}},
                {"key": "metadata.file_name", "match": {"text": scope_doc_id}},
            ]
        })
    if not must and not must_not:
        return None
    result: dict = {}
    if must:
        result["must"] = must
    if must_not:
        result["must_not"] = must_not
    return result


_TABLE_STOP_WORDS = frozenset({
    # Question structure words
    "what", "which", "where", "when", "who", "how", "does", "according", "listed",
    "appears", "dated", "where",
    # Generic English connectives
    "is", "the", "a", "an", "for", "on", "in", "at", "of", "to", "with", "and",
    "that", "this", "from", "have", "been",
    # Common table column names — these are structural words, not values
    "row", "amount", "total", "date", "number", "net", "value",
    "transaction", "transactions", "supplier", "beneficiary",
    "merchant", "category", "purchase", "expenditure",
    "department", "directorate", "authority",
    # Context words that appear in table-query phrasing but are not entity values
    "spreadsheet", "report", "spend", "published", "card",
})


def _extract_table_filter_token(query: str) -> str | None:
    """Extract the most distinctive value token from a table lookup query for Qdrant text filtering.

    Prefers numeric IDs (transaction numbers), then long proper-noun tokens not in the stop list.
    """
    # Prefer explicit numeric identifiers (transaction numbers, IDs) — skip 4-digit years
    numeric = re.findall(r"\b\d{5,}\b", query)
    if numeric:
        return max(numeric, key=len)

    # Plain alphanumeric tokens only — no & or . so cross-word combos aren't captured
    words = re.findall(r"[A-Za-z0-9]+", query)
    candidates = [w for w in words if w.lower() not in _TABLE_STOP_WORDS and len(w) >= 5]
    if not candidates:
        return None
    # Prefer tokens that look like proper values: contain digits or start with uppercase
    proper = [w for w in candidates if any(c.isdigit() for c in w) or w[0].isupper()]
    pool = proper if proper else candidates
    return max(pool, key=len)


def _extract_table_filter_terms(query: str) -> list[str]:
    """Return useful content terms for reordering table lookup hits.

    Qdrant receives one high-precision filter token to keep recall reasonable.
    The remaining distinctive terms are used as a soft row-match score after
    retrieval, which helps cases where a supplier/beneficiary name has multiple
    words or punctuation variants.
    """
    extra_stops = {"doncaster", "council", "q1", "april", "2025", "2026"}
    terms: list[str] = []
    for token in re.findall(r"[A-Za-z0-9]+", query):
        lowered = token.lower()
        if len(lowered) < 3:
            continue
        if lowered in _TABLE_STOP_WORDS or lowered in extra_stops:
            continue
        if lowered not in terms:
            terms.append(lowered)
    return terms


def infer_query_chunk_types(query: str) -> tuple[list[str] | None, list[str] | None]:
    """Infer chunk type routing for a query.

    Always returns (None, None) — search all chunk types for every query.
    sheet_summary chunks are small and few; column-overlap scoring in the
    reranking step surfaces them only when a sheet's columns match the query.
    """
    return None, None


def _qdrant_search(
    qdrant_url: str,
    collection: str,
    query_vec: list[float],
    top_k: int,
    filter_token: str | None = None,
    chunk_types: list[str] | None = None,
    exclude_chunk_types: list[str] | None = None,
    scope_doc_id: str | None = None,
    scope_doc_key: str = "metadata.doc_id",
) -> list[dict[str, Any]]:
    """Run a dense vector search against Qdrant, with optional payload filters."""
    base = qdrant_url.rstrip("/")
    url = f"{base}/collections/{collection}/points/search"
    body: dict[str, Any] = {"vector": query_vec, "limit": top_k, "with_payload": True}
    payload_filter = _metadata_filter(
        chunk_types=chunk_types,
        exclude_chunk_types=exclude_chunk_types,
        filter_token=filter_token,
        scope_doc_id=scope_doc_id,
        scope_doc_key=scope_doc_key,
    )
    if payload_filter:
        body["filter"] = payload_filter
    payload = json.dumps(body).encode("utf-8")
    req = Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(req, timeout=300) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Qdrant search failed ({exc.code}): {detail}") from exc
    except URLError as exc:
        raise RuntimeError(
            f"Could not connect to Qdrant at {qdrant_url}. Ensure the service is running."
        ) from exc

    result = body.get("result")
    if not isinstance(result, list):
        raise RuntimeError(f"Unexpected Qdrant response: {body}")
    return result


def _collection_has_sparse(qdrant_url: str, collection: str) -> bool:
    """Return True if the collection has sparse_vectors configured."""
    base = qdrant_url.rstrip("/")
    try:
        req = Request(f"{base}/collections/{collection}", method="GET")
        with urlopen(req, timeout=10) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        params = body.get("result", {}).get("config", {}).get("params", {})
        return bool(params.get("sparse_vectors"))
    except Exception:
        return False


def _qdrant_hybrid_search(
    qdrant_url: str,
    collection: str,
    query_vec: list[float],
    sparse_indices: list[int],
    sparse_values: list[float],
    top_k: int,
    filter_token: str | None = None,
    chunk_types: list[str] | None = None,
    exclude_chunk_types: list[str] | None = None,
    scope_doc_id: str | None = None,
    scope_doc_key: str = "metadata.doc_id",
) -> list[dict[str, Any]]:
    """Hybrid search using dense prefetch + sparse prefetch + RRF fusion."""
    base = qdrant_url.rstrip("/")
    url = f"{base}/collections/{collection}/points/query"
    body: dict[str, Any] = {
        "prefetch": [
            {"query": query_vec, "limit": top_k * 3},
            {"query": {"indices": sparse_indices, "values": sparse_values}, "using": "sparse", "limit": top_k * 3},
        ],
        "query": {"fusion": "rrf"},
        "limit": top_k,
        "with_payload": True,
    }
    payload_filter = _metadata_filter(
        chunk_types=chunk_types,
        exclude_chunk_types=exclude_chunk_types,
        filter_token=filter_token,
        scope_doc_id=scope_doc_id,
        scope_doc_key=scope_doc_key,
    )
    if payload_filter:
        body["filter"] = payload_filter
    payload = json.dumps(body).encode("utf-8")
    req = Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(req, timeout=300) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Qdrant hybrid query failed ({exc.code}): {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not connect to Qdrant at {qdrant_url}.") from exc

    result = body.get("result", {})
    # /query returns {"result": {"points": [...]}}
    points = result.get("points", []) if isinstance(result, dict) else []
    if not isinstance(points, list):
        raise RuntimeError(f"Unexpected Qdrant hybrid response: {body}")
    return points


def _qdrant_scroll_filter(
    qdrant_url: str,
    collection: str,
    limit: int,
    filter_token: str | None = None,
    chunk_types: list[str] | None = None,
    exclude_chunk_types: list[str] | None = None,
    scope_doc_id: str | None = None,
    scope_doc_key: str = "metadata.doc_id",
) -> list[dict[str, Any]]:
    """Fetch exact payload-filter matches without vector ranking."""
    base = qdrant_url.rstrip("/")
    url = f"{base}/collections/{collection}/points/scroll"
    payload_filter = _metadata_filter(
        chunk_types=chunk_types,
        exclude_chunk_types=exclude_chunk_types,
        filter_token=filter_token,
        scope_doc_id=scope_doc_id,
        scope_doc_key=scope_doc_key,
    )
    body: dict[str, Any] = {
        "limit": limit,
        "with_payload": True,
        "with_vector": False,
    }
    if payload_filter:
        body["filter"] = payload_filter
    payload = json.dumps(body).encode("utf-8")
    req = Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(req, timeout=300) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Qdrant scroll failed ({exc.code}): {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not connect to Qdrant at {qdrant_url}.") from exc

    points = body.get("result", {}).get("points", [])
    if not isinstance(points, list):
        raise RuntimeError(f"Unexpected Qdrant scroll response: {body}")
    return [{**point, "score": 1.0} for point in points]


def retrieve(
    query: str,
    embeddings_path: Path | None = None,
    top_k: int = 20,
    api_base: str = "http://127.0.0.1:11434",
    model_name: str = OLLAMA_EMBED_MODEL,
    qdrant_url: str = "http://127.0.0.1:7333",
    collection: str = "documents_chunks",
    use_qdrant: bool = True,
    filter_token: str | None = None,
    force_chunk_types: list[str] | None = None,
    force_exclude_chunk_types: list[str] | None = None,
    scope_doc_id: str | None = None,
) -> list[dict[str, Any]]:
    """Retrieve top-k relevant chunks from the full collection.

    force_chunk_types / force_exclude_chunk_types bypass infer_query_chunk_types.
    Use force_chunk_types=["document_summary"] for stage-1 doc routing.
    Use force_exclude_chunk_types=["sheet_row","sheet_table"] to pin PDF-only retrieval
    even when the query contains table-trigger keywords (e.g. "supplier", "invoice").
    scope_doc_id restricts results to a single document (e.g. "doc_001") via Qdrant
    source_file text filter — guarantees small docs appear even if globally outranked.
    """
    query_vec = _ollama_embed_query(api_base=api_base, model_name=model_name, query=query)

    if use_qdrant:
        if force_chunk_types is not None or force_exclude_chunk_types is not None:
            chunk_types = force_chunk_types
            exclude_chunk_types = force_exclude_chunk_types
        else:
            chunk_types, exclude_chunk_types = infer_query_chunk_types(query)
        filter_terms = _extract_table_filter_terms(query)
        qdrant_top_k = max(top_k, 100) if filter_terms else top_k

        def _search_with_scope(scope_doc_key: str) -> list[dict[str, Any]]:
            if _collection_has_sparse(qdrant_url=qdrant_url, collection=collection):
                try:
                    from src.sparse_embedder import get_sparse_embedder
                    sparse_indices, sparse_values = get_sparse_embedder().embed(query)
                    hybrid_points = _qdrant_hybrid_search(
                        qdrant_url=qdrant_url,
                        collection=collection,
                        query_vec=query_vec,
                        sparse_indices=sparse_indices,
                        sparse_values=sparse_values,
                        top_k=qdrant_top_k,
                        filter_token=filter_token,
                        chunk_types=chunk_types,
                        exclude_chunk_types=exclude_chunk_types,
                        scope_doc_id=scope_doc_id,
                        scope_doc_key=scope_doc_key,
                    )
                    if hybrid_points:
                        return hybrid_points
                except Exception:
                    pass
                return _qdrant_search(
                    qdrant_url=qdrant_url,
                    collection=collection,
                    query_vec=query_vec,
                    top_k=qdrant_top_k,
                    filter_token=filter_token,
                    chunk_types=chunk_types,
                    exclude_chunk_types=exclude_chunk_types,
                    scope_doc_id=scope_doc_id,
                    scope_doc_key=scope_doc_key,
                )
            return _qdrant_search(
                qdrant_url=qdrant_url,
                collection=collection,
                query_vec=query_vec,
                top_k=qdrant_top_k,
                filter_token=filter_token,
                chunk_types=chunk_types,
                exclude_chunk_types=exclude_chunk_types,
                scope_doc_id=scope_doc_id,
                scope_doc_key=scope_doc_key,
            )

        if _collection_has_sparse(qdrant_url=qdrant_url, collection=collection):
            try:
                from src.sparse_embedder import get_sparse_embedder
                sparse_indices, sparse_values = get_sparse_embedder().embed(query)
                points = _qdrant_hybrid_search(
                    qdrant_url=qdrant_url,
                    collection=collection,
                    query_vec=query_vec,
                    sparse_indices=sparse_indices,
                    sparse_values=sparse_values,
                    top_k=qdrant_top_k,
                    filter_token=filter_token,
                    chunk_types=chunk_types,
                    exclude_chunk_types=exclude_chunk_types,
                    scope_doc_id=scope_doc_id,
                )
            except Exception:
                points = _qdrant_search(
                    qdrant_url=qdrant_url,
                    collection=collection,
                    query_vec=query_vec,
                    top_k=qdrant_top_k,
                    filter_token=filter_token,
                    chunk_types=chunk_types,
                    exclude_chunk_types=exclude_chunk_types,
                    scope_doc_id=scope_doc_id,
                )
        else:
            points = _qdrant_search(
                qdrant_url=qdrant_url,
                collection=collection,
                query_vec=query_vec,
                top_k=qdrant_top_k,
                filter_token=filter_token,
                chunk_types=chunk_types,
                exclude_chunk_types=exclude_chunk_types,
                scope_doc_id=scope_doc_id,
            )
        if scope_doc_id:
            points = _search_with_scope("metadata.doc_id")
            if not points:
                points = _search_with_scope("metadata.source_file")
            if not points:
                # Fall back to file_name text index (works for PDF chunks that lack doc_id)
                points = _search_with_scope("metadata.file_name")
            if not points and filter_token:
                original_filter_token = filter_token
                filter_token = None
                points = _search_with_scope("metadata.doc_id")
                if not points:
                    points = _search_with_scope("metadata.source_file")
                if not points:
                    points = _search_with_scope("metadata.file_name")
                filter_token = original_filter_token
        if filter_token:
            try:
                scroll_chunk_types = chunk_types
                exact_points = _qdrant_scroll_filter(
                    qdrant_url=qdrant_url,
                    collection=collection,
                    limit=qdrant_top_k,
                    filter_token=filter_token,
                    chunk_types=scroll_chunk_types,
                    exclude_chunk_types=exclude_chunk_types,
                    scope_doc_id=scope_doc_id,
                    scope_doc_key="metadata.doc_id",
                )
                if scope_doc_id and not exact_points:
                    exact_points = _qdrant_scroll_filter(
                        qdrant_url=qdrant_url,
                        collection=collection,
                        limit=qdrant_top_k,
                        filter_token=filter_token,
                        chunk_types=scroll_chunk_types,
                        exclude_chunk_types=exclude_chunk_types,
                        scope_doc_id=scope_doc_id,
                        scope_doc_key="metadata.source_file",
                    )
                if not exact_points:
                    alternate_terms = [
                        term for term in sorted(filter_terms, key=len, reverse=True)
                        if term != filter_token.lower() and len(term) >= 4
                    ][:4]
                    for term in alternate_terms:
                        term_points = _qdrant_scroll_filter(
                            qdrant_url=qdrant_url,
                            collection=collection,
                            limit=max(10, qdrant_top_k // 4),
                            filter_token=term,
                            chunk_types=scroll_chunk_types,
                            exclude_chunk_types=exclude_chunk_types,
                            scope_doc_id=scope_doc_id,
                            scope_doc_key="metadata.doc_id",
                        )
                        if scope_doc_id and not term_points:
                            term_points = _qdrant_scroll_filter(
                                qdrant_url=qdrant_url,
                                collection=collection,
                                limit=max(10, qdrant_top_k // 4),
                                filter_token=term,
                                chunk_types=scroll_chunk_types,
                                exclude_chunk_types=exclude_chunk_types,
                                scope_doc_id=scope_doc_id,
                                scope_doc_key="metadata.source_file",
                            )
                        exact_points.extend(term_points)
                seen_ids = {point.get("id") for point in exact_points}
                points = exact_points + [point for point in points if point.get("id") not in seen_ids]
            except Exception:
                pass
        scored_from_qdrant: list[dict[str, Any]] = []
        for point in points:
            payload = point.get("payload", {}) or {}
            if not isinstance(payload, dict):
                payload = {}
            scored_from_qdrant.append(
                {
                    "score": point.get("score", 0.0),
                    "id": payload.get("source_id", point.get("id")),
                    "content": payload.get("content", ""),
                    "vector_text": payload.get("vector_text", ""),
                    "metadata": payload.get("metadata", {}),
                }
            )
        if filter_terms:
            def _table_term_score(hit: dict[str, Any]) -> int:
                content = (hit.get("content") or "").lower()
                return sum(1 for term in filter_terms if term in content)

            scored_from_qdrant.sort(key=lambda h: (_table_term_score(h), h.get("score", 0.0)), reverse=True)
            return scored_from_qdrant[:top_k]
        return scored_from_qdrant

    if embeddings_path is None:
        raise ValueError("`embeddings_path` is required when `use_qdrant=False`.")

    rows = json.loads(embeddings_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Expected a list in {embeddings_path}, got {type(rows).__name__}")
    if not rows:
        return []

    scored: list[dict[str, Any]] = []
    for row in rows:
        emb = row.get("embedding")
        if not isinstance(emb, list):
            continue
        score = _cosine_similarity(query_vec, emb)
        scored.append(
            {
                "score": score,
                "id": row.get("id"),
                "content": row.get("content", ""),
                "vector_text": row.get("vector_text", ""),
                "metadata": row.get("metadata", {}),
            }
        )

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Retrieve top-k relevant chunks from embeddings JSON.")
    parser.add_argument("--query", required=True, help="Search query text")
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=None,
        help="Path to embeddings JSON created by src/embedder.py (used when --no-use-qdrant).",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to return")
    parser.add_argument("--qdrant-url", default="http://127.0.0.1:7333", help="Qdrant base URL")
    parser.add_argument("--collection", default="documents_chunks", help="Qdrant collection name")
    parser.add_argument(
        "--use-qdrant",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Query Qdrant directly. Disable to use local embeddings JSON.",
    )
    parser.add_argument(
        "--api-base",
        default=os.getenv("OLLAMA_API_BASE", "http://127.0.0.1:11434"),
        help="Ollama base URL",
    )
    parser.add_argument(
        "--model-name",
        default=OLLAMA_EMBED_MODEL,
        help="Ollama embedding model name",
    )
    args = parser.parse_args()

    results = retrieve(
        query=args.query,
        embeddings_path=args.embeddings,
        top_k=args.top_k,
        qdrant_url=args.qdrant_url,
        collection=args.collection,
        use_qdrant=args.use_qdrant,
        api_base=args.api_base,
        model_name=args.model_name,
    )

    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
