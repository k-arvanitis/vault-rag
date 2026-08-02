"""First-stage chunk retrieval — embed a query and search Qdrant.

Embeds the query via Ollama, then runs dense / hybrid (dense + sparse RRF)
vector search against a Qdrant collection, with optional payload filters for
chunk-type routing, single-document scoping and content text matching. Also
supports an offline mode that scores a local embeddings JSON file directly.

Called by: src/tools/retrieval_tool.py (_fetch_docs / _resolve_scope call
retrieve() for the agent's search_knowledge_base tool) and the CLI main().
Calls into: Ollama embed API, Qdrant HTTP API, src/sparse_embedder.py
(get_sparse_embedder) for hybrid search, and src/config.py for defaults.
"""

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

from vault_rag.config import OLLAMA_EMBED_MODEL

# ---------------------------------------------------------------------------
# Embedding & similarity helpers
# ---------------------------------------------------------------------------


def _ollama_embed_query(api_base: str, model_name: str, query: str) -> list[float]:
    """Embed a single query string via the Ollama /api/embed endpoint."""
    # Build the POST request to Ollama's embed endpoint (num_gpu=0 keeps it CPU).
    url = f"{api_base.rstrip('/')}/api/embed"
    payload = json.dumps(
        {"model": model_name, "input": [query], "options": {"num_gpu": 0}}
    ).encode("utf-8")
    req = Request(
        url, data=payload, headers={"Content-Type": "application/json"}, method="POST"
    )
    # Send the request, turning HTTP/connection errors into clear RuntimeErrors.
    try:
        with urlopen(req, timeout=300) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Ollama embed request failed ({exc.code}): {detail}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(
            f"Could not connect to Ollama at {api_base}. Ensure `ollama serve` is running."
        ) from exc

    # Validate the response shape and return the single query's embedding vector.
    embeddings = body.get("embeddings")
    if not isinstance(embeddings, list) or not embeddings:
        raise RuntimeError(f"Unexpected Ollama response: {body}")
    return embeddings[0]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two equal-length embedding vectors."""
    # Reject mismatched dimensions early — usually a wrong-model bug.
    if len(a) != len(b):
        raise ValueError(f"Embedding dimension mismatch: query={len(a)} doc={len(b)}")
    # Dot product over magnitudes; guard against zero-vector division.
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Qdrant payload filters — chunk-type routing, doc scoping, text matching
# ---------------------------------------------------------------------------


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
    scope_doc_id: str | list[str] | None = None,
    scope_doc_key: str = "metadata.doc_id",
) -> dict | None:
    """Build a Qdrant filter for chunk-type routing and optional text filtering.

    scope_doc_id restricts results to one or more documents via a should-OR
    across metadata.doc_id, metadata.source_file, and metadata.file_name for
    each id — covering both old ingestions (source_file only) and new ones
    (doc_id set). A list ORs across documents (matches any one of them), not
    ANDs (a chunk only ever belongs to one document, so requiring it match
    every id in the list would always return nothing).
    scope_doc_key is kept for backward compat but is no longer used.
    """
    # Accumulate positive (must) and negative (must_not) filter conditions.
    must: list[dict] = []
    must_not: list[dict] = []
    # Restrict to / exclude specific chunk types (document_summary, sheet_summary, …).
    if chunk_types:
        must.append({"key": "metadata.chunk_type", "match": {"any": chunk_types}})
    if exclude_chunk_types:
        must_not.append(
            {"key": "metadata.chunk_type", "match": {"any": exclude_chunk_types}}
        )
    # Require a literal token to appear in the chunk's content text.
    if filter_token:
        must.append({"key": "content", "match": {"text": filter_token}})
    if scope_doc_id:
        scope_doc_ids = (
            [scope_doc_id] if isinstance(scope_doc_id, str) else list(scope_doc_id)
        )
        # OR across all three doc-id fields for every requested document —
        # older ingestions may only have source_file/file_name; newer ones set
        # metadata.doc_id explicitly.
        should: list[dict] = []
        for doc_id in scope_doc_ids:
            should.extend(
                [
                    {"key": "metadata.doc_id", "match": {"value": doc_id}},
                    {"key": "metadata.source_file", "match": {"text": doc_id}},
                    {"key": "metadata.file_name", "match": {"text": doc_id}},
                ]
            )
        must.append({"should": should})
    # No conditions means "no filter" — Qdrant should search everything.
    if not must and not must_not:
        return None
    # Assemble the filter dict, omitting empty must / must_not keys.
    result: dict = {}
    if must:
        result["must"] = must
    if must_not:
        result["must_not"] = must_not
    return result


# ---------------------------------------------------------------------------
# Table query analysis — pick distinctive value tokens for table lookups
# ---------------------------------------------------------------------------

_TABLE_STOP_WORDS = frozenset(
    {
        # Question structure words
        "what",
        "which",
        "where",
        "when",
        "who",
        "how",
        "does",
        "according",
        "listed",
        "appears",
        "dated",
        "where",
        # Generic English connectives
        "is",
        "the",
        "a",
        "an",
        "for",
        "on",
        "in",
        "at",
        "of",
        "to",
        "with",
        "and",
        "that",
        "this",
        "from",
        "have",
        "been",
        # Common table column names — these are structural words, not values
        "row",
        "amount",
        "total",
        "date",
        "number",
        "net",
        "value",
        "transaction",
        "transactions",
        "supplier",
        "beneficiary",
        "merchant",
        "category",
        "purchase",
        "expenditure",
        "department",
        "directorate",
        "authority",
        # Context words that appear in table-query phrasing but are not entity values
        "spreadsheet",
        "report",
        "spend",
        "published",
        "card",
    }
)


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
    candidates = [
        w for w in words if w.lower() not in _TABLE_STOP_WORDS and len(w) >= 5
    ]
    if not candidates:
        return None
    # Prefer tokens that look like proper values: contain digits or start with uppercase
    # Pick the longest such token (most distinctive) as the filter token.
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
    # Drop generic words and corpus-specific noise; keep distinct content terms.
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


# ---------------------------------------------------------------------------
# Chunk-type routing
# ---------------------------------------------------------------------------


def infer_query_chunk_types(query: str) -> tuple[list[str] | None, list[str] | None]:
    """Infer chunk type routing for a query.

    Always returns (None, None) — search all chunk types for every query.
    sheet_summary chunks are small and few; column-overlap scoring in the
    reranking step surfaces them only when a sheet's columns match the query.
    """
    return None, None


# ---------------------------------------------------------------------------
# Qdrant transport — dense, hybrid and scroll-filter search calls
# ---------------------------------------------------------------------------


def _qdrant_search(
    qdrant_url: str,
    collection: str,
    query_vec: list[float],
    top_k: int,
    filter_token: str | None = None,
    chunk_types: list[str] | None = None,
    exclude_chunk_types: list[str] | None = None,
    scope_doc_id: str | list[str] | None = None,
    scope_doc_key: str = "metadata.doc_id",
) -> list[dict[str, Any]]:
    """Run a dense vector search against Qdrant, with optional payload filters."""
    # Build the /points/search request body with the dense query vector.
    base = qdrant_url.rstrip("/")
    url = f"{base}/collections/{collection}/points/search"
    body: dict[str, Any] = {"vector": query_vec, "limit": top_k, "with_payload": True}
    # Attach the payload filter (chunk-type / doc-scope / token) if any.
    payload_filter = _metadata_filter(
        chunk_types=chunk_types,
        exclude_chunk_types=exclude_chunk_types,
        filter_token=filter_token,
        scope_doc_id=scope_doc_id,
        scope_doc_key=scope_doc_key,
    )
    if payload_filter:
        body["filter"] = payload_filter
    # Send the request, converting HTTP/connection errors into RuntimeErrors.
    payload = json.dumps(body).encode("utf-8")
    req = Request(
        url, data=payload, headers={"Content-Type": "application/json"}, method="POST"
    )
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

    # Validate and return the list of scored points.
    result = body.get("result")
    if not isinstance(result, list):
        raise RuntimeError(f"Unexpected Qdrant response: {body}")
    return result


def _collection_has_sparse(qdrant_url: str, collection: str) -> bool:
    """Return True if the collection has sparse_vectors configured."""
    # GET the collection config and inspect whether sparse_vectors is defined.
    # Any error (collection missing, service down) is treated as "no sparse".
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
    scope_doc_id: str | list[str] | None = None,
    scope_doc_key: str = "metadata.doc_id",
) -> list[dict[str, Any]]:
    """Hybrid search using dense prefetch + sparse prefetch + RRF fusion."""
    # Build the /points/query body: two prefetch branches (dense + sparse,
    # each over-fetching 3x) fused by reciprocal-rank fusion into top_k.
    base = qdrant_url.rstrip("/")
    url = f"{base}/collections/{collection}/points/query"
    body: dict[str, Any] = {
        "prefetch": [
            {"query": query_vec, "limit": top_k * 3},
            {
                "query": {"indices": sparse_indices, "values": sparse_values},
                "using": "sparse",
                "limit": top_k * 3,
            },
        ],
        "query": {"fusion": "rrf"},
        "limit": top_k,
        "with_payload": True,
    }
    # Attach the payload filter (chunk-type / doc-scope / token) if any.
    payload_filter = _metadata_filter(
        chunk_types=chunk_types,
        exclude_chunk_types=exclude_chunk_types,
        filter_token=filter_token,
        scope_doc_id=scope_doc_id,
        scope_doc_key=scope_doc_key,
    )
    if payload_filter:
        body["filter"] = payload_filter
    # Send the request, converting HTTP/connection errors into RuntimeErrors.
    payload = json.dumps(body).encode("utf-8")
    req = Request(
        url, data=payload, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urlopen(req, timeout=300) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Qdrant hybrid query failed ({exc.code}): {detail}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(f"Could not connect to Qdrant at {qdrant_url}.") from exc

    # /query nests the hits under result.points — validate and return them.
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
    scope_doc_id: str | list[str] | None = None,
    scope_doc_key: str = "metadata.doc_id",
) -> list[dict[str, Any]]:
    """Fetch exact payload-filter matches without vector ranking."""
    # Build the /points/scroll request — pure filter match, no vector ranking.
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
    # Send the request, converting HTTP/connection errors into RuntimeErrors.
    payload = json.dumps(body).encode("utf-8")
    req = Request(
        url, data=payload, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urlopen(req, timeout=300) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Qdrant scroll failed ({exc.code}): {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not connect to Qdrant at {qdrant_url}.") from exc

    # Validate, then return matches with a uniform score=1.0 (no ranking here).
    points = body.get("result", {}).get("points", [])
    if not isinstance(points, list):
        raise RuntimeError(f"Unexpected Qdrant scroll response: {body}")
    return [{**point, "score": 1.0} for point in points]


# ---------------------------------------------------------------------------
# Public entry point — embed the query and return ranked chunks
# ---------------------------------------------------------------------------


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
    scope_doc_id: str | list[str] | None = None,
) -> list[dict[str, Any]]:
    """Retrieve top-k relevant chunks from the full collection.

    force_chunk_types / force_exclude_chunk_types bypass infer_query_chunk_types.
    Use force_chunk_types=["document_summary"] for stage-1 doc routing.
    Use force_exclude_chunk_types=["sheet_row","sheet_table"] to pin PDF-only retrieval
    even when the query contains table-trigger keywords (e.g. "supplier", "invoice").
    scope_doc_id restricts results to a single document (e.g. "doc_001") via Qdrant
    source_file text filter — guarantees small docs appear even if globally outranked.
    """
    # Embed the query once — the same vector is reused for every search below.
    query_vec = _ollama_embed_query(
        api_base=api_base, model_name=model_name, query=query
    )

    if use_qdrant:
        # Decide chunk-type routing: explicit force_* overrides win, otherwise
        # infer from the query (currently always None — search all types).
        if force_chunk_types is not None or force_exclude_chunk_types is not None:
            chunk_types = force_chunk_types
            exclude_chunk_types = force_exclude_chunk_types
        else:
            chunk_types, exclude_chunk_types = infer_query_chunk_types(query)
        # When the query has table value terms, over-fetch (>=100) so the soft
        # term-match reordering at the end has enough candidates to work with.
        filter_terms = _extract_table_filter_terms(query)
        qdrant_top_k = max(top_k, 100) if filter_terms else top_k

        def _search_with_scope(scope_doc_key: str) -> list[dict[str, Any]]:
            """Run hybrid (or dense) search for the given doc-scope key field."""
            # Use hybrid search when the collection supports sparse vectors;
            # fall back to plain dense search on any sparse-path failure.
            if _collection_has_sparse(qdrant_url=qdrant_url, collection=collection):
                try:
                    from vault_rag.sparse_embedder import get_sparse_embedder

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

        # Primary search: hybrid when sparse vectors exist (dense fallback on
        # error), otherwise plain dense search.
        if _collection_has_sparse(qdrant_url=qdrant_url, collection=collection):
            try:
                from vault_rag.sparse_embedder import get_sparse_embedder

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
        # Single-doc scoping: the primary search above already applied
        # scope_doc_id (which itself ORs across doc_id/source_file/file_name
        # in _metadata_filter, see its docstring) -- scope_doc_key is a dead
        # parameter (_metadata_filter never reads it), so re-running
        # _search_with_scope under three different scope_doc_key values used
        # to issue three byte-identical, fully redundant Qdrant round-trips
        # per scoped call. If nothing matched and a filter_token is set, drop
        # the token and retry once more -- that IS a genuinely different query.
        if scope_doc_id and not points and filter_token:
            original_filter_token = filter_token
            filter_token = None
            points = _search_with_scope("metadata.doc_id")
            filter_token = original_filter_token
        # Exact-match boost: scroll the filter_token (and alternate terms) to
        # surface verbatim hits, then prepend them ahead of the vector results.
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
                # No exact hit for the primary token — try up to 4 longest
                # alternate query terms (handles multi-word/punctuated names).
                if not exact_points:
                    alternate_terms = [
                        term
                        for term in sorted(filter_terms, key=len, reverse=True)
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
                # Prepend exact matches, then append vector hits not already seen.
                seen_ids = {point.get("id") for point in exact_points}
                points = exact_points + [
                    point for point in points if point.get("id") not in seen_ids
                ]
            except Exception:
                pass
        # Normalize raw Qdrant points into the flat hit dicts callers expect.
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
        # Table queries: soft-reorder hits by how many value terms they contain
        # (primary key), breaking ties with the vector score, then cut to top_k.
        # Gated to genuine structured lookups -- filter_terms alone fires on any
        # prose query (any word >=3 chars survives the stop list), so also
        # require an ID-shaped filter_token (contains a digit) or that the
        # candidate pool actually contains spreadsheet row/table chunks.
        is_id_shaped = bool(filter_token and any(c.isdigit() for c in filter_token))
        has_table_chunks = any(
            (hit.get("metadata") or {}).get("chunk_type") in ("sheet_row", "sheet_table")
            for hit in scored_from_qdrant
        )
        if filter_terms and (is_id_shaped or has_table_chunks):

            def _table_term_score(hit: dict[str, Any]) -> int:
                """Count how many table value terms appear in a hit's content."""
                content = (hit.get("content") or "").lower()
                return sum(1 for term in filter_terms if term in content)

            scored_from_qdrant.sort(
                key=lambda h: (_table_term_score(h), h.get("score", 0.0)), reverse=True
            )
            return scored_from_qdrant[:top_k]
        return scored_from_qdrant[:top_k]

    # --- Offline mode: no Qdrant — score a local embeddings JSON file directly.
    if embeddings_path is None:
        raise ValueError("`embeddings_path` is required when `use_qdrant=False`.")

    # Load and validate the embeddings file.
    rows = json.loads(embeddings_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(
            f"Expected a list in {embeddings_path}, got {type(rows).__name__}"
        )
    if not rows:
        return []

    # Cosine-score every stored chunk against the query embedding.
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

    # Rank by similarity and return the top_k chunks.
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse CLI args, run a single retrieve() call and print the JSON results."""
    # Load .env so OLLAMA_API_BASE etc. are picked up before arg parsing.
    load_dotenv()

    # Define the CLI arguments (query, embeddings path, Qdrant settings, …).
    parser = argparse.ArgumentParser(
        description="Retrieve top-k relevant chunks from embeddings JSON."
    )
    parser.add_argument("--query", required=True, help="Search query text")
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=None,
        help="Path to embeddings JSON created by src/embedder.py (used when --no-use-qdrant).",
    )
    parser.add_argument(
        "--top-k", type=int, default=5, help="Number of chunks to return"
    )
    parser.add_argument(
        "--qdrant-url", default="http://127.0.0.1:7333", help="Qdrant base URL"
    )
    parser.add_argument(
        "--collection", default="documents_chunks", help="Qdrant collection name"
    )
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

    # Run retrieval with the parsed arguments.
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

    # Emit the ranked hits as pretty-printed JSON.
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
