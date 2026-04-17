import argparse
import json
import math
import os
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


def _qdrant_search(
    qdrant_url: str,
    collection: str,
    query_vec: list[float],
    top_k: int,
    filter_token: str | None = None,
) -> list[dict[str, Any]]:
    """Run a dense vector search against Qdrant, with an optional keyword filter."""
    base = qdrant_url.rstrip("/")
    url = f"{base}/collections/{collection}/points/search"
    body: dict[str, Any] = {"vector": query_vec, "limit": top_k, "with_payload": True}
    if filter_token:
        body["filter"] = _text_filter(filter_token)
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
    if filter_token:
        body["filter"] = _text_filter(filter_token)
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
) -> list[dict[str, Any]]:
    query_vec = _ollama_embed_query(api_base=api_base, model_name=model_name, query=query)

    if use_qdrant:
        if _collection_has_sparse(qdrant_url=qdrant_url, collection=collection):
            from src.sparse_embedder import get_sparse_embedder
            sparse_indices, sparse_values = get_sparse_embedder().embed(query)
            points = _qdrant_hybrid_search(
                qdrant_url=qdrant_url,
                collection=collection,
                query_vec=query_vec,
                sparse_indices=sparse_indices,
                sparse_values=sparse_values,
                top_k=top_k,
                filter_token=filter_token,
            )
        else:
            points = _qdrant_search(
                qdrant_url=qdrant_url,
                collection=collection,
                query_vec=query_vec,
                top_k=top_k,
                filter_token=filter_token,
            )
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
