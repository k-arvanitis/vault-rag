import argparse
import hashlib
import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from src.config import QDRANT_COLLECTION, QDRANT_URL


def _stable_id(file_name: str, chunk_idx: object) -> int:
    """Deterministic point ID so re-ingesting the same file updates existing points."""
    key = f"{file_name}::{chunk_idx}"
    return int(hashlib.sha1(key.encode()).hexdigest()[:15], 16)


def _request(method: str, url: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = Request(
        url, data=data, method=method, headers={"Content-Type": "application/json"}
    )
    try:
        with urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Vector store HTTP {exc.code} at {url}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(
            f"Could not connect to vector store at {url}. Is Qdrant running?"
        ) from exc


def _collection_has_sparse(info: dict, collection: str, verbose: bool = True) -> bool:
    """Check if the collection has sparse_vectors configured; warn if not."""
    params = info.get("result", {}).get("config", {}).get("params", {})
    has_sparse = bool(params.get("sparse_vectors"))
    if not has_sparse and verbose:
        print(
            f"[VECTOR_STORE] WARNING: collection '{collection}' has no sparse_vectors. "
            "Hybrid search will not be available. Reset and re-ingest to enable it."
        )
    return has_sparse


def ingest_embeddings(
    input_path: Path,
    url: str = QDRANT_URL,
    collection: str = QDRANT_COLLECTION,
    verbose: bool = True,
) -> None:
    if verbose:
        print(f"[VECTOR_STORE] Loading embeddings: {input_path}")
    rows = json.loads(input_path.read_text(encoding="utf-8"))
    vector_size = len(rows[0]["embedding"])
    base = url.rstrip("/")
    if verbose:
        print(f"[VECTOR_STORE] Connecting to: {base}")

    # 1) Create collection if it does not exist.
    try:
        info = _request("GET", f"{base}/collections/{collection}")
        if verbose:
            print(f"[VECTOR_STORE] Collection exists: {collection}")
        # Warn if sparse vectors are not configured (hybrid search will be unavailable).
        _collection_has_sparse(info, collection, verbose=verbose)
    except RuntimeError:
        _request(
            "PUT",
            f"{base}/collections/{collection}",
            {
                "vectors": {"size": vector_size, "distance": "Cosine"},
                "sparse_vectors": {"sparse": {}},
            },
        )
        if verbose:
            print(f"[VECTOR_STORE] Created collection: {collection}")

    # 2) Ingest chunks.
    from src.sparse_embedder import get_sparse_embedder

    points = []
    for i, row in enumerate(rows):
        vector_text = row.get("vector_text", "") or row.get("content", "")
        meta = row.get("metadata", {}) or {}
        file_name = meta.get("file_name") or meta.get("source_file", "")
        chunk_idx = meta.get("chunk_index", i)
        point_id = _stable_id(file_name, chunk_idx) if file_name else i
        point: dict = {
            "id": point_id,
            "vector": {"": row["embedding"]},
            "payload": {
                "source_id": row.get("id", i),
                "content": row.get("content", ""),
                "vector_text": row.get("vector_text", ""),
                "metadata": row.get("metadata", {}),
            },
        }
        try:
            sparse_indices, sparse_values = get_sparse_embedder().embed(vector_text)
            point["vector"]["sparse"] = {"indices": sparse_indices, "values": sparse_values}
        except Exception:
            pass  # Fall back to dense-only point
        points.append(point)

    _request(
        "PUT",
        f"{base}/collections/{collection}/points?wait=true",
        {"points": points},
    )
    if verbose:
        print(f"[VECTOR_STORE] Ingested {len(points)} points into {collection}")


def scroll_all_payloads(url: str, collection: str) -> list[dict]:
    """Return all point payloads from the collection (for listing ingested files)."""
    base = url.rstrip("/")
    payloads: list[dict] = []
    offset = None
    while True:
        body: dict = {"limit": 100, "with_payload": True, "with_vector": False}
        if offset is not None:
            body["offset"] = offset
        try:
            result = _request(
                "POST", f"{base}/collections/{collection}/points/scroll", body
            )
        except RuntimeError as exc:
            if "404" in str(exc) or "doesn't exist" in str(exc):
                return []
            raise
        points = result.get("result", {}).get("points", [])
        payloads.extend(p.get("payload", {}) for p in points)
        offset = result.get("result", {}).get("next_page_offset")
        if not offset:
            break
    return payloads


def get_document_summary(url: str, collection: str, file_name: str) -> str | None:
    """Return the document summary content for a given file, or None if not found."""
    base = url.rstrip("/")
    try:
        result = _request(
            "POST",
            f"{base}/collections/{collection}/points/scroll",
            {
                "filter": {
                    "must": [
                        {
                            "key": "metadata.chunk_type",
                            "match": {"value": "document_summary"},
                        },
                        {"key": "metadata.file_name", "match": {"value": file_name}},
                    ]
                },
                "limit": 1,
                "with_payload": True,
                "with_vector": False,
            },
        )
    except RuntimeError:
        return None
    points = result.get("result", {}).get("points", [])
    if not points:
        return None
    return points[0].get("payload", {}).get("content") or None


def get_chunks_by_file(url: str, collection: str, source_file: str) -> list[dict]:
    """Return all chunk payloads for a given source_file (used by the inspector)."""
    base = url.rstrip("/")
    payloads: list[dict] = []
    offset = None
    while True:
        body: dict = {
            "filter": {
                "must": [
                    {"key": "metadata.source_file", "match": {"value": source_file}},
                ]
            },
            "limit": 250,
            "with_payload": True,
            "with_vector": False,
        }
        if offset is not None:
            body["offset"] = offset
        try:
            result = _request(
                "POST", f"{base}/collections/{collection}/points/scroll", body
            )
        except RuntimeError as exc:
            if "404" in str(exc) or "doesn't exist" in str(exc):
                return []
            raise
        points = result.get("result", {}).get("points", [])
        payloads.extend(p.get("payload", {}) for p in points)
        offset = result.get("result", {}).get("next_page_offset")
        if not offset:
            break
    return payloads


def delete_by_file(url: str, collection: str, file_name: str) -> int:
    """Delete all PDF chunk points for a given file_name. Returns deleted count."""
    base = url.rstrip("/")
    try:
        result = _request(
            "POST",
            f"{base}/collections/{collection}/points/count",
            {
                "filter": {
                    "must": [
                        {"key": "metadata.source_file", "match": {"value": file_name}}
                    ]
                }
            },
        )
    except RuntimeError as exc:
        if "404" in str(exc) or "doesn't exist" in str(exc):
            return 0
        raise
    count = result.get("result", {}).get("count", 0)
    if count > 0:
        _request(
            "POST",
            f"{base}/collections/{collection}/points/delete?wait=true",
            {
                "filter": {
                    "must": [
                        {"key": "metadata.source_file", "match": {"value": file_name}}
                    ]
                }
            },
        )
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--url", default=QDRANT_URL)
    parser.add_argument("--collection", default=QDRANT_COLLECTION)
    args = parser.parse_args()
    ingest_embeddings(input_path=args.input, url=args.url, collection=args.collection)


if __name__ == "__main__":
    main()
