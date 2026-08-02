"""One-time backfill: add the missing named "sparse" vector to every existing
Qdrant point. ingest_embeddings() used to send sparse vectors under a
top-level "sparse_vectors" key that Qdrant's point-upsert API doesn't
recognize -- silently dropped, no error -- leaving every previously-ingested
point dense-only despite the collection being configured for hybrid search.

Re-embeds from each point's own payload (vector_text or content), keeps the
existing dense vector and id unchanged, and re-upserts in place. Safe to
re-run: it's an idempotent overwrite, not an append.
"""

from __future__ import annotations

import requests

from vault_rag.config import QDRANT_COLLECTION, QDRANT_URL
from vault_rag.sparse_embedder import get_sparse_embedder


def backfill(url: str = QDRANT_URL, collection: str = QDRANT_COLLECTION, batch_size: int = 64) -> None:
    """Scroll every point, compute its sparse vector, and re-upsert in place."""
    embedder = get_sparse_embedder()
    base = url.rstrip("/")
    offset = None
    total = 0
    while True:
        body: dict = {"limit": batch_size, "with_payload": True, "with_vector": True}
        if offset is not None:
            body["offset"] = offset
        resp = requests.post(f"{base}/collections/{collection}/points/scroll", json=body)
        resp.raise_for_status()
        result = resp.json()["result"]
        points = result["points"]
        if not points:
            break

        upsert_points = []
        for p in points:
            vector = p["vector"]
            dense = vector[""] if isinstance(vector, dict) else vector
            payload = p["payload"]
            text = payload.get("vector_text") or payload.get("content") or ""
            indices, values = embedder.embed(text)
            upsert_points.append(
                {
                    "id": p["id"],
                    "vector": {"": dense, "sparse": {"indices": indices, "values": values}},
                    "payload": payload,
                }
            )

        put_resp = requests.put(
            f"{base}/collections/{collection}/points?wait=true",
            json={"points": upsert_points},
        )
        put_resp.raise_for_status()
        total += len(upsert_points)
        print(f"[BACKFILL] {total} points updated")

        offset = result.get("next_page_offset")
        if offset is None:
            break

    print(f"[BACKFILL] done, {total} points total")


if __name__ == "__main__":
    backfill()
