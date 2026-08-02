"""One-time backfill: set metadata.doc_id on every existing point that's
missing it. Only the document_summary chunk ever got a doc_id (see
src/chunker.py) -- every narrative/page chunk had none, so any retrieval
call scoped to a specific document (routing directives, cross-document
comparison retries) silently searched nothing for that chunk type. Derives
doc_id from metadata.file_name the same way src/chunker.py now does at
ingest time. Safe to re-run: only touches points missing doc_id.
"""

from __future__ import annotations

import re

import requests

from vault_rag.config import QDRANT_COLLECTION, QDRANT_URL

_DOC_ID_RE = re.compile(r"doc_\d+")


def backfill(url: str = QDRANT_URL, collection: str = QDRANT_COLLECTION, batch_size: int = 200) -> None:
    """Scroll every point; set metadata.doc_id where missing, derived from file_name."""
    base = url.rstrip("/")
    offset = None
    updated = 0
    skipped = 0
    unresolved = 0
    while True:
        body: dict = {"limit": batch_size, "with_payload": True, "with_vector": False}
        if offset is not None:
            body["offset"] = offset
        resp = requests.post(f"{base}/collections/{collection}/points/scroll", json=body)
        resp.raise_for_status()
        result = resp.json()["result"]
        points = result["points"]
        if not points:
            break

        for p in points:
            metadata = p["payload"].get("metadata", {}) or {}
            if metadata.get("doc_id"):
                skipped += 1
                continue
            file_name = metadata.get("file_name") or metadata.get("source_file") or ""
            match = _DOC_ID_RE.search(file_name)
            if not match:
                unresolved += 1
                continue
            metadata["doc_id"] = match.group(0)
            set_resp = requests.post(
                f"{base}/collections/{collection}/points/payload?wait=true",
                json={"points": [p["id"]], "payload": {"metadata": metadata}},
            )
            set_resp.raise_for_status()
            updated += 1

        offset = result.get("next_page_offset")
        if offset is None:
            break

    print(f"[BACKFILL] updated={updated} already_had_doc_id={skipped} unresolved={unresolved}")


if __name__ == "__main__":
    backfill()
