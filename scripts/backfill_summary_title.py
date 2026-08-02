"""One-time backfill: add a literal "Title:" line to already-ingested
document_summary chunks, matching src/chunker.py's ingest-time fix.

generate_document_summary() only ever produces an LLM paraphrase, never the
literal cover-page title -- so a "what is the title of this document"
question had nothing verbatim to match against, and the agent tended to pick
a more prominent section heading deeper in the document instead (e.g.
doc_001 answering "V. Purchasing and Contracting Policy" instead of the real
cover-page title). Only touches document_summary points whose file_name
resolves to a processed markdown file on disk and doesn't already carry a
Title: line.
"""

from __future__ import annotations

import re
from pathlib import Path

import requests

from vault_rag.chunker import extract_literal_title
from vault_rag.config import (
    EMBED_API_BASE,
    OLLAMA_EMBED_MODEL,
    QDRANT_COLLECTION,
    QDRANT_URL,
)
from vault_rag.retriever import _ollama_embed_query
from vault_rag.sparse_embedder import get_sparse_embedder

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data/output/processed"
_TITLE_LINE_RE = re.compile(r"\nTitle: [^\n]*")


def backfill(url: str = QDRANT_URL, collection: str = QDRANT_COLLECTION) -> None:
    """Scroll every document_summary point and add a missing Title: line."""
    base = url.rstrip("/")
    embedder = get_sparse_embedder()
    offset = None
    updated = 0
    skipped = 0
    while True:
        body: dict = {
            "limit": 100,
            "with_payload": True,
            "with_vector": True,
            "filter": {
                "must": [
                    {
                        "key": "metadata.chunk_type",
                        "match": {"value": "document_summary"},
                    }
                ]
            },
        }
        if offset is not None:
            body["offset"] = offset
        resp = requests.post(
            f"{base}/collections/{collection}/points/scroll", json=body
        )
        resp.raise_for_status()
        result = resp.json()["result"]
        points = result["points"]
        if not points:
            break

        for p in points:
            payload = p["payload"]
            orig_content = payload.get("content", "")
            orig_vector_text = payload.get("vector_text", "") or orig_content
            # Strip any Title: line this script previously inserted (a first
            # pass here used a naive first-heading heuristic that got several
            # documents wrong -- e.g. picking a boilerplate label or a bare
            # dateline as the title). Recomputing from scratch every run makes
            # this idempotent and self-correcting instead of skip-on-sight.
            content = _TITLE_LINE_RE.sub("", orig_content, count=1)
            vector_text = _TITLE_LINE_RE.sub("", orig_vector_text, count=1)

            file_name = payload.get("metadata", {}).get("file_name")
            md_path = PROCESSED_DIR / file_name if file_name else None
            if not md_path or not md_path.exists():
                skipped += 1
                continue

            title = extract_literal_title(md_path.read_text(encoding="utf-8"))
            if not title:
                # No confident title -- if a wrong one was previously inserted,
                # the strip above already removed it; re-upsert only if that
                # actually changed anything.
                if content == orig_content and vector_text == orig_vector_text:
                    skipped += 1
                    continue
                new_content, new_vector_text = content, vector_text
            else:
                new_content = content.replace(
                    "\nFile: ", "\nTitle: " + title + "\nFile: ", 1
                )
                if new_content == content:
                    lines = content.split("\n", 1)
                    new_content = (
                        lines[0]
                        + f"\n\nTitle: {title}"
                        + ("\n" + lines[1] if len(lines) > 1 else "")
                    )
                new_vector_text = vector_text.replace(
                    "\nFile: ", "\nTitle: " + title + "\nFile: ", 1
                )

            dense = _ollama_embed_query(
                EMBED_API_BASE, OLLAMA_EMBED_MODEL, new_vector_text
            )
            sp_indices, sp_values = embedder.embed(new_vector_text)

            payload["content"] = new_content
            payload["vector_text"] = new_vector_text
            upsert_point = {
                "id": p["id"],
                "vector": {
                    "": dense,
                    "sparse": {"indices": sp_indices, "values": sp_values},
                },
                "payload": payload,
            }
            put_resp = requests.put(
                f"{base}/collections/{collection}/points?wait=true",
                json={"points": [upsert_point]},
            )
            put_resp.raise_for_status()
            updated += 1
            print(
                f"[BACKFILL] {payload.get('metadata', {}).get('doc_id')}: title = {title!r}"
            )

        offset = result.get("next_page_offset")
        if offset is None:
            break

    print(f"[BACKFILL] done: updated={updated} skipped={skipped}")


if __name__ == "__main__":
    backfill()
