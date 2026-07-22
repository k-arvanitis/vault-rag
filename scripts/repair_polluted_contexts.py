"""One-time repair: 133 of 862 corpus chunks were embedded with a literal LLM
rate-limit error string ("CONTEXT: Error: Error code: 429 - {'error': ...}")
as their embedding prefix — src/chunker.py's contextualize_chunk used to
return f"Error: {e}" on any enrichment failure and bake it straight into
chunk.vector_text (fixed at the source, see contextualize_chunk /
chunk_markdown). This script repairs the already-ingested fallout: it
re-runs enrichment for every polluted chunk in the 4 affected documents,
updates the chunks JSON, the embeddings JSON, and upserts the corrected
points into Qdrant (dense + sparse) so retrieval sees clean text.

Affected files (data/output/chunks/):
  doc_001_procurement_policy_chunks.json   (69 chunks)
  doc_015_food_sop_manual_chunks.json      (52 chunks)
  doc_016a_original_lease_chunks.json      (6 chunks)
  doc_016c_second_amendment_chunks.json    (6 chunks)

Idempotent: a second run finds 0 polluted chunks per file and exits cleanly.
One chunk's enrichment failing does not abort the run — it's logged and left
empty (never re-poisoned with an error string), so no re-run can regress.

Usage: uv run python -m scripts.repair_polluted_contexts [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from openai import OpenAI

from src.chunker import contextualize_chunk
from src.config import (
    CHUNK_LLM_API_BASE,
    CHUNK_LLM_API_KEY,
    CHUNK_LLM_MODEL,
    OLLAMA_API_BASE,
    OLLAMA_EMBED_MODEL,
    QDRANT_COLLECTION,
    QDRANT_URL,
)
from src.retriever import _ollama_embed_query
from src.vector_store import ingest_embeddings

REPO_ROOT = Path(__file__).resolve().parents[1]
CHUNKS_DIR = REPO_ROOT / "data/output/chunks"
EMBEDDINGS_DIR = REPO_ROOT / "data/output/embeddings"
BACKUP_DIR = REPO_ROOT / "data/output/backup_20260722"

AFFECTED_STEMS = [
    "doc_001_procurement_policy",
    "doc_015_food_sop_manual",
    "doc_016a_original_lease",
    "doc_016c_second_amendment",
]

_ENRICH_WINDOW = 2
_MAX_RETRIES = 3
_RETRY_SLEEP_S = 15


def _is_polluted(chunk: dict) -> bool:
    """True when a chunk's context (top-level or metadata) is the leaked error string."""
    context = chunk.get("context") or chunk.get("metadata", {}).get("context", "")
    return str(context).lower().startswith("error")


def _doc_context_for(chunks: list[dict], position: int) -> str:
    """Build the same '{doc_summary}\\n\\n--- Nearby text ---\\n{neighbours}' context
    chunk_markdown uses in its non-whole-doc enrichment branch.

    Uses the document_summary chunk's stored content as the summary text (a
    close approximation of the raw doc_summary string chunk_markdown had at
    ingest time, which isn't preserved standalone anywhere on disk) plus the
    ±_ENRICH_WINDOW neighbouring chunks by list position (chunk_index equals
    list position in every chunks JSON produced by chunk_markdown)."""
    doc_summary_chunk = next(
        (c for c in chunks if c.get("metadata", {}).get("chunk_type") == "document_summary"),
        None,
    )
    doc_summary = doc_summary_chunk["content"] if doc_summary_chunk else ""
    lo = max(0, position - _ENRICH_WINDOW)
    hi = min(len(chunks), position + _ENRICH_WINDOW + 1)
    neighbours = "\n\n".join(c["content"] for c in chunks[lo:hi])
    return f"{doc_summary}\n\n--- Nearby text ---\n{neighbours}"


def _reenrich_with_retry(client: OpenAI, model_name: str, doc_context: str, content: str) -> str:
    """Call contextualize_chunk, retrying a rate-limit failure up to _MAX_RETRIES
    times before giving up and returning "" (never an error string)."""
    for attempt in range(_MAX_RETRIES):
        result = contextualize_chunk(client, model_name, doc_context, content)
        if not result.lower().startswith("error"):
            return result
        if attempt < _MAX_RETRIES - 1:
            time.sleep(_RETRY_SLEEP_S)
    return ""


def repair_file(stem: str, *, dry_run: bool) -> None:
    """Repair one document's polluted chunks + embeddings + Qdrant points."""
    chunks_path = CHUNKS_DIR / f"{stem}_chunks.json"
    embeddings_path = EMBEDDINGS_DIR / f"{stem}_chunks_embeddings.json"
    if not chunks_path.exists():
        print(f"[REPAIR] {stem}: chunks file not found, skipping")
        return

    chunks: list[dict] = json.loads(chunks_path.read_text(encoding="utf-8"))
    polluted_positions = [i for i, c in enumerate(chunks) if _is_polluted(c)]
    if not polluted_positions:
        print(f"[REPAIR] {stem}: 0 polluted chunks, nothing to do")
        return
    print(f"[REPAIR] {stem}: {len(polluted_positions)} polluted chunk(s) found")

    if dry_run:
        print(f"[REPAIR] {stem}: dry-run, not writing anything")
        return

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(chunks_path, BACKUP_DIR / chunks_path.name)
    if embeddings_path.exists():
        shutil.copy2(embeddings_path, BACKUP_DIR / embeddings_path.name)

    client = OpenAI(base_url=CHUNK_LLM_API_BASE, api_key=CHUNK_LLM_API_KEY or "no-key")
    embeddings: list[dict] = (
        json.loads(embeddings_path.read_text(encoding="utf-8"))
        if embeddings_path.exists()
        else []
    )
    embeddings_by_idx = {
        row.get("metadata", {}).get("chunk_index"): row for row in embeddings
    }

    repaired = 0
    still_empty = 0
    for position in polluted_positions:
        chunk = chunks[position]
        content = chunk["content"]
        doc_context = _doc_context_for(chunks, position)
        context = _reenrich_with_retry(client, CHUNK_LLM_MODEL, doc_context, content)

        if context:
            repaired += 1
            vector_text = f"CONTEXT: {context}\n\nCONTENT:\n{content}"
        else:
            still_empty += 1
            vector_text = None
            print(f"[REPAIR][WARN] {stem} chunk {position}: still failing after retries, leaving empty")

        chunk["context"] = context
        chunk["metadata"]["context"] = context
        chunk["vector_text"] = vector_text

        emb_row = embeddings_by_idx.get(position)
        if emb_row is None:
            continue
        emb_row["context"] = context
        emb_row["metadata"]["context"] = context
        emb_row["vector_text"] = vector_text
        text_for_embedding = vector_text or content
        emb_row["embedding"] = _ollama_embed_query(
            api_base=OLLAMA_API_BASE, model_name=OLLAMA_EMBED_MODEL, query=text_for_embedding
        )

    chunks_path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    if embeddings:
        embeddings_path.write_text(
            json.dumps(embeddings, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # Upsert only the repaired points into Qdrant — _stable_id keyed on
    # (file_name, chunk_index) overwrites the existing polluted point in
    # place, same patch-in-place precedent as the doc_008 figure-label fix.
    repaired_rows = [
        embeddings_by_idx[p] for p in polluted_positions if p in embeddings_by_idx
    ]
    if repaired_rows:
        tmp_path = BACKUP_DIR / f"{stem}_repaired_subset.json"
        tmp_path.write_text(json.dumps(repaired_rows, ensure_ascii=False), encoding="utf-8")
        ingest_embeddings(tmp_path, url=QDRANT_URL, collection=QDRANT_COLLECTION)

    print(f"[REPAIR] {stem}: repaired={repaired} still_empty={still_empty}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true", help="Only report polluted-chunk counts, write nothing."
    )
    args = parser.parse_args()
    for stem in AFFECTED_STEMS:
        repair_file(stem, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
