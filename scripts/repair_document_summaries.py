"""One-time repair: 3 of 6 live `document_summary` chunks in Qdrant were
embedded with a literal LLM error string ("Summary unavailable: Error code:
400 - ...") as their content and vector_text — src/chunker.py's
generate_document_summary used to return f"Summary unavailable: {e}" on any
failure, the same anti-pattern already fixed for contextualize_chunk (see
scripts/repair_polluted_contexts.py). Root cause: CHUNK_LLM_MODEL in .env
points at a Groq model decommissioned by the provider, so every call raised.

This script regenerates the 3 polluted summaries with a model that actually
works (gpt-oss-120b via the local free-llm-api proxy, CHUNK_LLM_API_BASE —
CHUNK_LLM_MODEL itself is left untouched since .env is not this script's to
edit), re-embeds via Ollama bge-m3, and upserts in place by the same stable
point id scheme _stable_id(file_name, chunk_index) already live in Qdrant —
never deletes a point.

Usage: uv run python -m scripts.repair_document_summaries [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from openai import OpenAI

from src import query_cache
from src.chunker import generate_document_summary
from src.config import (
    CHUNK_LLM_API_BASE,
    CHUNK_LLM_API_KEY,
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
PROCESSED_DIR = REPO_ROOT / "data/output/processed"
BACKUP_DIR = REPO_ROOT / "data/output/backup_20260730"

# CHUNK_LLM_MODEL in .env is the dead "llama-3.1-8b-instant" (decommissioned
# by Groq) -- .env is the user's to edit, not this script's, so the working
# model is passed explicitly here instead.
REPAIR_MODEL = "gpt-oss-120b"

AFFECTED_STEMS = [
    "doc_001_procurement_policy",
    "doc_002_services_contract_terms",
    "doc_016a_original_lease",
]


def _is_polluted(chunk: dict) -> bool:
    """True when a document_summary chunk's content is the leaked error string."""
    if chunk.get("metadata", {}).get("chunk_type") != "document_summary":
        return False
    return "unavailable" in str(chunk.get("content", "")).lower()


def repair_file(stem: str, client: OpenAI, *, dry_run: bool) -> None:
    """Regenerate the polluted document_summary chunk for one document and
    upsert the corrected point into Qdrant, if it exists and is polluted."""
    chunks_path = CHUNKS_DIR / f"{stem}_chunks.json"
    embeddings_path = EMBEDDINGS_DIR / f"{stem}_chunks_embeddings.json"
    markdown_path = PROCESSED_DIR / f"{stem}.md"
    if not chunks_path.exists():
        print(f"[REPAIR] {stem}: chunks file not found, skipping")
        return

    chunks: list[dict] = json.loads(chunks_path.read_text(encoding="utf-8"))
    position = next((i for i, c in enumerate(chunks) if _is_polluted(c)), None)
    if position is None:
        print(f"[REPAIR] {stem}: 0 polluted document_summary chunks, nothing to do")
        return
    print(f"[REPAIR] {stem}: polluted document_summary chunk found at index {position}")

    if dry_run:
        print(f"[REPAIR] {stem}: dry-run, not writing anything")
        return

    if not markdown_path.exists():
        print(f"[REPAIR][WARN] {stem}: no source markdown at {markdown_path}, skipping")
        return
    markdown = markdown_path.read_text(encoding="utf-8")

    summary = generate_document_summary(client, REPAIR_MODEL, markdown)
    if not summary:
        print(f"[REPAIR][WARN] {stem}: regeneration still failed, leaving as-is")
        return

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(chunks_path, BACKUP_DIR / chunks_path.name)
    if embeddings_path.exists():
        shutil.copy2(embeddings_path, BACKUP_DIR / embeddings_path.name)

    chunk = chunks[position]
    old_content = chunk["content"]
    # Preserve the "## Document Summary\n\nDocument ID: ...\nFile: ...\n[Title: ...]\n\n"
    # header exactly as it was; only the trailing summary text was polluted.
    header_end = old_content.rfind("Summary unavailable:")
    header = old_content[:header_end]
    new_content = f"{header}{summary}"
    new_vector_text = f"DOCUMENT SUMMARY:\n{header}{summary}"
    chunk["content"] = new_content
    chunk["vector_text"] = new_vector_text
    chunks_path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")

    embeddings: list[dict] = (
        json.loads(embeddings_path.read_text(encoding="utf-8"))
        if embeddings_path.exists()
        else []
    )
    emb_row = next(
        (r for r in embeddings if r.get("metadata", {}).get("chunk_index") == chunk["metadata"]["chunk_index"]
         and r.get("metadata", {}).get("chunk_type") == "document_summary"),
        None,
    )
    if emb_row is None:
        print(f"[REPAIR][WARN] {stem}: no matching embeddings row, skipping Qdrant upsert")
        return

    emb_row["content"] = new_content
    emb_row["vector_text"] = new_vector_text
    emb_row["embedding"] = _ollama_embed_query(
        api_base=OLLAMA_API_BASE, model_name=OLLAMA_EMBED_MODEL, query=new_vector_text
    )
    embeddings_path.write_text(
        json.dumps(embeddings, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Upsert only the repaired point -- _stable_id keyed on (file_name,
    # chunk_index) overwrites the existing polluted point in place, same
    # patch-in-place precedent as scripts/repair_polluted_contexts.py.
    tmp_path = BACKUP_DIR / f"{stem}_summary_repaired_subset.json"
    tmp_path.write_text(json.dumps([emb_row], ensure_ascii=False), encoding="utf-8")
    ingest_embeddings(tmp_path, url=QDRANT_URL, collection=QDRANT_COLLECTION)

    print(f"[REPAIR] {stem}: summary regenerated and upserted")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true", help="Only report polluted-summary counts, write nothing."
    )
    args = parser.parse_args()
    client = OpenAI(base_url=CHUNK_LLM_API_BASE, api_key=CHUNK_LLM_API_KEY or "no-key")
    for stem in AFFECTED_STEMS:
        repair_file(stem, client, dry_run=args.dry_run)
    if not args.dry_run:
        # Corpus changed -- see repair_polluted_contexts.py's identical note on
        # why the query cache must be cleared after a direct Qdrant upsert.
        query_cache.clear()
        print("[REPAIR] query cache cleared (corpus changed)")


if __name__ == "__main__":
    main()
