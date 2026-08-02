"""Gold-evidence rank@1 harness (M-2 measurement, MITIGATION_PLAN.md).

For every gold question whose document is live in the collection, locates the
chunk containing the gold evidence quote (via `retrieve()`'s raw pool, same
lenient matcher as `repro3_goldrank.py`) and reports the rank of that
chunk_index in the *tool's actual returned order* — i.e. the block order of
`search_knowledge_base`'s formatted output, not the reranker in isolation.

Matching is done on chunk_index against each block's `chunk=K` header, not on
quote text against block content: `_format_hits` injects neighbour chunks
([prev chunk]/[next chunk]) and truncates long prose via `_best_snippet`
(FM-7), either of which would make substring-matching the block text produce
a rank attributable to the wrong chunk. The header always carries the real
chunk_index (retrieval_tool.py:818) regardless of neighbour/snippet mutation.

Also prints each query's top reranker score (raw BGE logit, same pool) next
to whether the gold chunk landed at rank 1 — this is the tuning data for the
M-2 confidence-floor guard.

Run before and after a change to `_fetch_docs`'s doc-scoped head-insert
(`src/tools/retrieval_tool.py` ~1246) to see the effect on the order the
agent actually sees.

ponytail: HyDE is disabled (use_hyde=False) so the run is deterministic and
needs no LLM credentials/spend — it does not touch the mechanism under test
(the raw_hits[:3]-vs-reranked_hits merge), only adds an extra candidate
source. Re-enable if HyDE itself is ever the thing being measured.

Usage:
    uv run python eval/gold_rank_at_1.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vault_rag.config import (  # noqa: E402
    GENERATION_API_BASE,
    GENERATION_MODEL,
    QDRANT_COLLECTION,
    QDRANT_URL,
    RERANK_TOP_N,
    RERANKER_DEVICE,
    RERANKER_MODEL,
    RETRIEVAL_TOP_K,
)
from vault_rag.reranker import BGEReranker  # noqa: E402
from vault_rag.retriever import retrieve  # noqa: E402
from vault_rag.tools.retrieval_tool import _make_unified_tool  # noqa: E402

LIVE_DOCS = {"doc_001", "doc_002"}
QA_FILES = [
    "eval/data/qa_pairs/doc_001_procurement_policy_qa.json",
    "eval/data/qa_pairs/doc_002_services_contract_terms_qa.json",
    "eval/data/qa_pairs/doc_001_doc_002_cross_document_qa.json",
]
_BLOCK_RE = re.compile(r"\n\n(?=\[\d+\] file=)")
_CHUNK_HEADER_RE = re.compile(r"^\[\d+\] file=.*? chunk=(\d+) .*?score=(-?[\d.]+)")


def _norm(s: str) -> str:
    """Lowercase and strip to alphanumerics/spaces for lenient quote matching."""
    return re.sub(r"[^a-z0-9 ]", " ", s.lower())


def _contains(text: str, quote: str) -> bool:
    """True if `quote` appears in `text` verbatim (normalised) or via an
    8-consecutive-word window — tolerates OCR/markdown drift."""
    c = " ".join(_norm(text).split())
    q = " ".join(_norm(quote).split())
    if len(q) < 12:
        return False
    if q in c:
        return True
    words = q.split()
    return any(" ".join(words[i : i + 8]) in c for i in range(max(1, len(words) - 7)))


def _gold_chunk_index(question: str, doc: str, quote: str) -> int | None:
    """Locate the chunk_index of the raw hit containing `quote`, over the same
    doc-scoped retrieve() pool repro3_goldrank.py uses."""
    hits = retrieve(
        query=question,
        top_k=RETRIEVAL_TOP_K,
        qdrant_url=QDRANT_URL,
        collection=QDRANT_COLLECTION,
        use_qdrant=True,
        filter_token=None,
        force_exclude_chunk_types=["sheet_summary", "document_summary"],
        scope_doc_id=doc,
    )
    for h in hits:
        if _contains(h.get("content") or "", quote):
            return (h.get("metadata") or {}).get("chunk_index")
    return None


def _tool_rank_and_top_score(
    content: str | None, gold_chunk_index: int | None
) -> tuple[int | None, float | None]:
    """Return (1-indexed block rank of gold_chunk_index, block-1's score) by
    reading each block's `chunk=K score=S` header, in the tool's own order."""
    if not content:
        return None, None
    blocks = _BLOCK_RE.split(content)
    top_score = None
    rank = None
    for i, block in enumerate(blocks, start=1):
        m = _CHUNK_HEADER_RE.match(block)
        if not m:
            continue
        chunk_idx, score = int(m.group(1)), float(m.group(2))
        if i == 1:
            top_score = score
        if gold_chunk_index is not None and chunk_idx == gold_chunk_index and rank is None:
            rank = i
    return rank, top_score


def main() -> None:
    """Run the rank@1 harness over all live gold evidence quotes and print the table."""
    ranker = BGEReranker(model_name=RERANKER_MODEL, device=RERANKER_DEVICE)
    tool, _limits = _make_unified_tool(
        qdrant_url=QDRANT_URL,
        collection=QDRANT_COLLECTION,
        retrieval_top_k=RETRIEVAL_TOP_K,
        rerank_top_n=RERANK_TOP_N,
        ranker=ranker,
        generation_api_base=GENERATION_API_BASE,
        generation_model=GENERATION_MODEL,
        use_hyde=False,
    )

    rows = []
    for f in QA_FILES:
        for qa in json.load(open(REPO_ROOT / f)):
            for ev in qa.get("gold_evidence") or []:
                doc = (ev.get("doc_id") or "")[:7]
                if doc not in LIVE_DOCS:
                    continue
                quote = ev.get("quote") or ""
                gold_idx = _gold_chunk_index(qa["question"], doc, quote)
                content, _artifact = tool.func(query=qa["question"], doc_id=doc)
                rank, top_score = _tool_rank_and_top_score(content, gold_idx)
                label = f"{f.split('/')[-1][:28]}:{qa['qa_id']}"
                rows.append((label, doc, gold_idx, rank, top_score))
                print(
                    f"{label:42s} {doc} gold_chunk={gold_idx} tool_rank={rank} "
                    f"top_score={top_score}"
                )

    located = [r for r in rows if r[2] is not None]
    at_1 = sum(1 for r in located if r[3] == 1)
    print(f"\n=== {len(located)}/{len(rows)} gold quotes located in the doc-scoped pool")
    print(f"gold chunk at rank 1 in tool's returned order : {at_1}/{len(located)}")


if __name__ == "__main__":
    main()
