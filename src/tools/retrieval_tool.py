"""Document-search tool (search_knowledge_base) for the RAG agent.

Extracted from rag_agent.py: the unified retrieval tool plus its exclusive
helpers (HyDE, snippet selection, table formatting, neighbour expansion).

_make_unified_tool builds the LangChain StructuredTool the agent invokes. Per
call it: resolves scope (which doc(s) to search, filter token, stage-1 boosts),
runs first-stage retrieval, reranks, expands neighbour chunks and formats the
numbered result block the LLM consumes.

Called by: src/rag_agent.py (build_agent calls _make_unified_tool to register
search_knowledge_base; ask_agent mutates the returned _limits dict on retry).
Calls into: src/retriever.py (retrieve, infer_query_chunk_types,
_extract_table_filter_token), src/reranker.py (QwenReranker), src/file_resolver.py
(resolve_original_name), src/llm_utils.py (LLM helpers for HyDE) and Qdrant's
scroll API directly for neighbour-chunk fetching.
"""

from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

from langchain_core.tools import StructuredTool
from langsmith import traceable
from pydantic import BaseModel

from src.config import DOC_MIN_SCORE, MAX_CHUNK_CHARS, MAX_TABLE_CHARS, MAX_TOOL_RESULTS
from src.file_resolver import resolve_original_name
from src.llm_utils import _is_thinking_model, _llm_call, _to_openai_base
from src.reranker import QwenReranker
from src.retriever import (
    _extract_table_filter_token,
    infer_query_chunk_types,
    retrieve,
)

_RETRIEVAL_DEBUG = bool(os.getenv("RETRIEVAL_DEBUG"))

# Set by answer_pipeline.answer_one when the UI's source-scope control names a
# specific document — overrides whatever doc_id the LLM passes to
# search_knowledge_base. A prompt-only routing directive was verified
# unreliable: the model sometimes queries a different document entirely
# despite being told which one to use (reproduced directly, not hypothetical).
# ContextVar rather than a module-global: each API request runs its own
# executor thread (see api.py's run_in_executor), and a plain global would
# leak one request's forced scope into a concurrent request on another thread.
FORCED_DOC_ID: ContextVar[str | list[str] | None] = ContextVar(
    "FORCED_DOC_ID", default=None
)

# Same ContextVar-per-request rationale as FORCED_DOC_ID above. route_question
# (rag_agent.py) already detects, with a confidence gate, when a question's
# top retrieved hits are majority-Excel/CSV and prepends a routing directive
# telling the agent to use query_excel instead -- but that's a prompt nudge
# only, verified unreliable on its own: the agent sometimes calls
# search_knowledge_base anyway, gets text-chunk evidence for a question that
# needed an aggregate/SQL answer, and returns Unsupported. Unlike FORCED_DOC_ID
# (which redirects search_knowledge_base to a different doc_id), there's no
# equivalent redirect into query_excel -- it's a different tool entirely -- so
# this makes search_knowledge_base refuse outright and name the right tool,
# forcing the agent's next step to actually call query_excel.
FORCED_MODALITY: ContextVar[str | None] = ContextVar("FORCED_MODALITY", default=None)

# ---------------------------------------------------------------------------
# Scope plan — the routing decision passed from _resolve_scope to _fetch_docs
# ---------------------------------------------------------------------------


@dataclass
class _ScopePlan:
    """Routing decision from _resolve_scope: where to search and how."""

    search_query: str
    retrieval_query: str
    chunk_types: Any
    effective_scope: str | list[str] | None
    parallel_doc_ids: list[str]
    explicitly_mentioned_doc_ids: set[str]
    filter_token: str | None
    stage1_doc_ids: set[str]
    stem_match_doc_ids: set[str]


# ---------------------------------------------------------------------------
# HyDE — generate a hypothetical answer to embed instead of the raw query
# ---------------------------------------------------------------------------


@traceable(name="hyde-expansion")
def _hyde(query: str, api_base: str, model_name: str) -> str:
    """Generate a hypothetical answer to embed instead of the raw query (HyDE)."""
    # Suppress chain-of-thought for thinking models so only the passage returns.
    no_think = "/no_think " if _is_thinking_model(model_name) else ""
    # Ask the LLM for a short passage that would answer the question.
    return _llm_call(
        f"{no_think}Write a short passage (2-3 sentences) that would directly answer "
        f"this question. Use the same language and terminology as the likely source document."
        f"\n\nQuestion: {query}",
        api_base,
        model_name,
    )


# ---------------------------------------------------------------------------
# Snippet selection — keep the most query-relevant window of a long chunk
# ---------------------------------------------------------------------------


def _query_terms(query: str) -> list[str]:
    """Extract useful lexical anchors for snippet selection."""
    # Strip stop-words and keep distinct content tokens of length >= 3.
    stop_words = {
        "about",
        "according",
        "after",
        "also",
        "and",
        "are",
        "before",
        "between",
        "does",
        "document",
        "during",
        "from",
        "have",
        "into",
        "issued",
        "must",
        "that",
        "the",
        "their",
        "there",
        "this",
        "what",
        "when",
        "where",
        "which",
        "while",
        "with",
        "within",
        "would",
        "your",
        "approved",
        "policy",
    }
    terms = []
    for term in re.findall(r"[A-Za-z0-9][A-Za-z0-9'/-]{2,}", query.lower()):
        if term not in stop_words and term not in terms:
            terms.append(term)
    return terms


def _best_snippet(content: str, query: str, max_chars: int) -> str:
    """Keep the most query-relevant window from a long text chunk."""
    # Short chunks fit whole — return unchanged.
    if len(content) <= max_chars:
        return content
    # Find every position where a query term occurs in the chunk.
    terms = _query_terms(query)
    lower = content.lower()
    positions: list[int] = []
    for term in terms:
        positions.extend(match.start() for match in re.finditer(re.escape(term), lower))
    # No term matched anywhere — just take the head of the chunk.
    if not positions:
        return content[:max_chars] + "…"

    # Score candidate windows instead of anchoring on the first matching token.
    # Long chunks often mention generic query terms early ("contract", "document")
    # while the answer is later near a denser cluster ("contract term", "extension",
    # "up to two years"). Choosing the highest-density window keeps the answer.
    weighted_terms = {term: max(1, min(len(term), 12)) for term in terms}
    best_start = 0
    best_position = 0
    best_score = -1
    best_distance = len(content)
    # Score a max_chars window centred on each term occurrence.
    for position in positions:
        start = max(0, position - max_chars // 2)
        end = min(len(content), start + max_chars)
        start = max(0, end - max_chars)
        window = lower[start:end]
        score = 0
        # Reward windows that contain (and repeat) more weighted query terms.
        for term, weight in weighted_terms.items():
            occurrences = len(re.findall(re.escape(term), window))
            if occurrences:
                score += weight + min(occurrences, 3)
        # Prefer windows with answer-like numeric phrases when the query asks for
        # amounts, dates, periods, counts, or durations.
        if re.search(
            r"\b(how many|amount|date|period|longer|total|cost|rate|range)\b",
            query,
            re.IGNORECASE,
        ):
            if re.search(
                r"\b(up to|maximum|total|invoice date|please pay|contract term|closed\s*[-–]\s*implemented)\b",
                window,
            ):
                score += 14
            if re.search(
                r"\b(months?|years?|percent|transactions?|districts?|areas?|recommendations?)\b",
                window,
            ):
                score += 6
            if re.search(
                r"\b(up to|maximum|total|date|amount|range|percent|months?|years?)\b.{0,80}\d",
                window,
            ):
                score += 8
            if re.search(
                r"\d.{0,40}\b(months?|years?|percent|transactions?|districts?|areas?|recommendations?)\b",
                window,
            ):
                score += 8
        # Keep the highest-scoring window; tie-break toward better-centred ones.
        distance = abs(position - (start + max_chars // 2))
        if score > best_score or (score == best_score and distance < best_distance):
            best_start = start
            best_position = position
            best_score = score
            best_distance = distance

    # Re-derive the final window bounds from the winning start offset.
    start = best_start
    end = min(len(content), start + max_chars)
    start = max(0, end - max_chars)

    # Prefer clean line boundaries when possible.
    if start:
        line_start = content.find("\n", start)
        if 0 <= line_start < min(end, start + 300) and line_start < best_position:
            start = line_start + 1
    # Add ellipsis markers when the window is not at the start/end of the chunk.
    prefix = "…\n" if start else ""
    suffix = "\n…" if end < len(content) else ""
    return prefix + content[start:end].strip() + suffix


# ---------------------------------------------------------------------------
# Table formatting — render retrieved table text as explicit key/value rows
# ---------------------------------------------------------------------------


def _split_markdown_row(line: str) -> list[str]:
    """Split a simple markdown table row into cells."""
    stripped = line.strip()
    if not stripped.startswith("|") or not stripped.endswith("|"):
        return []
    return [cell.strip() for cell in stripped.strip("|").split("|")]


def _normalize_table_match_text(text: str) -> str:
    """Lower-case text and collapse non-alphanumeric runs to single spaces."""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _table_match_terms(query: str) -> list[str]:
    """Extract distinct content terms from a query for table-row matching."""
    terms: list[str] = []
    for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9&./_-]{1,}", query):
        normalized = _normalize_table_match_text(token)
        if len(normalized) < 3:
            continue
        if normalized in {
            "the",
            "and",
            "for",
            "row",
            "what",
            "which",
            "where",
            "when",
            "with",
            "dated",
            "date",
            "amount",
            "total",
            "net",
            "supplier",
            "department",
            "directorate",
            "transaction",
            "transactions",
            "council",
            "doncaster",
            "published",
            "spend",
            "report",
            "purchase",
            "card",
            "q1",
            "2025",
            "2026",
        }:
            continue
        if normalized not in terms:
            terms.append(normalized)
    return terms


def _format_key_value_rows(
    headers: list[str], rows: list[list[str]], query: str, max_rows: int = 4
) -> str:
    """Render table rows as explicit field/value lines for reliable cell extraction."""
    # Score each row by how many query terms its cells contain.
    terms = _table_match_terms(query)
    scored: list[tuple[int, int, list[str]]] = []
    for idx, row in enumerate(rows):
        row_text = _normalize_table_match_text(" ".join(row))
        score = sum(1 for term in terms if term in row_text)
        scored.append((score, -idx, row))

    # Keep the best-matching rows; fall back to the first rows if none matched.
    scored.sort(reverse=True)
    selected = [row for score, _, row in scored if score > 0][:max_rows]
    if not selected:
        selected = rows[:max_rows]

    # Emit each selected row as "Row N:" followed by header: value lines.
    blocks = ["Relevant table rows:"]
    for row_idx, row in enumerate(selected, start=1):
        blocks.append(f"Row {row_idx}:")
        for header, value in zip(headers, row):
            header = header.strip()
            value = value.strip()
            if header and value:
                blocks.append(f"{header}: {value}")
    return "\n".join(blocks)


def _table_context_as_key_values(content: str, query: str) -> str | None:
    """Convert retrieved markdown or pipe-delimited table text to key/value rows."""
    lines = [line for line in content.splitlines() if line.strip()]

    # Case 1: a real markdown table — parse rows, drop the separator line and
    # any short/ragged rows, then render header + matching data rows.
    table_lines = [line for line in lines if line.strip().startswith("|")]
    if table_lines:
        parsed = [_split_markdown_row(line) for line in table_lines]
        parsed = [row for row in parsed if row]
        if len(parsed) >= 2:
            headers = parsed[0]
            rows = [
                row
                for row in parsed[1:]
                if not all(re.fullmatch(r":?-{3,}:?", cell.strip()) for cell in row)
            ]
            rows = [row for row in rows if len(row) >= len(headers)]
            if rows:
                return _format_key_value_rows(headers, rows, query)

    # Case 2: a single "label: value | label: value" line — split it into rows.
    pipe_pair_lines = [
        line
        for line in lines
        if ": " in line and " | " in line and not line.lstrip().startswith("[")
    ]
    if pipe_pair_lines:
        converted: list[str] = ["Relevant table rows:", "Row 1:"]
        for part in pipe_pair_lines[0].split(" | "):
            if ": " in part:
                label, value = part.split(": ", 1)
                if label.strip() and value.strip():
                    converted.append(f"{label.strip()}: {value.strip()}")
        if len(converted) > 2:
            return "\n".join(converted)

    return None


# ---------------------------------------------------------------------------
# Hit merging & neighbour-chunk expansion
# ---------------------------------------------------------------------------


def _merge_hits(
    primary: list[dict[str, Any]], secondary: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Merge retrieval hits while preserving order and removing duplicate chunks."""
    merged: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    # Walk primary then secondary; a composite identity key dedupes chunks.
    for hit in primary + secondary:
        meta = hit.get("metadata", {}) or {}
        key = (
            meta.get("doc_id"),
            meta.get("source_file") or meta.get("file_name"),
            meta.get("chunk_index"),
            meta.get("part"),
            meta.get("sheet_name"),
            meta.get("row_ref"),
            (hit.get("content") or "")[:120],
        )
        if key in seen:
            continue
        seen.add(key)
        merged.append(hit)
    return merged


def _fetch_neighbor_chunks(
    qdrant_url: str,
    collection: str,
    source_file: str,
    indices: list[int],
) -> dict[int, str]:
    """Return {chunk_index: content} for the requested neighbor chunks of a single file.

    Used to expand a retrieved chunk with surrounding context — addresses chunker
    boundary misses where the answer ends up split across consecutive chunks
    (e.g. section header in chunk N, table of values in chunk N+1).
    """
    # Nothing to fetch without both a file and at least one index.
    if not indices or not source_file:
        return {}
    # Scroll-filter Qdrant for the requested chunk indices within one file.
    from urllib.request import Request, urlopen

    base = qdrant_url.rstrip("/")
    url = f"{base}/collections/{collection}/points/scroll"
    body = json.dumps(
        {
            "limit": len(indices) + 2,
            "with_payload": True,
            "with_vector": False,
            "filter": {
                "must": [
                    {"key": "metadata.source_file", "match": {"value": source_file}},
                    {"key": "metadata.chunk_index", "match": {"any": indices}},
                ]
            },
        }
    ).encode("utf-8")
    # Any failure is non-fatal — neighbour expansion is best-effort.
    req = Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return {}
    # Map each returned point's chunk_index to its content text.
    out: dict[int, str] = {}
    for point in data.get("result", {}).get("points", []) or []:
        payload = point.get("payload") or {}
        meta = payload.get("metadata") or {}
        idx = meta.get("chunk_index")
        if isinstance(idx, int):
            out[idx] = (payload.get("content") or "").strip()
    return out


# ---------------------------------------------------------------------------
# Tool factory — assembles the search_knowledge_base StructuredTool
# ---------------------------------------------------------------------------


def _make_unified_tool(
    qdrant_url: str,
    collection: str,
    retrieval_top_k: int,
    rerank_top_n: int,
    ranker: QwenReranker | None,
    generation_api_base: str,
    generation_model: str,
    use_hyde: bool = True,
    doc_registry: dict[str, str] | None = None,
) -> tuple[StructuredTool, dict]:
    """Build the search_knowledge_base tool plus its mutable _limits dict."""

    class SearchInput(BaseModel):
        query: str
        doc_id: str = ""  # Optional: restrict to a specific doc (e.g. "doc_006"). Leave empty for all docs.

    # Tokens that look like document references (appear in source_file metadata, not chunk content).
    # These must be routed to scope_doc_id (metadata filter), not used as content text filters.
    # Pattern: short alphanumeric prefix + underscore + digits (e.g. "doc_001", "inv_2024", "rpt_003").
    # Extend this pattern if your corpus uses a different naming convention.
    _DOC_ID_RE = re.compile(r"\b[a-z]{2,6}_\d+\b", re.IGNORECASE)
    # Detect content ID-like tokens: mixed alphanumeric codes (invoice numbers, transaction IDs, etc.)
    # that appear verbatim inside chunk text and are useful as Qdrant text-match filters.
    _ID_RE = re.compile(r"\b(?=[^\s]*[A-Za-z])(?=[^\s]*\d)[\w/-]{4,}\b")

    # Mutable limits dict — allows ask_agent to reduce limits on overflow retry
    _limits: dict = {
        "max_table_chars": MAX_TABLE_CHARS,
        "max_chunk_chars": MAX_CHUNK_CHARS,
        "rerank_top_n": min(rerank_top_n, MAX_TOOL_RESULTS),
    }

    # Reranked-but-not-selected candidates from the most recent _fetch_docs call —
    # UI-only, for the "retrieved but rejected" trust panel. Not read by the LLM.
    _last_rejected: list[dict] = []

    def _resolve_scope(query: str, doc_id: str | list[str]) -> _ScopePlan:
        """Resolve retrieval scope, parallel fan-out, filter token and the
        stage-1 doc-routing boosts from the query."""
        # A list of doc_ids is the UI's multi-select source scope, forced in via
        # FORCED_DOC_ID — already-validated ids, not LLM-inferred text, so skip
        # the single-doc regex/fuzzy-match/stage1-routing inference below entirely
        # and search across exactly those documents (same precedent as the
        # existing single-doc hard override).
        if isinstance(doc_id, list):
            doc_ids = [d for d in doc_id if d]
            return _ScopePlan(
                search_query=query,
                retrieval_query=query,
                chunk_types=None,
                effective_scope=doc_ids or None,
                parallel_doc_ids=[],
                explicitly_mentioned_doc_ids=set(),
                filter_token=None,
                stage1_doc_ids=set(),
                stem_match_doc_ids=set(),
            )
        # Accept the passed doc_id only if it matches the doc_XXX id pattern.
        doc_id = (doc_id or "").strip()
        valid_doc_scope = doc_id if _DOC_ID_RE.fullmatch(doc_id) else ""

        # If the LLM passed a document title (not a doc_XXX id), fuzzy-match against the registry.
        if not valid_doc_scope and doc_id and doc_registry:
            doc_id_lower = doc_id.lower()
            for stem, mapped_id in doc_registry.items():
                if stem in doc_id_lower or doc_id_lower in stem:
                    valid_doc_scope = mapped_id
                    break

        # If a non-id doc title was passed, fold it into the search text so the
        # embedding picks up the title; a valid id is handled via scope instead.
        search_query = query if not doc_id or valid_doc_scope else f"{query} {doc_id}"

        # Routing and filter token are derived from the original question, not HyDE,
        # because table-routing keywords ("row", "supplier", "transaction number") are
        # present in the question but typically absent in a hypothetical document.
        chunk_types, exclude_chunk_types = infer_query_chunk_types(search_query)

        # doc_XXX identifiers in the query → use as scope filter (metadata), not content filter.
        # Real content IDs (invoice/transaction numbers) → use as filter_token (content text match).
        inferred_doc_ids = [m.lower() for m in _DOC_ID_RE.findall(search_query)]
        if valid_doc_scope:
            effective_scope = valid_doc_scope
        elif len(inferred_doc_ids) == 1:
            effective_scope = inferred_doc_ids[0]
        else:
            effective_scope = (
                None  # cross-doc or no scope — parallel retrieval handles this below
            )
        # Whether to run parallel scoped retrieval instead of a single unscoped search.
        # Activates when exactly two distinct doc_ids appear in the query and no explicit
        # scope was passed from outside — equivalent to LangGraph Send fan-out per doc.
        _parallel_doc_ids: list[str] = (
            inferred_doc_ids
            if len(inferred_doc_ids) == 2 and not valid_doc_scope
            else []
        )
        # Explicitly mentioned doc_ids get maximum stage1 boost regardless of embedding rank.
        explicitly_mentioned_doc_ids: set[str] = set(inferred_doc_ids)

        non_doc_ids = [
            m for m in _ID_RE.findall(search_query) if not _DOC_ID_RE.match(m)
        ]
        filter_token: str | None = non_doc_ids[0] if non_doc_ids else None
        # _ID_RE requires a letter+digit mix, so pure-digit codes (invoice numbers, IDs)
        # are missed. Extract them explicitly when the query mentions "invoice" or "transaction".
        if filter_token is None:
            _pure_digit_id = re.search(r"\b(\d{5,})\b", search_query)
            if _pure_digit_id and any(
                w in search_query.lower() for w in ("invoice", "transaction", "record")
            ):
                filter_token = _pure_digit_id.group(1)
        if filter_token is None:
            filter_token = _extract_table_filter_token(search_query)

        # Routing/filter tokens are derived from search_query (the raw question + any
        # doc_id); the same text is used as the retrieval/embedding query.
        retrieval_query = search_query

        # Stage 1 (doc routing): search document_summaries (all doc types) to identify
        # likely source documents. Results used to BOOST chunk ranking — not as exclusive
        # scope, so wrong Stage 1 picks don't break retrieval for other question types.
        stage1_doc_ids: set[str] = set()
        stem_match_doc_ids: set[str] = set()
        if not effective_scope:
            doc_summary_hits = retrieve(
                query=retrieval_query,
                top_k=5,
                qdrant_url=qdrant_url,
                collection=collection,
                use_qdrant=True,
                filter_token=None,
                force_chunk_types=["document_summary"],
                force_exclude_chunk_types=None,
                scope_doc_id=None,
            )
            for ds_hit in doc_summary_hits:
                ds_doc_id = (ds_hit.get("metadata") or {}).get("doc_id", "")
                if _DOC_ID_RE.fullmatch(ds_doc_id):
                    stage1_doc_ids.add(ds_doc_id)
            # Explicitly mentioned doc_ids always get the boost regardless of Stage 1 rank.
            stage1_doc_ids |= explicitly_mentioned_doc_ids

            # Filename-token boost: when a doc's stem shares >=2 content tokens with the
            # query, add it to stage1 even if dense search ranked it low. Catches cases
            # like "procurement policy" → doc_001_procurement_policy.pdf where the title
            # literally matches but generic policy phrasing dominates the embedding.
            stem_match_scores: dict[str, int] = {}
            if doc_registry:
                _STEM_STOPWORDS = {
                    "the",
                    "and",
                    "for",
                    "with",
                    "from",
                }
                q_tokens = {
                    t
                    for t in re.findall(r"[a-z]{3,}", search_query.lower())
                    if t not in _STEM_STOPWORDS
                }
                seen_doc_ids: set[str] = set()
                for stem, mapped_id in doc_registry.items():
                    if mapped_id in seen_doc_ids or not _DOC_ID_RE.fullmatch(mapped_id):
                        continue
                    stem_tokens = {
                        t
                        for t in re.findall(r"[a-z]{3,}", stem)
                        if t not in _STEM_STOPWORDS and not t.startswith("doc")
                    }
                    overlap = len(q_tokens & stem_tokens)
                    stem_match_scores[mapped_id] = max(
                        stem_match_scores.get(mapped_id, 0), overlap
                    )
                    if overlap >= 2:
                        stage1_doc_ids.add(mapped_id)
                        stem_match_doc_ids.add(mapped_id)
                        seen_doc_ids.add(mapped_id)

            # Auto-scope: when a single doc dominates filename-token overlap (>=3 tokens,
            # gap of >=2 over the next-best doc), scope retrieval to that doc. Prevents
            # topically-similar but query-unrelated docs from competing for slots and
            # confusing the LLM. Stays inert for symmetric cross-doc queries (where two
            # docs tie or are close).
            if not effective_scope and not _parallel_doc_ids and stem_match_scores:
                sorted_pairs = sorted(
                    stem_match_scores.items(), key=lambda kv: kv[1], reverse=True
                )
                top_id, top_score = sorted_pairs[0]
                second_score = sorted_pairs[1][1] if len(sorted_pairs) > 1 else 0
                if top_score >= 3 and (top_score - second_score) >= 2:
                    effective_scope = top_id
        # Bundle every routing decision into the immutable plan _fetch_docs reads.
        return _ScopePlan(
            search_query=search_query,
            retrieval_query=retrieval_query,
            chunk_types=chunk_types,
            effective_scope=effective_scope,
            parallel_doc_ids=_parallel_doc_ids,
            explicitly_mentioned_doc_ids=explicitly_mentioned_doc_ids,
            filter_token=filter_token,
            stage1_doc_ids=stage1_doc_ids,
            stem_match_doc_ids=stem_match_doc_ids,
        )

    def _format_hits(
        top_hits: list[dict[str, Any]],
        filter_token: str | None,
        reranked_used: bool,
        retrieval_query: str,
        search_query: str,
        max_table_chars: int,
        max_chunk_chars: int,
    ) -> str | None:
        """Expand the top hits with neighbour chunks and render them as the
        numbered [N] file=… blocks the agent consumes. None if nothing survives."""
        # Neighbor-chunk expansion: chunker boundaries can split related content
        # (section header in chunk N, table values in N+1). For the top-5 PDF/text
        # chunks, fetch chunk_index ± 1 from the same file and append. Generic fix
        # for "right chunk, missing detail" misses across all docs.
        _NEIGHBOR_EXCLUDE_TYPES = {
            "sheet_summary",
            "document_summary",
            "sheet_table",
            "sheet_row",
        }
        neighbor_requests: dict[str, set[int]] = {}
        for h in top_hits:
            meta = h.get("metadata") or {}
            if meta.get("chunk_type", "") in _NEIGHBOR_EXCLUDE_TYPES:
                continue
            src = meta.get("source_file")
            idx = meta.get("chunk_index")
            if not src or not isinstance(idx, int):
                continue
            wanted = neighbor_requests.setdefault(src, set())
            if idx > 0:
                wanted.add(idx - 1)
            wanted.add(idx + 1)
        neighbors_by_file: dict[str, dict[int, str]] = {}
        for src, idxs in neighbor_requests.items():
            existing = {
                (h.get("metadata") or {}).get("chunk_index")
                for h in top_hits
                if (h.get("metadata") or {}).get("source_file") == src
            }
            needed = sorted(i for i in idxs if i not in existing)
            if needed:
                neighbors_by_file[src] = _fetch_neighbor_chunks(
                    qdrant_url, collection, src, needed
                )

        # Render each surviving hit into one numbered [N] block.
        parts: list[str] = []
        for i, h in enumerate(top_hits, start=1):
            meta = h.get("metadata", {}) or {}
            score = h.get("rerank_score", h.get("score", 0))
            # DOC_MIN_SCORE only applies to dense cosine scores (0-1 range).
            # Reranker returns raw logits (can be negative) — top_n already limits relevance.
            if not filter_token and not reranked_used and score < DOC_MIN_SCORE:
                continue
            # Resolve display name, location label and chunk-type flags.
            file_name = meta.get("file_name") or meta.get("source_file", "unknown")
            file_name = resolve_original_name(file_name)
            chunk_type = meta.get("chunk_type", "")
            sheet_name = meta.get("sheet_name")
            location = (
                f"sheet={sheet_name}"
                if sheet_name
                else f"chunk={meta.get('chunk_index', meta.get('part', '?'))}"
            )
            content = (h.get("content", "") or "").strip()
            is_pdf_table = "[TABLE_START]" in content
            is_sheet_table = chunk_type == "sheet_table"
            is_sheet_row = chunk_type == "sheet_row"
            # Trim content per type: matching table rows only, table truncation,
            # or query-focused snippet for long prose chunks.
            # When filter_token matched an ID, extract only header + matching rows
            if filter_token and is_sheet_table:
                lines = content.splitlines()
                header_lines = [
                    ln
                    for ln in lines
                    if not ln.startswith("|") or ln.startswith("| ---")
                ]
                table_lines = [ln for ln in lines if ln.startswith("|")]
                matching = [ln for ln in table_lines if filter_token in ln]
                if matching:
                    # keep description + table header rows (first 2 table lines) + matching rows
                    sep_idx = next(
                        (
                            i
                            for i, ln in enumerate(table_lines)
                            if ln.startswith("| ---")
                        ),
                        1,
                    )
                    content = "\n".join(
                        header_lines[
                            : header_lines.index(table_lines[0])
                            if table_lines[0] in header_lines
                            else len(header_lines)
                        ]
                        + table_lines[: sep_idx + 1]
                        + matching
                    )
            elif (is_pdf_table or is_sheet_table) and len(content) > max_table_chars:
                content = content[:max_table_chars] + "\n… (truncated)"
            elif (
                not is_pdf_table
                and not is_sheet_table
                and not is_sheet_row
                and len(content) > max_chunk_chars
            ):
                content = _best_snippet(content, retrieval_query, max_chunk_chars)
            # For sheet tables/rows, prepend an explicit key/value rendering so
            # the LLM can read cell values reliably.
            if is_sheet_table or is_sheet_row:
                table_context = _table_context_as_key_values(content, search_query)
                if table_context:
                    content = (
                        f"{table_context}\n\nOriginal retrieved table text:\n{content}"
                    )

            # Inject neighbor chunks (N-1, N+1) for PDF/text chunks in the top-5
            src_path = meta.get("source_file")
            chunk_idx = meta.get("chunk_index")
            if (
                src_path in neighbors_by_file
                and isinstance(chunk_idx, int)
                and chunk_type not in _NEIGHBOR_EXCLUDE_TYPES
            ):
                file_neighbors = neighbors_by_file[src_path]
                prev_text = file_neighbors.get(chunk_idx - 1, "")
                next_text = file_neighbors.get(chunk_idx + 1, "")
                budget = max_chunk_chars
                if prev_text:
                    if len(prev_text) > budget:
                        prev_text = "…" + prev_text[-budget:]
                    content = f"[prev chunk]\n{prev_text}\n\n[this chunk]\n{content}"
                if next_text:
                    if len(next_text) > budget:
                        next_text = next_text[:budget] + "…"
                    content = f"{content}\n\n[next chunk]\n{next_text}"

            page = meta.get("page")
            doc_id = meta.get("doc_id") or "none"
            title = quote(meta.get("title") or "none")
            page_field = f" page={page if page is not None else 'none'}"
            parts.append(
                f"[{i}] file={file_name} {location}{page_field} score={score:.4f}"
                f" doc_id={doc_id} title={title}\n{content}"
            )

        # Nothing survived the score/format filters.
        if not parts:
            return None
        # Append a usage instruction and join all blocks into one string.
        parts.append(
            "Instruction: Use the retrieved results above to answer the user's question. "
            "Do not repeat the same search unless a different missing fact is required."
        )
        return "\n\n".join(parts)

    @traceable(
        name="fetch-docs",
        metadata={"top_k": retrieval_top_k, "rerank_top_n": rerank_top_n},
    )
    def _fetch_docs(query: str, doc_id: str | list[str] = "") -> str | None:
        """Retrieve, rerank and format knowledge-base chunks for one query; return the formatted block, or None when nothing relevant is found."""
        _last_rejected.clear()
        if _RETRIEVAL_DEBUG:
            print(f"[RETRIEVAL_DEBUG] tool call: query={query!r} doc_id={doc_id!r}")
        # Read the current (mutable) size limits and the LLM endpoint.
        max_table_chars = _limits["max_table_chars"]
        max_chunk_chars = _limits["max_chunk_chars"]
        _rerank_top_n = _limits["rerank_top_n"]
        api_base = _to_openai_base(generation_api_base)

        # Resolve routing (scope, filter token, stage-1 boosts) and unpack it.
        scope = _resolve_scope(query, doc_id)
        search_query = scope.search_query
        retrieval_query = scope.retrieval_query
        chunk_types = scope.chunk_types
        effective_scope = scope.effective_scope
        _parallel_doc_ids = scope.parallel_doc_ids
        explicitly_mentioned_doc_ids = scope.explicitly_mentioned_doc_ids
        filter_token = scope.filter_token
        if _RETRIEVAL_DEBUG:
            print(
                f"[RETRIEVAL_DEBUG] resolved scope: retrieval_query={retrieval_query!r} "
                f"effective_scope={effective_scope!r} filter_token={filter_token!r} "
                f"chunk_types={chunk_types!r} parallel_doc_ids={_parallel_doc_ids!r}"
            )
        stage1_doc_ids = scope.stage1_doc_ids
        stem_match_doc_ids = scope.stem_match_doc_ids

        # Fix 1: for single-doc scoped queries the answer chunk may not rank in the global
        # top-8 after reranking. Give the reranker a wider pool — context is bounded since
        # chunks come from one document only. Cap at 12 to avoid flooding context with
        # similar chunks from dense docs (OCR invoices, long reports) that confuse the LLM.
        if effective_scope:
            _rerank_top_n = max(_rerank_top_n, 12)

        # sheet_summary chunks contain only column names — filter_token (a specific row
        # value like a supplier name or transaction ID) would eliminate all of them.
        # Fetch sheet_summaries separately with no filter_token, then merge with the
        # main retrieval which applies filter_token to PDF/text chunks.
        sheet_summary_hits = retrieve(
            query=retrieval_query,
            top_k=300,
            qdrant_url=qdrant_url,
            collection=collection,
            use_qdrant=True,
            filter_token=None,
            force_chunk_types=["sheet_summary"],
            force_exclude_chunk_types=None,
            scope_doc_id=effective_scope,
        )

        if _parallel_doc_ids:
            # Parallel scoped retrieval: fan out to each doc simultaneously then merge.
            # Equivalent to LangGraph Send fan-out but within a synchronous tool call.
            def _retrieve_scoped(doc_id: str) -> list[dict[str, Any]]:
                """Retrieve content chunks scoped to a single doc_id.

                filter_token is a hard content must-match — merges in an
                unfiltered pass so the reranker (not the filter) decides
                relevance when the answer chunk doesn't literally contain the
                token (see the non-parallel branch below for the reproduced
                case)."""
                filtered = retrieve(
                    query=retrieval_query,
                    top_k=retrieval_top_k,
                    qdrant_url=qdrant_url,
                    collection=collection,
                    use_qdrant=True,
                    filter_token=filter_token,
                    force_chunk_types=chunk_types,
                    force_exclude_chunk_types=["sheet_summary", "document_summary"],
                    scope_doc_id=doc_id,
                )
                if not filter_token:
                    return filtered
                try:
                    unfiltered = retrieve(
                        query=retrieval_query,
                        top_k=retrieval_top_k,
                        qdrant_url=qdrant_url,
                        collection=collection,
                        use_qdrant=True,
                        filter_token=None,
                        force_chunk_types=chunk_types,
                        force_exclude_chunk_types=["sheet_summary", "document_summary"],
                        scope_doc_id=doc_id,
                    )
                    return _merge_hits(filtered, unfiltered)
                except Exception:
                    return filtered

            with ThreadPoolExecutor(max_workers=len(_parallel_doc_ids)) as pool:
                futures = [pool.submit(_retrieve_scoped, d) for d in _parallel_doc_ids]
                raw_hits = _merge_hits(futures[0].result(), futures[1].result())
        else:
            raw_hits = retrieve(
                query=retrieval_query,
                top_k=retrieval_top_k,
                qdrant_url=qdrant_url,
                collection=collection,
                use_qdrant=True,
                filter_token=filter_token,
                force_chunk_types=chunk_types,
                force_exclude_chunk_types=["sheet_summary", "document_summary"],
                scope_doc_id=effective_scope,
            )
            # filter_token is a hard content must-match tuned for table/ID
            # lookups; on prose questions it can exclude the answer chunk
            # entirely (reproduced live: query "...LACERA procurement
            # policy?" -> filter_token="LACERA", but the answer chunk's own
            # text reads "L/CERA" from an OCR'd logo, so it never contains
            # the literal token and was silently dropped from every result;
            # without the filter the reranker placed it #3). Merge in an
            # unfiltered pass so the reranker arbitrates relevance instead
            # of the filter deciding recall. Filtered hits stay first in the
            # merge, preserving the exact-match boost for genuine ID lookups.
            if filter_token:
                try:
                    unfiltered_hits = retrieve(
                        query=retrieval_query,
                        top_k=retrieval_top_k,
                        qdrant_url=qdrant_url,
                        collection=collection,
                        use_qdrant=True,
                        filter_token=None,
                        force_chunk_types=chunk_types,
                        force_exclude_chunk_types=["sheet_summary", "document_summary"],
                        scope_doc_id=effective_scope,
                    )
                    raw_hits = _merge_hits(raw_hits, unfiltered_hits)
                except Exception:
                    pass

        # Older PDF chunks have no metadata.doc_id — if the doc_id filter returned
        # nothing, retry with metadata.source_file text match (contains the doc_id
        # stem in the filename path, e.g. "doc_008_gao_24_106915.pdf").
        if not raw_hits and effective_scope:
            raw_hits = retrieve(
                query=retrieval_query,
                top_k=retrieval_top_k,
                qdrant_url=qdrant_url,
                collection=collection,
                use_qdrant=True,
                filter_token=filter_token,
                force_chunk_types=chunk_types,
                force_exclude_chunk_types=["sheet_summary", "document_summary"],
                scope_doc_id=effective_scope,
            )

        # HyDE: embed a hypothetical answer as a second query and retrieve with
        # it too — catches answer-phrased chunks the literal question misses.
        hyde_hits: list[dict[str, Any]] = []
        if use_hyde:
            try:
                embed_query = _hyde(retrieval_query, api_base, generation_model)
                if (
                    embed_query
                    and embed_query.strip().lower() != retrieval_query.strip().lower()
                ):
                    hyde_hits = retrieve(
                        query=embed_query,
                        top_k=retrieval_top_k,
                        qdrant_url=qdrant_url,
                        collection=collection,
                        use_qdrant=True,
                        filter_token=filter_token,
                        force_chunk_types=chunk_types,
                        force_exclude_chunk_types=["sheet_summary", "document_summary"],
                        scope_doc_id=effective_scope,
                    )
            except Exception:
                hyde_hits = []

        # Combine literal and HyDE hits, dedupe and cap the candidate pool.
        hits = _merge_hits(raw_hits, hyde_hits)[: max(retrieval_top_k, _rerank_top_n)]

        # Inject chunks from stage1-boosted docs, scoped to each doc directly — not
        # just when the doc is entirely absent from the pool. Reproduced live: a
        # narrow-field question ("who is the authorizing manager...") landed
        # stage1 on doc_001, and doc_001 chunks WERE in the broad pool (other
        # pages ranked well against the generic embedding) -- so the old
        # "missing doc" check never fired, yet the one chunk actually holding
        # the field ("Authorizing Manager: Ricki Contreras...") never made the
        # global top-k on its own weak embedding match and was silently absent.
        # A doc-scoped fetch (same query, far smaller candidate pool) reliably
        # surfaces it at rank 0 -- verified directly with _direct_answer_from_context
        # on the scoped vs. unscoped pool (5/5 correct scoped, 5/5 Unsupported
        # unscoped). Unconditional per stage1 doc, same as the union pattern
        # used for filter_token above; _merge_hits dedupes so an already-present
        # doc's chunks just get reinforced, not duplicated.
        if stage1_doc_ids and not effective_scope:
            # Older ingestions only set metadata.doc_id on document_summary chunks;
            # PDF chunks may lack it. Also used below to reserve must-include slots.
            def _hit_doc_id(hit: dict) -> str:
                """Return a hit's doc_id, falling back to one parsed from its filename."""
                meta = hit.get("metadata") or {}
                if meta.get("doc_id"):
                    return meta["doc_id"]
                src = (meta.get("source_file") or meta.get("file_name") or "").lower()
                m = _DOC_ID_RE.search(src)
                return m.group(0) if m else ""

            for stage1_id in stage1_doc_ids:
                if not stage1_id:
                    continue
                try:
                    injected = retrieve(
                        query=retrieval_query,
                        top_k=5,
                        qdrant_url=qdrant_url,
                        collection=collection,
                        use_qdrant=True,
                        filter_token=None,
                        force_chunk_types=chunk_types,
                        force_exclude_chunk_types=["sheet_summary", "document_summary"],
                        scope_doc_id=stage1_id,
                    )
                    hits = _merge_hits(hits, injected)
                except Exception:
                    continue

        # Prepend sheet_summary hits so the reranking split below always has them
        hits = sheet_summary_hits + hits

        # When scoped to a doc but got no hits, retry without chunk-type constraints
        # so PDF page_content chunks are included. force_chunk_types=[] means
        # "explicitly no filter" vs None="infer from query".
        # sheet_summaries are always fetched separately above; exclude here to avoid dups.
        if not hits and effective_scope and chunk_types:
            hits = retrieve(
                query=retrieval_query,
                top_k=retrieval_top_k,
                qdrant_url=qdrant_url,
                collection=collection,
                use_qdrant=True,
                filter_token=filter_token,
                force_chunk_types=[],
                force_exclude_chunk_types=["sheet_summary", "document_summary"],
                scope_doc_id=effective_scope,
            )
            hits = sheet_summary_hits + hits

        # Nothing retrieved at all — let the caller report "not found".
        if not hits:
            return None

        # Drop bodyless chunks -- a page-continuation heading like
        # "## V. Purchasing and Contracting Policy (Continued)" gets its own
        # chunk (the _has_section_header() merge guard never dissolves a named
        # heading into a neighbour, even when it repeats an already-seen
        # section title with no new content under it). These carry no answer
        # text but can still out-rank the real content chunk and get cited as
        # "evidence" that's just a heading. Reproduced live: chunk 25 (55
        # chars, heading only) got cited over chunk 26 (1768 chars, the real
        # Sole Source Procurement text) for "who must approve a Sole Source
        # Procurement?". Filtering here (not in the chunker) needs no
        # re-ingest and leaves the merge guard itself untouched.
        non_empty_hits = [
            h
            for h in hits
            if any(
                ln.strip() and not ln.strip().startswith("#")
                for ln in (h.get("content") or "").splitlines()
            )
        ]
        if non_empty_hits:
            hits = non_empty_hits

        # Split hits by chunk type: sheet_summaries are ranked by column-name keyword
        # overlap (cross-encoders are trained on passage-answer pairs, not metadata
        # discovery chunks, so they return noise scores for sheet_summaries).
        # All other chunks go through the normal BGE reranker path.
        sheet_hits = [
            h
            for h in hits
            if (h.get("metadata") or {}).get("chunk_type") == "sheet_summary"
        ]
        other_hits = [
            h
            for h in hits
            if (h.get("metadata") or {}).get("chunk_type") != "sheet_summary"
        ]

        _COL_STOPWORDS = {
            "what",
            "which",
            "where",
            "when",
            "who",
            "how",
            "is",
            "are",
            "was",
            "were",
            "the",
            "a",
            "an",
            "and",
            "or",
            "of",
            "for",
            "to",
            "in",
            "on",
            "at",
            "by",
            "with",
            "from",
            "that",
            "this",
            "these",
            "those",
            "be",
            "do",
            "does",
            "did",
            "have",
            "has",
            "had",
            "as",
            "it",
            "its",
            "into",
            "any",
            "all",
            "summary",
            "data",
            "table",
            "sheet",
            "report",
            "row",
            "column",
            "value",
        }
        # Distinct content terms of the query, used for sheet column matching.
        q_terms = {
            t
            for t in re.sub(r"[^a-z0-9 ]", " ", retrieval_query.lower()).split()
            if t not in _COL_STOPWORDS and len(t) >= 3
        }

        def _col_overlap(hit: dict) -> int:
            """Count query terms that appear in a sheet_summary hit's column list."""
            content = (hit.get("content") or "").lower()
            col_line = next(
                (ln for ln in content.splitlines() if ln.startswith("columns:")), ""
            )
            col_tokens = {
                t
                for t in re.sub(r"[^a-z0-9 ]", " ", col_line).split()
                if t not in _COL_STOPWORDS and len(t) >= 3
            }
            return len(q_terms & col_tokens)

        def _sheet_sort_key(hit: dict) -> tuple[int, int]:
            """Sort key for sheet_summary hits: (column overlap, stage-1 doc bonus)."""
            overlap = _col_overlap(hit)
            # Stage 1 boost: prefer sheets from Stage 1-identified docs as a tiebreaker.
            stage1_bonus = (
                1
                if (hit.get("metadata") or {}).get("doc_id", "") in stage1_doc_ids
                else 0
            )
            return (overlap, stage1_bonus)

        ranked_sheet_hits = sorted(sheet_hits, key=_sheet_sort_key, reverse=True)

        # Rerank the non-sheet hits with the cross-encoder when a ranker exists.
        reranked_used = False
        if ranker is not None and other_hits:
            docs = [h.get("content", "") for h in other_hits]
            # Rerank ALL candidates so we can reserve slots for stage1 docs after the
            # fact. With the previous top_n=_rerank_top_n cut, stage1 chunks the reranker
            # ranked 11th+ were silently dropped before we could promote them.
            reranked = ranker.rerank(retrieval_query, docs, top_n=len(docs))
            reranked_hits = [
                {**other_hits[r["index"]], "rerank_score": r["score"]} for r in reranked
            ]
            reranked_used = True

            if stage1_doc_ids and not effective_scope:
                # Stem-matched docs (filename literally matches query terms) are a
                # strong signal — guarantee top-2 chunks each before filling rest
                # with reranker order. Stops one popular stage1 doc (e.g. HR policy)
                # from monopolizing reserved slots and squeezing out the actually-
                # named doc (e.g. procurement_policy).
                must_include = stem_match_doc_ids | explicitly_mentioned_doc_ids
                guaranteed: list[dict[str, Any]] = []
                guaranteed_keys: set[tuple[Any, ...]] = set()
                for doc in must_include:
                    doc_chunks = [h for h in reranked_hits if _hit_doc_id(h) == doc][:2]
                    for h in doc_chunks:
                        meta = h.get("metadata") or {}
                        key = (meta.get("source_file"), meta.get("chunk_index"))
                        if key not in guaranteed_keys:
                            guaranteed.append(h)
                            guaranteed_keys.add(key)
                remaining_slots = max(_rerank_top_n - len(guaranteed), 0)
                rest = [
                    h
                    for h in reranked_hits
                    if (
                        (h.get("metadata") or {}).get("source_file"),
                        (h.get("metadata") or {}).get("chunk_index"),
                    )
                    not in guaranteed_keys
                ][:remaining_slots]
                top_other = guaranteed + rest
            else:
                # M-2 (MITIGATION_PLAN.md): this used to head-insert
                # raw_hits[:3] ahead of the cross-encoder's own ranking,
                # overriding it for the first three slots the agent ever
                # sees. Over 18 gold evidence quotes the reranker beat
                # retrieve()'s pre-rerank order 11:2 on rank-1 hits (see
                # eval/gold_rank_at_1.py for current numbers). A
                # confidence-floor guard (fall back to raw_hits[:1] only
                # when the reranker's top score is low) was tried first and
                # dropped: correct top-1 chunks and wrong ones both commonly
                # score negative, so no floor separates the cases -- any
                # threshold either never fires or fires on nearly every
                # query. Let the reranker's order stand unconditionally.
                head = (
                    raw_hits[: min(3, _rerank_top_n)]
                    if filter_token and any(c.isdigit() for c in filter_token)
                    else []
                )
                top_other = _merge_hits(head, reranked_hits)[:_rerank_top_n]
        else:
            top_other = other_hits[:_rerank_top_n]

        if reranked_used:
            selected_keys = {
                (
                    (h.get("metadata") or {}).get("source_file"),
                    (h.get("metadata") or {}).get("chunk_index"),
                )
                for h in top_other
            }
            for h in reranked_hits:
                meta = h.get("metadata") or {}
                key = (meta.get("source_file"), meta.get("chunk_index"))
                if key in selected_keys:
                    continue
                _last_rejected.append(
                    {
                        "filename": meta.get("source_file")
                        or meta.get("file_name")
                        or "unknown",
                        "score": round(float(h.get("rerank_score", 0.0)), 4),
                    }
                )
            _last_rejected.sort(key=lambda r: r["score"], reverse=True)
            del _last_rejected[3:]

        # sheet_summary chunks signal "the relevant sheet has these columns" so the
        # agent can route to the Excel tool — they are not answer-bearing content.
        # Require >=2 token overlap (after stopword filter) to avoid generic matches
        # like "and/the/of"; cap at 2 to avoid crowding PDF/text chunks.
        top_sheet = [h for h in ranked_sheet_hits if _col_overlap(h) >= 2][:2]
        top_hits = (top_sheet + top_other)[:_rerank_top_n]

        # Graceful degrade for pure-spreadsheet documents (no page_content chunks
        # at all -- only sheet_summary): a question about the FILE ITSELF ("what
        # year is this titled for") mentions no real column name, so the
        # col_overlap>=2 gate above rejects every sheet_summary hit even when
        # they're highly relevant, and top_other is empty (nothing else exists to
        # fall back on) -- reproduced live: 9/9 tool calls on doc_011 (a pure
        # .xlsx corpus doc) returned "No relevant information found." despite the
        # top sheet_summary hit scoring 0.83. Only fires when nothing else
        # survived; never overrides the column-based routing above.
        if not top_hits and ranked_sheet_hits:
            top_hits = ranked_sheet_hits[:1]

        if _RETRIEVAL_DEBUG:
            returned = [
                (
                    (h.get("metadata") or {}).get("source_file")
                    or (h.get("metadata") or {}).get("file_name"),
                    (h.get("metadata") or {}).get("chunk_index"),
                )
                for h in top_hits
            ]
            print(f"[RETRIEVAL_DEBUG] returned {len(top_hits)} chunks: {returned}")

        # Expand neighbours and render the final numbered block for the agent.
        return _format_hits(
            top_hits,
            filter_token,
            reranked_used,
            retrieval_query,
            search_query,
            max_table_chars,
            max_chunk_chars,
        )

    def search_knowledge_base(query: str, doc_id: str = "") -> tuple[str, dict]:
        """Search the knowledge base for relevant documents and tables.

        Returns (content, artifact). The artifact carries the reranked-but-not-
        selected candidates for the UI's "retrieved but rejected" trust panel; it
        rides on the ToolMessage and is not shown to the LLM.
        """
        if FORCED_MODALITY.get() == "excel":
            return (
                "This question needs structured/aggregate data (a sum, count, "
                "maximum, or specific row lookup over spreadsheet data) — use the "
                "query_excel tool instead of search_knowledge_base.",
                {"rejected": []},
            )
        # Run the full retrieve/rerank/format pipeline; map None → not-found text.
        forced = FORCED_DOC_ID.get()
        result = _fetch_docs(query, doc_id=forced if forced else doc_id)
        content = result if result else "No relevant information found."
        artifact = {"rejected": list(_last_rejected)}
        return content, artifact

    # Wrap the function as a LangChain tool and return it with its limits dict.
    tool = StructuredTool.from_function(
        func=search_knowledge_base,
        name="search_knowledge_base",
        response_format="content_and_artifact",
        description=(
            "Search unstructured documents (PDFs, reports, policies, handbooks) for "
            "text passages. Input: a focused sub-question. This tool retrieves text — it "
            "cannot compute sums, counts, maxima, or other aggregates over Excel/CSV "
            "data. For any structured-data or arithmetic question, use query_excel instead."
        ),
        args_schema=SearchInput,
    )
    return tool, _limits
