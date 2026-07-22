"""Answer post-processing for the RAG agent — abstention normalization,
multi-part query splitting, coverage-driven repair, and the non-agentic
fallback retrieval paths.

Extracted from rag_agent.py; imported back by ask_agent / stream_agent
(and their helpers _context_fallback_answer / _retrieval_only_answer /
_normalize_final) which run these functions as the answer-finalization pipeline.

Calls into: src/retriever.py (retrieve), src/llm_utils.py (_llm_call,
_is_thinking_model), src/file_resolver.py (resolve_original_name), src/config.py.
"""

from __future__ import annotations

import json
import re
from typing import Any

from src.config import MAX_TOOL_RESULTS, QDRANT_COLLECTION, QDRANT_URL
from src.file_resolver import resolve_original_name
from src.llm_utils import _is_thinking_model, _llm_call
from src.retriever import retrieve

# ---------------------------------------------------------------------------
# Text / JSON parsing utilities — clean model output before inspecting it
# ---------------------------------------------------------------------------


def _strip_think(text: str) -> str:
    """Remove reasoning blocks emitted by thinking models."""
    return re.sub(r"(?is)<think>.*?</think>\s*", "", text).strip()


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Parse the first JSON object from model output."""
    cleaned = _strip_think(text).strip()
    cleaned = re.sub(
        r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE | re.MULTILINE
    ).strip()
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        parsed = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


# ---------------------------------------------------------------------------
# Bad-answer detection — classify answers worth retrying or rejecting
# ---------------------------------------------------------------------------


_SOFT_REFUSAL_RE = re.compile(
    r"does not (?:provide|contain)|cannot (?:perform|determine|answer)|"
    r"no information (?:is )?available|unable to (?:determine|answer|find)",
    re.IGNORECASE,
)


def _looks_like_bad_final_answer(text: str) -> bool:
    """Return True for transport/tool artifacts or abstentions worth retrying.

    Includes soft-refusal prose ("the retrieved content does not provide...")
    -- not just the literal "Unsupported" token -- so these fall through to
    the context/retrieval fallbacks below instead of being accepted as a
    final answer. Reproduced: a comparison question whose first attempt only
    retrieved one document produced "The retrieved content does not provide
    information on..." -- a bad answer with a strong, targeted fallback
    (_direct_retrieval_answer's clause-split re-retrieval) available, but the
    narrow exact-string check let it slip through un-retried.
    """
    cleaned = _strip_think(text).strip()
    if not cleaned:
        return True
    lowered = cleaned.lower()
    return (
        lowered in {"unsupported", "no answer generated.", "no answer generated"}
        or lowered.startswith("sorry, need more steps")
        or lowered.startswith("<function=")
        or lowered.startswith("function=")
        or "search_knowledge_base" in lowered
        and ("<function" in lowered or "</function>" in lowered)
        or bool(_SOFT_REFUSAL_RE.search(lowered))
    )


# ---------------------------------------------------------------------------
# Abstention normalization — collapse verbose 'not found' replies to 'Unsupported'
# ---------------------------------------------------------------------------

# Hedging phrases that indicate the model failed to find the requested value.
_NOT_PROVIDED_PHRASES = (
    "not provided",
    "not available",
    "not included",
    "not contained",
    "not answerable",
    "not found in",
    "not in this dataset",
    "not in the dataset",
    "does not contain",
    "not present in",
    "cannot be determined",
    "cannot be found",
    "no information",
    "not specified",
    "not stated",
    "none of the",
    "none of these",
    "no document",
    "not in any",
    "not listed in",
    "not given",
    "no such information",
    "does not specify",
    "does not mention",
    "does not explicitly",
    "is not explicitly",
    "not explicitly stated",
    "not explicitly mentioned",
    "does not exist",
)


# Stronger phrases that warrant abstention even mid-sentence.
_STRONG_NOT_FOUND_PHRASES = (
    "none of the provided documents",
    "none of the documents",
    "no document in",
    "not present in any",
    "not found in any",
)


def _normalize_unsupported(answer: str) -> str:
    """Convert verbose 'not found' answers to the canonical 'Unsupported' token."""
    if "Unsupported" in answer:
        # Collapse "hedging preamble + Unsupported" → "Unsupported" when the text
        # before "Unsupported" is pure hedging with no real answer values.
        # Preserve multi-part answers that have real values (Label: value pattern).
        idx = answer.index("Unsupported")
        before = answer[:idx].strip()
        after = answer[idx + len("Unsupported") :].strip().lstrip(".")
        if before and not after:
            lowered_before = before.lower()
            is_hedging = any(
                phrase in lowered_before for phrase in _NOT_PROVIDED_PHRASES
            )
            has_real_value = bool(re.search(r"\w[\w\s]{1,30}:\s+\S", before))
            if is_hedging and not has_real_value:
                return "Unsupported"
        return answer
    # No literal "Unsupported" token — abstain if the whole answer is hedging.
    lowered = answer.lower()
    if any(phrase in lowered for phrase in _STRONG_NOT_FOUND_PHRASES):
        return "Unsupported"
    if any(phrase in lowered for phrase in _NOT_PROVIDED_PHRASES):
        return "Unsupported"
    return answer


# ---------------------------------------------------------------------------
# Bare-filename detection — reject answers that name a file but extract no value
# ---------------------------------------------------------------------------

# Matches an ingested document filename (doc_NNN...ext) OR its extension-less
# stem (doc_NNN_word...) inside an answer -- the model sometimes names the
# stem alone (reproduced live: "doc_005_fueling_records_invoice" answered a
# question with no such value), which must trip this guard the same way the
# full filename does. A bare "doc_NNN" id with no trailing underscore-word is
# NOT matched: legitimate cross-document answers cite documents that way.
_BARE_FILENAME_RE = re.compile(
    r"\bdoc_\d+(?:[_a-z0-9-]*\.(?:pdf|csv|xlsx|xls|md|json|txt|tsv)|(?:_[a-z0-9-]+)+)\b",
    re.IGNORECASE,
)


# Framing/question-echo words removed before checking for substantive content.
_BARE_FN_STOP = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "am",
        "in",
        "on",
        "of",
        "for",
        "to",
        "from",
        "with",
        "by",
        "at",
        "as",
        "into",
        "that",
        "which",
        "this",
        "these",
        "those",
        "it",
        "its",
        "they",
        "them",
        "what",
        "who",
        "whom",
        "where",
        "when",
        "how",
        "why",
        "whose",
        "and",
        "or",
        "but",
        "not",
        "no",
        "nor",
        "so",
        "if",
        "then",
        "provides",
        "provide",
        "providing",
        "provided",
        "provider",
        "gives",
        "give",
        "given",
        "giving",
        "gave",
        "shows",
        "show",
        "showing",
        "shown",
        "showed",
        "contains",
        "contain",
        "containing",
        "contained",
        "has",
        "have",
        "had",
        "having",
        "states",
        "state",
        "stating",
        "stated",
        "mentions",
        "mention",
        "mentioning",
        "mentioned",
        "specifies",
        "specify",
        "specifying",
        "specified",
        "outlines",
        "outline",
        "outlining",
        "outlined",
        "details",
        "detail",
        "detailing",
        "detailed",
        "lists",
        "list",
        "listing",
        "listed",
        "refers",
        "refer",
        "referring",
        "referred",
        "reference",
        "references",
        "document",
        "documents",
        "doc",
        "docs",
        "file",
        "files",
        "filename",
        "report",
        "reports",
        "policy",
        "policies",
        "page",
        "section",
        "chapter",
        "row",
        "table",
        "csv",
        "pdf",
        "xlsx",
        "xls",
        "md",
        "json",
        "txt",
        "source",
        "sources",
        "based",
        "according",
        "answer",
        "answers",
        "above",
        "below",
        "found",
        "find",
    }
)


def _is_bare_filename_answer(query: str, answer: str) -> bool:
    """Return True if the answer is dominated by a filename with no extracted value.

    The agent occasionally responds to "which document gives X" / "what is X" by
    naming a topically-related document filename without actually extracting any
    value matching the question. That is a refusal failure dressed up as an
    answer. This check fires when, after removing filenames, framing words and
    question-echo, no substantive tokens or numeric values remain.

    The guard is general: it does not key on doc IDs, question patterns, or
    specific data types — only on whether the answer carries content beyond a
    filename + restated question.
    """
    if not answer or not answer.strip():
        return False
    # Strip only markers that wrap content; preserve underscores inside identifiers.
    cleaned = re.sub(r"[*`#]+", "", answer.strip())
    cleaned = re.sub(r"\[\d+\]", "", cleaned)
    filenames = _BARE_FILENAME_RE.findall(cleaned)
    if not filenames:
        return False
    stripped = cleaned
    for fn in filenames:
        stripped = stripped.replace(fn, " ")
    # Numbers that already appear in the query are echo, not extracted values.
    query_nums = set(re.findall(r"\d[\d,.$%/-]*", query))
    nums = [n for n in re.findall(r"\d[\d,.$%/-]*", stripped) if n not in query_nums]
    if nums:
        return False
    # No new numbers — bad answer only if fewer than 2 substantive (non-framing,
    # non-question-echo) word tokens remain after stripping filenames.
    query_terms = set(re.findall(r"\b[a-z][a-z]{2,}\b", query.lower()))
    tokens = re.findall(r"\b[A-Za-z][A-Za-z'/-]{1,}\b", stripped.lower())
    substantive = [t for t in tokens if t not in _BARE_FN_STOP and t not in query_terms]
    return len(substantive) < 2


def _verify_grounded(
    query: str,
    answer: str,
    tool_contexts: list[str],
    api_base: str,
    model_name: str,
) -> bool:
    """Post-generation check: is this answer actually supported by retrieved context?

    One LLM call, binary verdict — not per-claim scoring (that's the eval
    judge's job, offline). This is the runtime counterpart: catches an answer
    that isn't grounded in what was actually retrieved before it reaches the
    user, so the query pipeline can downgrade it to Unsupported instead.
    Fails open (returns True = "grounded") on any judge error — a broken
    verifier should not turn every live answer into a refusal.
    """
    context = "\n\n".join(ctx.strip() for ctx in tool_contexts if ctx and ctx.strip())
    if not context:
        return True
    context = context[:8000]
    no_think = "/no_think " if _is_thinking_model(model_name) else ""
    prompt = (
        f"{no_think}Question: {query}\n\n"
        f"Answer: {answer}\n\n"
        f"Retrieved context:\n{context}\n\n"
        "Is every factual claim in the Answer supported by — or directly "
        "inferable from — the Retrieved context above? A claim is supported "
        "even if worded differently, as long as the facts are present. "
        "Reply with exactly one word: YES or NO."
    )
    try:
        # A reasoning model (e.g. gpt-oss) spends tokens on hidden chain-of-
        # thought before any visible YES/NO -- observed up to ~170 reasoning
        # tokens on this exact prompt shape, so 10 (enough for a non-reasoning
        # model's bare word) silently starves it to an empty answer. 128
        # gives headroom without meaningfully raising cost/latency.
        verdict = _llm_call(prompt, api_base, model_name, max_tokens=128, temperature=0)
    except Exception:
        return True
    return "no" not in verdict.strip().lower()[:3]


# ---------------------------------------------------------------------------
# Query analysis — detect unanswerable, multi-part, and decomposable questions
# ---------------------------------------------------------------------------


def _has_empty_reference_placeholder(query: str) -> bool:
    """Detect questions with missing reference placeholders (e.g. 'in and the cost shown in ?').

    These questions are unanswerable by construction — a reference (doc id, table, sheet)
    was meant to fill a slot but the slot is empty. Any non-Unsupported answer is a
    fabrication.
    """
    q = re.sub(r"\s+", " ", query).strip().lower()
    return bool(
        re.search(r"\bin and\b", q)
        or re.search(r"\bin\s+and\s+the\b", q)
        or re.search(r"\bin\s+\?", q)
        or re.search(r"\bshown in\s*\?", q)
        or re.search(r"\bfrom\s+\?", q)
    )


def _is_multi_part_query(query: str) -> bool:
    """Return True when the query asks for several distinct facts at once."""
    q = query.lower()
    return (
        " and what " in q
        or "? according to " in q
        or "? in the " in q
        or " respectively" in q
        or " both " in q
        or " compared with " in q
        or " versus " in q
        or bool(re.search(r"\band,?\s+in\b", q))
        or bool(re.search(r"\bthe two\b.{0,40}\bdocuments?\b", q))
        or bool(re.search(r"\band the\b.{3,50}\brow\b", q))
        or bool(re.search(r"\beach\s+(?:document|doc|file|allow|has|contain)\b", q))
        or bool(re.search(r"\?\s+and\s+for\b", q))
        or bool(re.search(r"\band\s+which\s+(?:to|document|doc|file|report)\b", q))
        or bool(
            re.search(
                r"\b(which|what)\b[^?]{1,40}\b(document|policy|file|report|invoice|contract|dataset)\b[^?]{1,150}\bor\b",
                q,
            )
        )
        or bool(re.search(r"\bcomparing\b.{1,80}\band\b", q))
    )


def _split_multi_part_query(query: str) -> list[str]:
    """Split obvious multi-hop questions into answerable retrieval sub-queries."""
    q = query.strip()
    parts: list[str] = []

    # Try three split patterns in priority order: a "?" boundary before a new
    # question, an "and what" conjunction, or a trailing "respectively". The
    # boundary alternation must stay in sync with _is_multi_part_query's own
    # "\?\s+and\s+for\b" detection pattern -- that pattern flagged a question
    # as multi-part with no matching split rule here, so it silently fell
    # through to "return the whole question unsplit" (reproduced directly:
    # eval qa_id doc_006_doc_007_cross_document_qa__qa_2 detected as
    # multi-part but never split, silently dropping the first sub-question's
    # answer entirely).
    question_boundary = re.search(
        r"\?\s+(?=(According to|In the|In |What|Which|And for)\b)", q
    )
    if question_boundary:
        first = q[: question_boundary.start() + 1].strip()
        second = q[question_boundary.end() :].strip()
        if first:
            parts.append(first)
        if second:
            parts.append(second)
    elif re.search(r"\band what\b", q, flags=re.IGNORECASE):
        first, second = re.split(r"\band what\b", q, maxsplit=1, flags=re.IGNORECASE)
        first = first.strip(" ,;")
        second = "What " + second.strip(" ,;")
        if first:
            parts.append(first)
        if second:
            parts.append(second)
    elif " respectively" in q.lower():
        # Keep the original wording; retrieval can still use both named entities.
        parts.append(q)

    # Keep only non-trivial, unique parts; fall back to the whole query if <2.
    cleaned: list[str] = []
    for part in parts:
        part = re.sub(r"\s+", " ", part).strip()
        if len(part) >= 12 and part not in cleaned:
            cleaned.append(part)
    return cleaned if len(cleaned) >= 2 else [q]


# ---------------------------------------------------------------------------
# Non-agentic fallback answer paths — re-answer from context or fresh retrieval
# ---------------------------------------------------------------------------


def _direct_answer_from_context(
    query: str, contexts: list[str], api_base: str, model_name: str
) -> str:
    """Fallback answer pass over retrieved text, without another tool-planning loop."""
    usable_contexts = [ctx.strip() for ctx in contexts if ctx and ctx.strip()]
    if not usable_contexts:
        return "Unsupported"

    # Cheap path first: pull a "Field: Value" match straight from the text.
    extracted = _extract_key_value_answer(query, usable_contexts)
    if extracted:
        return extracted

    # Pack contexts into a character budget before sending them to the LLM.
    packed_contexts: list[str] = []
    used_chars = 0
    max_context_chars = 16000
    per_context_chars = 2500
    for ctx in usable_contexts:
        piece = ctx[:per_context_chars]
        if used_chars + len(piece) > max_context_chars:
            remaining = max_context_chars - used_chars
            if remaining > 0:
                packed_contexts.append(piece[:remaining])
            break
        packed_contexts.append(piece)
        used_chars += len(piece)
    context = "\n\n".join(packed_contexts)

    # Single grounded-answer LLM call over the packed context.
    no_think = "/no_think " if _is_thinking_model(model_name) else ""
    prompt = (
        f"{no_think}Answer the question using only the retrieved context below.\n"
        "Each retrieved passage is prefixed with [N] file=<filename> and optionally subquery=<sub-query>. "
        "For multi-document questions, use the file= label to match each answer value to its correct source document — "
        "do not mix values from different files.\n"
        "If the requested answer is not present in the context, output only: Unsupported\n"
        "Do not output Unsupported when the context contains the requested value under a matching field label.\n"
        "For multi-part questions, answer each part from its matching source and only mark missing parts Unsupported.\n"
        "Lead with the direct answer value. Do not expose tool calls or reasoning.\n\n"
        f"Question: {query}\n\n"
        f"Retrieved context:\n{context}\n\n"
        "Answer:"
    )
    answer = _llm_call(prompt, api_base, model_name, max_tokens=256, temperature=0)
    return _strip_think(answer) or "Unsupported"


def _llm_split_subqueries(query: str, api_base: str, model_name: str) -> list[str]:
    """Use the LLM to decompose a multi-part question into self-contained retrieval sub-queries.

    Each sub-query preserves enough context (entity names, document references) to retrieve
    the right chunk without relying on surrounding query text.
    """
    no_think = "/no_think " if _is_thinking_model(model_name) else ""
    prompt = (
        f"{no_think}Decompose the following question into 2-3 focused, self-contained search queries.\n"
        "Rules:\n"
        "- Each sub-query must include the key entity names and document references from the original question.\n"
        "- Preserve temporal context (dates, periods like 'after three rate cuts', 'Q1 2025/26') in each sub-query.\n"
        "- Do NOT produce generic sub-queries that lose context (e.g. avoid 'What is the total cost?').\n"
        "- Return ONLY a JSON array of strings, no explanation.\n\n"
        f"Question: {query}\n\n"
        "JSON array:"
    )
    # Ask the LLM for a JSON array of sub-queries; on any failure fall back to
    # the deterministic regex-based splitter.
    try:
        raw = _llm_call(prompt, api_base, model_name, max_tokens=200, temperature=0)
        raw = _strip_think(raw)
        match = re.search(r"\[.*?\]", raw, re.DOTALL)
        if match:
            parts = json.loads(match.group())
            if isinstance(parts, list) and len(parts) >= 2:
                return [str(p).strip() for p in parts if str(p).strip()]
    except Exception:
        pass
    return _split_multi_part_query(query)


def _direct_retrieval_answer(query: str, api_base: str, model_name: str) -> str:
    """Last-resort non-agentic RAG path for malformed or skipped tool calls."""
    # Decompose the query, then retrieve a share of the token budget per sub-query.
    contexts = []
    context_index = 1
    subqueries = _llm_split_subqueries(query, api_base, model_name)
    per_query_top_k = min(
        MAX_TOOL_RESULTS, max(4, MAX_TOOL_RESULTS // max(1, len(subqueries)))
    )
    for subquery in subqueries:
        hits = retrieve(
            query=subquery,
            top_k=per_query_top_k,
            qdrant_url=QDRANT_URL,
            collection=QDRANT_COLLECTION,
            use_qdrant=True,
        )
        for hit in hits:
            meta = hit.get("metadata", {}) or {}
            file_name = meta.get("file_name") or meta.get("source_file", "unknown")
            file_name = resolve_original_name(file_name)
            content = (hit.get("content") or "").strip()
            contexts.append(
                f"[{context_index}] subquery={subquery}\nfile={file_name}\n{content}"
            )
            context_index += 1
    return _direct_answer_from_context(query, contexts, api_base, model_name)


# ---------------------------------------------------------------------------
# Coverage-driven repair — detect and refill missing parts of multi-part answers
# ---------------------------------------------------------------------------


def _coverage_check(
    query: str,
    answer: str,
    contexts: list[str],
    api_base: str,
    model_name: str,
) -> dict[str, Any]:
    """Check whether an answer covers every requested part of the question."""
    context = "\n\n".join(ctx[:1500] for ctx in contexts[:8] if ctx.strip())
    no_think = "/no_think " if _is_thinking_model(model_name) else ""
    prompt = (
        f"{no_think}You are a strict RAG answer coverage checker.\n"
        "Decide whether the answer covers every distinct fact requested by the question.\n"
        "If any requested fact is missing, write focused search queries for the missing facts.\n"
        "Do not judge style or citation quality. Do not require facts the question did not ask for.\n"
        "Return ONLY one JSON object with this schema:\n"
        '{"complete": true|false, "missing_queries": ["focused search query", ...]}\n'
        "Use at most 2 missing_queries.\n\n"
        f"Question:\n{query}\n\n"
        f"Answer:\n{answer}\n\n"
        f"Retrieved context summaries:\n{context}\n"
    )
    raw = _llm_call(prompt, api_base, model_name, max_tokens=300, temperature=0)
    parsed = _extract_json_object(raw) or {}
    missing = parsed.get("missing_queries", [])
    if not isinstance(missing, list):
        missing = []
    missing = [str(item).strip() for item in missing if str(item).strip()][:2]
    return {
        "complete": bool(parsed.get("complete")) and not missing,
        "missing_queries": missing,
    }


def _context_source_count(contexts: list[str]) -> int:
    """Count the distinct source files referenced across retrieved contexts."""
    sources: set[str] = set()
    for ctx in contexts:
        for match in re.finditer(
            r"(?:^|\n)(?:\[\d+\]\s*)?(?:repair_query=.*\n)?file=([^\s\n]+)", ctx
        ):
            sources.add(match.group(1))
    return len(sources)


def _missing_source_queries(
    query: str, answer: str, contexts: list[str], api_base: str, model_name: str
) -> list[str]:
    """Generate follow-up searches when a multi-part answer used too few sources."""
    context_refs = "\n".join(ctx.splitlines()[0] for ctx in contexts[:8] if ctx.strip())
    no_think = "/no_think " if _is_thinking_model(model_name) else ""
    prompt = (
        f"{no_think}The question likely needs evidence from more than one source, "
        "but the current answer used too few retrieved sources.\n"
        "Write up to 2 focused vector-store search queries for the missing independent facts or sources.\n"
        "Do not repeat facts already answered from the retrieved source.\n"
        'Return ONLY JSON: {"missing_queries": ["query", ...]}\n\n'
        f"Question:\n{query}\n\n"
        f"Current answer:\n{answer}\n\n"
        f"Retrieved source refs:\n{context_refs}\n"
    )
    raw = _llm_call(prompt, api_base, model_name, max_tokens=220, temperature=0)
    parsed = _extract_json_object(raw) or {}
    missing = parsed.get("missing_queries", [])
    if not isinstance(missing, list):
        return []
    return [str(item).strip() for item in missing if str(item).strip()][:2]


def _unsupported_count(answer: str) -> int:
    """Count occurrences of the 'Unsupported' token in an answer."""
    return len(re.findall(r"\bUnsupported\b", answer, flags=re.IGNORECASE))


def _is_better_multi_answer(candidate: str, current: str) -> bool:
    """Return True if candidate is a better multi-part answer than current.

    Prefers answers with fewer Unsupported tokens and more distinct value separators
    (semicolons, sentence boundaries with different numeric content).
    """
    c_unsupported = _unsupported_count(candidate)
    cur_unsupported = _unsupported_count(current)
    if c_unsupported < cur_unsupported:
        return True
    if c_unsupported > cur_unsupported:
        return False
    # Same Unsupported count — prefer the answer that has more distinct numeric values.
    c_nums = set(re.findall(r"\b\d[\d,.$%½¼¾]+", candidate))
    cur_nums = set(re.findall(r"\b\d[\d,.$%½¼¾]+", current))
    if len(c_nums) > len(cur_nums):
        return True
    # Prefer longer answer when both have the same numeric richness (more complete).
    return len(candidate) > len(current) + 30


def _repair_incomplete_answer(
    query: str,
    answer: str,
    contexts: list[str],
    api_base: str,
    model_name: str,
) -> str:
    """Run a small coverage-driven retrieval repair for incomplete multi-part answers."""
    if not _is_multi_part_query(query):
        return answer

    source_count = _context_source_count(contexts)
    missing_queries: list[str] = []

    # Cross-document failures often happen before synthesis: the model makes only one
    # tool call for a two-part question, then the coverage checker calls the answer
    # "complete" because the one retrieved source was internally answered. When the
    # evidence came from fewer than two sources, force a decomposed retrieval pass.
    if source_count < 2:
        missing_queries = _llm_split_subqueries(query, api_base, model_name)
        if len(missing_queries) < 2:
            try:
                missing_queries = _missing_source_queries(
                    query, answer, contexts, api_base, model_name
                )
            except Exception as exc:
                print(
                    f"[WARN] Missing-source query generation failed ({type(exc).__name__}): {exc}"
                )
                missing_queries = []
        if not missing_queries:
            missing_queries = [query]
    else:
        try:
            coverage = _coverage_check(query, answer, contexts, api_base, model_name)
        except Exception as exc:
            print(f"[WARN] Coverage check failed ({type(exc).__name__}): {exc}")
            return answer

        missing_queries = coverage.get("missing_queries") or []
        if coverage.get("complete"):
            return answer

    if not missing_queries:
        return answer

    # Avoid repeating identical searches when LLM decomposition returns near duplicates.
    deduped_queries: list[str] = []
    for missing_query in missing_queries:
        normalized = re.sub(r"\s+", " ", missing_query).strip().lower()
        if normalized and normalized not in {
            re.sub(r"\s+", " ", q).strip().lower() for q in deduped_queries
        }:
            deduped_queries.append(missing_query)
    missing_queries = deduped_queries

    # Retrieve extra chunks for each missing query and append them to the context.
    repaired_contexts = list(contexts)
    context_index = len(repaired_contexts) + 1
    for missing_query in missing_queries[:2]:
        try:
            hits = retrieve(
                query=missing_query,
                top_k=min(MAX_TOOL_RESULTS, 6),
                qdrant_url=QDRANT_URL,
                collection=QDRANT_COLLECTION,
                use_qdrant=True,
            )
        except Exception as exc:
            print(
                f"[WARN] Coverage repair retrieval failed ({type(exc).__name__}): {exc}"
            )
            continue
        for hit in hits[:4]:
            meta = hit.get("metadata", {}) or {}
            file_name = meta.get("file_name") or meta.get("source_file", "unknown")
            file_name = resolve_original_name(file_name)
            content = (hit.get("content") or "").strip()
            if content:
                repaired_contexts.append(
                    f"[{context_index}] repair_query={missing_query}\nfile={file_name}\n{content}"
                )
                context_index += 1

    # No new chunks were found — nothing to repair with.
    if len(repaired_contexts) == len(contexts):
        return answer

    # Re-answer over the enlarged context; keep the original answer on failure.
    try:
        repaired = _direct_answer_from_context(
            query, repaired_contexts, api_base, model_name
        )
    except Exception as exc:
        print(f"[WARN] Coverage repair answer failed ({type(exc).__name__}): {exc}")
        return answer

    if _looks_like_bad_final_answer(repaired):
        return answer

    # Source-diversity acceptance check.
    # When the original answer was a grounded abstention (bare Unsupported), only
    # accept the repair if it actually brought in chunks from a NEW source. Without
    # this guard, repair retrieval that re-surfaces chunks from the same doc the
    # agent already saw lets the synthesis step fabricate a "complete" multi-part
    # answer — converting correct refusals into wrong answers.
    if _looks_like_bad_final_answer(answer):
        if _context_source_count(repaired_contexts) <= source_count:
            return answer

    # Accept the repair only when it is genuinely better than the original.
    if source_count < 2 and _is_better_multi_answer(repaired, answer):
        return repaired
    if _unsupported_count(repaired) < _unsupported_count(answer):
        return repaired
    # The coverage-check path (source_count >= 2) had no acceptance rule at all:
    # a confidently-worded but incomplete answer (e.g. only addressing one of two
    # named documents) with >=2 coincidental sources from UNRELATED documents
    # would reach here and always fall through to the stale original, even when
    # _coverage_check correctly flagged it incomplete and repair retrieval found
    # the actually-missing document. Accept when repair genuinely added a new
    # source the original didn't have — the same "did we actually gain evidence"
    # signal already used for the bad-answer branch above.
    #
    # But gaining a source isn't enough on its own: verified this breaks a case
    # where the original was ALREADY fully correct and complete, coverage_check
    # still (wrongly) flagged it incomplete, and the resulting repair retrieval
    # pulled in a wrong chunk that overwrote a correct answer with a shorter,
    # incorrect one. Require the original to have shown some sign it knew it was
    # incomplete (a hedge phrase, e.g. "does not appear to be referenced") before
    # trusting a same-or-fewer-source replacement over it — self-admitted gaps are
    # a much stronger repair-worthy signal than a bare source-count proxy alone.
    original_admits_gap = any(p in answer.lower() for p in _NOT_PROVIDED_PHRASES) or (
        "does not appear" in answer.lower() or "do not appear" in answer.lower()
    )
    if source_count >= 2 and _context_source_count(repaired_contexts) > source_count:
        if original_admits_gap or _is_better_multi_answer(repaired, answer):
            return repaired
    return answer


# ---------------------------------------------------------------------------
# Key-value extraction — pull 'Field: Value' answers straight from context text
# ---------------------------------------------------------------------------


def _label_candidates_from_query(query: str) -> list[str]:
    """Infer possible field labels from common factoid question phrasing."""
    q = query.lower()
    patterns = [
        r"what\s+is\s+the\s+(.+?)(?:\?|$)",
        r"what\s+(.+?)\s+is\s+listed(?:\?|$)",
        r"what\s+(.+?)\s+is\s+shown(?:\?|$)",
        r"what\s+(.+?)\s+is\s+given(?:\?|$)",
        r"what\s+(.+?)\s+does\s+.+?\s+list(?:\?|$)",
    ]
    candidates: list[str] = []
    cut_words = (
        " in ",
        " for ",
        " from ",
        " within ",
        " according ",
        " on ",
        " dated ",
        " with ",
        " of the ",
    )
    # For each phrasing pattern, extract the label phrase and trim trailing scope.
    for pattern in patterns:
        match = re.search(pattern, q)
        if not match:
            continue
        label = match.group(1)
        for cut in cut_words:
            if cut in label:
                label = label.split(cut, 1)[0]
        label = re.sub(
            r"\b(listed|shown|given|provided|document|policy|report|table|row)\b",
            " ",
            label,
        )
        label = re.sub(r"\s+", " ", label).strip(" :.-")
        if len(label) >= 3 and label not in candidates:
            candidates.append(label)
    return candidates


def _extract_key_value_answer(query: str, contexts: list[str]) -> str | None:
    """Extract answers from generic 'Field: Value' text when the model abstains."""
    labels = _label_candidates_from_query(query)
    if not labels:
        return None

    # Scan each context for the label as a "Label: value" line or a markdown
    # table cell, returning the first cleaned value found.
    for ctx in contexts:
        text = re.sub(r"[*_`#]+", "", ctx)
        for label in labels:
            label_pattern = r"\s+".join(re.escape(part) for part in label.split())
            patterns = [
                rf"(?im)^\s*{label_pattern}\s*:\s*(?P<value>[^\n|]+)",
                rf"(?im)\|\s*{label_pattern}\s*\|\s*(?P<value>[^|\n]+)\|",
            ]
            for pattern in patterns:
                match = re.search(pattern, text)
                if not match:
                    continue
                value = re.sub(
                    r"<br\s*/?>", " ", match.group("value"), flags=re.IGNORECASE
                )
                value = re.sub(r"\s+", " ", value).strip(" :-")
                if value:
                    return value
    return None
