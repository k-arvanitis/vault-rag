"""Shared answer pipeline for the live API and the eval harness.

Both api.py's /query endpoint and eval/run_eval.py need identical routing,
retry, and multi-part-split behavior. Before this module existed, that logic
lived only inside api.py's request handler — eval called stream_agent
directly and never exercised it, so eval silently measured a weaker pipeline
than what real users actually got (e.g. a fix verified live in the app would
not move the eval numbers at all). Centralizing it here is the fix: both
callers now go through the same code.
"""

from __future__ import annotations

import itertools
import re
from typing import Any, Generator
from urllib.parse import unquote

from src.answer_quality import _is_multi_part_query, _split_multi_part_query
from src.config import MAX_TOOL_RESULTS, QDRANT_COLLECTION, QDRANT_URL
from src.file_resolver import resolve_original_name as _resolve_original_name
from src.llm_utils import _llm_call
from src.rag_agent import (
    FinalCorrection,
    route_question,
    routing_directive,
    stream_agent,
)
from src.retriever import retrieve
from src.tools.excel import ORIGINAL_QUESTION
from src.tools.retrieval_tool import FORCED_DOC_ID, FORCED_MODALITY
from src.vector_store import _stable_id

_CALL_BOUNDARY = "---CALL_BOUNDARY---"

_RETRY_INSTRUCTION = (
    "\n\nIMPORTANT: This is a retry. The previous attempt returned Unsupported. "
    "You MUST follow the doc-routing protocol strictly: "
    "(1) call search_knowledge_base with the topic words alone to identify the relevant "
    "document_summary chunk and read its Document ID (doc_XXX). "
    "(2) call search_knowledge_base again with the original question, passing that doc_id "
    "as the doc_id argument to scope the search to that specific document."
)

# Detects "what is the title of this document" style questions. Answered
# directly from a document_summary chunk's literal "Title:" line (see
# src/chunker.py's extract_literal_title) instead of letting the agent
# generate an answer -- verified live (qwen3-32b) that generation ignores
# the Title: line even when it's the #1-ranked retrieved chunk after
# reranking, picking a more prominent section heading instead. A prompt rule
# telling the model to prefer it is a coin flip against that nondeterminism;
# a deterministic lookup is not.
_TITLE_QUESTION_RE = re.compile(
    r"\btitle of\b"
    r"|\bthe title\b"
    r"|\b(?:document|file)(?:'s)? title\b"
    r"|\bwhat is this (?:document|file) (?:titled|called)\b",
    re.IGNORECASE,
)
_TITLE_LINE_RE = re.compile(r"^Title: (.+)$", re.MULTILINE)


def _title_shortcut_answer(question: str) -> tuple[str, list[str]] | None:
    """Return (answer, collected) from a document_summary's Title: line, or
    None if the question isn't a title question or no chunk has that line."""
    if not _TITLE_QUESTION_RE.search(question):
        return None
    try:
        hits = retrieve(
            query=question,
            qdrant_url=QDRANT_URL,
            collection=QDRANT_COLLECTION,
            top_k=3,
            use_qdrant=True,
            force_chunk_types=["document_summary"],
        )
    except Exception:
        return None
    for hit in hits:
        content = hit.get("content") or ""
        m = _TITLE_LINE_RE.search(content)
        if not m:
            continue
        meta = hit.get("metadata") or {}
        file_name = meta.get("file_name") or meta.get("source_file") or "unknown"
        score = hit.get("score") or 1.0
        collected = [f"[1] file={file_name} chunk=-1 score={score:.4f}\n{content}"]
        return m.group(1).strip(), collected
    return None


# Detects "compare X and Y" / "which document does A, which does B" style questions —
# these require evidence from two distinct sources. A prompt rule alone was verified
# not to reliably stop the agent from answering half from general knowledge instead
# of making the second required tool call; this check forces a real second retrieval.
_COMPARISON_RE = re.compile(
    r"\b(?:compare|comparing|versus|vs\.?|both .+ and\b|between .+ and\b|"
    r"which .+ and which|which .+\bor the .+\?|while the other\b)",
    re.IGNORECASE,
)

# doc_XXX ids named directly in a comparison question -- used to check whether
# the final answer actually has evidence for every document the question asked
# about. More robust than pattern-matching the answer's wording for a refusal:
# reproduced live, the model wrote off a side with several different phrasings
# across runs ("Unsupported", "No details ... are provided") -- a keyword check
# alone misses whichever phrasing wasn't anticipated, but checking "does the
# named document actually appear in the sources" doesn't depend on wording.
_MENTIONED_DOC_ID_RE = re.compile(r"\bdoc_\d+\b", re.IGNORECASE)


def _missing_mentioned_docs(
    question_text: str, srcs: list[dict], resolved_ids: list[str] | None = None
) -> list[str]:
    """doc_XXX ids named in the question, or already resolved for it (M-6's
    shared resolver -- see _resolve_comparison_doc_ids), with no matching
    source. resolved_ids widens this past literal "doc_XXX" text so a
    natural-language comparison (no doc id typed) still gets checked."""
    mentioned = {m.lower() for m in _MENTIONED_DOC_ID_RE.findall(question_text)}
    mentioned |= {d.lower() for d in (resolved_ids or [])}
    if len(mentioned) < 2:
        return []
    covered = {(s.get("document_id") or "").lower() for s in srcs} | {
        s["filename"].lower() for s in srcs
    }
    return [d for d in mentioned if not any(d in c for c in covered)]


def _comparison_incompleteness(
    question_text: str,
    ans: str,
    coll: list[str],
    resolved_ids: list[str] | None = None,
) -> tuple[list[str], bool, int]:
    """(missing_named_docs, has_partial_unsupported_fragment, n_distinct_sources)."""
    sources = parse_sources(coll)
    n_sources = len({s["filename"] for s in sources})
    missing = _missing_mentioned_docs(question_text, sources, resolved_ids)
    has_partial = ans.strip().lower() != "unsupported" and "unsupported" in ans.lower()
    return missing, has_partial, n_sources


# ---------------------------------------------------------------------------
# Deterministic comparison path
#
# The agent-based comparison retry above (answer_one's is_comparison branch)
# is inherently probabilistic: it lets the ReAct agent decide whether/how to
# make a second search_knowledge_base call, then detects incompleteness after
# the fact and re-prompts. That works most of the time but not every time.
# This path instead retrieves independently for every resolved document
# BEFORE any generation happens, guaranteeing (not hoping for) at least one
# evidence item per resolvable document, then makes a single synthesis call
# over the combined evidence. Falls back to None (caller uses the existing
# agent-based path) whenever the requested documents can't be confidently
# resolved -- e.g. "compare the two most recent versions" names no document,
# so forcing this path onto it would be guessing, not determinism.
# ---------------------------------------------------------------------------

_STEM_STOPWORDS = {"the", "and", "for", "with", "from"}

# Splits "which X ..., while the other Y ...?" into its two contrasted
# clauses -- see _retrieve_for_doc's docstring for why this exists.
_COMPARISON_SPLIT_RE = re.compile(
    r"^(?P<a>.*?),?\s+while the other\s+(?P<b>.*)$", re.IGNORECASE
)


def _split_comparison_clauses(question: str) -> tuple[str, str] | None:
    """Split a "which X, while the other Y" comparison into its two clauses,
    or None when the question doesn't match that shape.

    Only handles this one connector today -- the shape our real demo/gold
    questions use. Returns None (caller falls back to the full question)
    rather than guessing at other comparison phrasings.
    """
    # Strip a "Comparing doc_001 and doc_002:" lead-in before splitting --
    # otherwise it stays attached to clause A and pollutes its retrieval
    # query with doc-id tokens that dilute the real match (reproduced live:
    # clause A's actual match score dropped from 0.83 to 0.33 with the
    # lead-in still attached).
    question = re.sub(
        r"^\s*comparing\s+\S+\s+and\s+\S+\s*:\s*", "", question, flags=re.IGNORECASE
    )
    m = _COMPARISON_SPLIT_RE.search(question)
    if not m:
        return None
    a = m.group("a").strip().rstrip("?").strip()
    b = m.group("b").strip().rstrip("?").strip()
    if len(a) < 10 or len(b) < 10:
        return None
    return a, b


def _per_doc_retrieval_queries(question: str, doc_ids: list[str]) -> dict[str, str]:
    """Pick which query to retrieve with for each doc: the full question by
    default, or -- when the question splits into two contrasted clauses --
    whichever clause actually scores best for that doc.

    _retrieve_for_doc passing the whole comparison question to every doc
    dilutes retrieval: a question like "...requires legal review..., while
    the other focuses on customer-issued notices..." is dominated by the
    "legal review" vocabulary, so the doc whose real evidence is the
    "customer notices" clause has that clause buried near the bottom of its
    own scoped results (reproduced live: doc_002's actual answer clause
    ranked last of 10 doc_002-scoped hits on the full question, but first
    when queried with its own clause text alone). Splitting the question
    fixes that -- but WHICH clause belongs to WHICH doc_id isn't reliable
    from question order (a comparison question's clause order need not
    match doc_ids' sorted-mention order). So instead of guessing, this
    tries both clauses against both docs and lets each doc's own top
    retrieval score pick its clause -- self-aligning without an order
    assumption. Falls back to the full question on any lookup failure or a
    non-two-way split.
    """
    fallback = {doc_id: question for doc_id in doc_ids}
    if len(doc_ids) != 2:
        return fallback
    clauses = _split_comparison_clauses(question)
    if clauses is None:
        return fallback
    try:
        # score[doc][clause] -- each doc's best hit for each clause.
        score: list[list[float]] = []
        for doc_id in doc_ids:
            row = []
            for clause in clauses:
                hits = retrieve(
                    query=clause,
                    top_k=1,
                    qdrant_url=QDRANT_URL,
                    collection=QDRANT_COLLECTION,
                    scope_doc_id=doc_id,
                )
                row.append(hits[0]["score"] if hits else -1.0)
            score.append(row)
        # Pick the ASSIGNMENT (one distinct clause per doc) with the better
        # total, not each doc's own argmax independently. Scores from two
        # different query embeddings aren't on a comparable scale, so an
        # independent argmax can hand both docs the same clause -- reproduced
        # live: doc_002 scored the doc_001 clause higher in absolute terms and
        # both docs ended up retrieving with it, re-diluting exactly the
        # retrieval this split exists to separate. Comparing the two whole
        # assignments only needs the scores to be comparable WITHIN a doc,
        # which they are. A comparison question has one clause per document by
        # construction, so a shared clause is never the intended answer.
        straight = score[0][0] + score[1][1]
        crossed = score[0][1] + score[1][0]
        order = (0, 1) if straight >= crossed else (1, 0)
        return {doc_ids[i]: clauses[order[i]] for i in range(2)}
    except Exception:
        return fallback


_DOC_RESOLUTION_PROMPT = (
    "Below is a catalogue of documents, then a question that compares exactly two of them.\n\n"
    "CATALOGUE:\n{catalog}\n\n"
    "QUESTION: {q}\n\n"
    "Work out which two documents the question contrasts. Think briefly, then end your "
    "reply with a final line in exactly this form, quoting the exact substring copied from "
    "each document's own catalogue entry above that justifies picking it:\n"
    'ANSWER: doc_xxx ("<exact phrase copied from doc_xxx\'s catalogue entry>"), '
    'doc_yyy ("<exact phrase copied from doc_yyy\'s catalogue entry>")\n'
    "If the question does not clearly contrast two documents in the catalogue, end with:\n"
    "ANSWER: NONE"
)
_ANSWER_PAIR_RE = re.compile(r'(doc_\d+)\s*\(\s*"([^"]+)"\s*\)', re.IGNORECASE)

# Generic English function words only, stripped before comparing a quoted
# phrase's vocabulary against the question's -- same restriction excel.py's
# _TABLE_STOPWORDS documents: strip connectives, never real title/content
# words like "report" or "policy", which are exactly what distinguishes one
# document's catalogue entry from another's.
_DOC_RESOLUTION_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "in", "on", "for", "to", "with",
    "which", "is", "are", "this", "that",
}


# A picked document's own catalogue entry must share at least this many
# content words with the question. 2 (not 1) because a single shared word is
# routinely an accident of a publisher name or a shared domain term.
_DOC_ENTRY_OVERLAP_FLOOR = 2


def _entry_matches_question(entry: str, question: str) -> bool:
    """Does the picked document's catalogue entry share real vocabulary with
    the question, ignoring generic words?

    Substring-in-catalogue membership alone (verified by the caller) isn't
    enough: reproduced live, the model quoted a genuine, verifiable phrase
    from the WRONG document's own catalogue entry when the right answer
    couldn't be derived from titles at all ("which document identifies 42
    new topic areas" against a catalogue with no such topic-count anywhere).

    Compared against the whole ENTRY, not the quoted phrase. Measured over
    all 17 gold cross-document questions: phrase-vs-question overlap rejects
    correct and incorrect picks alike (the two spreadsheet documents are
    picked correctly from a question naming "Supplier Name"/"NET Amount",
    but the quoted title phrase shares no words with it) -- 8/17 correct.
    Entry-vs-question overlap keeps the same zero-wrong-pair property while
    recovering those: 12/17 correct, 0 wrong. The question's terms do appear
    in the right document's summary/column text, and don't appear in an
    unrelated document's entry, which is the signal that actually separates
    the two cases.
    """
    q_words = set(re.findall(r"[a-z]{3,}", question.lower())) - _DOC_RESOLUTION_STOPWORDS
    e_words = set(re.findall(r"[a-z]{3,}", entry.lower())) - _DOC_RESOLUTION_STOPWORDS
    return len(q_words & e_words) >= _DOC_ENTRY_OVERLAP_FLOOR


def _resolve_comparison_doc_ids_llm(
    question: str,
    catalogue: dict[str, str],
    api_base: str,
    model_name: str,
) -> list[str] | None:
    """M-6's LLM resolution step: ask the model which two documents in
    `catalogue` (doc_id -> "title (source_file)") a comparison question
    names, purely from document identity -- never retrieval scores, and
    never the eval manifest's aliases (those leak the gold questions'
    own phrasing).

    Hard verification gate, same shape as excel.py's
    _column_matches_question: the model must also quote, for each pick, the
    exact phrase from that document's own catalogue entry that justifies it.
    A hallucinated id can't produce a phrase that verifies, so it degrades to
    None (today's fallback to the agent path) instead of a confidently wrong
    pair -- the failure mode measured and rejected in the first attempt at
    this (removed 2026-07-30, see MITIGATION_PLAN.md M-6). Requires a large
    max_tokens: gpt-oss-120b spends the budget on hidden reasoning before any
    visible output, so a small budget silently returns '' and looks like the
    model "can't" resolve anything.
    """
    if len(catalogue) < 2:
        return None
    catalog_text = "\n".join(
        f"{doc_id}: {desc}" for doc_id, desc in sorted(catalogue.items())
    )
    prompt = _DOC_RESOLUTION_PROMPT.format(catalog=catalog_text, q=question)
    try:
        reply = _llm_call(prompt, api_base, model_name, max_tokens=1024, temperature=0)
    except Exception:
        return None
    m = re.search(r"ANSWER:\s*(.+)$", reply or "", re.MULTILINE)
    tail = (m.group(1) if m else (reply or "")).strip()
    verified: list[str] = []
    for doc_id, phrase in _ANSWER_PAIR_RE.findall(tail):
        doc_id = doc_id.lower()
        entry = catalogue.get(doc_id, "")
        phrase = phrase.strip()
        if (
            entry
            and phrase.lower() in entry.lower()
            and _entry_matches_question(entry, question)
        ):
            verified.append(doc_id)
    distinct = list(dict.fromkeys(verified))
    return distinct if len(distinct) == 2 else None


def _resolve_comparison_doc_ids(
    question: str,
    forced_doc_id: str | list[str] | None,
    doc_registry: dict[str, str],
    catalogue: dict[str, str] | None = None,
    api_base: str | None = None,
    model_name: str | None = None,
) -> list[str] | None:
    """Resolve which distinct documents a comparison question is actually
    asking about. Returns None when fewer than two can be confidently
    resolved, signaling the caller to fall back to the agent-based path.

    Checked in order: literal doc_XXX ids in the question, the UI's
    multi-select scope, fuzzy filename-stem overlap, then -- only if
    `catalogue`/`api_base`/`model_name` are all given -- M-6's LLM
    resolution step over the document catalogue as a last resort. See
    _resolve_comparison_doc_ids_llm's docstring and MITIGATION_PLAN.md's M-6
    for why that step is gated behind a verification check rather than
    trusted on its own.
    """
    if isinstance(forced_doc_id, list):
        resolved: list[str] = []
        for f in forced_doc_id:
            stem = re.sub(r"\.[^.]+$", "", f.rsplit("/", 1)[-1]).lower()
            doc_id = doc_registry.get(stem) or next(
                (v for k, v in doc_registry.items() if stem in k or k in stem), None
            )
            resolved.append(doc_id or stem)
        distinct = list(dict.fromkeys(resolved))
        return distinct if len(distinct) >= 2 else None

    mentioned = {m.lower() for m in _MENTIONED_DOC_ID_RE.findall(question)}
    if len(mentioned) >= 2:
        return sorted(mentioned)

    # Fuzzy title match against the doc registry -- same "stem token overlap"
    # heuristic retrieval_tool.py's own auto-scope uses, kept conservative:
    # only trusted when exactly two candidates clear the overlap bar and the
    # rest of the corpus doesn't (an ambiguous/no-name question won't).
    if doc_registry:
        q_tokens = set(re.findall(r"[a-z]{3,}", question.lower())) - _STEM_STOPWORDS
        scored: dict[str, int] = {}
        for stem, doc_id in doc_registry.items():
            stem_tokens = {
                t
                for t in re.findall(r"[a-z]{3,}", stem)
                if t not in _STEM_STOPWORDS and not t.startswith("doc")
            }
            overlap = len(q_tokens & stem_tokens)
            if overlap >= 2:
                scored[doc_id] = max(scored.get(doc_id, 0), overlap)
        top = sorted(scored, key=lambda d: scored[d], reverse=True)
        if len(top) == 2:
            return top

    if catalogue and api_base and model_name:
        return _resolve_comparison_doc_ids_llm(question, catalogue, api_base, model_name)

    return None


def _retrieve_for_doc(agent: Any, question: str, doc_id: str) -> str:
    """Call search_knowledge_base directly, scoped to exactly one document --
    bypasses the ReAct agent's own decision of whether/how to call it."""
    tool = getattr(agent, "_tools_by_name", {}).get("search_knowledge_base")
    if tool is None:
        return ""
    token = FORCED_DOC_ID.set(doc_id)
    try:
        content, _artifact = tool.func(question, doc_id=doc_id)
    finally:
        FORCED_DOC_ID.reset(token)
    return content or ""


_COMPARISON_SYNTHESIS_PROMPT = (
    "Answer this comparison question using ONLY the evidence passages below. "
    "Each passage is numbered and belongs to one of the documents being compared. "
    "Cite passages inline using their bracketed number, e.g. [1]. Do not use "
    "outside/general knowledge -- if a claim isn't supported by a passage below, "
    "do not make it. If a note below says a document has no relevant evidence, "
    "state plainly that this comparison has no support for that document instead "
    "of guessing.\n\nQuestion: {question}\n\nEvidence:\n{evidence}\n\nAnswer:"
)


def answer_comparison_deterministic(
    agent: Any,
    question: str,
    forced_doc_id: str | list[str] | None = None,
    trace: Any = None,
    doc_ids: list[str] | None = None,
) -> dict | None:
    """Deterministic comparison path: resolve doc ids, retrieve independently
    from each, synthesize once. Returns None to signal "not applicable here,
    use the existing agent-based comparison path" -- either the documents
    couldn't be confidently resolved, or every one of them came back empty.

    doc_ids: pass the already-resolved pair (answer_query resolves once and
    shares it with answer_one's coverage retry, M-7) to skip resolving here
    a second time. Left None, this resolves internally -- kept for direct
    callers/tests that don't need the shared resolution.
    """
    doc_registry = getattr(agent, "_doc_registry", None) or {}
    if doc_ids is None:
        # M-6's LLM resolution step is deliberately NOT reached here:
        # answer_query (the only production caller) always resolves once,
        # including the LLM step, and passes the result through as doc_ids
        # -- even when that result is None -- so this branch only runs for
        # direct callers/tests that skip pre-resolution. Passing the
        # catalogue here too would re-run the (nondeterministic, LLM-backed)
        # resolution a second time per question in production. Only the
        # free/deterministic checks run in this branch.
        doc_ids = _resolve_comparison_doc_ids(question, forced_doc_id, doc_registry)
    if (
        not doc_ids
        or not hasattr(agent, "_tools_by_name")
        or not hasattr(agent, "_llm")
    ):
        return None

    collected: list[str] = []
    covered_doc_ids: list[str] = []
    missing_doc_ids: list[str] = []
    marker = 1
    retrieval_queries = _per_doc_retrieval_queries(question, doc_ids)
    for doc_id in doc_ids:
        raw = _retrieve_for_doc(agent, retrieval_queries[doc_id], doc_id)
        if not raw.strip() or raw.strip() == "No relevant information found.":
            missing_doc_ids.append(doc_id)
            continue
        covered_doc_ids.append(doc_id)
        # raw is one or more "[N] file=... \nbody" numbered blocks already --
        # renumber into the shared collected sequence so parse_sources/
        # citation_map treat this exactly like a normal single agent run.
        for block in re.split(r"(?=^\[\d+\]\s+file=)", raw, flags=re.MULTILINE):
            block = block.strip()
            if not block:
                continue
            collected.append(re.sub(r"^\[\d+\]", f"[{marker}]", block, count=1))
            marker += 1

    if len(covered_doc_ids) < 2 and not (covered_doc_ids and missing_doc_ids):
        # Nothing usable, or only one requested doc even exists -- can't
        # produce a real comparison. Let the caller fall back.
        return None

    evidence = "\n\n".join(collected)
    prompt = _COMPARISON_SYNTHESIS_PROMPT.format(question=question, evidence=evidence)
    try:
        from langchain_core.messages import HumanMessage

        resp = agent._llm.invoke([HumanMessage(content=prompt)])
        answer = (resp.content or "").strip()
    except Exception:
        return None
    if not answer:
        return None

    if missing_doc_ids:
        notes = "\n\n".join(
            f"No relevant evidence was found for {mid} in this document set."
            for mid in missing_doc_ids
        )
        answer = f"{answer}\n\n{notes}"

    sources = parse_sources(collected)
    citation_map = build_citation_map(collected, sources)
    # marker - 1 = the real highest marker number assigned across both docs'
    # combined evidence, which can legitimately exceed MAX_TOOL_RESULTS (a
    # bound that only holds for a single tool call) -- see
    # _strip_inline_citation's docstring.
    answer = strip_leaked_headers(
        answer, citation_map=citation_map, max_plausible_marker=marker - 1
    )
    answer, sources = _renumber_citations_sequentially(answer, sources)
    answer = _narrow_quotes_to_answer(sources, answer)
    if trace is not None:
        trace.span(
            name="comparison-deterministic",
            input=question,
            output=answer,
            metadata={
                "covered_doc_ids": covered_doc_ids,
                "missing_doc_ids": missing_doc_ids,
            },
        )
    return {
        "answer": answer,
        "sources": sources,
        "sql": [],
        "tools": ["search_knowledge_base"] * len(covered_doc_ids),
        "rejected_sources": [],
        "collected": collected,
    }


_COMPARISON_RETRY_INSTRUCTION = (
    "\n\nIMPORTANT: This is a retry. This is a two-part comparison question and the "
    "previous attempt only retrieved evidence from one source. Do NOT answer the other "
    "part from general knowledge. Make a second search_knowledge_base call scoped to "
    "the other document or topic named in the question before finalizing your answer."
)

# Distinct from the retry above: evidence for BOTH sides was already retrieved
# (both documents appear in the source list) but the answer still wrote off one
# side as Unsupported — reproduced directly: a document whose relevant content
# was split across several related sub-topics (e.g. multiple distinct leave
# types) got 12 real chunks back, yet the answer said "Unsupported" for it
# instead of picking the most relevant sub-topic. Telling the model to search
# again wouldn't help here since it already has the evidence; the fix is a
# synthesis instruction, not a retrieval one.
_COMPARISON_PARTIAL_RETRY_INSTRUCTION = (
    "\n\nIMPORTANT: This is a retry. The previous attempt already retrieved relevant "
    "content for both documents in this comparison, but still answered one side as "
    "Unsupported. If a document's retrieved content covers several related "
    "sub-topics instead of one single answer (e.g. multiple distinct leave types, "
    "multiple different clauses), do not default to Unsupported — pick the most "
    "directly relevant sub-topic and answer using it, briefly noting that other "
    "related sub-topics exist if that's genuinely useful. Only answer Unsupported "
    "for a side if nothing relevant was actually retrieved for it."
)

# Chunk header format: "[1] file=name.pdf chunk=5 page=4 score=0.8312"
# or                   "[1] file=name.xlsx sheet=Sheet1 score=0.91"
_HEADER_RE = re.compile(
    r"^\[?\d+\]?\s+file=(?P<file>[^\s]+)"
    r"(?:\s+(?P<loc_key>chunk|sheet|part|repair_query|subquery)=(?P<loc_val>[^\s]+))?"
    r"(?:\s+page=(?P<page>[^\s]+))?"
    r"(?:\s+score=(?P<score>[^\s]+))?",
)
_MD_HEADING_RE = re.compile(r"^#{1,3}\s+(.+)$", re.MULTILINE)
_TABLE_MARKER_RE = re.compile(r"\[TABLE_START\]|\[TABLE_END\]")
# A figure's VLM description is synthetic text, not present in the PDF itself —
# leaving it in a citation's quote would break fitz.search_for-based highlighting.
_FIGURE_BLOCK_RE = re.compile(r"\[FIGURE_START\].*?\[FIGURE_END\]", re.DOTALL)
_FIGURE_BBOX_RE = re.compile(r"<!-- bbox:\[([\d.,\s]+)\] -->")
_OCR_BBOX_RE = re.compile(r"<!-- ocr_bbox:\[([\d.,\s]+)\] -->")
# _format_hits (retrieval_tool.py) sometimes wraps a chunk's own content with
# "[prev chunk]"/"[next chunk]" neighbor context for the LLM's benefit — that
# wrapper isn't part of the chunk's real text, so it must not leak into the
# excerpt/quote a citation shows or fitz text-search would never match the PDF.
_THIS_CHUNK_RE = re.compile(
    r"\[this chunk\]\n(.*?)(?:\n\n\[next chunk\]|\Z)", re.DOTALL
)
# doc_id/title trail the header line's fixed file/loc/page/score sequence,
# so they're picked up separately rather than folded into _HEADER_RE's ordered groups.
_DOC_META_RE = re.compile(r"\bdoc_id=(?P<doc_id>\S+)\s+title=(?P<title>\S+)")

# The model sometimes copies a raw retrieved-chunk header line
# ("[1] file=doc.pdf chunk=2 ...") or a dangling "Sources:" label into its final
# answer. Strip those whole lines so they never reach the user. Inline citation
# markers like "[1]" are NOT matched — the pattern requires "file=" after them.
_LEAKED_HEADER_RE = re.compile(
    r"^[ \t]*(?:\[\d+\][ \t]+file=\S+.*|sources?[ \t]*:[ \t]*)$",
    re.MULTILINE | re.IGNORECASE,
)
# Same leak as _LEAKED_HEADER_RE, but glued mid-sentence instead of on its own
# line ("$297 billion. Source: file=doc_003.pdf") -- _LEAKED_HEADER_RE's ^...$
# anchors never match that, so it reached the user raw. Matches the same
# file=/chunk=/page=/score= field sequence as _HEADER_RE.
_INLINE_LEAKED_SOURCE_RE = re.compile(
    r"\s*Sources?\s*:\s*file=\S+"
    r"(?:\s+(?:chunk|sheet|part|repair_query|subquery)=\S+)?"
    r"(?:\s+page=\S+)?(?:\s+score=\S+)?",
    re.IGNORECASE,
)
# Same leak, cut short before the model ever wrote "file=..." -- a dangling
# "Source:"/"Sources:" label with nothing after it on that line. Only matches
# when nothing follows before end-of-line/string, so real prose like "the
# source: the annual report" (content on the same line) is left alone.
_DANGLING_SOURCE_LABEL_RE = re.compile(
    r"[ \t]*Sources?\s*:[ \t]*(?=\n|$)", re.IGNORECASE | re.MULTILINE
)
# Inline [N] citation markers — dropped because the answer's numbering does not
# correspond to the trace panel's, so they would be misleading. Only [N] within
# the retrieved-chunk range (1..MAX_TOOL_RESULTS) is treated as a citation — a
# bracketed year like [2024] or any larger number is left alone.
_INLINE_CITATION_RE = re.compile(r"([ \t]*)\[(\d+)\]")
_BLANK_LINES_RE = re.compile(r"\n{3,}")

# gpt-oss's "harmony" response format splits output into named channels
# (analysis = hidden chain-of-thought, final = the real answer); reproduced
# live via OpenRouter streaming that this channel boundary isn't always
# cleanly stripped before content reaches the client, leaking the bare
# channel-name word glued directly onto the real answer with no space
# ("1. final5239.0", "finalJuly 1 2024 ..."). Matched only when glued with no
# separating whitespace to an uppercase letter or digit -- genuine prose use
# of these words ("the final amount", "further analysis shows") always has a
# space after, so this can't strip real content.
_LEAKED_CHANNEL_MARKER_RE = re.compile(r"\b(?:analysis|final)(?=[A-Z0-9])")
# gpt-oss also emits citations as a dagger-glued marker naming the raw source
# file inside the brackets ("[3†file=doc_002_services_contract_terms.pdf]") --
# _INLINE_CITATION_RE's \[(\d+)\] never matches (a † follows the digit, not ]),
# so this used to leak the raw internal filename into user-facing answers and
# get unconditionally stripped. Reproduced live 2026-07-21: the marker's N is
# a REAL, resolvable citation index (same numbering space as plain [N]) --
# blind stripping was throwing away the model's only citation, leaving an
# answer with zero [N] markers and the UI falling back to showing every
# retrieved candidate as if cited. Resolve N via citation_map like [N] does;
# only strip when unresolvable (leading whitespace included).
_LEAKED_DAGGER_CITATION_RE = re.compile(r"([ \t]*)\[(\d+)†[^\]]*\]")
# The same source-file token left bare (not inside brackets), glued to the end of
# a sentence after a now-stripped "[N]" marker ("...investigation. file=doc_010.pdf")
# or written as a parenthetical ("(file=doc_010_...pdf)"). "file=" is never
# legitimate answer prose, so stripping it plus optional wrapping parens is safe.
_LEAKED_BARE_FILE_RE = re.compile(r"[ \t]*\(?file=\S+?\.\w+\)?")
# The model's own tool names ("calculate", "search_knowledge_base",
# "query_excel"), leaked inline as a bracketed tag instead of a real [N]
# citation -- reproduced live: "2,587.67, the monthly rent... [calculate]"
# for a value it derived by dividing a retrieved annual figure by 12. Not a
# citation (no digit inside, _ANSWER_CITATION_RE/_INLINE_CITATION_RE never
# match it), so it survives every existing strip pass and shows up as literal
# clutter in the answer. These tool names are never legitimate answer prose
# on their own in brackets, so stripping is content-safe.
_LEAKED_TOOL_NAME_TAG_RE = re.compile(
    r"[ \t]*\[(?:calculate|search_knowledge_base|query_excel)\]", re.IGNORECASE
)
# The literal placeholder "[N]" -- the prompt's own instructions use "[N]" as
# example citation syntax ("prefixed with [N] file=...", "cite the [N] of the
# passage..."), and the model sometimes echoes that literal template token
# instead of substituting a real retrieved index -- reproduced live:
# "$31,052.08, the annual rent for the first year of the lease[N]". Not a
# citation (no digit, _INLINE_CITATION_RE never matches "N"), so it survived
# every existing strip pass. Case-sensitive: a lowercase "[n]" is never the
# model referencing this convention.
_LEAKED_PLACEHOLDER_N_RE = re.compile(r"[ \t]*\[N\]")
_INLINE_CITATION_RE = re.compile(r"([ \t]*)\[(\d+)\]")
# gpt-oss occasionally never separates hidden chain-of-thought / a raw tool-call
# payload from the real answer at all -- reproduced live: a full raw
# "<|channel|>commentary<|message|>We need to query the sheet..." reasoning dump,
# and separately a bare '{"action": "search_knowledge_base", "arguments": {...}}'
# tool-call JSON object, each returned as the entire "answer". Neither can ever
# be legitimate answer content, so detecting either and forcing Unsupported is
# content-safe -- unlike _LEAKED_CHANNEL_MARKER_RE (which strips a stray glued
# marker from an otherwise-real answer), this is a full-generation failure.
# The two JSON-shape patterns below are intentionally NOT anchored to the
# start of the string -- reproduced live forcing the excel-modality hard
# block (FORCED_MODALITY): the model narrated its own next move in plain
# prose first ("We must call search_knowledge_base ... Let's call.") before
# the leaked JSON blob, so a "^" anchor that only caught a leak that WAS the
# entire generation missed it. No legitimate answer ever contains a raw
# {"action": ...}/{"tool": ...} tool-call object as a substring, wherever it
# sits in the text, so matching anywhere is still content-safe.
_MALFORMED_GENERATION_RE = re.compile(
    r"<\|(?:channel|message|start|end|constrain)\|>"
    r'|\{\s*"action"\s*:'
    # A third leaked-tool-call syntax, reproduced live 2026-07-21:
    # "[search_knowledge_base: topic=\"...\"]" / "[search_knowledge_base
    # query=\"...\"]" -- the model emitting its own tool invocation as visible
    # bracket-wrapped text instead of a real tool call. Never legitimate
    # answer content (no real answer starts with a raw tool name in brackets).
    r"|\[(?:search_knowledge_base|query_excel)\b"
    # A fourth variant, reproduced live 2026-07-22 (doc_015_food_sop_manual_qa__qa_13):
    # '{"tool": "search_knowledge_base", "parameters": {...}}' -- a different
    # key name than the already-caught "action" shape, same failure class (a
    # raw tool-call object emitted as visible text instead of a real call).
    r'|\{\s*"tool"\s*:'
    # A sixth variant, reproduced live forcing the excel-modality hard block:
    # narrated reasoning with no JSON/bracket artifact at all -- "We must
    # call search_knowledge_base with topic words alone to identify relevant
    # doc summary chunk." The JSON/bracket patterns above all assume some
    # structured leftover; this has none, so match the bare tool name
    # itself. A real user-facing answer never mentions these internal
    # function names -- no legitimate content about a lease, a policy, or a
    # spreadsheet total needs the words "search_knowledge_base" or
    # "query_excel" at all, so this is safe wherever it appears.
    r"|\bsearch_knowledge_base\b|\bquery_excel\b",
    re.IGNORECASE,
)


def _is_malformed_generation(text: str) -> bool:
    """True when `text` is raw leaked reasoning or a raw tool-call payload
    rather than a real answer (see _MALFORMED_GENERATION_RE)."""
    return bool(_MALFORMED_GENERATION_RE.search(text))


# Only the last few turns are kept -- enough for a follow-up pronoun ("that
# notice", "who else") to resolve, without growing the condense prompt or
# risking an unrelated earlier turn's document bleeding into routing.
_HISTORY_TURNS = 4

_CONDENSE_PROMPT = (
    "Rewrite the follow-up question as a standalone question, using the "
    "conversation so far to resolve any pronouns or implicit references. "
    "Keep it short and preserve its original meaning. If the follow-up is "
    "already standalone, return it unchanged. Reply with ONLY the rewritten "
    "question, no explanation.\n\n"
    "Conversation:\n{history}\n\n"
    "Follow-up: {question}\n"
    "Standalone question:"
)


def _condense_followup_question(
    question: str, history: list[dict] | None, agent: Any = None
) -> str:
    """Rewrite `question` into a standalone question using prior turns.

    No-op (returns `question` unchanged) when there's no history -- this
    keeps eval/run_eval.py, which never passes history, byte-for-byte
    unaffected. On any LLM failure or malformed/empty output, also falls
    back to the raw question rather than risk a broken rewrite.

    Uses the same generation model/endpoint the agent itself answers with
    (agent._generation_api_base / _generation_model, set in build_rag_agent)
    via the shared _llm_call helper, rather than a separately-configured
    "cheap" model -- CHUNK_LLM_MODEL is tuned for ingestion-time enrichment
    and isn't guaranteed to be a valid model on whatever endpoint is live.
    """
    if not history:
        return question
    api_base = getattr(agent, "_generation_api_base", None)
    model_name = getattr(agent, "_generation_model", None)
    if not api_base or not model_name:
        return question
    turns = history[-_HISTORY_TURNS:]
    transcript = "\n".join(
        f"Q: {t.get('question', '')}\nA: {t.get('answer', '')}" for t in turns
    )
    prompt = _CONDENSE_PROMPT.format(history=transcript, question=question)
    try:
        rewritten = _llm_call(
            prompt, api_base, model_name, max_tokens=200, temperature=0
        )
    except Exception:
        return question
    if not rewritten or _is_malformed_generation(rewritten):
        return question
    return rewritten


_BLANK_LINES_RE = re.compile(r"\n{3,}")


def _chunk_identity(raw: str) -> tuple[int, str, str] | None:
    """Parse one collected chunk's (marker_number, filename, location) identity,
    the same filename/location derivation parse_sources uses, so a citation_map
    built from this lines up with the final sources list's own (filename,
    location) pairs. Returns None for a malformed/headerless entry."""
    lines = raw.strip().splitlines()
    if not lines:
        return None
    m = _HEADER_RE.match(lines[0])
    if not m:
        return None
    marker_m = re.match(r"^\[?(\d+)\]?", lines[0])
    if not marker_m:
        return None
    filename = m.group("file")
    if filename.startswith("eval/data/raw/"):
        filename = filename[len("eval/data/raw/") :]
    filename = _resolve_original_name(filename)
    loc_key, loc_val = m.group("loc_key") or "", m.group("loc_val") or ""
    body = "\n".join(lines[1:]).strip()
    is_doc_summary = loc_key == "chunk" and loc_val == "-1"
    is_sheet_summary = loc_key == "sheet" and "Sheet summary:" in body[:200]
    if is_doc_summary:
        location = "document summary"
    elif is_sheet_summary:
        location = f"sheet summary: {loc_val}"
    elif loc_key == "chunk":
        location = f"chunk {loc_val}"
    elif loc_key == "sheet":
        location = f"sheet: {loc_val}"
    elif loc_key == "part":
        location = f"part {loc_val}"
    else:
        location = ""
    return int(marker_m.group(1)), filename, location


def build_citation_map(collected: list[str], sources: list[dict]) -> dict[int, int]:
    """Map every tool call's raw [N] marker numbers to their 1-based position
    in the final `sources` list, so inline [N] citations the model emits can be
    renumbered to match what the UI actually shows instead of being stripped.

    Every call is mapped directly by its raw marker, not just the last: raw
    numbering is now globally unique across calls within one answer (search_
    knowledge_base is renumbered at the tool boundary in src/rag_agent.py's
    _renumber_tool_markers, keyed by the same running total
    answer_comparison_deterministic already applies to its own independent
    calls) -- before this, every call restarted at [1], so the last call's
    markers silently overwrote any earlier call's identical numbers here.
    A marker whose source didn't survive into the final (deduped, capped)
    sources list is simply absent from the map, and falls back to the old
    strip-on-sight behavior in strip_leaked_headers.
    """
    citation_map: dict[int, int] = {}
    final_positions = {
        (s["filename"], s["location"]): i + 1 for i, s in enumerate(sources)
    }
    for raw in collected:
        if raw == _CALL_BOUNDARY:
            continue
        identity = _chunk_identity(raw)
        if identity is None:
            continue
        marker, filename, location = identity
        position = final_positions.get((filename, location))
        if position is not None:
            citation_map[marker] = position
    return citation_map


def _max_marker(collected: list[str]) -> int:
    """Highest raw [N] marker across every call in `collected`. Markers are
    now globally unique across calls within one answer (see
    build_citation_map's docstring), so a multi-call answer's true ceiling
    can legitimately exceed MAX_TOOL_RESULTS, the same reasoning
    answer_comparison_deterministic already applies via its own `marker - 1`
    -- callers should pass max(MAX_TOOL_RESULTS, this) as
    strip_leaked_headers' max_plausible_marker so a genuine high marker from
    a second call isn't mistaken for a literal number and left leaking, or
    conversely a hallucinated one isn't left un-stripped.
    """
    best = 0
    for raw in collected:
        if raw == _CALL_BOUNDARY:
            continue
        identity = _chunk_identity(raw)
        if identity is not None:
            best = max(best, identity[0])
    return best


def _strip_inline_citation(
    match: re.Match, citation_map: dict[int, int] | None, max_plausible_marker: int
) -> str:
    """Renumber a [N] marker to its real position in `sources` when resolvable,
    otherwise drop it (leading whitespace included) if N is a plausible (but
    unresolvable) retrieved-chunk index -- i.e. N was actually shown to the
    model as a real chunk marker, just didn't survive into the final (deduped,
    capped) sources list. `max_plausible_marker` is the true upper bound of
    marker numbers the model could have seen in THIS call: normally
    MAX_TOOL_RESULTS (a single tool call is capped there), but
    answer_comparison_deterministic renumbers markers across two independent
    per-document retrieval calls combined, so its real ceiling can legitimately
    exceed MAX_TOOL_RESULTS -- passing the global constant there caused a real,
    high (but genuine) marker like [15] to be misread as "probably a literal
    year in prose" and left leaking raw into the answer. Reproduced live
    2026-07-21."""
    leading, n = match.group(1), int(match.group(2))
    if citation_map and n in citation_map:
        return f"{leading}[{citation_map[n]}]"
    return "" if 1 <= n <= max_plausible_marker else match.group(0)


def _strip_dagger_citation(match: re.Match, citation_map: dict[int, int] | None) -> str:
    """Renumber a "[N†file=...]" dagger citation to its real position in
    `sources` when resolvable, otherwise drop it (leading whitespace
    included) -- unlike a plain [N], the dagger form is never legitimate
    prose (a bracketed year is never followed by "†file="), so an
    unresolvable marker is always a leak and always gets fully stripped,
    never left dangling in the answer."""
    leading, n = match.group(1), int(match.group(2))
    if citation_map and n in citation_map:
        return f"{leading}[{citation_map[n]}]"
    return ""


def strip_leaked_headers(
    text: str,
    citation_map: dict[int, int] | None = None,
    max_plausible_marker: int | None = None,
) -> str:
    """Remove raw chunk-header lines, leaked reasoning-channel markers, and
    inline [N] citation markers the LLM echoes into its answer.

    citation_map (see build_citation_map): when a [N] marker resolves to a
    real source, it's renumbered to that source's position instead of being
    stripped — true inline citations. Unresolvable markers are stripped as
    before (the trace panel/eval evidence is the source list).

    max_plausible_marker (see _strip_inline_citation): the true upper bound of
    marker numbers the model could have legitimately seen in this call.
    Defaults to MAX_TOOL_RESULTS (correct for a normal single-tool-call
    answer); answer_comparison_deterministic passes its own real per-call
    marker count instead, since it can legitimately exceed MAX_TOOL_RESULTS."""
    if max_plausible_marker is None:
        max_plausible_marker = MAX_TOOL_RESULTS
    # The model sometimes emits fullwidth brackets ("【1】") instead of ASCII
    # ("[1]") -- every citation regex downstream (here, renumbering, quote
    # narrowing, the frontend's chip rendering) is ASCII-only, so an
    # unnormalized fullwidth marker leaks into the answer as raw text and
    # never resolves to a real source. Normalize once, at the earliest point.
    text = text.translate(str.maketrans("【】", "[]"))
    cleaned = _LEAKED_HEADER_RE.sub("", text)
    cleaned = _INLINE_LEAKED_SOURCE_RE.sub("", cleaned)
    cleaned = _LEAKED_CHANNEL_MARKER_RE.sub("", cleaned)
    cleaned = _LEAKED_DAGGER_CITATION_RE.sub(
        lambda m: _strip_dagger_citation(m, citation_map), cleaned
    )
    cleaned = _LEAKED_BARE_FILE_RE.sub("", cleaned)
    cleaned = _LEAKED_TOOL_NAME_TAG_RE.sub("", cleaned)
    cleaned = _LEAKED_PLACEHOLDER_N_RE.sub("", cleaned)
    cleaned = _INLINE_CITATION_RE.sub(
        lambda m: _strip_inline_citation(m, citation_map, max_plausible_marker), cleaned
    )
    # Runs AFTER the [N] pass, not before: "Source: [1]" is only a leak when
    # its [N] didn't resolve to a real citation and got stripped above,
    # leaving a bare dangling "Source:" -- a RESOLVED citation ("Source: [2]")
    # is the model's own legitimate citation style and must survive.
    cleaned = _DANGLING_SOURCE_LABEL_RE.sub("", cleaned)
    cleaned = _BLANK_LINES_RE.sub("\n\n", cleaned)
    return cleaned.strip()


def _attach_excel_marker_if_missing(
    text: str, excel_citations: list[dict], sources: list[dict]
) -> str:
    """Attach a deterministic [N] marker when `text` cites nothing but was
    answered from Excel, so it isn't silently excluded from "Sources used".

    query_excel's result never enters the [N] chunk-marker stream
    search_knowledge_base's does (it's SQL output, not a numbered tool
    result), so the model has nothing to cite for a SQL-derived fact.
    excel_citations names the real source deterministically -- unlike a
    retrieved chunk's rank, which reflects relevance to the query, not
    what the answer was actually derived from, and is not a safe stand-in
    (reproduced live: the answer's own top-ranked chunk and the chunk it
    actually cited were different chunks; a comparison-question fix in
    this same file was needed for the reranker parking the real evidence
    at rank 3, not rank 1). Only ever attaches a marker backed by this
    specific, causal signal -- never a guess.
    """
    if not text or text.lower() == "unsupported" or _ANSWER_CITATION_RE.search(text):
        return text
    if not excel_citations:
        return text
    citation = excel_citations[0]
    key = (
        _resolve_original_name(citation.get("source_file") or ""),
        citation.get("sheet_name"),
    )
    position = next(
        (
            i + 1
            for i, s in enumerate(sources)
            if (s["filename"], s.get("sheet")) == key
        ),
        None,
    )
    return f"{text.rstrip()} [{position}]" if position is not None else text


def parse_sources(collected: list[str]) -> list[dict]:
    """Parse collected tool chunks into source cards.

    Groups chunks by tool call (boundaries marked by `---CALL_BOUNDARY---`) and
    iterates calls in REVERSE order — later calls are typically scoped/refined
    and contain the answer-bearing chunks the LLM actually used. This prevents
    the first broad call's noise chunks from monopolizing the displayed list.
    """
    calls: list[list[str]] = [[]]
    for raw in collected:
        if raw == _CALL_BOUNDARY:
            calls.append([])
        else:
            calls[-1].append(raw)
    calls = [c for c in calls if c]

    seen: set[tuple[str, str]] = set()
    sources: list[dict] = []
    for call_chunks in reversed(calls):
        for raw in call_chunks:
            lines = raw.strip().splitlines()
            if not lines:
                continue
            m = _HEADER_RE.match(lines[0])
            if not m:
                # Not a real "[N] file=..." chunk at all -- a tool's own
                # plain error/no-results message, e.g. "No relevant results
                # found for query X", ended up in `collected` (see
                # stream_agent's regex-split fallback for unformatted tool
                # output). Reproduced live 2026-07-21: this used to become a
                # fabricated "unknown"-filename source card. It was never a
                # real chunk, so it must never become a citation.
                continue
            filename = m.group("file")
            loc_key = m.group("loc_key") or ""
            loc_val = m.group("loc_val") or ""
            score_str = m.group("score")
            page_str = m.group("page")
            page = int(page_str) if page_str and page_str.isdigit() else None

            doc_meta_m = _DOC_META_RE.search(lines[0])
            document_id = (
                doc_meta_m.group("doc_id")
                if doc_meta_m and doc_meta_m.group("doc_id") != "none"
                else None
            )
            document_title = (
                unquote(doc_meta_m.group("title"))
                if doc_meta_m and doc_meta_m.group("title") != "none"
                else None
            )

            body_lines = lines[1:] if len(lines) > 1 else []
            body = "\n".join(body_lines).strip()
            this_chunk_m = _THIS_CHUNK_RE.search(body)
            if this_chunk_m:
                body = this_chunk_m.group(1).strip()

            if filename.startswith("eval/data/raw/"):
                filename = filename[len("eval/data/raw/") :]
            filename = _resolve_original_name(filename)

            heading_m = _MD_HEADING_RE.search(body[:600])
            if heading_m:
                section = heading_m.group(1).strip()
            elif loc_key == "sheet":
                section = loc_val
            else:
                section = ""

            is_doc_summary = loc_key == "chunk" and loc_val == "-1"
            is_sheet_summary = loc_key == "sheet" and "Sheet summary:" in body[:200]

            if is_doc_summary:
                location = "document summary"
            elif is_sheet_summary:
                location = f"sheet summary: {loc_val}"
            elif loc_key == "chunk":
                location = f"chunk {loc_val}"
            elif loc_key == "sheet":
                location = f"sheet: {loc_val}"
            elif loc_key == "part":
                location = f"part {loc_val}"
            else:
                location = ""

            # Only surface a figure crop when the figure IS the chunk (e.g. a
            # standalone chart/image chunk) -- a page header logo that happens
            # to share a text chunk with unrelated prose would otherwise show
            # up as an irrelevant crop next to a citation about that prose.
            # This 40%-of-body heuristic isn't sufficient on its own though:
            # reproduced live, a short chunk pairing real answer text
            # ("Authorizing Manager: Ricki Contreras...") with a verbose logo
            # description (LACERA's letterhead image) let the logo's own text
            # dominate the chunk's length, passing this gate for a citation
            # whose actual claim has nothing to do with the figure. The
            # figure's own description text is kept alongside the bbox (under
            # a leading-underscore key, stripped before this dict is returned)
            # so _narrow_quotes_to_answer can null the bbox out when the
            # figure isn't actually what the answer drew from.
            figure_block_m = _FIGURE_BLOCK_RE.search(body)
            bbox_m = _FIGURE_BBOX_RE.search(body)
            figure_bbox = (
                [float(v) for v in bbox_m.group(1).split(",")]
                if bbox_m
                and figure_block_m
                and len(figure_block_m.group(0)) > 0.4 * len(body)
                else None
            )
            figure_text = (
                _FIGURE_BBOX_RE.sub("", figure_block_m.group(0))
                .replace("[FIGURE_START]", "")
                .replace("[FIGURE_END]", "")
                .strip()
                if figure_bbox is not None
                else ""
            )

            # Scanned-page (unstructured/CPU OCR) chunks carry a bbox comment
            # before each OCR'd element -- a chunk usually spans several
            # elements (a heading, several clauses), so the first one isn't
            # necessarily anywhere near whichever sentence the answer
            # actually cites. Keep every (bbox, text) segment here; once
            # _narrow_quotes_to_answer knows the real cited sentence it picks
            # the best-overlapping segment's bbox instead of just the first.
            ocr_split = _OCR_BBOX_RE.split(body)
            ocr_segments = (
                [
                    ([float(v) for v in ocr_split[i].split(",")], ocr_split[i + 1])
                    for i in range(1, len(ocr_split), 2)
                ]
                if len(ocr_split) > 1
                else []
            )
            ocr_bbox = ocr_segments[0][0] if ocr_segments else None

            plain = _FIGURE_BLOCK_RE.sub("", body)
            plain = _OCR_BBOX_RE.sub("", plain)
            # A chunk boundary can split a figure block so only [FIGURE_END]
            # (no matching [FIGURE_START]) lands in this chunk's body -- the
            # paired regex above doesn't match an orphan, so it leaks through
            # as literal visible text. Strip any leftover lone tag too.
            plain = re.sub(r"\[FIGURE_START\]|\[FIGURE_END\]", "", plain)
            plain = _TABLE_MARKER_RE.sub("", plain)
            # Page-boundary markers ("<!-- PAGE 6 pymupdf4llm -->", or the
            # inspector's "<!-- PAGE N | label -->") are pipeline bookkeeping,
            # not text on the page -- left in, they show up as literal HTML
            # comments in the Evidence panel's quote.
            plain = re.sub(r"<!--\s*PAGE\s+\d+.*?-->", "", plain, flags=re.IGNORECASE)
            # Strip only the "#" markup, not the heading's own text -- reproduced
            # live: pymupdf4llm sometimes renders an actual answer-bearing field
            # ("## **Authorizing Manager: Ricki Contreras...**") as a markdown
            # heading, not just navigational section titles. Deleting the whole
            # line dropped the one sentence that answered the question from the
            # displayed quote, even though the answer itself cited it -- the
            # heading text still ends up in `section` too (see heading_m above),
            # which is fine, it's the same sentence appearing in both places.
            plain = re.sub(r"^#{1,3}\s+", "", plain, flags=re.MULTILINE).strip()
            plain = re.sub(r"<br\s*/?>", " ", plain, flags=re.IGNORECASE)
            # "**bold**" is pymupdf4llm's own markdown, not text in the PDF --
            # left in, it shows literal asterisks in the citation quote and
            # breaks fitz.search_for's exact-text PDF highlight lookup.
            plain = plain.replace("**", "")
            # A chunk can itself be a raw markdown table (pymupdf4llm's own
            # rendering of a PDF table, not wrapped in TABLE_START/END) --
            # drop separator rows and pipe characters so the quote reads as
            # prose instead of leaking "|---|---|" table syntax.
            plain = re.sub(r"^\s*\|?[-:\s|]+\|?\s*$", "", plain, flags=re.MULTILINE)
            plain = plain.replace("|", " ")
            # Capped generously here, not to 350 -- a duplicated-text artifact
            # in a chunk (a repeated preamble phrase, seen in practice) can
            # make the answer-bearing sentence land well past 350 chars.
            # Truncating that early would drop it before _narrow_quotes_to_answer
            # ever gets a chance to select it; the real 350-char display cap is
            # applied there, after narrowing, to whatever text was actually kept.
            excerpt = _truncate_at_sentence_boundary(" ".join(plain.split()), 2000)

            score = float(score_str) if score_str else None
            # 1.0 is the retriever's placeholder for filter/scroll fetches
            # (doc-routed chunks carry no similarity score) — drop it so the UI
            # doesn't show a misleading "perfect match" chip.
            if score == 1.0:
                score = None

            # Dedupe on (filename, location) when location is known (chunk/
            # sheet/part) so two chunks from the same file+page/sheet never
            # both survive as separate-looking "duplicate" source rows just
            # because their excerpts differ slightly (e.g. neighbour-window
            # padding) -- location is also citation_map's own identity key,
            # so this keeps that mapping unambiguous too. Falls back to the
            # excerpt for loc_keys without a location (repair_query/subquery),
            # where "" would otherwise over-collapse distinct passages.
            key = (filename, location or excerpt[:80])
            if key in seen:
                continue
            seen.add(key)
            chunk_id = (
                _stable_id(m.group("file"), loc_val)
                if m and loc_key == "chunk"
                else None
            )
            sources.append(
                {
                    "filename": filename,
                    "document_id": document_id,
                    "document_title": document_title or filename,
                    "section": section,
                    "location": location,
                    "page": page,
                    "sheet": loc_val if loc_key == "sheet" else None,
                    "excerpt": excerpt,
                    "quote": excerpt,
                    "chunk_id": chunk_id,
                    "score": round(score, 4) if score else None,
                    "figure_bbox": figure_bbox,
                    "_figure_text": figure_text,
                    "ocr_bbox": ocr_bbox,
                    "_ocr_segments": ocr_segments,
                }
            )
    # Cap at 8, allocated round-robin by filename (each file's rank-1, then
    # each file's rank-2, ...) so the cap is shared fairly across documents --
    # reproduced directly: with "one guaranteed slot per file, then fill in
    # list order" (the previous scheme), a two-document comparison emitting
    # 12 chunks per document kept 7 of the first document's chunks and only
    # the second document's own rank-1 chunk, discarding every other
    # genuinely retrieved chunk from that document -- e.g. its rank-2 chunk,
    # even when that's the one the answer actually cited. Round-robin still
    # keeps the original guarantee this replaces (every file gets its rank-1
    # slot before any file gets a rank-2 -- the fix for a redundant re-query
    # of an already-answered document crowding out a different document
    # entirely), it just no longer stops there: each file gets an even share
    # up to the cap, preserving each file's internal rank order. A
    # single-file `sources` list is unaffected (identical order), since
    # round-robin over one group is just that group in order.
    by_file: dict[str, list[dict]] = {}
    for s in sources:
        by_file.setdefault(s["filename"], []).append(s)
    interleaved: list[dict] = []
    for chunks in itertools.zip_longest(*by_file.values()):
        interleaved.extend(c for c in chunks if c is not None)
    return interleaved[:8]


# Splits after sentence punctuation, and also before a definitions-list term
# like "Amendment: " so a preamble ("...construed to have the following
# meaning:") doesn't stay glued to the actual definition it introduces --
# there's no period between them, just a colon run-on. The third alternative
# handles a quoted-term glossary ("Purchase" means ...; "Term" means ...) --
# a source PDF table (e.g. doc_002's Appendix C definitions) collapses to
# semicolon-joined prose once parse_sources strips its "|" table syntax, with
# no periods at all between entries. Without this, the whole glossary reads
# as one unsplittable "sentence" and _narrow_quotes_to_answer's oversized-
# sentence fallback keeps it whole -- reproduced live: a "Term" citation
# showed the entire multi-definition block instead of just the Term clause.
_SENTENCE_SPLIT_RE = re.compile(
    r'(?<=[.!?])\s+|\s+(?=[A-Z][a-zA-Z]{2,24}:\s)|(?<=;)\s+(?=[“"])'
)
_WORD_RE = re.compile(r"[a-z0-9]+")
# Generic connective words, excluded from _narrow_quotes_to_answer's overlap
# count -- reproduced live: a contract's glossary preamble sentence shared 5
# words with the answer ("the", "with", "term", "services", "terms") purely
# by genericness, clearing the overlap>=3 rescue threshold and getting kept
# alongside the real answer-bearing "Term" definition sentence. Since `kept`
# joins in original order and the preamble sentence came first, the 350-char
# display cap then cut the citation off mid-preamble, dropping the real
# definition entirely. Stripping these before counting leaves only content
# words, so overlap reflects topical relevance, not shared filler.
_STOPWORDS = {
    "the",
    "and",
    "or",
    "of",
    "in",
    "to",
    "for",
    "as",
    "by",
    "is",
    "are",
    "was",
    "were",
    "has",
    "have",
    "had",
    "with",
    "from",
    "that",
    "this",
    "these",
    "those",
    "may",
    "shall",
    "such",
    "any",
    "all",
    "not",
    "but",
    "means",
    "term",
    "terms",
}
_ANSWER_CITATION_RE = re.compile(r"\[(\d+)\]")


def _renumber_citations_sequentially(
    answer: str, sources: list[dict]
) -> tuple[str, list[dict]]:
    """Renumber [N] markers (and reorder `sources` to match) so cited sources
    always read 1, 2, 3... in order of first appearance.

    citation_map numbers a marker by its raw position in the full retrieved
    sources list -- e.g. an answer that cites only one source out of 8
    retrieved can show up as "[7]", which reads as arbitrary/confusing in the
    UI ("why 7, not 1?"). Uncited sources keep trailing after the cited ones
    in their existing relative order (still shown as "N more retrieved
    candidates").
    """
    seen_order: list[int] = []
    for m in _ANSWER_CITATION_RE.finditer(answer):
        n = int(m.group(1))
        if 1 <= n <= len(sources) and n not in seen_order:
            seen_order.append(n)
    if not seen_order or seen_order == list(range(1, len(seen_order) + 1)):
        return answer, sources
    remap = {old: new for new, old in enumerate(seen_order, start=1)}
    new_answer = _ANSWER_CITATION_RE.sub(
        lambda m: (
            f"[{remap[int(m.group(1))]}]" if int(m.group(1)) in remap else m.group(0)
        ),
        answer,
    )
    cited = [sources[old - 1] for old in seen_order]
    uncited = [s for i, s in enumerate(sources, start=1) if i not in remap]
    return new_answer, cited + uncited


def _truncate_at_sentence_boundary(text: str, limit: int) -> str:
    """Cut `text` to about `limit` chars without severing a sentence mid-word.

    A raw text[:limit] slice can stop the excerpt (and, downstream, the PDF
    highlight search text) in the middle of the one sentence the answer
    actually cites -- accumulate whole sentences up to the budget instead;
    if even the first sentence alone exceeds it, keep it whole anyway (a
    slightly long excerpt beats a mid-sentence stub).
    """
    if len(text) <= limit:
        return text
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    kept: list[str] = []
    total = 0
    for sentence in sentences:
        if kept and total + len(sentence) + 1 > limit:
            break
        kept.append(sentence)
        total += len(sentence) + 1
    return " ".join(kept) if kept else text[:limit]


# Minimum word-overlap a CITED source's best sentence must clear, or its
# [N] marker gets stripped from the answer entirely (see _narrow_quotes_to_answer).
# Calibrated live against two runs of the same cross-doc question: a correctly-
# cited "Term" definition scored 6, a correctly-cited "Sole Source" clause
# scored 5, but a wrongly-cited VAT/invoice clause -- attributed to a claim
# about contract extension periods it has nothing to do with -- scored 2.
# Same threshold as the existing overlap>=3 rescue below, reused rather than
# inventing a second magic number.
_CITATION_OVERLAP_FLOOR = 3


def _narrow_quotes_to_answer(sources: list[dict], answer: str) -> str:
    """Trim each source's quote/excerpt down to only the sentence(s) that
    actually overlap the final answer text, in place, and return `answer`
    with any citation markers that point at an off-topic source stripped.

    A retrieved chunk is the retrieval unit, not the attribution unit -- it
    often carries surrounding context (a preceding "Whenever the following
    words appear..." preamble, neighbour-chunk padding) that helped the
    chunk get retrieved but isn't what the answer actually drew from. Left
    in, that preamble both misleads the Evidence panel's blockquote and
    makes the PDF-highlight endpoint search for a multi-sentence string that
    frequently fails to match the real text layout, degrading to a partial
    fragment. Falls back to the untouched excerpt when nothing overlaps
    (e.g. a paraphrased/abstractive answer) -- a full-chunk highlight beats
    guessing at fake precision.

    Sources are already renumbered (see _renumber_citations_sequentially)
    before this runs, so a source's 1-based position in `sources` IS its
    [N] marker -- reproduced live: the model sometimes cites a real [N] but
    points it at the wrong retrieved chunk entirely (a VAT/invoice clause
    cited as if it supported a contract-extension-period claim). No amount
    of quote-narrowing fixes that -- the chunk itself is off-topic. When a
    cited source's best sentence can't clear _CITATION_OVERLAP_FLOOR, strip
    that marker so the answer degrades to unpinned (the Evidence panel's
    empty state) instead of showing confidently wrong evidence.
    """
    cited_markers = {int(n) for n in _ANSWER_CITATION_RE.findall(answer)}
    # A token is kept when it is long enough to be a content word OR contains
    # a digit -- a short figure ("42", "97") is the most specific thing an
    # answer can share with its source, and the >2 length filter alone
    # discarded it before any overlap was measured.
    answer_words = {
        w
        for w in _WORD_RE.findall(answer.lower())
        if (len(w) > 2 or any(ch.isdigit() for ch in w)) and w not in _STOPWORDS
    }
    for position, source in enumerate(sources, start=1):
        excerpt = source["excerpt"]
        final = excerpt
        best_sentence_overlap = 0
        if answer_words:
            sentences = [
                s.strip() for s in _SENTENCE_SPLIT_RE.split(excerpt) if s.strip()
            ]
            kept: list[tuple[int, str]] = []
            for sentence in sentences:
                words = {
                    w
                    for w in _WORD_RE.findall(sentence.lower())
                    if (len(w) > 2 or any(ch.isdigit() for ch in w))
                    and w not in _STOPWORDS
                }
                if not words:
                    continue
                shared = words & answer_words
                overlap = len(shared)
                # A shared NUMBER counts for the whole floor. A terse numeric
                # answer ("$297 billion", "42") has too few content words to
                # ever reach a 3-word overlap, so the floor below stripped the
                # citation off exactly the answers whose evidence is most
                # specific -- reproduced by test_each_parts_citation_survives.
                # A figure matching between answer and source sentence is
                # stronger attribution evidence than three shared prose words,
                # not weaker.
                if any(any(ch.isdigit() for ch in w) for w in shared):
                    overlap = max(overlap, _CITATION_OVERLAP_FLOOR)
                best_sentence_overlap = max(best_sentence_overlap, overlap)
                # Ratio alone punishes a long, correct sentence: reproduced live, a
                # numbered clause's heading ("2. Term.") scored ratio=1.0 (its one
                # real word, "term", is in the answer) and got kept, while the
                # actual answer-bearing sentence right after it ("The term of this
                # Lease shall commence on July 1, 2020... and end on June 30,
                # 2024...") scored ~35% -- diluted by legitimate extra detail
                # (dates, "commencement", "unless terminated earlier") -- and got
                # dropped, leaving the citation quote as the single word "Term."
                # A minimum absolute overlap count catches that case regardless of
                # sentence length, without loosening the ratio gate for short
                # sentences (which still need the ratio to avoid keeping every
                # sentence that happens to share one generic word).
                if overlap / len(words) >= 0.5 or overlap >= 3:
                    kept.append((overlap, sentence))
            if len(sentences) > 1 and kept:
                # Highest-overlap sentence first, and budget the 350-char cap
                # directly off that ranking -- reproduced live: a marginal
                # rescue-threshold sentence (a glossary's opening line,
                # overlap=4) sorted ahead of the real answer-bearing sentence
                # (overlap=8) in original document order. Joining in that
                # order and re-truncating via _truncate_at_sentence_boundary
                # (below) doesn't fix it either: that function re-splits the
                # joined text with the same sentence regex, and two sentences
                # reordered out of their natural document sequence often have
                # no valid split boundary between them (no period, no
                # semicolon-plus-quote) -- so it degrades to "keep the whole
                # oversized blob," silently dragging the irrelevant sentence
                # back in at full length. Building `final` directly from the
                # ranked list, greedily within budget, sidesteps that re-split
                # entirely.
                kept.sort(key=lambda pair: pair[0], reverse=True)
                budgeted: list[str] = []
                total_len = 0
                for _, sentence in kept:
                    if budgeted and total_len + len(sentence) + 1 > 350:
                        continue
                    budgeted.append(sentence)
                    total_len += len(sentence) + 1
                final = " ".join(budgeted)
        final = _truncate_at_sentence_boundary(final, 350)
        source["excerpt"] = final
        source["quote"] = final

        # A SQL-derived citation is exempt from the word-overlap floor: its
        # source is a spreadsheet row ("NET Amount=0.53") and the answer is
        # the bare value, so there is no prose to overlap -- the floor
        # deleted every Excel marker, silently undoing the deterministic
        # attach _attach_excel_marker_if_missing exists to provide. The
        # floor guards against a MODEL-emitted marker pointing at unrelated
        # content; these markers are built from query_excel's own
        # name-checked citation, so that risk doesn't apply.
        sql_derived = source.pop("_sql_derived", False)
        if (
            position in cited_markers
            and not sql_derived
            and best_sentence_overlap < _CITATION_OVERLAP_FLOOR
        ):
            answer = re.sub(rf"\[{position}\](?!\d)", "", answer)

        # A figure crop is only real evidence for THIS citation if the
        # figure's own description is what the answer actually drew from --
        # reproduced live: a chunk pairing real answer text with a verbose
        # logo description passed parse_sources' "figure is >40% of the
        # chunk" gate even though the answer had nothing to do with the
        # logo. Null the bbox out here, where the answer text is actually
        # known, unless the figure's own words meaningfully overlap it.
        figure_text = source.pop("_figure_text", "")
        if source.get("figure_bbox") is not None and answer_words:
            figure_words = {
                w for w in _WORD_RE.findall(figure_text.lower()) if len(w) > 2
            }
            if (
                not figure_words
                or len(figure_words & answer_words) / len(figure_words) < 0.3
            ):
                source["figure_bbox"] = None

        # Re-point the OCR crop at whichever element's own text actually
        # overlaps the answer, instead of leaving it on the chunk's first
        # element (parse_sources' default) which is often just a heading.
        ocr_segments = source.pop("_ocr_segments", None)
        if ocr_segments and answer_words:
            best_bbox, best_overlap = None, 0
            for bbox, text in ocr_segments:
                words = {w for w in _WORD_RE.findall(text.lower()) if len(w) > 2}
                overlap = len(words & answer_words)
                if overlap > best_overlap:
                    best_overlap, best_bbox = overlap, bbox
            if best_bbox is not None:
                source["ocr_bbox"] = best_bbox
        source["quote"] = final
    return answer


def _excel_citations_to_sources(
    citations: list[dict], existing: list[dict]
) -> list[dict]:
    """Convert query_excel's {source_file, sheet_name, quote} citations into
    source cards, in the same shape parse_sources produces.

    query_excel never emits retrieval chunks (see run_once) so these are the
    only sources a SQL-answered question ever gets — without this, SpreadsheetEvidence
    has nothing to render and the question shows no evidence at all. Skips a
    citation whose filename+sheet is already covered by a parsed source.
    """
    seen = {(s["filename"], s.get("sheet")) for s in existing}
    out: list[dict] = []
    for c in citations:
        source_file = c.get("source_file")
        sheet_name = c.get("sheet_name")
        if not source_file or not sheet_name:
            continue
        filename = _resolve_original_name(source_file)
        key = (filename, sheet_name)
        if key in seen:
            continue
        seen.add(key)
        quote = (c.get("quote") or "").strip()
        out.append(
            {
                "filename": filename,
                "document_id": None,
                "document_title": filename,
                "section": "",
                "location": f"sheet: {sheet_name}",
                "page": None,
                "sheet": sheet_name,
                "excerpt": quote[:350],
                "quote": quote[:350],
                "chunk_id": None,
                "score": None,
                "figure_bbox": None,
                "ocr_bbox": None,
                "is_aggregate": bool(c.get("is_aggregate")),
                # Marks a citation built deterministically from query_excel's
                # own name-checked excel_citations, not emitted by the model.
                # Stripped before the dict leaves _narrow_quotes_to_answer,
                # same convention as _figure_text/_ocr_segments.
                "_sql_derived": True,
            }
        )
    return out


def run_once(
    agent: Any,
    question: str,
    attempt: str = "initial",
    trace: Any = None,
    skip_grounding_check: bool = False,
    usage: dict | None = None,
) -> tuple[str, list[str], dict]:
    """Run the agent once for a question; return (answer, collected chunks, trace dict).

    Emits one Langfuse span per tool call actually made (name + retrieved
    chunk group, or the SQL for query_excel calls) plus one span for the
    attempt itself, so retries show up as distinct, inspectable steps.

    skip_grounding_check: passed through to stream_agent — set for comparison
    questions, whose own doc-coverage retry (see answer_one) already checks
    what matters here without the grounding judge's false-positive risk on
    comparative claims (see stream_agent's docstring).

    usage: if provided, token counts from this attempt are summed into it in
    place (see stream_agent's usage param).
    """
    collected: list[str] = []
    tokens: list[str] = []
    sql_trace: list[str] = []
    tool_calls: list[str] = []
    rejected: list[dict] = []
    excel_citations: list[dict] = []
    # query_excel's column-match gate reads this to check against what was
    # actually asked, not whatever text the agent's own internal retry loop
    # rewrites into a later query_excel call — see ORIGINAL_QUESTION's
    # docstring (src/tools/excel.py) for the reproduced case.
    question_token = ORIGINAL_QUESTION.set(question)
    try:
        for token in stream_agent(
            agent,
            question,
            collected_chunks=collected,
            sql_trace=sql_trace,
            tool_calls=tool_calls,
            rejected_chunks=rejected,
            excel_citations=excel_citations,
            trace=trace,
            skip_grounding_check=skip_grounding_check,
            usage=usage,
        ):
            tokens.append(token)
    finally:
        ORIGINAL_QUESTION.reset(question_token)
    answer = "".join(tokens).strip()
    trace_holder = {
        "sql": sql_trace,
        "tools": tool_calls,
        "rejected": rejected,
        "excel_citations": excel_citations,
    }

    if trace is not None:
        # collected_chunks only gets a "---CALL_BOUNDARY---" per non-excel
        # tool call (query_excel's result is SQL, not chunks — see
        # stream_agent's collected_chunks guard) — so only zip against the
        # non-excel names, or an excel+search mix silently mislabels spans.
        groups = [
            g.strip()
            for g in "\n\n".join(collected).split("---CALL_BOUNDARY---")
            if g.strip()
        ]
        retrieval_names = [t for t in tool_calls if t != "query_excel"]
        for name, group in zip(retrieval_names, groups):
            trace.span(name=name, input={"question": question}, output=group[:2000])
        for sql in sql_trace:
            trace.span(
                name="query_excel", input={"question": question}, output=sql[:2000]
            )
        trace.span(name=f"attempt:{attempt}", input=question, output=answer)

    return answer, collected, trace_holder


_TABLE_EXTS = (".xlsx", ".xls", ".csv")


def answer_one(
    agent: Any,
    question: str,
    trace: Any = None,
    forced_doc_id: str | list[str] | None = None,
    usage: dict | None = None,
    resolved_doc_ids: list[str] | None = None,
) -> tuple[str, list[str], dict]:
    """Answer one question, with a forced retry on a bare Unsupported.

    First resolves the tool deterministically: route_question matches the
    question against document summaries in Qdrant and, if it lands on a
    spreadsheet vs a text document, a routing directive is prepended so the
    agent uses query_excel vs search_knowledge_base accordingly — instead of
    guessing the tool from question wording.

    forced_doc_id: when the UI's source-scope control names a specific
    document, route directly to it instead of semantic auto-detection —
    the user already told us which document, no need to guess.

    resolved_doc_ids: the comparison doc ids answer_query already resolved
    with the shared M-6 resolver, reused here (M-7) to widen the
    doc-coverage retry trigger past literal "doc_XXX" text in the question —
    real users rarely type doc ids, so without this the coverage check below
    was silently a no-op on almost every natural-language comparison.

    Groq inference at temp=0 still has small nondeterminism; the agent
    occasionally skips doc-routing on the first attempt and returns
    Unsupported despite the answer existing. The retry forces the protocol.

    A second, separate retry covers a different failure: on a comparison
    question, the agent sometimes retrieves only one of the two required
    sources and answers the other half from general knowledge instead of
    making the second tool call. Detected after the fact by checking how
    many distinct source files the retrieved chunks actually span.
    """
    is_comparison = bool(_COMPARISON_RE.search(question))
    is_forced_doc_modality = None
    if isinstance(forced_doc_id, list):
        # Multi-select source scope: always text documents (query_excel has no
        # multi-source scoping), and there's no single filename to name in the
        # directive, so nudge generically — the hard tool-layer enforcement below
        # is what actually constrains retrieval, not this prompt text.
        is_forced_doc_modality = "document"
        route = {"modality": "document", "source_file": "the selected documents"}
    elif forced_doc_id:
        is_forced_doc_modality = (
            "excel" if forced_doc_id.lower().endswith(_TABLE_EXTS) else "document"
        )
        route = {"modality": is_forced_doc_modality, "source_file": forced_doc_id}
    else:
        route = {} if is_comparison else route_question(question)
    q = routing_directive(route) + question

    # The routing directive above is only a prompt nudge -- verified unreliable
    # on its own (the model sometimes calls search_knowledge_base on a
    # different document despite being told which one to use, or on an excel
    # question, calls search_knowledge_base at all instead of query_excel).
    # Hard-enforce it at the tool layer instead: for document modality,
    # search_knowledge_base gets redirected to the right doc_id; for excel
    # modality it has no such redirect target (different tool entirely), so
    # it refuses outright and names query_excel (see FORCED_MODALITY).
    token = None
    modality_token = None
    if is_forced_doc_modality == "document":
        token = FORCED_DOC_ID.set(forced_doc_id)
    if route.get("modality") == "excel":
        modality_token = FORCED_MODALITY.set("excel")
    try:
        ans, coll, tr = run_once(
            agent,
            q,
            attempt="initial",
            trace=trace,
            skip_grounding_check=is_comparison,
            usage=usage,
        )
        if is_comparison:
            # One unified check-and-retry pass. Reproduced live: a flat
            # "Unsupported" first attempt used to be caught by an earlier,
            # separate check that returned the OLD retry's result immediately
            # -- before the partial/missing-doc checks below ever ran on it,
            # so an incomplete retry answer (e.g. still missing one named
            # document) went out unexamined. All three failure signals now
            # feed into the same accept-or-keep-original decision.
            is_flat_unsupported = ans.strip().lower() == "unsupported"
            missing, has_partial, n_sources = _comparison_incompleteness(
                question, ans, coll, resolved_doc_ids
            )
            if is_flat_unsupported or missing or n_sources < 2 or has_partial:
                if missing:
                    retry_instruction = (
                        f"\n\nIMPORTANT: This is a retry. This comparison is about "
                        f"{', '.join(missing)}, but the previous answer has no evidence "
                        f"from {'it' if len(missing) == 1 else 'them'}. Make a "
                        f"search_knowledge_base call scoped to {' and '.join(missing)} "
                        "(pass it as the doc_id argument). If its relevant content covers "
                        "several related sub-topics, pick the most directly relevant one "
                        "and answer using it instead of skipping that document."
                    )
                elif is_flat_unsupported or n_sources < 2:
                    retry_instruction = _COMPARISON_RETRY_INSTRUCTION
                else:
                    retry_instruction = _COMPARISON_PARTIAL_RETRY_INSTRUCTION
                r_ans, r_coll, r_tr = run_once(
                    agent,
                    q + retry_instruction,
                    attempt="comparison-retry",
                    trace=trace,
                    skip_grounding_check=True,
                    usage=usage,
                )
                r_flat = r_ans.strip().lower() == "unsupported"
                r_missing, r_has_partial, r_sources = _comparison_incompleteness(
                    question, r_ans, r_coll, resolved_doc_ids
                )
                improved = (
                    (is_flat_unsupported and not r_flat)
                    or len(r_missing) < len(missing)
                    or r_sources > n_sources
                    or (has_partial and not r_has_partial)
                )
                if r_ans and improved:
                    return r_ans, r_coll, r_tr
            return ans, coll, tr
        # An empty answer (e.g. the entire generation was a hidden reasoning
        # channel that never reached "final" -- see _strip_channels in
        # rag_agent.py) is strictly worse than a bare "Unsupported" and gets
        # the same forced retry.
        #
        # skip_grounding_check=True: reproduced live -- the grounding judge
        # itself is flaky (~1 in 3-4 calls) and can independently downgrade a
        # correct retry answer the same way it downgraded the original, which
        # defeats the retry entirely (both attempts nuked, correct answer
        # never reaches the user despite correct evidence). Same rationale as
        # the comparison-retry above; the retry is already a stricter re-ask
        # via _RETRY_INSTRUCTION and still passes through _normalize_final and
        # answer_query's own no-sources-forces-Unsupported guard.
        if not ans.strip() or ans.lower() == "unsupported":
            r_ans, r_coll, r_tr = run_once(
                agent,
                q + _RETRY_INSTRUCTION,
                attempt="unsupported-retry",
                trace=trace,
                skip_grounding_check=True,
                usage=usage,
            )
            if r_ans.strip() and r_ans.lower() != "unsupported":
                return r_ans, r_coll, r_tr
            return ans if ans.strip() else "Unsupported", coll, tr
    finally:
        if token is not None:
            FORCED_DOC_ID.reset(token)
        if modality_token is not None:
            FORCED_MODALITY.reset(modality_token)
    return ans, coll, tr


def answer_query(
    agent: Any,
    question: str,
    trace: Any = None,
    forced_doc_id: str | list[str] | None = None,
    history: list[dict] | None = None,
) -> dict:
    """Full answer pipeline: split, answer each part, merge, and format sources.

    Multi-part questions are split and answered one sub-question at a time,
    then merged here deterministically. The agent's single-pass synthesis
    intermittently drops a part, so we never rely on it for that — each
    sub-question runs on its own and every part is guaranteed in the output.

    Title questions ("what is the title of...") are answered directly from
    a document_summary chunk's Title: line, bypassing the agent entirely —
    see _title_shortcut_answer. Skipped when forced_doc_id is set, since that
    shortcut's own retrieval isn't scoped to the requested document.

    forced_doc_id: the UI's source-scope control, when the user picked "one
    document" instead of "all sources" — see answer_one.

    history: prior {"question", "answer"} turns from this chat, if any. When
    given, a follow-up like "who must that notice be given to?" is rewritten
    into a standalone question before routing/splitting/retrieval even see
    it -- otherwise the dangling pronoun defeats route_question, comparison
    detection, and the title shortcut alike, all of which run on the raw
    string. eval/run_eval.py never passes history, so this is a no-op there.
    """
    _zero_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    question = _condense_followup_question(question, history, agent)

    shortcut = None if forced_doc_id else _title_shortcut_answer(question)
    if shortcut is not None:
        title, collected = shortcut
        if trace is not None:
            trace.span(name="title-shortcut", input=question, output=title)
        return {
            "answer": title,
            "sources": parse_sources(collected),
            "sql": [],
            "tools": ["search_knowledge_base"],
            "rejected_sources": [],
            "collected": collected,
            "usage": dict(_zero_usage),
        }

    # Try the deterministic comparison path first: only engages when at least
    # two distinct documents can be confidently resolved (named doc_ids, the
    # UI's multi-select scope, a clear two-way title match, or M-6's verified
    # LLM resolution step as a last resort) -- returns None for anything more
    # ambiguous, which falls through to the existing agent-based comparison
    # path below unchanged. Resolved once here and reused (M-7) by
    # answer_one's doc-coverage retry below, rather than re-resolving.
    resolved_doc_ids: list[str] | None = None
    if _COMPARISON_RE.search(question):
        doc_registry = getattr(agent, "_doc_registry", None) or {}
        resolved_doc_ids = _resolve_comparison_doc_ids(
            question,
            forced_doc_id,
            doc_registry,
            catalogue=getattr(agent, "_doc_catalogue", None),
            api_base=getattr(agent, "_generation_api_base", None),
            model_name=getattr(agent, "_generation_model", None),
        )
        det = answer_comparison_deterministic(
            agent,
            question,
            forced_doc_id=forced_doc_id,
            trace=trace,
            doc_ids=resolved_doc_ids,
        )
        if det is not None:
            det.setdefault("usage", dict(_zero_usage))
            return det

    usage = dict(_zero_usage)

    # Comparison questions are kept whole: splitting strips the "Comparing X
    # and Y" clause that binds each fragment to a specific document, so a
    # fragment like "which one prohibits evergreen contracts" reaches routing
    # with no document context and lands on the wrong doc or Unsupported.
    # Keeping it whole lets _COMPARISON_RE match in answer_one() and its
    # two-source retry actually do its job. Reproduced directly: without this,
    # a "Comparing X and Y, which..." question came back "1. Unsupported /
    # 2. Unsupported" -- both fragments split apart and lost the comparison.
    parts = (
        [question]
        if _COMPARISON_RE.search(question)
        else _split_multi_part_query(question)
        if _is_multi_part_query(question)
        else [question]
    )
    is_single_part = len(parts) == 1
    if is_single_part:
        answer, collected, excel_trace = answer_one(
            agent,
            question,
            trace=trace,
            forced_doc_id=forced_doc_id,
            usage=usage,
            resolved_doc_ids=resolved_doc_ids,
        )
    else:
        sub_parts: list[tuple[str, list[str], list[dict]]] = []
        collected = []
        sql_all: list[str] = []
        tools_all: list[str] = []
        rejected_all: list[dict] = []
        excel_citations_all: list[dict] = []
        for part in parts:
            p_ans, p_coll, p_trace = answer_one(
                agent, part, trace=trace, forced_doc_id=forced_doc_id, usage=usage
            )
            p_excel_citations = p_trace.get("excel_citations") or []
            sub_parts.append((p_ans.strip(), p_coll, p_excel_citations))
            collected += p_coll
            sql_all += p_trace.get("sql") or []
            tools_all += p_trace.get("tools") or []
            rejected_all += p_trace.get("rejected") or []
            excel_citations_all += p_excel_citations
        excel_trace = {
            "sql": sql_all,
            "tools": tools_all,
            "rejected": rejected_all,
            "excel_citations": excel_citations_all,
        }

    sources = parse_sources(collected)
    sources += _excel_citations_to_sources(
        excel_trace.get("excel_citations") or [], existing=sources
    )
    if is_single_part:
        citation_map = build_citation_map(collected, sources)
        answer = strip_leaked_headers(
            answer,
            citation_map=citation_map,
            max_plausible_marker=max(MAX_TOOL_RESULTS, _max_marker(collected)),
        )
        if _is_malformed_generation(answer) or not answer.strip():
            answer = "Unsupported"
        else:
            answer = _attach_excel_marker_if_missing(
                answer, excel_trace.get("excel_citations") or [], sources
            )
    else:
        # Each part ran as its own agent call, with its own [N] numbering
        # reset at the start of that call (see _MARKER_OFFSET in
        # src/rag_agent.py) -- a single citation_map built from the merged
        # `collected` would still collide between parts even though each
        # part's own numbering is internally unique. Building one map per
        # part (matched against the final merged `sources` list by chunk
        # identity, not position) gives every part real, resolvable
        # citations instead of the "no markers -> show every retrieved
        # candidate" fallback this used to hit.
        cleaned_parts = []
        for p_ans, p_coll, p_excel_citations in sub_parts:
            part_map = build_citation_map(p_coll, sources)
            cleaned = strip_leaked_headers(
                p_ans,
                citation_map=part_map,
                max_plausible_marker=max(MAX_TOOL_RESULTS, _max_marker(p_coll)),
            )
            if _is_malformed_generation(cleaned):
                cleaned = "Unsupported"
            cleaned = _attach_excel_marker_if_missing(
                cleaned, p_excel_citations, sources
            )
            cleaned_parts.append(cleaned)
        # Blank line between parts (a single \n is only a soft break in
        # markdown); number them so a terse part still reads as its own answer.
        kept = [a for a in cleaned_parts if a]
        answer = (
            "\n\n".join(f"{i}. {a}" for i, a in enumerate(kept, 1)) or "Unsupported"
        )
    # No real evidence at all (no parsed sources, no SQL) -- reproduced live
    # 2026-07-21: with zero chunks retrieved, the agent can still answer from
    # its own tool-call arguments (e.g. a doc_id string containing a year).
    # Checked against `sources` (post parse_sources), not raw `collected` --
    # `collected` can hold a tool's own plain error/no-results text (now
    # skipped by parse_sources instead of becoming a fake "unknown" source,
    # see its own comment), so `collected` alone isn't a reliable "real
    # evidence" signal. A "verify every answer" product must never show a
    # confident answer with nothing behind it -- force an honest refusal
    # instead.
    #
    # Exception: a genuine clarifying question ("Which policy area --
    # procurement, travel, conduct?") is not a confident, ungrounded claim --
    # it's the agent correctly declining to guess on an ambiguous broad
    # question (see route_question's _CROSS_DOCUMENT_RE). Reproduced live
    # 2026-07-24: "Summarize the main policies across all documents" made no
    # tool call and got a real clarifying question back, which this guard
    # used to silently discard and replace with a bare, unhelpful
    # "Unsupported" -- worse than just showing the question.
    is_clarifying_question = not sources and answer.strip().endswith("?")
    if (
        not sources
        and not (excel_trace.get("sql") or [])
        and answer.strip().lower() != "unsupported"
        and not is_clarifying_question
    ):
        answer = "Unsupported"
        sources = []
    answer, sources = _renumber_citations_sequentially(answer, sources)
    answer = _narrow_quotes_to_answer(sources, answer)
    sql_list = [s for s in (excel_trace.get("sql") or []) if s]
    kept_filenames = {s["filename"] for s in sources}
    rejected_sources = []
    seen_rejected: set[str] = set()
    for r in excel_trace.get("rejected") or []:
        name = _resolve_original_name(r.get("filename", "unknown"))
        if name in kept_filenames or name in seen_rejected:
            continue
        seen_rejected.add(name)
        rejected_sources.append({"filename": name, "score": r.get("score")})

    return {
        "answer": answer,
        "sources": sources,
        "sql": sql_list,
        "tools": excel_trace.get("tools") or [],
        "rejected_sources": rejected_sources,
        "collected": collected,
        "usage": usage,
    }


def stream_answer(
    agent: Any,
    question: str,
    forced_doc_id: str | list[str] | None = None,
    history: list[dict] | None = None,
) -> Generator[dict, None, None]:
    """Stream an answer token-by-token; yields {"token": str} events, then one
    final {"done": True, **answer_query-shaped result}.

    Only a single, non-comparison, non-multi-part question streams live —
    those are the common case (a single agent run, one attempt). Comparison
    questions and multi-part splits need their answer's *full text* before
    deciding whether to retry (see answer_one/answer_query) or before the
    parts can be merged, so live token-by-token output isn't available for
    them; they fall back to the complete non-streaming pipeline (which keeps
    its retry/merge logic intact) and arrive as a single lump token event.

    The single-part live path keeps answer_one's Unsupported-retry safety net
    too — it's exactly the question type that retry exists for (Groq/temp=0
    nondeterminism occasionally returns Unsupported despite the answer
    existing, see answer_one). A first-attempt "Unsupported" here re-runs
    once, non-streamed, and the retry's result (if it improved) replaces the
    streamed answer in the final event — no token events for the retry
    itself, since the client already replaces its accumulated text with the
    "done" event's answer rather than appending to it.

    The tokens streamed live are the model's raw, uncleaned output — leaked
    header lines, un-renumbered [N] markers — since that cleanup (see
    strip_leaked_headers/build_citation_map) needs the complete text and the
    tool results. The final "done" event's "answer" field is the clean,
    authoritative version; callers should replace the streamed text with it
    once the stream ends, not keep the raw concatenation.

    history: same as answer_query's -- resolved into a standalone question
    before any of the routing/split checks below run, since they all key off
    the raw question string the same way answer_query's do.
    """
    question = _condense_followup_question(question, history, agent)
    is_comparison = bool(_COMPARISON_RE.search(question))
    is_multi_part = _is_multi_part_query(question)
    if is_comparison or is_multi_part:
        result = answer_query(agent, question, forced_doc_id=forced_doc_id)
        if result["answer"]:
            yield {"token": result["answer"]}
        yield {"done": True, **result}
        return

    is_forced_doc_modality = None
    if isinstance(forced_doc_id, list):
        is_forced_doc_modality = "document"
        route = {"modality": "document", "source_file": "the selected documents"}
    elif forced_doc_id:
        is_forced_doc_modality = (
            "excel" if forced_doc_id.lower().endswith(_TABLE_EXTS) else "document"
        )
        route = {"modality": is_forced_doc_modality, "source_file": forced_doc_id}
    else:
        route = route_question(question)
    q = routing_directive(route) + question

    token = None
    if is_forced_doc_modality == "document":
        token = FORCED_DOC_ID.set(forced_doc_id)

    collected: list[str] = []
    sql_trace: list[str] = []
    tool_calls: list[str] = []
    rejected: list[dict] = []
    excel_citations: list[dict] = []
    raw_tokens: list[str] = []
    correction: str | None = None
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    try:
        for tok in stream_agent(
            agent,
            q,
            collected_chunks=collected,
            sql_trace=sql_trace,
            tool_calls=tool_calls,
            rejected_chunks=rejected,
            excel_citations=excel_citations,
            live_tokens=True,
            usage=usage,
        ):
            if isinstance(tok, FinalCorrection):
                # The repair pass / grounding check changed the text after
                # the live tokens above already streamed the raw version --
                # this replaces it rather than appending (see stream_agent's
                # live_tokens docstring). Not sent as its own "token" SSE
                # event: the client waits for the "done" event's answer,
                # same as any other correction (e.g. the Unsupported-retry
                # below).
                correction = tok.text
            else:
                raw_tokens.append(tok)
                yield {"token": tok}
    finally:
        if token is not None:
            FORCED_DOC_ID.reset(token)

    raw_answer = correction if correction is not None else "".join(raw_tokens).strip()
    if raw_answer.lower() == "unsupported":
        r_ans, r_coll, r_tr = run_once(
            agent, q + _RETRY_INSTRUCTION, attempt="unsupported-retry", usage=usage
        )
        if r_ans and r_ans.lower() != "unsupported":
            raw_answer = r_ans
            collected = r_coll
            sql_trace = r_tr.get("sql") or []
            tool_calls = r_tr.get("tools") or []
            rejected = r_tr.get("rejected") or []
            excel_citations = r_tr.get("excel_citations") or []

    sources = parse_sources(collected)
    sources += _excel_citations_to_sources(excel_citations, existing=sources)
    citation_map = build_citation_map(collected, sources)
    answer = strip_leaked_headers(
        raw_answer,
        citation_map=citation_map,
        max_plausible_marker=max(MAX_TOOL_RESULTS, _max_marker(collected)),
    )
    # answer_query (the non-streaming twin of this function) attaches this;
    # stream_answer never did -- reproduced live: a correct, single-fact
    # Excel table-lookup answer ("Doncaster Mbc, the supplier...") came back
    # from the live UI (which calls this streaming path, not answer_query)
    # with zero [N] markers and a blank Evidence panel, even though
    # excel_citations named the real source deterministically the whole time.
    answer = _attach_excel_marker_if_missing(answer, excel_citations, sources)
    answer, sources = _renumber_citations_sequentially(answer, sources)
    answer = _narrow_quotes_to_answer(sources, answer)
    sql_list = [s for s in sql_trace if s]
    kept_filenames = {s["filename"] for s in sources}
    rejected_sources = []
    seen_rejected: set[str] = set()
    for r in rejected:
        name = _resolve_original_name(r.get("filename", "unknown"))
        if name in kept_filenames or name in seen_rejected:
            continue
        seen_rejected.add(name)
        rejected_sources.append({"filename": name, "score": r.get("score")})

    yield {
        "done": True,
        "answer": answer,
        "sources": sources,
        "sql": sql_list,
        "tools": tool_calls,
        "rejected_sources": rejected_sources,
        "usage": usage,
    }
