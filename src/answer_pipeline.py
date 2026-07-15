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

import re
from typing import Any
from urllib.parse import unquote

from src.answer_quality import _is_multi_part_query, _split_multi_part_query
from src.config import MAX_TOOL_RESULTS, QDRANT_COLLECTION, QDRANT_URL
from src.file_resolver import resolve_original_name as _resolve_original_name
from src.rag_agent import route_question, routing_directive, stream_agent
from src.retriever import retrieve
from src.tools.retrieval_tool import FORCED_DOC_ID
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
    r"which .+ and which|which .+\bor the .+\?)",
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


def _missing_mentioned_docs(question_text: str, srcs: list[dict]) -> list[str]:
    """doc_XXX ids named in the question with no matching source."""
    mentioned = {m.lower() for m in _MENTIONED_DOC_ID_RE.findall(question_text)}
    if len(mentioned) < 2:
        return []
    covered = {(s.get("document_id") or "").lower() for s in srcs} | {
        s["filename"].lower() for s in srcs
    }
    return [d for d in mentioned if not any(d in c for c in covered)]


def _comparison_incompleteness(
    question_text: str, ans: str, coll: list[str]
) -> tuple[list[str], bool, int]:
    """(missing_named_docs, has_partial_unsupported_fragment, n_distinct_sources)."""
    sources = parse_sources(coll)
    n_sources = len({s["filename"] for s in sources})
    missing = _missing_mentioned_docs(question_text, sources)
    has_partial = (
        ans.strip().lower() != "unsupported" and "unsupported" in ans.lower()
    )
    return missing, has_partial, n_sources

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
# Inline [N] citation markers — dropped because the answer's numbering does not
# correspond to the trace panel's, so they would be misleading. Only [N] within
# the retrieved-chunk range (1..MAX_TOOL_RESULTS) is treated as a citation — a
# bracketed year like [2024] or any larger number is left alone.
_INLINE_CITATION_RE = re.compile(r"[ \t]*\[(\d+)\]")
_BLANK_LINES_RE = re.compile(r"\n{3,}")


def _strip_inline_citation(match: re.Match) -> str:
    """Drop a [N] marker only when N is a plausible retrieved-chunk index."""
    return "" if 1 <= int(match.group(1)) <= MAX_TOOL_RESULTS else match.group(0)


def strip_leaked_headers(text: str) -> str:
    """Remove raw chunk-header lines and inline [N] citation markers the LLM
    echoes into its answer — the trace panel/eval evidence is the source list."""
    cleaned = _LEAKED_HEADER_RE.sub("", text)
    cleaned = _INLINE_CITATION_RE.sub(_strip_inline_citation, cleaned)
    cleaned = _BLANK_LINES_RE.sub("\n\n", cleaned)
    return cleaned.strip()


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
            filename = m.group("file") if m else "unknown"
            loc_key = m.group("loc_key") or "" if m else ""
            loc_val = m.group("loc_val") or "" if m else ""
            score_str = m.group("score") if m else None
            page_str = m.group("page") if m else None
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

            bbox_m = _FIGURE_BBOX_RE.search(body)
            figure_bbox = (
                [float(v) for v in bbox_m.group(1).split(",")] if bbox_m else None
            )

            plain = _FIGURE_BLOCK_RE.sub("", body)
            plain = _TABLE_MARKER_RE.sub("", plain)
            plain = re.sub(r"^#{1,3}\s+.+$", "", plain, flags=re.MULTILINE).strip()
            excerpt = " ".join(plain.split())[:350]

            score = float(score_str) if score_str else None
            # 1.0 is the retriever's placeholder for filter/scroll fetches
            # (doc-routed chunks carry no similarity score) — drop it so the UI
            # doesn't show a misleading "perfect match" chip.
            if score == 1.0:
                score = None

            key = (filename, excerpt[:80])
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
                }
            )
    # Cap at 8, but guarantee every distinct file that produced a chunk keeps at
    # least one slot before a second chunk from an already-represented file
    # takes one -- reproduced directly: in a two-document comparison, a
    # redundant re-query of the document that already had a good answer filled
    # the cap with more of its own chunks (it's processed first, being the
    # latest call) and silently dropped the other document's genuinely
    # retrieved chunks from the source list entirely.
    seen_files: set[str] = set()
    diverse: list[dict] = []
    rest: list[dict] = []
    for s in sources:
        if s["filename"] not in seen_files:
            seen_files.add(s["filename"])
            diverse.append(s)
        else:
            rest.append(s)
    return (diverse + rest)[:8]


def run_once(
    agent: Any, question: str, attempt: str = "initial", trace: Any = None
) -> tuple[str, list[str], dict]:
    """Run the agent once for a question; return (answer, collected chunks, trace dict).

    Emits one Langfuse span per tool call actually made (name + retrieved
    chunk group, or the SQL for query_excel calls) plus one span for the
    attempt itself, so retries show up as distinct, inspectable steps.
    """
    collected: list[str] = []
    tokens: list[str] = []
    sql_trace: list[str] = []
    tool_calls: list[str] = []
    rejected: list[dict] = []
    for token in stream_agent(
        agent,
        question,
        collected_chunks=collected,
        sql_trace=sql_trace,
        tool_calls=tool_calls,
        rejected_chunks=rejected,
        trace=trace,
    ):
        tokens.append(token)
    answer = "".join(tokens).strip()
    trace_holder = {"sql": sql_trace, "tools": tool_calls, "rejected": rejected}

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
    # different document despite being told which one to use). For the
    # document modality, hard-enforce it at the tool layer instead; query_excel
    # has no per-source scoping param to override, so excel-forced questions
    # still rely on the directive alone (see TODO.md).
    token = None
    if is_forced_doc_modality == "document":
        token = FORCED_DOC_ID.set(forced_doc_id)
    try:
        ans, coll, tr = run_once(agent, q, attempt="initial", trace=trace)
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
                question, ans, coll
            )
            if is_flat_unsupported or missing or n_sources < 2 or has_partial:
                if missing:
                    retry_instruction = (
                        f"\n\nIMPORTANT: This is a retry. This comparison question names "
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
                )
                r_flat = r_ans.strip().lower() == "unsupported"
                r_missing, r_has_partial, r_sources = _comparison_incompleteness(
                    question, r_ans, r_coll
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
        if ans.lower() == "unsupported":
            r_ans, r_coll, r_tr = run_once(
                agent, q + _RETRY_INSTRUCTION, attempt="unsupported-retry", trace=trace
            )
            if r_ans.lower() != "unsupported" and r_ans:
                return r_ans, r_coll, r_tr
            return ans, coll, tr
    finally:
        if token is not None:
            FORCED_DOC_ID.reset(token)
    return ans, coll, tr


def answer_query(
    agent: Any,
    question: str,
    trace: Any = None,
    forced_doc_id: str | list[str] | None = None,
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
    """
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
        }

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
    if len(parts) == 1:
        answer, collected, excel_trace = answer_one(
            agent, question, trace=trace, forced_doc_id=forced_doc_id
        )
    else:
        sub_answers: list[str] = []
        collected = []
        sql_all: list[str] = []
        tools_all: list[str] = []
        rejected_all: list[dict] = []
        for part in parts:
            p_ans, p_coll, p_trace = answer_one(
                agent, part, trace=trace, forced_doc_id=forced_doc_id
            )
            sub_answers.append(p_ans.strip())
            collected += p_coll
            sql_all += p_trace.get("sql") or []
            tools_all += p_trace.get("tools") or []
            rejected_all += p_trace.get("rejected") or []
        # Blank line between parts (a single \n is only a soft break in
        # markdown); number them so a terse part still reads as its own answer.
        kept = [a for a in sub_answers if a]
        answer = (
            "\n\n".join(f"{i}. {a}" for i, a in enumerate(kept, 1)) or "Unsupported"
        )
        excel_trace = {"sql": sql_all, "tools": tools_all, "rejected": rejected_all}

    answer = strip_leaked_headers(answer)
    sources = parse_sources(collected)
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
    }
