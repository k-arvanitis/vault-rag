"""Agentic RAG using LangGraph ReAct agent with native tool calling (requires vLLM with
--enable-auto-tool-choice --tool-call-parser hermes).

Single tool: search_knowledge_base
  Searches documents (Qdrant) and structured tables (Postgres) in parallel.
  The agent calls it once — no routing decisions needed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Generator

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langsmith import traceable
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel

from src.excel_agent import build_excel_agent_tools
from src.excel_tool import DuckDBStore
from src.file_resolver import resolve_original_name
from src.reranker import BGEReranker, QwenReranker
from src.retriever import (
    _extract_table_filter_token,
    infer_query_chunk_types,
    retrieve,
)
from src.config import (
    GENERATION_API_BASE,
    GENERATION_MODEL,
    LITELLM_MASTER_KEY,
    MAX_CHUNK_CHARS as _CFG_MAX_CHUNK_CHARS,
    MAX_TABLE_CHARS as _CFG_MAX_TABLE_CHARS,
    DOC_MIN_SCORE as _CFG_DOC_MIN_SCORE,
    QDRANT_COLLECTION,
    QDRANT_URL,
    RERANK_TOP_N,
    RERANKER_DEVICE,
    RERANKER_MODEL,
    RETRIEVAL_TOP_K,
)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_TOOLS_BLOCK_WITH_EXCEL = """You have two tools:

1. **search_knowledge_base** — semantic search over all ingested documents: PDFs, reports, and table sheet summaries.
2. **query_excel** — answers any question about structured data (Excel/CSV) stored in DuckDB. Pass the full question as 'question'. The agent inside discovers the right table(s), generates SQL, and retries automatically.

Tool routing:
- For structured data questions (Excel, CSV, spend reports, transactions): call **query_excel** directly with the complete question — include every filter detail (dates, amounts, supplier names, transaction numbers, departments) verbatim. Do NOT call search_knowledge_base first for Excel questions.
- For PDFs, policies, reports, findings: use **search_knowledge_base** only — never query_excel.
- If a question mixes PDF and Excel sources, use both tools.
"""

_TOOLS_BLOCK_SEARCH_ONLY = """You have one tool:

1. **search_knowledge_base** — searches all knowledge sources: unstructured documents (PDFs, reports) and structured table rows (CSV/Excel) ingested into the vector store.
"""

_RULES_BLOCK = """Rules:
- You MUST call a tool before answering every question, no exceptions. Never answer from your own knowledge without searching first.
- Use a focused, specific sub-question as the search query. Include key entity names (supplier names, transaction IDs, beneficiary names, dates) verbatim so they match the indexed content.
- MULTI-PART QUESTIONS: when a question asks about two or more distinct pieces of information, decompose it into separate sub-questions and issue one tool call per sub-question. If all parts are from the same Excel dataset, pass the full multi-part question to query_excel in one call.
- **CROSS-DOCUMENT QUESTIONS**: for PDF/text docs — two separate search_knowledge_base calls, each scoped to one doc_id. For Excel/CSV cross-document questions — one query_excel call with the full question.
- **DOC_ID IS MANDATORY**: whenever the question names or implies a specific document (by title, publisher, or alias), first call search_knowledge_base with the document name as the query to retrieve its document_summary chunk — that chunk contains the Document ID (e.g. "doc_001"). Then use that doc_id in your follow-up call to scope retrieval. For two-part questions naming two different documents, resolve each doc_id separately.
- **EXACT QUALIFIER IN QUERY**: when the question includes an exact qualifier — a date, a count category, a status label, or a precise descriptor — copy that exact phrase verbatim into your search query. This is critical for retrieving the passage that matches the qualifier, not a nearby passage with a different value.
- For table row lookups: include ALL distinguishing attributes from the question (supplier name, date, transaction ID, department) in your search query to land on the exact row.
- After you receive tool results, answer from those results. Do not repeat identical tool calls.
"""

_CLARIFICATION_BLOCK = """CLARIFICATION RULE (apply BEFORE answering or returning Unsupported):
- If the question is too broad to answer with a specific value — e.g. "what about HR policies?", "tell me about finance", "anything on procurement?" — and no specific entity, date, or value is being asked for, do NOT list files and do NOT synthesize a generic summary.
- Instead, output exactly: "Clarify: <one short question listing 2-4 specific topics derived from the retrieved chunks>". Example: "Clarify: which HR topic — leave policy, harassment, equity, or monitoring?"
- Trigger this rule when the retrieved chunks span 3+ unrelated documents OR contain only document_summary chunks with no detail-level content matching the question.
- Do not use this rule when the question names a specific value, entity, date, or document — answer those normally.
"""

_ANSWERING_BLOCK = """When answering:
- Lead with the direct answer value — a name, number, date, or phrase — before adding any context. Do not open with "According to..." or "The X is..."; just state the value.
- Only state values that appear in the retrieved text. Do not interpolate, infer, or calculate anything not explicitly present.
- Each retrieved passage is prefixed with [N] file=<filename>. For multi-document questions, use the file= label to match each answer value to its correct source — do not mix values from different files.
- If retrieved table rows are shown as "Relevant table rows", use the field labels to select the requested cell value exactly.
- For multi-part questions, answer each part on its own line with a brief source label (e.g. "Source A: <value>. Source B: <value>."). Only mark a part Unsupported if that part's value is absent.
- **TIME-SCOPED NUMBERS**: when the question specifies an exact date or time qualifier, report only the number explicitly paired with that exact date in the retrieved text. Do NOT report numbers paired with a different date, even if both appear in the same passage.
- **MULTI-NUMBER DISAMBIGUATION**: when a passage contains multiple numbers with different descriptors, read the question to identify which descriptor it asks about, then report only the number paired with that descriptor. Never report the first number you see.
- Markdown headings (lines starting with #) in the retrieved text are document titles — quote them exactly when asked for a title.
- Never perform arithmetic. If a sum or average is not pre-computed in the source, list the raw values and note the calculation is unavailable.
- **VERBATIM VALUES**: when stating a specific number, rate, date, or named quantity, copy it exactly as it appears in the source. Preserve original formatting — do not normalize fractions, units, or date formats.
- **SHEET COUNT QUESTIONS**: when asked whether a document contains one sheet or multiple sheets, count the number of distinct sheet_summary chunks returned for that document. If more than one, answer "No" (it has multiple sheets).
"""

_ABSTENTION_BLOCK = """ABSTENTION RULE (CRITICAL — follow exactly):
- If the retrieved text does not contain the requested answer, you MUST output only the single word: Unsupported
- Do not output Unsupported when the retrieved text contains the requested value under a matching field label or in a matching sentence.
- No explanation. No hedging. No "I cannot find...". Just the single word: Unsupported
- This applies unconditionally to: personal phone numbers, home addresses, passwords, login credentials, government ID numbers (SSN, passport), GPS coordinates, salaries or pay of named individuals, and any other detail not present verbatim in the retrieved text.
- Do not use your general knowledge to fill gaps — if it is not in the retrieved text, output: Unsupported. Only answers explicitly stated in the retrieved passages are valid.
- **FILENAMES AND INTERNAL PATHS are not answers**: if the retrieved text only contains a filename or a file path, do not return that as the answer value — output: Unsupported
- **DOCUMENT IDENTITY CHECK**: when the question asks about a specific document by title or alias, verify the retrieved text's file= label matches that specific document. If the retrieved content is from a different document, do not use it — search again with the correct doc_id.
"""

_CITATION_BLOCK = """Always cite your sources:
- Document chunks: [1], [2], etc.
- Table results: mention the sheet/file name from the tool output.
"""


def _compose_system_prompt(*, with_excel: bool) -> str:
    """Compose the agent system prompt from shared blocks plus the right tools header."""
    tools = _TOOLS_BLOCK_WITH_EXCEL if with_excel else _TOOLS_BLOCK_SEARCH_ONLY
    return (
        "You are an intelligent RAG assistant.\n\n"
        f"{tools}\n"
        f"{_RULES_BLOCK}\n"
        f"{_CLARIFICATION_BLOCK}\n"
        f"{_ANSWERING_BLOCK}\n"
        f"{_ABSTENTION_BLOCK}\n"
        f"{_CITATION_BLOCK}"
    )


_SYSTEM_PROMPT_SEARCH_ONLY = _compose_system_prompt(with_excel=False)
_SYSTEM_PROMPT_WITH_EXCEL = _compose_system_prompt(with_excel=True)


def _is_thinking_model(model_name: str) -> bool:
    """Return True for models that emit <think> blocks and accept /no_think."""
    name = model_name.lower()
    return any(k in name for k in ("qwen", "qwq", "deepseek-r", "r1"))


def _build_system_prompt(model_name: str) -> str:
    """Build system prompt. query_excel is always registered so the Excel variant is always used."""
    prefix = "/no_think " if _is_thinking_model(model_name) else ""
    return prefix + _SYSTEM_PROMPT_WITH_EXCEL


# Keep a module-level default for callers that don't pass a model name
SYSTEM_PROMPT = _build_system_prompt(GENERATION_MODEL)

# ---------------------------------------------------------------------------
# Unified tool
# ---------------------------------------------------------------------------


DOC_MIN_SCORE = float(os.getenv("DOC_MIN_SCORE", str(_CFG_DOC_MIN_SCORE)))
MAX_CHUNK_CHARS = int(os.getenv("MAX_CHUNK_CHARS", str(_CFG_MAX_CHUNK_CHARS)))
MAX_TABLE_CHARS = int(os.getenv("MAX_TABLE_CHARS", str(_CFG_MAX_TABLE_CHARS)))
MAX_TOOL_RESULTS = int(os.getenv("MAX_TOOL_RESULTS", "8"))


def _llm_call(prompt: str, api_base: str, model_name: str, api_key: str = "", max_tokens: int = 128, temperature: float = 0.0) -> str:
    import openai
    # Resolve key: explicit arg → LiteLLM master key → Groq → OpenAI → dummy
    key = (
        api_key
        or LITELLM_MASTER_KEY
        or os.getenv("GROQ_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or "EMPTY"
    )
    client = openai.OpenAI(base_url=api_base, api_key=key)
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    raw = resp.choices[0].message.content
    # Strip <think> blocks emitted by reasoning models (Qwen3, QwQ, DeepSeek-R1, etc.)
    return re.sub(r"(?s)<think>.*?</think>", "", raw).strip()


@traceable(name="hyde-expansion")
def _hyde(query: str, api_base: str, model_name: str) -> str:
    """Generate a hypothetical answer to embed instead of the raw query (HyDE)."""
    no_think = "/no_think " if _is_thinking_model(model_name) else ""
    return _llm_call(
        f"{no_think}Write a short passage (2-3 sentences) that would directly answer "
        f"this question. Use the same language and terminology as the likely source document."
        f"\n\nQuestion: {query}",
        api_base, model_name,
    )


def _strip_think(text: str) -> str:
    """Remove reasoning blocks emitted by thinking models."""
    return re.sub(r"(?is)<think>.*?</think>\s*", "", text).strip()


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Parse the first JSON object from model output."""
    cleaned = _strip_think(text).strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE | re.MULTILINE).strip()
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


def _looks_like_bad_final_answer(text: str) -> bool:
    """Return True for transport/tool artifacts or empty abstentions worth retrying."""
    cleaned = _strip_think(text).strip()
    if not cleaned:
        return True
    lowered = cleaned.lower()
    return (
        lowered in {"unsupported", "no answer generated.", "no answer generated"}
        or lowered.startswith("sorry, need more steps")
        or lowered.startswith("<function=")
        or lowered.startswith("function=")
        or "search_knowledge_base" in lowered and ("<function" in lowered or "</function>" in lowered)
    )


_NOT_PROVIDED_PHRASES = (
    "not provided", "not available", "not included", "not contained",
    "not answerable", "not found in", "not in this dataset", "not in the dataset",
    "does not contain", "not present in", "cannot be determined",
    "cannot be found", "no information", "not specified", "not stated",
    "none of the", "none of these", "no document", "not in any",
    "not listed in", "not given", "no such information",
    "does not specify", "does not mention", "does not explicitly",
    "is not explicitly", "not explicitly stated", "not explicitly mentioned",
    "does not exist",
)


_STRONG_NOT_FOUND_PHRASES = (
    "none of the provided documents", "none of the documents", "no document in",
    "not present in any", "not found in any",
)


def _normalize_unsupported(answer: str) -> str:
    """Convert verbose 'not found' answers to the canonical 'Unsupported' token."""
    if "Unsupported" in answer:
        # Collapse "hedging preamble + Unsupported" → "Unsupported" when the text
        # before "Unsupported" is pure hedging with no real answer values.
        # Preserve multi-part answers that have real values (Label: value pattern).
        idx = answer.index("Unsupported")
        before = answer[:idx].strip()
        after = answer[idx + len("Unsupported"):].strip().lstrip(".")
        if before and not after:
            lowered_before = before.lower()
            is_hedging = any(phrase in lowered_before for phrase in _NOT_PROVIDED_PHRASES)
            has_real_value = bool(re.search(r"\w[\w\s]{1,30}:\s+\S", before))
            if is_hedging and not has_real_value:
                return "Unsupported"
        return answer
    lowered = answer.lower()
    if any(phrase in lowered for phrase in _STRONG_NOT_FOUND_PHRASES):
        return "Unsupported"
    if any(phrase in lowered for phrase in _NOT_PROVIDED_PHRASES):
        return "Unsupported"
    return answer


_BARE_FILENAME_RE = re.compile(
    r"\bdoc_\d+[_a-z0-9-]*\.(?:pdf|csv|xlsx|xls|md|json|txt|tsv)\b",
    re.IGNORECASE,
)

_BARE_FN_STOP = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "am",
    "in", "on", "of", "for", "to", "from", "with", "by", "at", "as", "into",
    "that", "which", "this", "these", "those", "it", "its", "they", "them",
    "what", "who", "whom", "where", "when", "how", "why", "whose",
    "and", "or", "but", "not", "no", "nor", "so", "if", "then",
    "provides", "provide", "providing", "provided", "provider",
    "gives", "give", "given", "giving", "gave",
    "shows", "show", "showing", "shown", "showed",
    "contains", "contain", "containing", "contained",
    "has", "have", "had", "having",
    "states", "state", "stating", "stated",
    "mentions", "mention", "mentioning", "mentioned",
    "specifies", "specify", "specifying", "specified",
    "outlines", "outline", "outlining", "outlined",
    "details", "detail", "detailing", "detailed",
    "lists", "list", "listing", "listed",
    "refers", "refer", "referring", "referred", "reference", "references",
    "document", "documents", "doc", "docs",
    "file", "files", "filename",
    "report", "reports", "policy", "policies",
    "page", "section", "chapter", "row", "table",
    "csv", "pdf", "xlsx", "xls", "md", "json", "txt",
    "source", "sources", "based", "according", "answer", "answers",
    "above", "below", "found", "find",
})


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
    query_terms = set(re.findall(r"\b[a-z][a-z]{2,}\b", query.lower()))
    tokens = re.findall(r"\b[A-Za-z][A-Za-z'/-]{1,}\b", stripped.lower())
    substantive = [t for t in tokens if t not in _BARE_FN_STOP and t not in query_terms]
    return len(substantive) < 2


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

    question_boundary = re.search(r"\?\s+(?=(According to|In the|In |What|Which)\b)", q)
    if question_boundary:
        first = q[: question_boundary.start() + 1].strip()
        second = q[question_boundary.end():].strip()
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

    cleaned: list[str] = []
    for part in parts:
        part = re.sub(r"\s+", " ", part).strip()
        if len(part) >= 12 and part not in cleaned:
            cleaned.append(part)
    return cleaned if len(cleaned) >= 2 else [q]


def _query_terms(query: str) -> list[str]:
    """Extract useful lexical anchors for snippet selection."""
    stop_words = {
        "about", "according", "after", "also", "and", "are", "before", "between", "does",
        "document", "during", "from", "have", "into", "issued", "must", "that", "the",
        "their", "there", "this", "what", "when", "where", "which", "while", "with",
        "within", "would", "your", "approved", "policy",
    }
    terms = []
    for term in re.findall(r"[A-Za-z0-9][A-Za-z0-9'/-]{2,}", query.lower()):
        if term not in stop_words and term not in terms:
            terms.append(term)
    return terms


def _best_snippet(content: str, query: str, max_chars: int) -> str:
    """Keep the most query-relevant window from a long text chunk."""
    if len(content) <= max_chars:
        return content
    terms = _query_terms(query)
    lower = content.lower()
    positions: list[int] = []
    for term in terms:
        positions.extend(match.start() for match in re.finditer(re.escape(term), lower))
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
    for position in positions:
        start = max(0, position - max_chars // 2)
        end = min(len(content), start + max_chars)
        start = max(0, end - max_chars)
        window = lower[start:end]
        score = 0
        for term, weight in weighted_terms.items():
            occurrences = len(re.findall(re.escape(term), window))
            if occurrences:
                score += weight + min(occurrences, 3)
        # Prefer windows with answer-like numeric phrases when the query asks for
        # amounts, dates, periods, counts, or durations.
        if re.search(r"\b(how many|amount|date|period|longer|total|cost|rate|range)\b", query, re.IGNORECASE):
            if re.search(r"\b(up to|maximum|total|invoice date|please pay|contract term|closed\s*[-–]\s*implemented)\b", window):
                score += 14
            if re.search(r"\b(months?|years?|percent|transactions?|districts?|areas?|recommendations?)\b", window):
                score += 6
            if re.search(r"\b(up to|maximum|total|date|amount|range|percent|months?|years?)\b.{0,80}\d", window):
                score += 8
            if re.search(r"\d.{0,40}\b(months?|years?|percent|transactions?|districts?|areas?|recommendations?)\b", window):
                score += 8
        distance = abs(position - (start + max_chars // 2))
        if score > best_score or (score == best_score and distance < best_distance):
            best_start = start
            best_position = position
            best_score = score
            best_distance = distance

    start = best_start
    end = min(len(content), start + max_chars)
    start = max(0, end - max_chars)

    # Prefer clean line boundaries when possible.
    if start:
        line_start = content.find("\n", start)
        if 0 <= line_start < min(end, start + 300) and line_start < best_position:
            start = line_start + 1
    prefix = "…\n" if start else ""
    suffix = "\n…" if end < len(content) else ""
    return prefix + content[start:end].strip() + suffix


def _split_markdown_row(line: str) -> list[str]:
    """Split a simple markdown table row into cells."""
    stripped = line.strip()
    if not stripped.startswith("|") or not stripped.endswith("|"):
        return []
    return [cell.strip() for cell in stripped.strip("|").split("|")]


def _normalize_table_match_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _table_match_terms(query: str) -> list[str]:
    terms: list[str] = []
    for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9&./_-]{1,}", query):
        normalized = _normalize_table_match_text(token)
        if len(normalized) < 3:
            continue
        if normalized in {
            "the", "and", "for", "row", "what", "which", "where", "when", "with",
            "dated", "date", "amount", "total", "net", "supplier", "department",
            "directorate", "transaction", "transactions", "council", "doncaster",
            "published", "spend", "report", "purchase", "card", "q1", "2025", "2026",
        }:
            continue
        if normalized not in terms:
            terms.append(normalized)
    return terms


def _format_key_value_rows(headers: list[str], rows: list[list[str]], query: str, max_rows: int = 4) -> str:
    """Render table rows as explicit field/value lines for reliable cell extraction."""
    terms = _table_match_terms(query)
    scored: list[tuple[int, int, list[str]]] = []
    for idx, row in enumerate(rows):
        row_text = _normalize_table_match_text(" ".join(row))
        score = sum(1 for term in terms if term in row_text)
        scored.append((score, -idx, row))

    scored.sort(reverse=True)
    selected = [row for score, _, row in scored if score > 0][:max_rows]
    if not selected:
        selected = rows[:max_rows]

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

    table_lines = [line for line in lines if line.strip().startswith("|")]
    if table_lines:
        parsed = [_split_markdown_row(line) for line in table_lines]
        parsed = [row for row in parsed if row]
        if len(parsed) >= 2:
            headers = parsed[0]
            rows = [
                row for row in parsed[1:]
                if not all(re.fullmatch(r":?-{3,}:?", cell.strip()) for cell in row)
            ]
            rows = [row for row in rows if len(row) >= len(headers)]
            if rows:
                return _format_key_value_rows(headers, rows, query)

    pipe_pair_lines = [
        line for line in lines
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


def _direct_answer_from_context(query: str, contexts: list[str], api_base: str, model_name: str) -> str:
    """Fallback answer pass over retrieved text, without another tool-planning loop."""
    usable_contexts = [ctx.strip() for ctx in contexts if ctx and ctx.strip()]
    if not usable_contexts:
        return "Unsupported"

    extracted = _extract_key_value_answer(query, usable_contexts)
    if extracted:
        return extracted

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
    contexts = []
    context_index = 1
    subqueries = _llm_split_subqueries(query, api_base, model_name)
    per_query_top_k = min(MAX_TOOL_RESULTS, max(4, MAX_TOOL_RESULTS // max(1, len(subqueries))))
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
            contexts.append(f"[{context_index}] subquery={subquery}\nfile={file_name}\n{content}")
            context_index += 1
    return _direct_answer_from_context(query, contexts, api_base, model_name)


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
    sources: set[str] = set()
    for ctx in contexts:
        for match in re.finditer(r"(?:^|\n)(?:\[\d+\]\s*)?(?:repair_query=.*\n)?file=([^\s\n]+)", ctx):
            sources.add(match.group(1))
    return len(sources)


def _missing_source_queries(query: str, answer: str, contexts: list[str], api_base: str, model_name: str) -> list[str]:
    """Generate follow-up searches when a multi-part answer used too few sources."""
    context_refs = "\n".join(ctx.splitlines()[0] for ctx in contexts[:8] if ctx.strip())
    no_think = "/no_think " if _is_thinking_model(model_name) else ""
    prompt = (
        f"{no_think}The question likely needs evidence from more than one source, "
        "but the current answer used too few retrieved sources.\n"
        "Write up to 2 focused vector-store search queries for the missing independent facts or sources.\n"
        "Do not repeat facts already answered from the retrieved source.\n"
        "Return ONLY JSON: {\"missing_queries\": [\"query\", ...]}\n\n"
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
                missing_queries = _missing_source_queries(query, answer, contexts, api_base, model_name)
            except Exception as exc:
                print(f"[WARN] Missing-source query generation failed ({type(exc).__name__}): {exc}")
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
        if normalized and normalized not in {re.sub(r"\s+", " ", q).strip().lower() for q in deduped_queries}:
            deduped_queries.append(missing_query)
    missing_queries = deduped_queries

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
            print(f"[WARN] Coverage repair retrieval failed ({type(exc).__name__}): {exc}")
            continue
        for hit in hits[:4]:
            meta = hit.get("metadata", {}) or {}
            file_name = meta.get("file_name") or meta.get("source_file", "unknown")
            file_name = resolve_original_name(file_name)
            content = (hit.get("content") or "").strip()
            if content:
                repaired_contexts.append(f"[{context_index}] repair_query={missing_query}\nfile={file_name}\n{content}")
                context_index += 1

    if len(repaired_contexts) == len(contexts):
        return answer

    try:
        repaired = _direct_answer_from_context(query, repaired_contexts, api_base, model_name)
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

    if source_count < 2 and _is_better_multi_answer(repaired, answer):
        return repaired
    if _unsupported_count(repaired) < _unsupported_count(answer):
        return repaired
    return answer


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
        " in ", " for ", " from ", " within ", " according ", " on ", " dated ",
        " with ", " of the ",
    )
    for pattern in patterns:
        match = re.search(pattern, q)
        if not match:
            continue
        label = match.group(1)
        for cut in cut_words:
            if cut in label:
                label = label.split(cut, 1)[0]
        label = re.sub(r"\b(listed|shown|given|provided|document|policy|report|table|row)\b", " ", label)
        label = re.sub(r"\s+", " ", label).strip(" :.-")
        if len(label) >= 3 and label not in candidates:
            candidates.append(label)
    return candidates


def _extract_key_value_answer(query: str, contexts: list[str]) -> str | None:
    """Extract answers from generic 'Field: Value' text when the model abstains."""
    labels = _label_candidates_from_query(query)
    if not labels:
        return None

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
                value = re.sub(r"<br\s*/?>", " ", match.group("value"), flags=re.IGNORECASE)
                value = re.sub(r"\s+", " ", value).strip(" :-")
                if value:
                    return value
    return None


def _merge_hits(primary: list[dict[str, Any]], secondary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge retrieval hits while preserving order and removing duplicate chunks."""
    merged: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
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
    if not indices or not source_file:
        return {}
    from urllib.request import Request, urlopen
    base = qdrant_url.rstrip("/")
    url = f"{base}/collections/{collection}/points/scroll"
    body = json.dumps({
        "limit": len(indices) + 2,
        "with_payload": True,
        "with_vector": False,
        "filter": {
            "must": [
                {"key": "metadata.source_file", "match": {"value": source_file}},
                {"key": "metadata.chunk_index", "match": {"any": indices}},
            ]
        },
    }).encode("utf-8")
    req = Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return {}
    out: dict[int, str] = {}
    for point in data.get("result", {}).get("points", []) or []:
        payload = point.get("payload") or {}
        meta = payload.get("metadata") or {}
        idx = meta.get("chunk_index")
        if isinstance(idx, int):
            out[idx] = (payload.get("content") or "").strip()
    return out


def _enrich_search_query(query: str) -> str:
    """Pass-through hook for query enrichment.

    Kept as a single seam so future query-rewrite logic (e.g. learned expansion)
    can be wired in without touching call sites.
    """
    return query


def _build_doc_registry(qdrant_url: str, collection: str) -> dict[str, str]:
    """Return {lowercased_source_file_stem: doc_id} by scrolling a sample of Qdrant points.

    Used for fuzzy title matching when the LLM passes a document title instead of a doc_id.
    Returns empty dict on failure so the caller degrades gracefully.
    """
    from urllib.request import Request, urlopen

    base = qdrant_url.rstrip("/")
    url = f"{base}/collections/{collection}/points/scroll"
    body_bytes = json.dumps({
        "limit": 250,
        "with_payload": True,
        "with_vector": False,
        "filter": {"must": [{"key": "metadata.chunk_type", "match": {"value": "document_summary"}}]},
    }).encode()
    req = Request(url, data=body_bytes, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
    except Exception:
        return {}

    registry: dict[str, str] = {}
    for point in data.get("result", {}).get("points", []):
        meta = (point.get("payload") or {}).get("metadata") or {}
        doc_id = meta.get("doc_id", "")
        source_file = meta.get("source_file") or meta.get("file_name", "")
        if doc_id and source_file:
            stem = source_file.lower().rsplit(".", 1)[0]
            registry[stem] = doc_id
            # Also index the full filename for direct matches
            registry[source_file.lower()] = doc_id
    return registry


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
    class SearchInput(BaseModel):
        query: str
        doc_id: str = ""  # Optional: restrict to a specific doc (e.g. "doc_006"). Leave empty for all docs.

    # Tokens that look like document references (appear in source_file metadata, not chunk content).
    # These must be routed to scope_doc_id (metadata filter), not used as content text filters.
    # Pattern: short alphanumeric prefix + underscore + digits (e.g. "doc_001", "inv_2024", "rpt_003").
    # Extend this pattern if your corpus uses a different naming convention.
    _DOC_ID_RE = re.compile(r'\b[a-z]{2,6}_\d+\b', re.IGNORECASE)
    # Detect content ID-like tokens: mixed alphanumeric codes (invoice numbers, transaction IDs, etc.)
    # that appear verbatim inside chunk text and are useful as Qdrant text-match filters.
    _ID_RE = re.compile(r'\b(?=[^\s]*[A-Za-z])(?=[^\s]*\d)[\w/-]{4,}\b')

    # Mutable limits dict — allows ask_agent to reduce limits on overflow retry
    _limits: dict = {
        "max_table_chars": MAX_TABLE_CHARS,
        "max_chunk_chars": MAX_CHUNK_CHARS,
        "rerank_top_n": min(rerank_top_n, MAX_TOOL_RESULTS),
    }

    @traceable(name="fetch-docs", metadata={"top_k": retrieval_top_k, "rerank_top_n": rerank_top_n})
    def _fetch_docs(query: str, doc_id: str = "") -> str | None:
        max_table_chars = _limits["max_table_chars"]
        max_chunk_chars = _limits["max_chunk_chars"]
        _rerank_top_n = _limits["rerank_top_n"]
        api_base = _to_openai_base(generation_api_base)
        doc_id = (doc_id or "").strip()
        valid_doc_scope = doc_id if _DOC_ID_RE.fullmatch(doc_id) else ""

        # If the LLM passed a document title (not a doc_XXX id), fuzzy-match against the registry.
        if not valid_doc_scope and doc_id and doc_registry:
            doc_id_lower = doc_id.lower()
            for stem, mapped_id in doc_registry.items():
                if stem in doc_id_lower or doc_id_lower in stem:
                    valid_doc_scope = mapped_id
                    break

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
            effective_scope = None  # cross-doc or no scope — parallel retrieval handles this below
        # Whether to run parallel scoped retrieval instead of a single unscoped search.
        # Activates when exactly two distinct doc_ids appear in the query and no explicit
        # scope was passed from outside — equivalent to LangGraph Send fan-out per doc.
        _parallel_doc_ids: list[str] = inferred_doc_ids if len(inferred_doc_ids) == 2 and not valid_doc_scope else []
        # Explicitly mentioned doc_ids get maximum stage1 boost regardless of embedding rank.
        explicitly_mentioned_doc_ids: set[str] = set(inferred_doc_ids)

        non_doc_ids = [m for m in _ID_RE.findall(search_query) if not _DOC_ID_RE.match(m)]
        filter_token: str | None = non_doc_ids[0] if non_doc_ids else None
        # _ID_RE requires a letter+digit mix, so pure-digit codes (invoice numbers, IDs)
        # are missed. Extract them explicitly when the query mentions "invoice" or "transaction".
        if filter_token is None:
            _pure_digit_id = re.search(r'\b(\d{5,})\b', search_query)
            if _pure_digit_id and any(w in search_query.lower() for w in ("invoice", "transaction", "record")):
                filter_token = _pure_digit_id.group(1)
        if filter_token is None:
            filter_token = _extract_table_filter_token(search_query)

        retrieval_query = _enrich_search_query(search_query)

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
                    "the", "and", "for", "with", "from",
                }
                q_tokens = {
                    t for t in re.findall(r"[a-z]{3,}", search_query.lower())
                    if t not in _STEM_STOPWORDS
                }
                seen_doc_ids: set[str] = set()
                for stem, mapped_id in doc_registry.items():
                    if mapped_id in seen_doc_ids or not _DOC_ID_RE.fullmatch(mapped_id):
                        continue
                    stem_tokens = {
                        t for t in re.findall(r"[a-z]{3,}", stem)
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
            if (
                not effective_scope
                and not _parallel_doc_ids
                and stem_match_scores
            ):
                sorted_pairs = sorted(stem_match_scores.items(), key=lambda kv: kv[1], reverse=True)
                top_id, top_score = sorted_pairs[0]
                second_score = sorted_pairs[1][1] if len(sorted_pairs) > 1 else 0
                if top_score >= 3 and (top_score - second_score) >= 2:
                    effective_scope = top_id

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
                return retrieve(
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

        hyde_hits: list[dict[str, Any]] = []
        if use_hyde:
            try:
                embed_query = _hyde(retrieval_query, api_base, generation_model)
                if embed_query and embed_query.strip().lower() != retrieval_query.strip().lower():
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

        hits = _merge_hits(raw_hits, hyde_hits)[: max(retrieval_top_k, _rerank_top_n)]

        # Inject chunks from stage1-boosted docs that the dense+sparse search missed.
        # The boost only reorders candidates — if doc_001's chunks aren't in the pool,
        # there's nothing to reorder. Force-fetch a few chunks per missing boosted doc
        # so the reranker gets to see them.
        if stage1_doc_ids and not effective_scope:
            # Older ingestions only set metadata.doc_id on document_summary chunks; PDF
            # chunks may lack it. Detect doc presence via filename regex too.
            def _hit_doc_id(hit: dict) -> str:
                meta = hit.get("metadata") or {}
                if meta.get("doc_id"):
                    return meta["doc_id"]
                src = (meta.get("source_file") or meta.get("file_name") or "").lower()
                m = _DOC_ID_RE.search(src)
                return m.group(0) if m else ""

            present_doc_ids = {_hit_doc_id(h) for h in hits}
            missing_doc_ids = [d for d in stage1_doc_ids if d and d not in present_doc_ids]
            for missing_id in missing_doc_ids:
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
                        scope_doc_id=missing_id,
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

        if not hits:
            return None

        # Split hits by chunk type: sheet_summaries are ranked by column-name keyword
        # overlap (cross-encoders are trained on passage-answer pairs, not metadata
        # discovery chunks, so they return noise scores for sheet_summaries).
        # All other chunks go through the normal BGE reranker path.
        sheet_hits = [h for h in hits if (h.get("metadata") or {}).get("chunk_type") == "sheet_summary"]
        other_hits = [h for h in hits if (h.get("metadata") or {}).get("chunk_type") != "sheet_summary"]

        _COL_STOPWORDS = {
            "what", "which", "where", "when", "who", "how", "is", "are", "was", "were",
            "the", "a", "an", "and", "or", "of", "for", "to", "in", "on", "at", "by",
            "with", "from", "that", "this", "these", "those", "be", "do", "does", "did",
            "have", "has", "had", "as", "it", "its", "into", "any", "all",
            "summary", "data", "table", "sheet", "report", "row", "column", "value",
        }
        q_terms = {
            t for t in re.sub(r"[^a-z0-9 ]", " ", retrieval_query.lower()).split()
            if t not in _COL_STOPWORDS and len(t) >= 3
        }

        def _col_overlap(hit: dict) -> int:
            content = (hit.get("content") or "").lower()
            col_line = next((ln for ln in content.splitlines() if ln.startswith("columns:")), "")
            col_tokens = {
                t for t in re.sub(r"[^a-z0-9 ]", " ", col_line).split()
                if t not in _COL_STOPWORDS and len(t) >= 3
            }
            return len(q_terms & col_tokens)

        def _sheet_sort_key(hit: dict) -> tuple[int, int]:
            overlap = _col_overlap(hit)
            # Stage 1 boost: prefer sheets from Stage 1-identified docs as a tiebreaker.
            stage1_bonus = 1 if (hit.get("metadata") or {}).get("doc_id", "") in stage1_doc_ids else 0
            return (overlap, stage1_bonus)

        ranked_sheet_hits = sorted(sheet_hits, key=_sheet_sort_key, reverse=True)

        reranked_used = False
        if ranker is not None and other_hits:
            docs = [h.get("content", "") for h in other_hits]
            # Rerank ALL candidates so we can reserve slots for stage1 docs after the
            # fact. With the previous top_n=_rerank_top_n cut, stage1 chunks the reranker
            # ranked 11th+ were silently dropped before we could promote them.
            reranked = ranker.rerank(retrieval_query, docs, top_n=len(docs))
            reranked_hits = [{**other_hits[r["index"]], "rerank_score": r["score"]} for r in reranked]
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
                    h for h in reranked_hits
                    if ((h.get("metadata") or {}).get("source_file"),
                        (h.get("metadata") or {}).get("chunk_index")) not in guaranteed_keys
                ][:remaining_slots]
                top_other = guaranteed + rest
            else:
                top_other = _merge_hits(raw_hits[: min(3, _rerank_top_n)], reranked_hits)[:_rerank_top_n]
        else:
            top_other = other_hits[:_rerank_top_n]

        # sheet_summary chunks signal "the relevant sheet has these columns" so the
        # agent can route to the Excel tool — they are not answer-bearing content.
        # Require >=2 token overlap (after stopword filter) to avoid generic matches
        # like "and/the/of"; cap at 2 to avoid crowding PDF/text chunks.
        top_sheet = [h for h in ranked_sheet_hits if _col_overlap(h) >= 2][:2]
        top_hits = (top_sheet + top_other)[:_rerank_top_n]

        # Neighbor-chunk expansion: chunker boundaries can split related content
        # (section header in chunk N, table values in N+1). For the top-5 PDF/text
        # chunks, fetch chunk_index ± 1 from the same file and append. Generic fix
        # for "right chunk, missing detail" misses across all docs.
        _NEIGHBOR_EXCLUDE_TYPES = {"sheet_summary", "document_summary", "sheet_table", "sheet_row"}
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
                neighbors_by_file[src] = _fetch_neighbor_chunks(qdrant_url, collection, src, needed)

        parts: list[str] = []
        for i, h in enumerate(top_hits, start=1):
            meta = h.get("metadata", {}) or {}
            score = h.get("rerank_score", h.get("score", 0))
            # DOC_MIN_SCORE only applies to dense cosine scores (0-1 range).
            # Reranker returns raw logits (can be negative) — top_n already limits relevance.
            if not filter_token and not reranked_used and score < DOC_MIN_SCORE:
                continue
            file_name = meta.get("file_name") or meta.get("source_file", "unknown")
            file_name = resolve_original_name(file_name)
            chunk_type = meta.get("chunk_type", "")
            sheet_name = meta.get("sheet_name")
            location = f"sheet={sheet_name}" if sheet_name else f"chunk={meta.get('chunk_index', meta.get('part', '?'))}"
            content = (h.get("content", "") or "").strip()
            is_pdf_table = "[TABLE_START]" in content
            is_sheet_table = chunk_type == "sheet_table"
            is_sheet_row = chunk_type == "sheet_row"
            # When filter_token matched an ID, extract only header + matching rows
            if filter_token and is_sheet_table:
                lines = content.splitlines()
                header_lines = [ln for ln in lines if not ln.startswith("|") or ln.startswith("| ---")]
                table_lines = [ln for ln in lines if ln.startswith("|")]
                matching = [ln for ln in table_lines if filter_token in ln]
                if matching:
                    # keep description + table header rows (first 2 table lines) + matching rows
                    sep_idx = next((i for i, ln in enumerate(table_lines) if ln.startswith("| ---")), 1)
                    content = "\n".join(header_lines[:header_lines.index(table_lines[0]) if table_lines[0] in header_lines else len(header_lines)]
                                       + table_lines[:sep_idx + 1] + matching)
            elif (is_pdf_table or is_sheet_table) and len(content) > max_table_chars:
                content = content[:max_table_chars] + "\n… (truncated)"
            elif not is_pdf_table and not is_sheet_table and not is_sheet_row and len(content) > max_chunk_chars:
                content = _best_snippet(content, retrieval_query, max_chunk_chars)
            if is_sheet_table or is_sheet_row:
                table_context = _table_context_as_key_values(content, search_query)
                if table_context:
                    content = f"{table_context}\n\nOriginal retrieved table text:\n{content}"

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

            parts.append(f"[{i}] file={file_name} {location} score={score:.4f}\n{content}")

        if not parts:
            return None
        parts.append(
            "Instruction: Use the retrieved results above to answer the user's question. "
            "Do not repeat the same search unless a different missing fact is required."
        )
        return "\n\n".join(parts)

    def search_knowledge_base(query: str, doc_id: str = "") -> str:
        """Search the knowledge base for relevant documents and tables."""
        result = _fetch_docs(query, doc_id=doc_id)
        return result if result else "No relevant information found."

    tool = StructuredTool.from_function(
        func=search_knowledge_base,
        name="search_knowledge_base",
        description="Search all knowledge sources (documents and tables). Input: a focused sub-question.",
        args_schema=SearchInput,
    )
    return tool, _limits


# ---------------------------------------------------------------------------
# Agent builder
# ---------------------------------------------------------------------------


def _to_openai_base(api_base: str) -> str:
    base = api_base.rstrip("/")
    return base if base.endswith("/v1") else f"{base}/v1"


def build_rag_agent(
    *,
    qdrant_url: str = QDRANT_URL,
    collection: str = QDRANT_COLLECTION,
    retrieval_top_k: int = RETRIEVAL_TOP_K,
    rerank_top_n: int = RERANK_TOP_N,
    reranker_model_name: str | None = RERANKER_MODEL,
    model_name: str = GENERATION_MODEL,
    generation_api_base: str = GENERATION_API_BASE,
    use_hyde: bool = True,
) -> Any:
    ranker: BGEReranker | QwenReranker | None = None
    if reranker_model_name:
        try:
            if "bge" in reranker_model_name.lower() or "mxbai" in reranker_model_name.lower():
                ranker = BGEReranker(model_name=reranker_model_name, device=RERANKER_DEVICE)
            else:
                ranker = QwenReranker(model_name=reranker_model_name, device=RERANKER_DEVICE)
            # Warm up: force model init so first real query has no latency spike
            ranker.rerank("warmup", ["warmup text"], top_n=1)
            print(f"[INFO] Reranker '{reranker_model_name}' loaded and warmed up.")
        except Exception as e:
            print(f"[WARNING] Reranker failed to load: {e}. Falling back to no reranker.")
            ranker = None

    # API key resolution: LiteLLM master key → Groq → OpenAI → dummy
    # When routing through the LiteLLM proxy the master key is used (if set);
    # direct-to-provider calls fall back to the provider-specific key.
    _api_key = (
        LITELLM_MASTER_KEY
        or os.getenv("GROQ_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or "EMPTY"
    )
    llm = ChatOpenAI(
        model=model_name,
        base_url=_to_openai_base(generation_api_base),
        api_key=_api_key,
        temperature=0,
        max_tokens=2048,
    )

    doc_registry = _build_doc_registry(qdrant_url, collection)
    if doc_registry:
        print(f"[INFO] Doc registry built: {len(doc_registry)} source file entries.")

    tool, _rag_limits = _make_unified_tool(
        qdrant_url=qdrant_url,
        collection=collection,
        retrieval_top_k=retrieval_top_k,
        rerank_top_n=rerank_top_n,
        ranker=ranker,
        generation_api_base=generation_api_base,
        generation_model=model_name,
        use_hyde=use_hyde,
        doc_registry=doc_registry,
    )

    excel_store = DuckDBStore()
    tools = [tool] + build_excel_agent_tools(excel_store)

    system_prompt = _build_system_prompt(model_name)
    agent = create_react_agent(model=llm, tools=tools, prompt=system_prompt, name="vault-rag")
    agent._rag_limits = _rag_limits  # type: ignore[attr-defined]
    agent._system_prompt = system_prompt  # type: ignore[attr-defined]
    agent._generation_api_base = _to_openai_base(generation_api_base)  # type: ignore[attr-defined]
    agent._generation_model = model_name  # type: ignore[attr-defined]
    return agent


# ---------------------------------------------------------------------------
# Invoke
# ---------------------------------------------------------------------------


def _extract_refs(tool_content: str) -> str:
    """Return a compact reference summary from any tool result.

    For search_knowledge_base: shows [N] file=... chunk=... score=... headers.
    For query_excel: shows the first 3 lines of the DataFrame output.
    """
    lines: list[str] = []
    for line in tool_content.splitlines():
        stripped = line.strip()
        if re.match(r"^\[\d+\] file=", stripped):
            lines.append(stripped)
        elif stripped.startswith("Sources (sheets):") or stripped.startswith("Summary:"):
            lines.append(stripped)
    if lines:
        return "\n".join(lines)
    # query_excel or other plain-text tool — show a short preview
    preview = [ln for ln in tool_content.splitlines() if ln.strip()][:3]
    return "\n".join(preview) if preview else "(no results)"


def _get_langfuse():
    """Return a Langfuse client if configured, else None."""
    host = os.getenv("LANGFUSE_HOST")
    pk = os.getenv("LANGFUSE_PUBLIC_KEY")
    sk = os.getenv("LANGFUSE_SECRET_KEY")
    if not (host and pk and sk):
        return None
    try:
        from langfuse import Langfuse
        return Langfuse(host=host, public_key=pk, secret_key=sk)
    except ImportError:
        return None


def ask_agent(
    agent: Any,
    query: str,
    history: list[dict] | None = None,
    show_tool_uses: bool = False,
    retrieved_contexts: list[str] | None = None,
) -> str:
    """Run the RAG agent on a query and return the final answer.

    Args:
        history: Prior conversation turns as [{"role": "user"/"assistant", "content": str}].
        retrieved_contexts: If provided, tool result chunks are appended here (same contract as stream_agent).
    """
    from openai import BadRequestError

    if _has_empty_reference_placeholder(query):
        return "Unsupported"

    lf = _get_langfuse()
    trace = lf.trace(name="rag-agent", input=query) if lf else None

    messages: list = []
    for turn in (history or []):
        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["content"]))
        else:
            messages.append(AIMessage(content=turn["content"]))
    messages.append(HumanMessage(content=query))

    _invoke_input = {"messages": messages}
    _invoke_config = {"recursion_limit": 20}

    _limits: dict = getattr(agent, "_rag_limits", {})

    try:
        result = agent.invoke(_invoke_input, config=_invoke_config)
    except BadRequestError as exc:
        err_str = str(exc).lower()
        # Groq sometimes rejects the model's final-answer step as a malformed tool call.
        # The actual answer text is in the 'failed_generation' field — extract it directly.
        if "failed to call a function" in err_str or "tool_use_failed" in err_str:
            body = getattr(exc, "body", {}) or {}
            failed_gen = body.get("failed_generation", "") if isinstance(body, dict) else ""
            # Only use failed_generation if it looks like a final answer, not a reasoning
            # preamble ("I need to search...", "Let me look...") that never reached a conclusion.
            _reasoning_prefix = re.compile(
                r"^(i need to|let me|i will|i'll|to answer|first[,\s]|i should"
                r"|to find|in order to|i must|i'll need|i have to"
                r"|the (question|user) (ask|want|need|is)|based on the (question|query))",
                re.IGNORECASE,
            )
            if failed_gen and not _reasoning_prefix.match(failed_gen.strip()):
                failed_gen = failed_gen.strip()
                return "Unsupported" if _looks_like_bad_final_answer(failed_gen) else failed_gen
        if "input tokens" in err_str or "context" in err_str or "400" in err_str:
            print("[WARN] Context overflow — retrying with fewer chunks.")
            _limits["rerank_top_n"] = max(3, _limits.get("rerank_top_n", RERANK_TOP_N) // 2)
            try:
                result = agent.invoke(_invoke_input, config=_invoke_config)
            finally:
                _limits["rerank_top_n"] = min(RERANK_TOP_N, MAX_TOOL_RESULTS)
        else:
            raise

    messages: list[Any] = result.get("messages", [])

    tool_results: dict[str, str] = {}
    tool_contexts: list[str] = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            tool_results[msg.tool_call_id] = msg.content
            tool_contexts.append(msg.content)
            if retrieved_contexts is not None:
                parts = re.split(r"\n\n(?=\[\d+\])", msg.content.strip())
                cleaned = [p.strip() for p in parts if p.strip()]
                if cleaned:
                    retrieved_contexts.append("---CALL_BOUNDARY---")
                    retrieved_contexts.extend(cleaned)

    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                if show_tool_uses:
                    print(f"[TOOL_CALL] {tc['name']} args={json.dumps(tc['args'], ensure_ascii=False)}")
                if trace is not None:
                    result_content = tool_results.get(tc["id"], "")
                    trace.span(
                        name=tc["name"],
                        input=tc["args"],
                        output=_extract_refs(result_content),
                    )
        elif isinstance(msg, ToolMessage) and show_tool_uses:
            print(f"[TOOL_RESULT] {msg.name} ->\n{_extract_refs(msg.content)}\n")

    answer = "No answer generated."
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            text = str(msg.content).strip()
            answer = _strip_think(text)
            break

    api_base = getattr(agent, "_generation_api_base", None)
    model_name = getattr(agent, "_generation_model", None)
    if api_base and model_name and tool_contexts and (
        not _looks_like_bad_final_answer(answer) or _is_multi_part_query(query)
    ):
        answer = _repair_incomplete_answer(query, answer, tool_contexts, api_base, model_name)

    if _looks_like_bad_final_answer(answer) and tool_contexts:
        if api_base and model_name:
            try:
                fallback = _direct_answer_from_context(query, tool_contexts, api_base, model_name)
                if not _looks_like_bad_final_answer(fallback) or answer.strip() != "Unsupported":
                    answer = fallback
                if _looks_like_bad_final_answer(answer):
                    direct = _direct_retrieval_answer(query, api_base, model_name)
                    if not _looks_like_bad_final_answer(direct) or answer.strip() != "Unsupported":
                        answer = direct
            except Exception as exc:
                print(f"[WARN] Direct context answer fallback failed ({type(exc).__name__}): {exc}")
    elif _looks_like_bad_final_answer(answer) and not tool_contexts:
        api_base = getattr(agent, "_generation_api_base", None)
        model_name = getattr(agent, "_generation_model", None)
        if api_base and model_name:
            try:
                answer = _direct_retrieval_answer(query, api_base, model_name)
            except Exception as exc:
                print(f"[WARN] Direct retrieval answer fallback failed ({type(exc).__name__}): {exc}")
                if "search_knowledge_base" in answer:
                    answer = "Unsupported"
        elif "search_knowledge_base" in answer:
            answer = "Unsupported"

    answer = _normalize_unsupported(answer)
    if _is_bare_filename_answer(query, answer):
        answer = "Unsupported"

    if trace is not None:
        trace.update(output=answer)
        lf.flush()

    return answer


def stream_agent(
    agent: Any,
    query: str,
    history: list[dict] | None = None,
    show_tool_uses: bool = False,
    collected_chunks: list[str] | None = None,
) -> Generator[str, None, None]:
    """Stream the agent's final answer token-by-token.

    Yields string fragments as they arrive from the LLM.
    Tool calls are not yielded (optionally printed if show_tool_uses=True).
    Qwen3 <think>...</think> blocks are suppressed.

    Args:
        history: Prior conversation turns as [{"role": "user"/"assistant", "content": str}].
        collected_chunks: If provided, tool result chunks are appended to this list.
    """
    if _has_empty_reference_placeholder(query):
        yield "Unsupported"
        return

    messages: list = []
    for turn in (history or []):
        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["content"]))
        else:
            messages.append(AIMessage(content=turn["content"]))
    messages.append(HumanMessage(content=query))

    _invoke_input = {"messages": messages}
    _invoke_config = {"recursion_limit": 20}

    # State machine for stripping <think> blocks mid-stream
    _think_buf = ""
    _in_think = False

    def _filter(token: str) -> str:
        """Buffer tokens and strip <think>...</think> spans."""
        nonlocal _think_buf, _in_think
        _think_buf += token
        out = ""
        while True:
            if _in_think:
                end = _think_buf.find("</think>")
                if end == -1:
                    return out
                _think_buf = _think_buf[end + len("</think>"):]
                _in_think = False
            else:
                start = _think_buf.find("<think>")
                if start == -1:
                    out += _think_buf
                    _think_buf = ""
                    return out
                out += _think_buf[:start]
                _think_buf = _think_buf[start + len("<think>"):]
                _in_think = True

    from openai import APIError, BadRequestError

    _reasoning_prefix = re.compile(
        r"^(i need to|let me|i will|i'll|to answer|first[,\s]|i should"
        r"|to find|in order to|i must|i'll need|i have to"
        r"|the (question|user) (ask|want|need|is)|based on the (question|query))",
        re.IGNORECASE,
    )

    # Buffer text tokens until the first tool call completes so we don't yield
    # reasoning preambles ("I need to search...", "Let me look...") as answers.
    _tool_used = False
    _pre_tool_buf: list[str] = []
    _final_buf: list[str] = []
    _tool_contexts: list[str] = []
    _limits: dict = getattr(agent, "_rag_limits", {})

    try:
        for chunk, metadata in agent.stream(_invoke_input, config=_invoke_config, stream_mode="messages"):
            if isinstance(chunk, AIMessageChunk):
                if chunk.tool_call_chunks:
                    continue
                if chunk.content:
                    filtered = _filter(str(chunk.content))
                    if filtered:
                        if _tool_used:
                            _final_buf.append(filtered)
                        else:
                            _pre_tool_buf.append(filtered)
            elif isinstance(chunk, ToolMessage):
                _tool_used = True
                _pre_tool_buf.clear()
                _tool_contexts.append(chunk.content)
                if collected_chunks is not None:
                    parts = re.split(r"\n\n(?=\[\d+\])", chunk.content.strip())
                    cleaned = [p.strip() for p in parts if p.strip()]
                    if cleaned:
                        # Mark tool-call boundaries so the API can group/prioritize
                        # chunks per call (later/scoped calls usually contain the
                        # answer-bearing chunks; earlier broad calls return noise).
                        collected_chunks.append("---CALL_BOUNDARY---")
                        collected_chunks.extend(cleaned)
                if show_tool_uses:
                    print(f"\n[TOOL_RESULT] {chunk.name} ->\n{_extract_refs(chunk.content)}\n")
    except (BadRequestError, APIError) as exc:
        err_str = str(exc).lower()
        if "context_length_exceeded" in err_str or "context overflow" in err_str or "reduce the length" in err_str:
            _limits["rerank_top_n"] = max(3, _limits.get("rerank_top_n", MAX_TOOL_RESULTS) // 2)
            try:
                api_base = getattr(agent, "_generation_api_base", None)
                model_name = getattr(agent, "_generation_model", None)
                answer = (
                    _direct_retrieval_answer(query, api_base, model_name)
                    if api_base and model_name
                    else ask_agent(agent, query, history=history, retrieved_contexts=collected_chunks)
                )
            finally:
                _limits["rerank_top_n"] = min(RERANK_TOP_N, MAX_TOOL_RESULTS)
            yield answer
            return
        if "failed to call a function" in err_str or "tool_use_failed" in err_str:
            if _tool_used:
                # The provider can raise after the final answer. Flush buffered answer text
                # instead of losing it.
                answer = "".join(_final_buf).strip()
                if _looks_like_bad_final_answer(answer) and _tool_contexts:
                    api_base = getattr(agent, "_generation_api_base", None)
                    model_name = getattr(agent, "_generation_model", None)
                    if api_base and model_name:
                        try:
                            fallback = _direct_answer_from_context(query, _tool_contexts, api_base, model_name)
                            if not _looks_like_bad_final_answer(fallback) or answer != "Unsupported":
                                answer = fallback
                            if _looks_like_bad_final_answer(answer):
                                direct = _direct_retrieval_answer(query, api_base, model_name)
                                if not _looks_like_bad_final_answer(direct) or answer != "Unsupported":
                                    answer = direct
                        except Exception as fallback_exc:
                            print(
                                "[WARN] Direct context answer fallback failed "
                                f"({type(fallback_exc).__name__}): {fallback_exc}"
                            )
                if answer:
                    yield answer
                return
            # Error happened before any tool call; try to extract answer from failed_generation body.
            body = getattr(exc, "body", {}) or {}
            failed_gen = body.get("failed_generation", "") if isinstance(body, dict) else ""
            if failed_gen and not _reasoning_prefix.match(failed_gen.strip()):
                failed_gen = failed_gen.strip()
                yield "Unsupported" if _looks_like_bad_final_answer(failed_gen) else failed_gen
                return
        raise
    except Exception as exc:
        err_str = str(exc).lower()
        if "context_length_exceeded" in err_str or "context overflow" in err_str or "reduce the length" in err_str:
            _limits["rerank_top_n"] = max(3, _limits.get("rerank_top_n", MAX_TOOL_RESULTS) // 2)
            try:
                api_base = getattr(agent, "_generation_api_base", None)
                model_name = getattr(agent, "_generation_model", None)
                answer = (
                    _direct_retrieval_answer(query, api_base, model_name)
                    if api_base and model_name
                    else ask_agent(agent, query, history=history, retrieved_contexts=collected_chunks)
                )
            finally:
                _limits["rerank_top_n"] = min(RERANK_TOP_N, MAX_TOOL_RESULTS)
            yield answer
            return
        raise

    # If the model answered without calling any tool, flush the pre-tool buffer.
    if not _tool_used:
        answer = "".join(_pre_tool_buf).strip()
        if _think_buf and not _in_think:
            answer += _think_buf
        if _looks_like_bad_final_answer(answer):
            api_base = getattr(agent, "_generation_api_base", None)
            model_name = getattr(agent, "_generation_model", None)
            if api_base and model_name:
                try:
                    answer = _direct_retrieval_answer(query, api_base, model_name)
                except Exception as exc:
                    print(f"[WARN] Direct retrieval answer fallback failed ({type(exc).__name__}): {exc}")
                    if "search_knowledge_base" in answer:
                        answer = "Unsupported"
            elif "search_knowledge_base" in answer:
                answer = "Unsupported"
        if answer:
            normalized = _normalize_unsupported(answer)
            if _is_bare_filename_answer(query, normalized):
                normalized = "Unsupported"
            yield normalized
        return

    # Flush any remaining buffered text (e.g. trailing content after last </think>)
    if _think_buf and not _in_think:
        _final_buf.append(_think_buf)

    answer = "".join(_final_buf).strip()
    api_base = getattr(agent, "_generation_api_base", None)
    model_name = getattr(agent, "_generation_model", None)
    if api_base and model_name and _tool_contexts and (
        not _looks_like_bad_final_answer(answer) or _is_multi_part_query(query)
    ):
        answer = _repair_incomplete_answer(query, answer, _tool_contexts, api_base, model_name)

    if _looks_like_bad_final_answer(answer) and _tool_contexts:
        if api_base and model_name:
            try:
                fallback = _direct_answer_from_context(query, _tool_contexts, api_base, model_name)
                if not _looks_like_bad_final_answer(fallback) or answer != "Unsupported":
                    answer = fallback
                if _looks_like_bad_final_answer(answer):
                    direct = _direct_retrieval_answer(query, api_base, model_name)
                    if not _looks_like_bad_final_answer(direct) or answer != "Unsupported":
                        answer = direct
            except Exception as exc:
                print(f"[WARN] Direct context answer fallback failed ({type(exc).__name__}): {exc}")

    if answer:
        normalized = _normalize_unsupported(answer)
        if _is_bare_filename_answer(query, normalized):
            normalized = "Unsupported"
        yield normalized


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Agentic RAG: unified document + table search.")
    parser.add_argument("--query", required=True, help="User question")
    parser.add_argument("--qdrant-url", default=QDRANT_URL)
    parser.add_argument("--collection", default=QDRANT_COLLECTION)
    parser.add_argument("--top-k", type=int, default=RETRIEVAL_TOP_K)
    parser.add_argument("--rerank-top-n", type=int, default=RERANK_TOP_N)
    parser.add_argument(
        "--reranker-model-name",
        default=RERANKER_MODEL,
        help="Optional local reranker model name. Empty disables reranker.",
    )
    parser.add_argument("--model-name", default=GENERATION_MODEL)
    parser.add_argument("--generation-api-base", default=GENERATION_API_BASE)
    parser.add_argument("--show-tool-uses", action="store_true")
    parser.add_argument("--stream", action="store_true", help="Stream the answer token-by-token.")
    parser.add_argument("--no-hyde", action="store_true", help="Disable HyDE query expansion.")
    args = parser.parse_args()

    agent = build_rag_agent(
        qdrant_url=args.qdrant_url,
        collection=args.collection,
        retrieval_top_k=args.top_k,
        rerank_top_n=args.rerank_top_n,
        reranker_model_name=args.reranker_model_name or None,
        model_name=args.model_name,
        generation_api_base=args.generation_api_base,
        use_hyde=not args.no_hyde,
    )

    if args.stream:
        print(f"[query] {args.query}\n[answer] ", end="", flush=True)
        for token in stream_agent(agent, args.query, show_tool_uses=args.show_tool_uses):
            print(token, end="", flush=True)
        print()
    else:
        answer = ask_agent(agent, args.query, show_tool_uses=args.show_tool_uses)
        print(json.dumps({"query": args.query, "answer": answer}, ensure_ascii=False, indent=2))

    try:
        from langsmith import Client
        Client().flush()
    except Exception:
        pass


if __name__ == "__main__":
    main()
