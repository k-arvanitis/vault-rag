"""Agentic RAG using a LangGraph ReAct agent with native tool calling.

Two tools are registered: search_knowledge_base (text/PDF retrieval over Qdrant)
and query_excel (text-to-SQL over spreadsheet rows in DuckDB). route_question
resolves which tool a question needs from the modality of its best-matching
document summary, so the agent does not have to infer it from question wording.

Public API: build_rag_agent() constructs the agent; ask_agent() / stream_agent()
run it for a single query and finalize the answer.

Called by: src/api.py and slack_app.py (build + ask/stream), rag_cli.py (CLI).
Calls into: src/prompts.py (system prompt), src/retriever.py (route_question),
src/tools/retrieval_tool.py + src/tools/excel.py (the two agent tools),
src/duckdb_store.py, src/reranker.py, src/llm_utils.py, and src/answer_quality.py
for the answer-finalization pipeline.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Generator

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from src.answer_quality import (
    _direct_answer_from_context,
    _direct_retrieval_answer,
    _has_empty_reference_placeholder,
    _is_bare_filename_answer,
    _is_multi_part_query,
    _looks_like_bad_final_answer,
    _normalize_unsupported,
    _repair_incomplete_answer,
    _strip_think,
    _verify_grounded,
)
from src.config import (
    FREE_LLM_API_KEY,
    GENERATION_API_BASE,
    GENERATION_MODEL,
    GROQ_API_KEY,
    LITELLM_MASTER_KEY,
    LLM_REQUEST_TIMEOUT_S,
    MAX_TOOL_RESULTS,
    OPENROUTER_API_KEY,
    POST_GENERATION_VERIFY_ENABLED,
    QDRANT_COLLECTION,
    QDRANT_URL,
    RERANK_TOP_N,
    RERANKER_DEVICE,
    RERANKER_ENABLED,
    RERANKER_MODEL,
    RETRIEVAL_TOP_K,
)
from src.duckdb_store import DuckDBStore
from src.llm_utils import (
    _is_thinking_model,
    _openrouter_provider_extra_body,
    _to_openai_base,
)
from src.prompts import PROMPT_VERSION, compose_system_prompt
from src.reranker import BGEReranker, QwenReranker
from src.retriever import (
    retrieve,
)
from src.tools.calculator import build_calculator_tool
from src.tools.excel import build_excel_agent_tools
from src.tools.retrieval_tool import _make_unified_tool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt — templates live in src/prompts.py; this is the model-specific wiring
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_WITH_EXCEL = compose_system_prompt(with_excel=True)


def _build_system_prompt(model_name: str) -> str:
    """Build system prompt. query_excel is always registered so the Excel variant is always used."""
    prefix = "/no_think " if _is_thinking_model(model_name) else ""
    return prefix + _SYSTEM_PROMPT_WITH_EXCEL


# Keep a module-level default for callers that don't pass a model name
SYSTEM_PROMPT = _build_system_prompt(GENERATION_MODEL)


# A reasoning preamble ("I need to search...", "Let me look...") that a model
# emits instead of a final answer — used to reject failed_generation text that
# never reached a conclusion.
_REASONING_PREFIX_RE = re.compile(
    r"^(i need to|let me|i will|i'll|to answer|first[,\s]|i should"
    r"|to find|in order to|i must|i'll need|i have to"
    r"|the (question|user) (ask|want|need|is)|based on the (question|query))",
    re.IGNORECASE,
)


def _is_context_overflow(err_str: str) -> bool:
    """True for provider errors signalling the prompt exceeded the context window.

    Matches the specific overflow signatures emitted by OpenAI-compatible
    providers (OpenAI, Groq, vLLM) — not a bare "context" or HTTP "400", which
    also occur on unrelated errors. err_str must already be lower-cased; ask_agent
    and stream_agent share this so the same error is handled the same on both.
    """
    return any(
        sign in err_str
        for sign in (
            "context_length_exceeded",
            "context window",
            "context overflow",
            "maximum context",
            "reduce the length",
            "input tokens",
        )
    )


# ---------------------------------------------------------------------------
# Deterministic tool routing — decide search_knowledge_base vs query_excel from
# the modality of the document whose summary best matches the question, instead
# of leaving the choice to the LLM's reading of the question wording.
# ---------------------------------------------------------------------------

_TABLE_EXTS = (".xlsx", ".xls", ".csv")

# Questions about a spreadsheet's own title/description/coverage, not its row
# data — these are answered from the file's document/sheet summary text, never
# from SQL. See route_question() for why extension-based routing alone can't
# tell these apart from real data lookups.
_DOC_METADATA_QUESTION_RE = re.compile(
    r"\bwhat (?:year|date) is (?:this|the) .*titled\b"
    r"|\bdocument title\b"
    r"|\btitle (?:shown|is) at the top\b"
    r"|\bwhat is (?:being )?recorded\b"
    r"|\bwhat is this (?:document|dataset|file|spreadsheet|workbook) about\b",
    re.IGNORECASE,
)


def route_question(
    question: str,
    qdrant_url: str = QDRANT_URL,
    collection: str = QDRANT_COLLECTION,
) -> dict[str, str]:
    """Resolve which tool a question needs by matching it against the index.

    Spreadsheets keep their rows in DuckDB and only their summaries in Qdrant,
    while PDFs have their full text indexed as chunks. So a question whose top
    hits are .xlsx/.csv chunks is a structured-data question (-> query_excel),
    and one whose top hits are .pdf chunks is a document question
    (-> search_knowledge_base). Decision is the majority modality of the top-3
    hits. Returns {modality, source_file}; modality is "" when nothing matched,
    so the caller falls back to the agent's own tool choice.
    """
    # Retrieve the question's nearest chunks; any failure or empty result yields
    # an empty modality so the caller leaves tool choice to the agent.
    try:
        hits = retrieve(
            query=question,
            top_k=5,
            qdrant_url=qdrant_url,
            collection=collection,
            use_qdrant=True,
        )
    except Exception:
        return {"modality": "", "source_file": ""}
    if not hits:
        return {"modality": "", "source_file": ""}

    def _source(hit: dict) -> str:
        """Return a hit's source filename, or '' when absent."""
        meta = hit.get("metadata") or {}
        return meta.get("source_file") or meta.get("file_name") or ""

    # Confidence gate: don't emit a directive when the top-3 hits don't even
    # agree on which DOCUMENT is meant. A generic financial-lookup phrasing
    # ("total amount for transaction number X") can rank three unrelated
    # documents in the top-3 with no single one dominating -- reproduced
    # directly: hits[0] came back as a wrong document (an unrelated invoice
    # PDF) for a question whose real answer was in a completely different
    # spreadsheet that didn't even place in the top-5. The directive told the
    # agent "use X, not the other" with unearned confidence and it complied,
    # querying the wrong document instead of abstaining or asking the agent's
    # own judgment (which, undirected, answered this exact case correctly
    # every time in testing). Require at least 2 of the top-3 hits to agree
    # on the source document before trusting hits[0] enough to route on it.
    top = hits[:3]
    top_source = _source(hits[0])
    if sum(1 for h in top if _source(h) == top_source) < 2:
        return {"modality": "", "source_file": ""}

    # Majority vote over the top-3 hits: more table-extension sources -> excel.
    table_votes = sum(1 for h in top if _source(h).lower().endswith(_TABLE_EXTS))
    is_table = table_votes > len(top) - table_votes

    # Exception: a question about the FILE ITSELF (its title, what it records,
    # what year it covers) is answered by the summary chunk's own text, not by
    # querying rows — even though the source file is .xlsx/.csv. Row data never
    # enters Qdrant (only document/sheet summaries do), so every xlsx-sourced hit
    # here is a summary chunk regardless of question type; extension-based voting
    # alone can't tell "data question about this file" from "question about this
    # file" apart. Only override to document modality when the question's own
    # phrasing is clearly about the document/dataset itself, not a data lookup.
    if is_table and _DOC_METADATA_QUESTION_RE.search(question):
        is_table = False
    return {
        "modality": "excel" if is_table else "document",
        "source_file": _source(hits[0]),
    }


def routing_directive(route: dict[str, str]) -> str:
    """Build the routing instruction to prepend to a question.

    Returns "" when route_question() could not resolve a modality, leaving the
    agent's own tool selection untouched.
    """
    modality = route.get("modality")
    if not modality:
        return ""
    name = (route.get("source_file") or "").split("/")[-1] or "the matched document"
    if modality == "excel":
        tool, kind = "query_excel", "a spreadsheet (its rows live in a DuckDB table)"
    else:
        tool, kind = "search_knowledge_base", "a text document"
    return (
        f"[ROUTING — the document summaries indicate this question concerns "
        f'"{name}", {kind}. Use the {tool} tool and not the other one.]\n\n'
    )


def _build_doc_registry(qdrant_url: str, collection: str) -> dict[str, str]:
    """Return {lowercased_source_file_stem: doc_id} by scrolling a sample of Qdrant points.

    Used for fuzzy title matching when the LLM passes a document title instead of a doc_id.
    Returns empty dict on failure so the caller degrades gracefully.
    """
    from urllib.request import Request, urlopen

    # Scroll the Qdrant collection for document_summary points (one per document).
    base = qdrant_url.rstrip("/")
    url = f"{base}/collections/{collection}/points/scroll"
    body_bytes = json.dumps(
        {
            "limit": 250,
            "with_payload": True,
            "with_vector": False,
            "filter": {
                "must": [
                    {
                        "key": "metadata.chunk_type",
                        "match": {"value": "document_summary"},
                    }
                ]
            },
        }
    ).encode()
    req = Request(
        url,
        data=body_bytes,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
    except Exception:
        return {}

    # Index each document by both its filename stem and full filename -> doc_id.
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


# ---------------------------------------------------------------------------
# Agent construction — wire reranker, LLM, tools and doc registry into a ReAct agent
# ---------------------------------------------------------------------------


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
    """Build the LangGraph ReAct agent — reranker, LLM and the retrieval + Excel tools — with the limits/model metadata ask_agent reads."""
    # Load the cross-encoder reranker (BGE/mxbai vs Qwen by model name); failure
    # degrades gracefully to no reranker rather than aborting agent construction.
    ranker: BGEReranker | QwenReranker | None = None
    if RERANKER_ENABLED and reranker_model_name:
        try:
            if (
                "bge" in reranker_model_name.lower()
                or "mxbai" in reranker_model_name.lower()
            ):
                ranker = BGEReranker(
                    model_name=reranker_model_name, device=RERANKER_DEVICE
                )
            else:
                ranker = QwenReranker(
                    model_name=reranker_model_name, device=RERANKER_DEVICE
                )
            # Warm up: force model init so first real query has no latency spike
            ranker.rerank("warmup", ["warmup text"], top_n=1)
            print(f"[INFO] Reranker '{reranker_model_name}' loaded and warmed up.")
        except Exception as e:
            print(
                f"[WARNING] Reranker failed to load: {e}. Falling back to no reranker."
            )
            ranker = None

    # API key resolution keyed on the actual base URL — LITELLM_MASTER_KEY only
    # authenticates the LiteLLM proxy itself, so it must not be picked when
    # generation_api_base points directly at a provider (e.g. temporarily
    # bypassing the proxy, see .env). Bug found 2026-07-03: this used to pick
    # LITELLM_MASTER_KEY unconditionally whenever it was set, regardless of
    # which base URL was actually configured, silently sending the wrong key
    # to the real provider API. Uses src/config.py's parsed constants, not
    # os.getenv() — pydantic-settings reads .env directly and does not
    # populate os.environ, so os.getenv() silently returns None unless
    # something else already called load_dotenv() in this process.
    _base_for_key = generation_api_base.lower()
    if "localhost:4000" in _base_for_key or "127.0.0.1:4000" in _base_for_key:
        _api_key = LITELLM_MASTER_KEY or "EMPTY"
    elif "localhost:3011" in _base_for_key or "127.0.0.1:3011" in _base_for_key:
        _api_key = FREE_LLM_API_KEY or "EMPTY"
    elif "openrouter.ai" in _base_for_key:
        _api_key = OPENROUTER_API_KEY or "EMPTY"
    elif "groq.com" in _base_for_key:
        _api_key = GROQ_API_KEY or "EMPTY"
    else:
        _api_key = GROQ_API_KEY or LITELLM_MASTER_KEY or "EMPTY"
    # Generation LLM, deterministic (temperature 0) and capped at 2048 output tokens.
    logger.info("Building agent prompt_version=%s model=%s", PROMPT_VERSION, model_name)
    # max_retries is a native ChatOpenAI/OpenAI-client param — retries happen at the
    # HTTP layer. NOT .with_retry(): that wraps the model in a RunnableRetry, which
    # create_react_agent can't call .bind_tools() on (AttributeError at agent build).
    llm = ChatOpenAI(
        model=model_name,
        base_url=_to_openai_base(generation_api_base),
        api_key=_api_key,
        temperature=0,
        max_tokens=2048,
        max_retries=3,
        timeout=LLM_REQUEST_TIMEOUT_S,
        extra_body=_openrouter_provider_extra_body(generation_api_base),
    )

    # Filename -> doc_id lookup used to resolve titles the LLM passes instead of ids.
    doc_registry = _build_doc_registry(qdrant_url, collection)
    if doc_registry:
        print(f"[INFO] Doc registry built: {len(doc_registry)} source file entries.")

    # Build the retrieval tool (search_knowledge_base) and its runtime limits dict.
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

    # Add the Excel text-to-SQL tool (query_excel) backed by the DuckDB store, plus
    # the controlled arithmetic tool for prose-retrieved numbers.
    excel_store = DuckDBStore()
    tools = [tool] + build_excel_agent_tools(excel_store) + [build_calculator_tool()]

    # Assemble the ReAct agent and stash metadata ask_agent/stream_agent read back
    # off the agent object (the _rag_limits dict is mutated on context overflow).
    system_prompt = _build_system_prompt(model_name)
    agent = create_react_agent(
        model=llm, tools=tools, prompt=system_prompt, name="vault-rag"
    )
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
        elif stripped.startswith("Sources (sheets):") or stripped.startswith(
            "Summary:"
        ):
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


# ---------------------------------------------------------------------------
# Answer-finalization pipeline
#
# After the agent produces a raw answer, ask_agent and stream_agent run it
# through these stages, in this order:
#   1. _repair_incomplete_answer  (answer_quality) — coverage-driven re-retrieval
#                                  when a multi-part answer is missing a part
#   2. _context_fallback_answer   — re-answer from the retrieved context, then
#                                  from a fresh retrieval, when the answer is
#                                  bad AND tool contexts exist
#   3. _retrieval_only_answer     — last-resort retrieval answer when the answer
#                                  is bad and there were NO tool contexts
#   4. _normalize_final           — canonical 'Unsupported' / bare-filename checks
#
# (_overflow_fallback_answer, below, is a separate error path — context overflow.)
# ---------------------------------------------------------------------------


def _normalize_final(query: str, answer: str) -> str:
    """Collapse verbose 'not found' answers to 'Unsupported' and reject answers
    that are just a filename with no extracted value."""
    answer = _normalize_unsupported(answer)
    if _is_bare_filename_answer(query, answer):
        return "Unsupported"
    return answer


def _apply_grounding_check(
    query: str,
    answer: str,
    tool_contexts: list[str],
    api_base: str | None,
    model_name: str | None,
    excel_only: bool = False,
    skip: bool = False,
    trace: Any = None,
) -> str:
    """Downgrade to Unsupported if the post-generation grounding check fails.

    No-op when the flag is off, the answer is already Unsupported, there was
    no retrieved context to check against, the endpoint/model isn't known
    (e.g. called before the agent's metadata attributes are set), the answer
    came only from query_excel, or skip is set.

    The excel_only skip matters: query_excel's tool result is the final
    extracted VALUE (e.g. "Doncaster Mbc"), not row evidence — there's no prose
    context to verify a claim like "this supplier matches Department=X and
    NET Amount=Y" against, since the joining columns never appear in the tool
    output. Asking the check anyway means the judge sees answer == context
    verbatim and (correctly, from its perspective) can't confirm the unstated
    claim, so it says NO — throwing away a demonstrably correct answer.
    Verified directly: "Doncaster Mbc" / context "Doncaster Mbc" -> NO; the
    same answer with the actual row ("Department: BUSINESS DONCASTER, NET
    Amount: 206.0, Supplier Name: Doncaster Mbc") -> YES. The eval's own scorer
    already exempts Excel questions from faithfulness for the same reason
    (see run_eval.py's _is_excel_question); this brings the runtime check in
    line with that.

    The skip flag covers a separate case: comparison questions, which already
    get a dedicated doc-coverage retry in answer_pipeline.py. Reproduced live
    with gpt-oss-120b: its grounding judge is stricter on inferential/
    comparative claims than the generation model itself, flagging a correct
    answer ("X has the longer extension, based on Y vs Z") as ungrounded and
    downgrading it to Unsupported ~1/3 of the time even with both documents
    present in tool_contexts. The dedicated retry already checks the thing
    that matters here (does the answer cover every named document) without
    this false-positive risk.
    """
    if (
        not POST_GENERATION_VERIFY_ENABLED
        or not answer
        or answer.strip().lower() == "unsupported"
        or not tool_contexts
        or not api_base
        or not model_name
        or excel_only
        or skip
    ):
        return answer
    grounded = _verify_grounded(query, answer, tool_contexts, api_base, model_name)
    if trace is not None:
        trace.span(
            name="grounding-check",
            input={"answer": answer},
            output={"grounded": grounded},
        )
    if not grounded:
        return "Unsupported"
    return answer


def _context_fallback_answer(
    query: str,
    answer: str,
    contexts: list[str],
    api_base: str,
    model_name: str,
) -> str:
    """Re-answer a bad final answer from the retrieved context, then from a
    fresh non-agentic retrieval. A grounded 'Unsupported' is only overwritten
    by a fallback that is itself a real answer."""
    try:
        fallback = _direct_answer_from_context(query, contexts, api_base, model_name)
        if not _looks_like_bad_final_answer(fallback) or answer != "Unsupported":
            answer = fallback
        if _looks_like_bad_final_answer(answer):
            direct = _direct_retrieval_answer(query, api_base, model_name)
            if not _looks_like_bad_final_answer(direct) or answer != "Unsupported":
                answer = direct
    except Exception as exc:
        print(
            f"[WARN] Direct context answer fallback failed ({type(exc).__name__}): {exc}"
        )
    return answer


def _retrieval_only_answer(
    query: str,
    answer: str,
    api_base: str | None,
    model_name: str | None,
) -> str:
    """Last-resort non-agentic retrieval answer when the agent produced a bad
    answer and there were no tool contexts to fall back on."""
    if api_base and model_name:
        try:
            return _direct_retrieval_answer(query, api_base, model_name)
        except Exception as exc:
            print(
                f"[WARN] Direct retrieval answer fallback failed ({type(exc).__name__}): {exc}"
            )
    return "Unsupported" if "search_knowledge_base" in answer else answer


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

    # Guard: questions with an empty reference slot are unanswerable by construction.
    if _has_empty_reference_placeholder(query):
        return "Unsupported"

    # Optional Langfuse tracing — None when not configured.
    lf = _get_langfuse()
    trace = lf.trace(name="rag-agent", input=query) if lf else None

    # Replay prior turns as LangChain messages, then append the current question.
    messages: list = []
    for turn in history or []:
        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["content"]))
        else:
            messages.append(AIMessage(content=turn["content"]))
    messages.append(HumanMessage(content=query))

    _invoke_input = {"messages": messages}
    _invoke_config = {"recursion_limit": 20}

    _limits: dict = getattr(agent, "_rag_limits", {})

    # Run the agent; the except block handles two provider error classes below.
    try:
        result = agent.invoke(_invoke_input, config=_invoke_config)
    except BadRequestError as exc:
        err_str = str(exc).lower()
        # Groq sometimes rejects the model's final-answer step as a malformed tool call.
        # The actual answer text is in the 'failed_generation' field — extract it directly.
        if "failed to call a function" in err_str or "tool_use_failed" in err_str:
            body = getattr(exc, "body", {}) or {}
            failed_gen = (
                body.get("failed_generation", "") if isinstance(body, dict) else ""
            )
            # Only use failed_generation if it looks like a final answer, not a
            # reasoning preamble that never reached a conclusion.
            if failed_gen and not _REASONING_PREFIX_RE.match(failed_gen.strip()):
                failed_gen = failed_gen.strip()
                return (
                    "Unsupported"
                    if _looks_like_bad_final_answer(failed_gen)
                    else failed_gen
                )
        if _is_context_overflow(err_str):
            print("[WARN] Context overflow — retrying with fewer chunks.")
            _limits["rerank_top_n"] = max(
                3, _limits.get("rerank_top_n", RERANK_TOP_N) // 2
            )
            try:
                result = agent.invoke(_invoke_input, config=_invoke_config)
            finally:
                _limits["rerank_top_n"] = min(RERANK_TOP_N, MAX_TOOL_RESULTS)
        else:
            raise

    messages: list[Any] = result.get("messages", [])

    # True when query_excel was the only tool called — see _apply_grounding_check
    # for why that answer's evidence can't be verified by the grounding check.
    tool_names_used = {
        tc["name"]
        for msg in messages
        if isinstance(msg, AIMessage) and msg.tool_calls
        for tc in msg.tool_calls
    }
    excel_only = tool_names_used == {"query_excel"}

    # Collect every tool result: by call id (for tracing) and as raw contexts;
    # also push split chunks into the caller's retrieved_contexts list if given.
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

    # Emit tool-call/result diagnostics and record one Langfuse span per call.
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                if show_tool_uses:
                    print(
                        f"[TOOL_CALL] {tc['name']} args={json.dumps(tc['args'], ensure_ascii=False)}"
                    )
                if trace is not None:
                    result_content = tool_results.get(tc["id"], "")
                    trace.span(
                        name=tc["name"],
                        input=tc["args"],
                        output=_extract_refs(result_content),
                    )
        elif isinstance(msg, ToolMessage) and show_tool_uses:
            print(f"[TOOL_RESULT] {msg.name} ->\n{_extract_refs(msg.content)}\n")

    # The final answer is the last AIMessage that made no further tool calls.
    answer = "No answer generated."
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            text = str(msg.content).strip()
            answer = _strip_think(text)
            break

    # Answer-finalization pipeline (see the section header above): coverage repair,
    # context/retrieval fallbacks for bad answers, then canonical normalization.
    api_base = getattr(agent, "_generation_api_base", None)
    model_name = getattr(agent, "_generation_model", None)
    if (
        api_base
        and model_name
        and tool_contexts
        and (not _looks_like_bad_final_answer(answer) or _is_multi_part_query(query))
    ):
        answer = _repair_incomplete_answer(
            query, answer, tool_contexts, api_base, model_name
        )

    if (
        _looks_like_bad_final_answer(answer)
        and tool_contexts
        and api_base
        and model_name
    ):
        answer = _context_fallback_answer(
            query, answer, tool_contexts, api_base, model_name
        )
    elif _looks_like_bad_final_answer(answer) and not tool_contexts:
        answer = _retrieval_only_answer(query, answer, api_base, model_name)

    answer = _normalize_final(query, answer)
    answer = _apply_grounding_check(
        query,
        answer,
        tool_contexts,
        api_base,
        model_name,
        excel_only=excel_only,
        trace=trace,
    )

    if trace is not None:
        # Token/cost tracking — LangChain attaches usage_metadata to each AIMessage
        # on non-streaming invoke(); sum across turns for the whole agent run.
        usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        for msg in messages:
            if isinstance(msg, AIMessage) and msg.usage_metadata:
                for k in usage:
                    usage[k] += msg.usage_metadata.get(k, 0)
        trace.generation(
            name="final-answer",
            model=getattr(agent, "_generation_model", None),
            input=query,
            output=answer,
            usage=usage if usage["total_tokens"] else None,
        )
        trace.update(output=answer)
        lf.flush()

    return answer


def _overflow_fallback_answer(
    agent: Any,
    query: str,
    history: list[dict] | None,
    collected_chunks: list[str] | None,
    limits: dict,
) -> str:
    """Halve the rerank pool and answer once more after a context-overflow error."""
    limits["rerank_top_n"] = max(3, limits.get("rerank_top_n", MAX_TOOL_RESULTS) // 2)
    try:
        api_base = getattr(agent, "_generation_api_base", None)
        model_name = getattr(agent, "_generation_model", None)
        if api_base and model_name:
            return _direct_retrieval_answer(query, api_base, model_name)
        return ask_agent(
            agent, query, history=history, retrieved_contexts=collected_chunks
        )
    finally:
        limits["rerank_top_n"] = min(RERANK_TOP_N, MAX_TOOL_RESULTS)


def stream_agent(
    agent: Any,
    query: str,
    history: list[dict] | None = None,
    show_tool_uses: bool = False,
    collected_chunks: list[str] | None = None,
    sql_trace: list[str] | None = None,
    tool_calls: list[str] | None = None,
    rejected_chunks: list[dict] | None = None,
    trace: Any = None,
    skip_grounding_check: bool = False,
) -> Generator[str, None, None]:
    """Stream the agent's final answer token-by-token.

    Yields string fragments as they arrive from the LLM.
    Tool calls are not yielded (optionally printed if show_tool_uses=True).
    Qwen3 <think>...</think> blocks are suppressed.

    Args:
        history: Prior conversation turns as [{"role": "user"/"assistant", "content": str}].
        collected_chunks: If provided, tool result chunks are appended to this list.
        sql_trace: If provided, SQL generated by query_excel is appended to this list.
        tool_calls: If provided, the name of each tool the agent actually invoked
            is appended to this list, in call order (with repeats).
        rejected_chunks: If provided, reranked-but-not-selected candidates from
            search_knowledge_base calls are appended to this list (UI-only).
        trace: Optional Langfuse trace/span to record the grounding-check verdict on.
        skip_grounding_check: Comparison questions already get a dedicated
            doc-coverage retry in answer_pipeline.py -- the grounding check is
            redundant safety there, and was reproduced live flagging correct
            comparative answers ("X has the longer extension, based on Y vs Z")
            as ungrounded and downgrading them to Unsupported (gpt-oss-120b's
            judge is stricter on inferential/comparative claims than the
            generation model itself). Passed True for comparison questions.
    """
    # Guard: questions with an empty reference slot are unanswerable by construction.
    if _has_empty_reference_placeholder(query):
        yield "Unsupported"
        return

    # Replay prior turns as LangChain messages, then append the current question.
    messages: list = []
    for turn in history or []:
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
                _think_buf = _think_buf[end + len("</think>") :]
                _in_think = False
            else:
                start = _think_buf.find("<think>")
                if start == -1:
                    out += _think_buf
                    _think_buf = ""
                    return out
                out += _think_buf[:start]
                _think_buf = _think_buf[start + len("<think>") :]
                _in_think = True

    from openai import APIError, BadRequestError

    # Buffer text tokens until the first tool call completes so we don't yield
    # reasoning preambles ("I need to search...", "Let me look...") as answers.
    _tool_used = False
    _pre_tool_buf: list[str] = []
    _final_buf: list[str] = []
    _tool_contexts: list[str] = []
    _tool_names_used: set[str] = set()
    _limits: dict = getattr(agent, "_rag_limits", {})

    # Stream messages from the agent. Text chunks go to the pre-tool buffer until
    # the first tool result arrives, then to the final buffer; tool-call argument
    # chunks are skipped entirely.
    try:
        for chunk, metadata in agent.stream(
            _invoke_input, config=_invoke_config, stream_mode="messages"
        ):
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
                # First tool result: discard any buffered reasoning preamble and
                # record the tool name, SQL trace, and retrieved chunks.
                _tool_used = True
                _pre_tool_buf.clear()
                _tool_contexts.append(chunk.content)
                if chunk.name:
                    _tool_names_used.add(chunk.name)
                if tool_calls is not None and chunk.name:
                    tool_calls.append(chunk.name)
                if sql_trace is not None and chunk.name == "query_excel":
                    artifact = getattr(chunk, "artifact", None)
                    if isinstance(artifact, dict):
                        sql_trace.extend(s for s in (artifact.get("sql") or []) if s)
                if (
                    rejected_chunks is not None
                    and chunk.name == "search_knowledge_base"
                ):
                    artifact = getattr(chunk, "artifact", None)
                    if isinstance(artifact, dict):
                        rejected_chunks.extend(artifact.get("rejected") or [])
                # query_excel is a SQL tool, not a retriever — its result is
                # surfaced via the SQL trace, not the "Retrieved chunks" panel.
                if collected_chunks is not None and chunk.name != "query_excel":
                    parts = re.split(r"\n\n(?=\[\d+\])", chunk.content.strip())
                    cleaned = [p.strip() for p in parts if p.strip()]
                    if cleaned:
                        # Mark tool-call boundaries so the API can group/prioritize
                        # chunks per call (later/scoped calls usually contain the
                        # answer-bearing chunks; earlier broad calls return noise).
                        collected_chunks.append("---CALL_BOUNDARY---")
                        collected_chunks.extend(cleaned)
                if show_tool_uses:
                    print(
                        f"\n[TOOL_RESULT] {chunk.name} ->\n{_extract_refs(chunk.content)}\n"
                    )
    except (BadRequestError, APIError) as exc:
        # Provider errors: context overflow retries with fewer chunks; a malformed
        # final-answer tool call has its real text recovered below.
        err_str = str(exc).lower()
        if _is_context_overflow(err_str):
            yield _overflow_fallback_answer(
                agent, query, history, collected_chunks, _limits
            )
            return
        if "failed to call a function" in err_str or "tool_use_failed" in err_str:
            if _tool_used:
                # The provider can raise after the final answer. Flush buffered answer text
                # instead of losing it.
                answer = "".join(_final_buf).strip()
                api_base = getattr(agent, "_generation_api_base", None)
                model_name = getattr(agent, "_generation_model", None)
                if (
                    _looks_like_bad_final_answer(answer)
                    and _tool_contexts
                    and api_base
                    and model_name
                ):
                    answer = _context_fallback_answer(
                        query, answer, _tool_contexts, api_base, model_name
                    )
                if answer:
                    yield answer
                return
            # Error happened before any tool call; try to extract answer from failed_generation body.
            body = getattr(exc, "body", {}) or {}
            failed_gen = (
                body.get("failed_generation", "") if isinstance(body, dict) else ""
            )
            if failed_gen and not _REASONING_PREFIX_RE.match(failed_gen.strip()):
                failed_gen = failed_gen.strip()
                yield (
                    "Unsupported"
                    if _looks_like_bad_final_answer(failed_gen)
                    else failed_gen
                )
                return
        raise
    except Exception as exc:
        err_str = str(exc).lower()
        if _is_context_overflow(err_str):
            yield _overflow_fallback_answer(
                agent, query, history, collected_chunks, _limits
            )
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
            answer = _retrieval_only_answer(query, answer, api_base, model_name)
        if answer:
            yield _normalize_final(query, answer)
        return

    # Flush any remaining buffered text (e.g. trailing content after last </think>)
    if _think_buf and not _in_think:
        _final_buf.append(_think_buf)

    # Tool path: assemble the streamed answer, then run coverage repair and the
    # context fallback (same finalization stages as ask_agent).
    answer = "".join(_final_buf).strip()
    api_base = getattr(agent, "_generation_api_base", None)
    model_name = getattr(agent, "_generation_model", None)
    if (
        api_base
        and model_name
        and _tool_contexts
        and (not _looks_like_bad_final_answer(answer) or _is_multi_part_query(query))
    ):
        answer = _repair_incomplete_answer(
            query, answer, _tool_contexts, api_base, model_name
        )

    if (
        _looks_like_bad_final_answer(answer)
        and _tool_contexts
        and api_base
        and model_name
    ):
        answer = _context_fallback_answer(
            query, answer, _tool_contexts, api_base, model_name
        )

    if answer:
        answer = _normalize_final(query, answer)
        answer = _apply_grounding_check(
            query,
            answer,
            _tool_contexts,
            api_base,
            model_name,
            excel_only=_tool_names_used == {"query_excel"},
            skip=skip_grounding_check,
            trace=trace,
        )
        yield answer


# The command-line runner lives in rag_cli.py — this module is import-only.
