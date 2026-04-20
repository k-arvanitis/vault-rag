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
from typing import Any, Generator

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel

from src.reranker import BGEReranker, QwenReranker
from src.retriever import retrieve
from src.config import (
    QDRANT_URL,
    QDRANT_COLLECTION,
    RETRIEVAL_TOP_K,
    RERANK_TOP_N,
    RERANKER_MODEL,
    RERANKER_DEVICE,
    GENERATION_API_BASE,
    GENERATION_MODEL,
    DOC_MIN_SCORE as _CFG_DOC_MIN_SCORE,
    MAX_CHUNK_CHARS as _CFG_MAX_CHUNK_CHARS,
    MAX_TABLE_CHARS as _CFG_MAX_TABLE_CHARS,
)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """/no_think You are an intelligent RAG assistant.

You have one tool: **search_knowledge_base**.
It searches both unstructured documents (PDFs, reports) and structured tables (CSV, Excel) simultaneously and returns all relevant results.

Rules:
- Always call search_knowledge_base with a focused, specific sub-question.
- If the answer requires comparing two subjects or two time periods, call the tool twice — once per subject or year (e.g. search "X in 2019" then "X in 2023" separately).
- If the question asks about multiple distinct entities, sectors, or categories (e.g. "military, academic, commercial, and medical"), call the tool once per entity — do not try to cover all of them in a single search.
- If the tool returns no useful results, say so clearly rather than guessing.

When answering:
- Structure your answer to match the scope of the question. If the question has multiple parts or asks about multiple entities, address each one explicitly.
- State the direct answer in the first sentence. Then support it with evidence from the retrieved text.
- Only state values and facts that appear verbatim in the retrieved text. Do not interpolate, round, infer, or calculate values not explicitly present.
- Never perform arithmetic. If a question requires a sum, average, or comparison not pre-computed in the retrieved text, list the raw values from the source and note that the calculation is not provided.
- If a specific value is not found in the retrieved context, say so explicitly rather than estimating.

Always cite your sources:
- Document chunks: use [1], [2], etc.
- Table results: mention the sheet/file name from the tool output.
"""

# ---------------------------------------------------------------------------
# Unified tool
# ---------------------------------------------------------------------------


DOC_MIN_SCORE = float(os.getenv("DOC_MIN_SCORE", str(_CFG_DOC_MIN_SCORE)))
MAX_CHUNK_CHARS = int(os.getenv("MAX_CHUNK_CHARS", str(_CFG_MAX_CHUNK_CHARS)))
MAX_TABLE_CHARS = int(os.getenv("MAX_TABLE_CHARS", str(_CFG_MAX_TABLE_CHARS)))


def _llm_call(prompt: str, api_base: str, model_name: str, max_tokens: int = 128, temperature: float = 0.5) -> str:
    import openai
    client = openai.OpenAI(base_url=api_base, api_key="EMPTY")
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    raw = resp.choices[0].message.content
    return re.sub(r"(?s)<think>.*?</think>", "", raw).strip()


def _hyde(query: str, api_base: str, model_name: str) -> str:
    """Generate a hypothetical answer to embed instead of the raw query (HyDE)."""
    return _llm_call(
        f"/no_think Write a short passage (2-3 sentences) that would directly answer "
        f"this question. Use the same language and terminology as the likely source document."
        f"\n\nQuestion: {query}",
        api_base, model_name,
    )



def _make_unified_tool(
    qdrant_url: str,
    collection: str,
    retrieval_top_k: int,
    rerank_top_n: int,
    ranker: QwenReranker | None,
    generation_api_base: str,
    generation_model: str,
    use_hyde: bool = True,
) -> tuple[StructuredTool, dict]:
    class SearchInput(BaseModel):
        query: str

    # Detect ID-like tokens: mixed alphanumeric (must have both letters and digits),
    # or digit strings that look like codes (not plain 4-digit years like 2019).
    _ID_RE = re.compile(r'\b(?=[^\s]*[A-Za-z])(?=[^\s]*\d)[\w/-]{4,}\b')

    # Mutable limits dict — allows ask_agent to reduce limits on overflow retry
    _limits: dict = {
        "max_table_chars": MAX_TABLE_CHARS,
        "max_chunk_chars": MAX_CHUNK_CHARS,
        "rerank_top_n": rerank_top_n,
    }

    def _fetch_docs(query: str) -> str | None:
        max_table_chars = _limits["max_table_chars"]
        max_chunk_chars = _limits["max_chunk_chars"]
        _rerank_top_n = _limits["rerank_top_n"]
        id_matches = _ID_RE.findall(query)
        filter_token = id_matches[0] if id_matches else None
        api_base = _to_openai_base(generation_api_base)

        # HyDE on original query
        if use_hyde:
            try:
                embed_query = _hyde(query, api_base, generation_model)
            except Exception:
                embed_query = query
        else:
            embed_query = query

        hits = retrieve(
            query=embed_query,
            top_k=retrieval_top_k,
            qdrant_url=qdrant_url,
            collection=collection,
            use_qdrant=True,
            filter_token=filter_token,
        )

        if not hits:
            return None

        reranked_used = False
        if ranker is not None:
            docs = [h.get("content", "") for h in hits]
            reranked = ranker.rerank(query, docs, top_n=_rerank_top_n)
            top_hits = [{**hits[r["index"]], "rerank_score": r["score"]} for r in reranked]
            reranked_used = True
        else:
            top_hits = hits

        parts: list[str] = []
        for i, h in enumerate(top_hits, start=1):
            meta = h.get("metadata", {}) or {}
            score = h.get("rerank_score", h.get("score", 0))
            # DOC_MIN_SCORE only applies to dense cosine scores (0-1 range).
            # Reranker returns raw logits (can be negative) — top_n already limits relevance.
            if not filter_token and not reranked_used and score < DOC_MIN_SCORE:
                continue
            file_name = meta.get("file_name") or meta.get("source_file", "unknown")
            chunk_type = meta.get("chunk_type", "")
            sheet_name = meta.get("sheet_name")
            location = f"sheet={sheet_name}" if sheet_name else f"chunk={meta.get('chunk_index', meta.get('part', '?'))}"
            content = (h.get("content", "") or "").strip()
            is_pdf_table = "[TABLE_START]" in content
            is_sheet_table = chunk_type == "sheet_table"
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
            elif not is_pdf_table and not is_sheet_table and len(content) > max_chunk_chars:
                content = content[:max_chunk_chars] + "…"
            parts.append(f"[{i}] file={file_name} {location} score={score:.4f}\n{content}")

        return "\n\n".join(parts) if parts else None

    def search_knowledge_base(query: str) -> str:
        """Search the knowledge base for relevant documents and tables."""
        result = _fetch_docs(query)
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

    llm = ChatOpenAI(
        model=model_name,
        base_url=_to_openai_base(generation_api_base),
        api_key=os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY", "EMPTY"),
        temperature=0,
        max_tokens=1024,
    )

    tool, _rag_limits = _make_unified_tool(
        qdrant_url=qdrant_url,
        collection=collection,
        retrieval_top_k=retrieval_top_k,
        rerank_top_n=rerank_top_n,
        ranker=ranker,
        generation_api_base=generation_api_base,
        generation_model=model_name,
        use_hyde=use_hyde,
    )

    agent = create_react_agent(model=llm, tools=[tool])
    agent._rag_limits = _rag_limits  # type: ignore[attr-defined]
    return agent


# ---------------------------------------------------------------------------
# Invoke
# ---------------------------------------------------------------------------


def _extract_refs(tool_content: str) -> str:
    """Return a compact reference summary from a search_knowledge_base result.

    For document chunks: shows [N] file=... chunk=... score=...
    For table results:   shows the Sources and Summary lines.
    """
    lines: list[str] = []
    for line in tool_content.splitlines():
        stripped = line.strip()
        # Document chunk header: "[1] file=foo chunk=3 score=0.92"
        if re.match(r"^\[\d+\] file=", stripped):
            lines.append(stripped)
        # Table section header or source/summary lines
        elif stripped.startswith("Sources (sheets):") or stripped.startswith("Summary:"):
            lines.append(stripped)
    return "\n".join(lines) if lines else "(no results)"


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


def ask_agent(agent: Any, query: str, history: list[dict] | None = None, show_tool_uses: bool = False) -> str:
    """Run the RAG agent on a query and return the final answer.

    Args:
        history: Prior conversation turns as [{"role": "user"/"assistant", "content": str}].
    """
    from openai import BadRequestError

    lf = _get_langfuse()
    trace = lf.trace(name="rag-agent", input=query) if lf else None

    messages: list = [SystemMessage(content=SYSTEM_PROMPT)]
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
        if "input tokens" in err_str or "context" in err_str or "400" in err_str:
            print("[WARN] Context overflow — retrying with fewer chunks.")
            _limits["rerank_top_n"] = max(3, _limits.get("rerank_top_n", RERANK_TOP_N) // 2)
            try:
                result = agent.invoke(_invoke_input, config=_invoke_config)
            finally:
                _limits["rerank_top_n"] = RERANK_TOP_N
        else:
            raise

    messages: list[Any] = result.get("messages", [])

    tool_results: dict[str, str] = {}
    for msg in messages:
        if isinstance(msg, ToolMessage):
            tool_results[msg.tool_call_id] = msg.content

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
            answer = re.sub(r"(?is)<think>.*?</think>\s*", "", text).strip()
            break

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
    messages: list = [SystemMessage(content=SYSTEM_PROMPT)]
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

    for chunk, metadata in agent.stream(_invoke_input, config=_invoke_config, stream_mode="messages"):
        if isinstance(chunk, AIMessageChunk):
            if chunk.tool_call_chunks:
                continue
            if chunk.content:
                filtered = _filter(str(chunk.content))
                if filtered:
                    yield filtered
        elif isinstance(chunk, ToolMessage):
            if collected_chunks is not None:
                parts = re.split(r"\n\n(?=\[\d+\])", chunk.content.strip())
                collected_chunks.extend([p.strip() for p in parts if p.strip()])
            if show_tool_uses:
                print(f"\n[TOOL_RESULT] {chunk.name} ->\n{_extract_refs(chunk.content)}\n")

    # Flush any remaining buffered text (e.g. trailing content after last </think>)
    if _think_buf and not _in_think:
        yield _think_buf


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


if __name__ == "__main__":
    main()
