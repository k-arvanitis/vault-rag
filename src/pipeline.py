"""LangGraph pipelines for vault-rag.

Three graphs are exported:

build_reflection_pipeline()
    Wraps the RAG agent in a StateGraph with an explicit reflection node.
    If the agent returns "Unsupported" and the question is retryable, retries
    with a widened search hint.

build_supervisor_pipeline()
    Multi-agent supervisor graph that classifies each question and routes
    to a PDF-RAG agent node, an Excel-SQL agent node, or both (mixed
    questions get parallel branches merged before final synthesis).

build_decomposition_pipeline()
    Plan-first pipeline: an LLM decomposer splits multi-hop questions into
    focused sub-questions before the agent runs. The agent receives an
    explicit step-by-step plan and works through it via tool calls.
    Single-hop questions bypass decomposition and go directly to the agent.
"""
from __future__ import annotations

import json
import operator
import re
from typing import Annotated, Any

from langgraph.types import Send
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from src.config import GENERATION_API_BASE, GENERATION_MODEL, GROQ_API_KEY

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_DOC_ID_RE = re.compile(r"\bdoc_\d+\b", re.IGNORECASE)


def _is_unsupported(answer: str) -> bool:
    return answer.strip().lower() == "unsupported"


def _looks_excel(question: str) -> bool:
    """Heuristic: does this question target a structured Excel/CSV document?"""
    kw = ("transaction", "supplier", "beneficiary", "spend", "csv", "excel",
          "spreadsheet", "total amount", "net amount", "purchase card")
    return any(k in question.lower() for k in kw)


# ---------------------------------------------------------------------------
# Reflection pipeline
# ---------------------------------------------------------------------------

class _ReflectState(TypedDict):
    question: str
    answer: str
    retry_count: int
    drop_filter: bool
    retrieved_contexts: Annotated[list[str], operator.add]


def build_reflection_pipeline(agent: Any) -> Any:
    """Return a StateGraph that wraps *agent* with one automatic reflection retry.

    If the agent returns "Unsupported" and the question looks like a scoped
    PDF query (contains exactly one doc_id), the pipeline retries with a
    modified prompt that asks the agent to widen its search.

    Args:
        agent: A compiled LangGraph agent returned by build_rag_agent().
    """
    from src.rag_agent import ask_agent  # noqa: PLC0415

    def _run_node(state: _ReflectState) -> dict:
        question = state["question"]
        if state.get("drop_filter"):
            # Hint the agent to relax its text filters on retry
            question = (
                f"{question}\n\n"
                "(Retry with broader search — do not restrict to exact text matches.)"
            )
        contexts: list[str] = []
        answer = ask_agent(agent, question, retrieved_contexts=contexts)
        return {
            "answer": answer,
            "retrieved_contexts": contexts,
            "retry_count": state.get("retry_count", 0) + 1,
        }

    def _reflect_node(state: _ReflectState) -> dict:
        answer = state["answer"]
        retry_count = state.get("retry_count", 0)
        question = state["question"]
        doc_ids = _DOC_ID_RE.findall(question)
        retryable = (
            _is_unsupported(answer)
            and retry_count < 2
            and len(doc_ids) <= 2  # don't retry open-ended questions
        )
        return {"drop_filter": retryable}

    def _route_after_reflect(state: _ReflectState) -> str:
        return "run" if state.get("drop_filter") else END

    builder: StateGraph = StateGraph(_ReflectState)
    builder.add_node("run", _run_node)
    builder.add_node("reflect", _reflect_node)
    builder.set_entry_point("run")
    builder.add_edge("run", "reflect")
    builder.add_conditional_edges("reflect", _route_after_reflect, {"run": "run", END: END})
    return builder.compile()


def ask_with_reflection(pipeline: Any, question: str) -> str:
    """Invoke a reflection pipeline and return the final answer string."""
    result = pipeline.invoke({
        "question": question,
        "answer": "",
        "retry_count": 0,
        "drop_filter": False,
        "retrieved_contexts": [],
    })
    return result["answer"]


# ---------------------------------------------------------------------------
# Multi-agent supervisor pipeline
# ---------------------------------------------------------------------------

class _SupervisorState(TypedDict):
    question: str
    question_type: str          # "pdf" | "excel" | "mixed"
    doc_ids: list[str]
    pdf_results: Annotated[list[str], operator.add]   # parallel merge
    excel_result: str
    answer: str


class _ParallelPDFState(TypedDict):
    """State passed to each parallel PDF retrieval branch via Send."""
    question: str
    doc_id: str


def build_supervisor_pipeline(pdf_agent: Any, excel_agent: Any) -> Any:
    """Return a supervisor StateGraph routing between specialized agents.

    pdf_agent  — agent built with search_knowledge_base tool only.
    excel_agent — agent built with query_excel tool only.

    The supervisor:
      1. Classifies the question as pdf / excel / mixed.
      2. For PDF questions: fans out a parallel retrieval branch per doc_id
         (LangGraph Send API) then merges answers.
      3. For Excel questions: calls the Excel agent directly.
      4. For mixed questions: runs both branches and synthesises a combined answer.
    """
    from src.rag_agent import ask_agent  # noqa: PLC0415

    def _classify_node(state: _SupervisorState) -> dict:
        q = state["question"]
        doc_ids = list({d.lower() for d in _DOC_ID_RE.findall(q)})
        if _looks_excel(q) and not doc_ids:
            qtype = "excel"
        elif _looks_excel(q) and doc_ids:
            qtype = "mixed"
        else:
            qtype = "pdf"
        return {"question_type": qtype, "doc_ids": doc_ids}

    def _route_after_classify(state: _SupervisorState) -> list | str:
        qtype = state["question_type"]
        doc_ids = state["doc_ids"]
        if qtype == "excel":
            return "excel_node"
        if qtype == "mixed":
            # Fan out PDF retrieval per doc in parallel AND call excel
            sends = [Send("pdf_node", {"question": state["question"], "doc_id": d}) for d in doc_ids]
            return sends + ["excel_node"]
        if doc_ids:
            # Fan out one branch per mentioned doc_id
            return [Send("pdf_node", {"question": state["question"], "doc_id": d}) for d in doc_ids]
        return "pdf_node"  # no specific doc — single unscoped PDF search

    def _pdf_node(state: _ParallelPDFState) -> dict:
        q = state["question"]
        doc_id = state.get("doc_id", "")
        scoped_q = f"{q} (focus on {doc_id})" if doc_id else q
        answer = ask_agent(pdf_agent, scoped_q)
        return {"pdf_results": [f"[{doc_id or 'pdf'}] {answer}"]}

    def _excel_node(state: _SupervisorState) -> dict:
        answer = ask_agent(excel_agent, state["question"])
        return {"excel_result": answer}

    def _synthesise_node(state: _SupervisorState) -> dict:
        parts = state.get("pdf_results", []) + (
            [f"[excel] {state['excel_result']}"] if state.get("excel_result") else []
        )
        if not parts:
            return {"answer": "Unsupported"}
        if len(parts) == 1:
            # Strip the leading [doc_id] prefix for single-result answers
            text = re.sub(r"^\[[^\]]+\]\s*", "", parts[0])
            return {"answer": text}
        return {"answer": "\n\n".join(parts)}

    builder: StateGraph = StateGraph(_SupervisorState)
    builder.add_node("classify", _classify_node)
    builder.add_node("pdf_node", _pdf_node)
    builder.add_node("excel_node", _excel_node)
    builder.add_node("synthesise", _synthesise_node)

    builder.set_entry_point("classify")
    builder.add_conditional_edges(
        "classify",
        _route_after_classify,
        {"pdf_node": "pdf_node", "excel_node": "excel_node"},
    )
    builder.add_edge("pdf_node", "synthesise")
    builder.add_edge("excel_node", "synthesise")
    builder.add_edge("synthesise", END)
    return builder.compile()


def ask_supervisor(pipeline: Any, question: str) -> str:
    """Invoke a supervisor pipeline and return the final answer string."""
    result = pipeline.invoke({
        "question": question,
        "question_type": "",
        "doc_ids": [],
        "pdf_results": [],
        "excel_result": "",
        "answer": "",
    })
    return result["answer"]


# ---------------------------------------------------------------------------
# Decomposition pipeline
# ---------------------------------------------------------------------------

class _DecomposeState(TypedDict):
    question: str
    sub_questions: list[str]
    formatted_question: str
    answer: str


def build_decomposition_pipeline(
    agent: Any,
    *,
    generation_api_base: str = GENERATION_API_BASE,
    generation_model: str = GENERATION_MODEL,
) -> Any:
    """Return a plan-first StateGraph that decomposes multi-hop questions.

    The decompose node calls the LLM to split a complex question into 2-4
    focused sub-questions. The run node receives an explicit numbered plan
    and the original question; the ReAct agent works through each sub-question
    via tool calls before synthesising a final answer.

    Single-hop questions are detected (list length == 1) and bypass the plan
    formatting — the agent receives the original question unchanged.

    Args:
        agent: A compiled LangGraph agent returned by build_rag_agent().
        generation_api_base: OpenAI-compatible base URL for the decomposer LLM.
        generation_model: Model name for the decomposer LLM.
    """
    from openai import OpenAI  # noqa: PLC0415

    from src.rag_agent import ask_agent  # noqa: PLC0415

    _client = OpenAI(base_url=generation_api_base, api_key=GROQ_API_KEY or "none")

    _SYSTEM = (
        "You are a question decomposer for a document retrieval system. "
        "Given a complex multi-hop question, split it into 2–4 focused sub-questions "
        "that together fully answer the original. "
        "If the question is a single-fact lookup (no comparison, no multi-document reasoning), "
        "return it unchanged as a one-element list. "
        "Respond with a JSON array of strings only. No explanation, no markdown fences."
    )

    def _decompose_node(state: _DecomposeState) -> dict:
        """Call LLM to split the question; format as numbered plan if multi-hop."""
        question = state["question"]
        try:
            resp = _client.chat.completions.create(
                model=generation_model,
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": question},
                ],
                temperature=0,
                max_tokens=512,
            )
            raw = resp.choices[0].message.content or ""
            # Strip thinking tags (Qwen3 with extended thinking enabled)
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            # Strip accidental markdown fences
            raw = re.sub(r"^```[a-z]*\n?|```$", "", raw, flags=re.MULTILINE).strip()
            sub_questions: list[str] = json.loads(raw)
            if not isinstance(sub_questions, list) or not sub_questions:
                sub_questions = [question]
        except Exception:
            sub_questions = [question]

        if len(sub_questions) > 1:
            plan = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(sub_questions))
            formatted = (
                f"Answer the following question by working through each sub-question in order "
                f"and calling the appropriate search tools for each one:\n\n"
                f"{plan}\n\n"
                f"Original question (use this for your final answer): {question}"
            )
        else:
            formatted = question

        return {"sub_questions": sub_questions, "formatted_question": formatted}

    def _run_node(state: _DecomposeState) -> dict:
        """Run the ReAct agent on the (possibly plan-formatted) question."""
        answer = ask_agent(agent, state["formatted_question"])
        return {"answer": answer}

    builder: StateGraph = StateGraph(_DecomposeState)
    builder.add_node("decompose", _decompose_node)
    builder.add_node("run", _run_node)
    builder.set_entry_point("decompose")
    builder.add_edge("decompose", "run")
    builder.add_edge("run", END)
    return builder.compile()


def ask_with_decomposition(pipeline: Any, question: str) -> str:
    """Invoke a decomposition pipeline and return the final answer string."""
    result = pipeline.invoke({
        "question": question,
        "sub_questions": [],
        "formatted_question": "",
        "answer": "",
    })
    return result["answer"]
