"""LangGraph pipelines for vault-rag.

Two graphs are exported:

build_reflection_pipeline()
    Wraps the RAG agent in a StateGraph with an explicit reflection node.
    If the agent returns "Unsupported" and the question is retryable, retries
    with a widened search hint.

build_decomposition_pipeline()
    Plan-first pipeline: an LLM decomposer splits multi-hop questions into
    focused sub-questions before the agent runs. The agent receives an
    explicit step-by-step plan and works through it via tool calls.
    Single-hop questions bypass decomposition and go directly to the agent.
"""

from __future__ import annotations

import json
import logging
import operator
import re
from typing import Annotated, Any

from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from vault_rag.config import (
    FREE_LLM_API_KEY,
    GENERATION_API_BASE,
    GENERATION_MODEL,
    GROQ_API_KEY,
    LITELLM_MASTER_KEY,
    OPENROUTER_API_KEY,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_DOC_ID_RE = re.compile(r"\bdoc_\d+\b", re.IGNORECASE)


def _is_unsupported(answer: str) -> bool:
    """Return True when an answer is the bare 'Unsupported' abstention token."""
    return answer.strip().lower() == "unsupported"


def _looks_excel(question: str) -> bool:
    """Heuristic: does this question target a structured Excel/CSV document?"""
    kw = (
        "transaction",
        "supplier",
        "beneficiary",
        "spend",
        "csv",
        "excel",
        "spreadsheet",
        "total amount",
        "net amount",
        "purchase card",
    )
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
    from vault_rag.rag_agent import ask_agent  # noqa: PLC0415

    def _run_node(state: _ReflectState) -> dict:
        """Run the agent on the question, widening the search hint on a retry."""
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
        """Decide whether an Unsupported answer warrants one widened retry."""
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
        """Route back to the run node when a retry was requested, else END."""
        return "run" if state.get("drop_filter") else END

    builder: StateGraph = StateGraph(_ReflectState)
    builder.add_node("run", _run_node)
    builder.add_node("reflect", _reflect_node)
    builder.set_entry_point("run")
    builder.add_edge("run", "reflect")
    builder.add_conditional_edges(
        "reflect", _route_after_reflect, {"run": "run", END: END}
    )
    return builder.compile()


def ask_with_reflection(pipeline: Any, question: str) -> str:
    """Invoke a reflection pipeline and return the final answer string."""
    return ask_with_reflection_state(pipeline, question)["answer"]


def ask_with_reflection_state(pipeline: Any, question: str) -> dict:
    """Like ask_with_reflection but returns the full final state, so callers
    can read retrieved_contexts alongside the answer (see eval/run_eval.py's
    override instrumentation, which needs both)."""
    return pipeline.invoke(
        {
            "question": question,
            "answer": "",
            "retry_count": 0,
            "drop_filter": False,
            "retrieved_contexts": [],
        }
    )


# ---------------------------------------------------------------------------
# Decomposition pipeline
# ---------------------------------------------------------------------------


class _DecomposeState(TypedDict):
    question: str
    sub_questions: list[str]
    formatted_question: str
    answer: str
    contexts: list[str]


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

    from vault_rag.rag_agent import ask_agent  # noqa: PLC0415

    # API key resolution keyed on the actual base URL — see the identical fix in
    # build_rag_agent() (src/rag_agent.py) and _llm_call() (src/llm_utils.py).
    # This call site was missed by both earlier fixes: it hardcoded GROQ_API_KEY
    # regardless of generation_api_base, so decomposition silently 401'd and fell
    # back to sub_questions = [question] (single-hop) whenever GENERATION_API_BASE
    # pointed anywhere but Groq — e.g. the current OpenRouter config.
    _base_for_key = generation_api_base.lower()
    if "localhost:4000" in _base_for_key or "127.0.0.1:4000" in _base_for_key:
        _decompose_api_key = LITELLM_MASTER_KEY or "EMPTY"
    elif "localhost:3011" in _base_for_key or "127.0.0.1:3011" in _base_for_key:
        _decompose_api_key = FREE_LLM_API_KEY or "EMPTY"
    elif "openrouter.ai" in _base_for_key:
        _decompose_api_key = OPENROUTER_API_KEY or "EMPTY"
    elif "groq.com" in _base_for_key:
        _decompose_api_key = GROQ_API_KEY or "EMPTY"
    else:
        _decompose_api_key = GROQ_API_KEY or LITELLM_MASTER_KEY or "EMPTY"
    _client = OpenAI(base_url=generation_api_base, api_key=_decompose_api_key)

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
        except Exception:
            # A real API failure (auth, network, rate limit) — not a legitimate
            # single-hop decision. Log loudly so a wrong key or dead endpoint
            # can't silently degrade every multi-hop question to single-hop again.
            logger.exception(
                "Decomposer LLM call failed — falling back to single-hop for: %s",
                question,
            )
            sub_questions = [question]
        else:
            raw = ""
            try:
                raw = resp.choices[0].message.content or ""
                # Strip thinking tags (Qwen3 with extended thinking enabled)
                raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
                # Strip accidental markdown fences
                raw = re.sub(r"^```[a-z]*\n?|```$", "", raw, flags=re.MULTILINE).strip()
                sub_questions = json.loads(raw)
                if not isinstance(sub_questions, list) or not sub_questions:
                    sub_questions = [question]
            except Exception:
                # Model returned malformed JSON — a real decomposer bug, still
                # worth knowing about, but distinct from an API-layer failure.
                logger.warning(
                    "Decomposer returned unparseable output, treating as single-hop: %s",
                    raw,
                )
                sub_questions = [question]

        if len(sub_questions) > 1:
            plan = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(sub_questions))
            formatted = (
                f"Answer the following question by working through each sub-question in order "
                f"and calling the appropriate search tools for each one. For each sub-question, "
                f"retrieve and state the specific fact or value it asks for — identifying which "
                f"document it comes from is not a complete answer on its own:\n\n"
                f"{plan}\n\n"
                f"Original question (use this for your final answer): {question}"
            )
        else:
            formatted = question

        return {"sub_questions": sub_questions, "formatted_question": formatted}

    def _run_node(state: _DecomposeState) -> dict:
        """Run the ReAct agent on the (possibly plan-formatted) question."""
        chunks: list[str] = []
        answer = ask_agent(
            agent, state["formatted_question"], retrieved_contexts=chunks
        )
        return {"answer": answer, "contexts": chunks}

    builder: StateGraph = StateGraph(_DecomposeState)
    builder.add_node("decompose", _decompose_node)
    builder.add_node("run", _run_node)
    builder.set_entry_point("decompose")
    builder.add_edge("decompose", "run")
    builder.add_edge("run", END)
    return builder.compile()


def ask_with_decomposition(
    pipeline: Any, question: str, collected_chunks: list[str] | None = None
) -> str:
    """Invoke a decomposition pipeline and return the final answer string.

    If collected_chunks is provided, the tool-result chunks the agent retrieved
    while answering are appended to it (same contract as stream_agent/ask_agent),
    so callers like the eval harness can ground a faithfulness judge.
    """
    result = pipeline.invoke(
        {
            "question": question,
            "sub_questions": [],
            "formatted_question": "",
            "answer": "",
            "contexts": [],
        }
    )
    if collected_chunks is not None:
        collected_chunks.extend(result.get("contexts") or [])
    return result["answer"]
