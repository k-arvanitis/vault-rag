"""Pytest-compatible RAG evaluation using DeepEval's pytest integration.

Run with:
    deepeval test run eval/test_rag.py

Or plain pytest (scores are logged but no deepeval dashboard upload):
    pytest eval/test_rag.py -v

Requires eval/dataset.json — generate it first:
    python -m eval.generate_dataset

Environment variables (see eval/judge_llm.py):
    JUDGE_API_BASE   (default: GENERATION_API_BASE)
    JUDGE_MODEL      (default: GENERATION_MODEL)
    JUDGE_API_KEY    (default: not-needed)
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv
from deepeval import assert_test
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = REPO_ROOT / "eval" / "dataset.json"
METRIC_THRESHOLD = 0.7

sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Shared fixtures (initialised once per session)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def judge():
    from eval.judge_llm import OpenAICompatibleJudge
    return OpenAICompatibleJudge()


@pytest.fixture(scope="session")
def rag_agent():
    from src.config import (
        GENERATION_API_BASE,
        GENERATION_MODEL,
        QDRANT_COLLECTION,
        QDRANT_URL,
        RERANK_TOP_N,
        RERANKER_MODEL,
        RETRIEVAL_TOP_K,
    )
    from src.rag_agent import build_rag_agent
    return build_rag_agent(
        qdrant_url=QDRANT_URL,
        collection=QDRANT_COLLECTION,
        retrieval_top_k=RETRIEVAL_TOP_K,
        rerank_top_n=RERANK_TOP_N,
        reranker_model_name=RERANKER_MODEL or None,
        model_name=GENERATION_MODEL,
        generation_api_base=GENERATION_API_BASE,
    )


@pytest.fixture(scope="session")
def metrics(judge):
    return [
        AnswerRelevancyMetric(threshold=METRIC_THRESHOLD, model=judge, include_reason=True),
        FaithfulnessMetric(threshold=METRIC_THRESHOLD, model=judge, include_reason=True),
        ContextualRecallMetric(threshold=METRIC_THRESHOLD, model=judge, include_reason=True),
        ContextualPrecisionMetric(threshold=METRIC_THRESHOLD, model=judge, include_reason=True),
    ]


# ---------------------------------------------------------------------------
# Load dataset and parametrize
# ---------------------------------------------------------------------------

def _load_pairs() -> list[dict]:
    if not DATASET_PATH.exists():
        pytest.skip(f"Dataset not found: {DATASET_PATH} — run python -m eval.generate_dataset first")
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


def _parse_retrieval_context(tool_content: str) -> list[str]:
    if not tool_content or tool_content.strip() == "No relevant information found.":
        return []
    parts = re.split(r"\n\n(?=\[\d+\])", tool_content.strip())
    return [p.strip() for p in parts if p.strip()]


def _run_agent(agent, question: str, system_prompt: str) -> tuple[str, list[str]]:
    result = agent.invoke(
        {"messages": [SystemMessage(content=system_prompt), HumanMessage(content=question)]},
        config={"recursion_limit": 20},
    )
    messages: list = result.get("messages", [])

    retrieval_context: list[str] = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            retrieval_context.extend(_parse_retrieval_context(content))

    actual_output = "No answer generated."
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            text = re.sub(r"(?is)<think>.*?</think>\s*", "", str(msg.content)).strip()
            if text:
                actual_output = text
                break

    return actual_output, retrieval_context


# Build parametrize list at collection time (skips gracefully if no dataset)
_pairs = _load_pairs() if DATASET_PATH.exists() else []


@pytest.mark.parametrize("pair", _pairs, ids=[f"q{i}" for i in range(len(_pairs))])
def test_rag_pair(pair, rag_agent, metrics):
    from src.rag_agent import SYSTEM_PROMPT

    question = pair["input"]
    expected = pair["expected_output"]
    context_gt = pair.get("context", [])

    actual_output, retrieval_context = _run_agent(rag_agent, question, SYSTEM_PROMPT)

    test_case = LLMTestCase(
        input=question,
        actual_output=actual_output,
        expected_output=expected,
        context=context_gt,
        retrieval_context=retrieval_context,
    )

    assert_test(test_case, metrics)
