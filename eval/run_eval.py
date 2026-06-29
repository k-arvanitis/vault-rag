"""Run answer evaluation for the Vault RAG benchmark using the full agent.

Usage:
    uv run python eval/run_eval.py
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(override=True)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

from src.config import (  # noqa: E402
    GENERATION_API_BASE,
    GENERATION_MODEL,
    LITELLM_MASTER_KEY,
    QDRANT_COLLECTION,
    QDRANT_URL,
)
from src.pipeline import (  # noqa: E402
    _looks_excel,
    ask_with_decomposition,
    ask_with_reflection,
    build_decomposition_pipeline,
    build_reflection_pipeline,
)
from src.rag_agent import build_rag_agent, stream_agent  # noqa: E402
from src.retriever import retrieve  # noqa: E402

# ---------------------------------------------------------------------------
# Commented out: retrieval-metric imports (available for ablation runs)
# ---------------------------------------------------------------------------
# from src.retriever import retrieve, infer_query_chunk_types, _TABLE_CHUNK_TYPES


def _generation_api_key() -> str:
    """Return the API key that matches the configured generation endpoint."""
    base = GENERATION_API_BASE.lower()
    if "localhost:4000" in base or "127.0.0.1:4000" in base:
        return LITELLM_MASTER_KEY or "no-key"
    if "groq.com" in base:
        return os.getenv("GROQ_API_KEY", "")
    if "openrouter.ai" in base:
        return os.getenv("OPENROUTER_API_KEY", "")
    if "nvidia.com" in base:
        return os.getenv("NVIDIA_API_KEY", "")
    return LITELLM_MASTER_KEY or os.getenv("OPENAI_API_KEY", "no-key")


def _judge_config() -> tuple[str, str, str]:
    """Return the model, base URL, and API key for DeepEval judging.

    Keep judging separate from generation: answer generation can use local/Groq
    models, while DeepEval needs a schema-reliable model for JSON metric output.
    """
    judge_model = os.getenv("EVAL_JUDGE_MODEL", "gpt-4o-mini")
    judge_base = os.getenv("EVAL_JUDGE_API_BASE", "https://api.openai.com/v1")
    judge_key = os.getenv("EVAL_JUDGE_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not judge_key:
        judge_model = GENERATION_MODEL
        judge_base = GENERATION_API_BASE
        judge_key = _generation_api_key()
    return judge_model, judge_base, judge_key


def _trim_judge_context(contexts: list[str]) -> list[str]:
    """Keep DeepEval judge prompts below provider context limits."""
    max_total = int(os.getenv("EVAL_JUDGE_CONTEXT_MAX_CHARS", "4000"))
    max_each = int(os.getenv("EVAL_JUDGE_CONTEXT_CHUNK_MAX_CHARS", "1000"))
    trimmed: list[str] = []
    used = 0
    for context in contexts:
        piece = context[:max_each]
        if used + len(piece) > max_total:
            remaining = max_total - used
            if remaining > 0:
                trimmed.append(piece[:remaining])
            break
        trimmed.append(piece)
        used += len(piece)
    return trimmed or ["No context retrieved."]


def _select_judge_context(
    contexts: list[str],
    question: str,
    answer: str,
    gold_answer: str,
    max_total: int | None = None,
) -> str:
    """Pack the most relevant retrieved snippets for a compact judge prompt."""
    max_total = max_total or int(
        os.getenv("EVAL_CUSTOM_JUDGE_CONTEXT_MAX_CHARS", "12000")
    )
    anchors = {
        token.lower()
        for token in re.findall(
            r"[A-Za-z0-9$£€.,¼½¾/-]{2,}", f"{question} {answer} {gold_answer}"
        )
        if token.lower()
        not in {
            "the",
            "and",
            "for",
            "with",
            "from",
            "that",
            "what",
            "which",
            "according",
            "document",
            "documents",
            "report",
            "policy",
            "terms",
            "does",
            "about",
        }
    }
    scored: list[tuple[int, int, str]] = []
    for idx, context in enumerate(contexts):
        text = context.strip()
        lower = text.lower()
        score = sum(1 for anchor in anchors if anchor in lower)
        # Numeric and currency overlap is especially important for these evals.
        score += 3 * sum(
            1
            for token in re.findall(
                r"[$£€]?\d[\d,./]*|[¼½¾]", f"{answer} {gold_answer}"
            )
            if token in text
        )
        scored.append((score, -idx, text))

    selected = [text for score, _, text in sorted(scored, reverse=True) if text][:8]
    if not selected:
        selected = [ctx.strip() for ctx in contexts if ctx.strip()][:8]

    packed: list[str] = []
    used = 0
    for text in selected:
        remaining = max_total - used
        if remaining <= 0:
            break
        piece = text[: min(len(text), remaining)]
        packed.append(piece)
        used += len(piece)
    return "\n\n---\n\n".join(packed) if packed else "No context retrieved."


def _extract_json_scores(text: str) -> dict[str, Any] | None:
    cleaned = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        text.strip(),
        flags=re.IGNORECASE | re.MULTILINE,
    ).strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            parsed = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None


def _clamp_score(value: Any, fallback: float | None = None) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return fallback
    return max(0.0, min(1.0, score))


def _custom_judge_answer(
    question: dict[str, Any],
    normalized_answer: str,
    gold_answer: str,
    retrieved_contexts: list[str],
) -> dict[str, Any]:
    """Score answer quality with one compact JSON-only judge call."""
    import openai

    judge_model_name, judge_base_url, judge_api_key = _judge_config()
    if not judge_api_key:
        raise RuntimeError("No judge API key configured")

    context = _select_judge_context(
        retrieved_contexts,
        question["question"],
        normalized_answer,
        gold_answer,
    )
    prompt = (
        "You are grading a RAG system. Return ONLY valid JSON with numeric scores from 0 to 1.\n"
        "Use this schema exactly: "
        '{"correctness": number, "faithfulness": number, "answer_relevancy": number, "reason": string}\n\n'
        "Scoring rules:\n"
        "- correctness: compare ACTUAL ANSWER to EXPECTED ANSWER for the facts requested. "
        "Accept paraphrases, concise answers, source labels, currency symbols, and formatting differences. "
        "If all requested values/facts are present, score 1.0 even if wording differs.\n"
        "- faithfulness: judge at the CLAIM level (RAGAS-style) — a claim is supported if it can be "
        "inferred from the RETRIEVED CONTEXT, not only if it appears verbatim. A comparison, ranking, or "
        "conclusion that follows from facts that ARE present in the context (e.g. 'X allows a longer term "
        "than Y' when X's and Y's terms are both in the context) IS supported — score it faithful. "
        "Do not require exact wording; values under matching field labels count as support. "
        "Do not penalize missing citations. Penalize only claims that CONTRADICT the context, introduce "
        "facts ABSENT from the context, or mix values across the wrong sources.\n"
        "- answer_relevancy: score whether ACTUAL ANSWER directly addresses the QUESTION. "
        "Concise direct answers are relevant.\n\n"
        f"QUESTION:\n{question['question']}\n\n"
        f"EXPECTED ANSWER:\n{gold_answer}\n\n"
        f"ACTUAL ANSWER:\n{normalized_answer}\n\n"
        f"RETRIEVED CONTEXT:\n{context}\n"
    )

    client = openai.OpenAI(base_url=judge_base_url, api_key=judge_api_key)
    response = client.chat.completions.create(
        model=judge_model_name,
        messages=[
            {
                "role": "system",
                "content": "You are a strict but fair evaluation judge. Output valid JSON only.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=220,
        response_format={"type": "json_object"},
    )
    parsed = _extract_json_scores(response.choices[0].message.content or "")
    if not parsed:
        raise RuntimeError("custom judge returned invalid JSON")

    exact_score = _exact_match_score(normalized_answer, gold_answer)
    correctness = _clamp_score(parsed.get("correctness"), exact_score)
    # If deterministic key-fact matching is perfect, do not let the LLM under-score
    # a terse but correct answer such as "$297 billion; 42 new topic areas."
    if exact_score == 1.0 and correctness is not None:
        correctness = max(correctness, 1.0)

    return {
        "correctness": correctness,
        "faithfulness": _clamp_score(parsed.get("faithfulness"), None),
        "answer_relevancy": _clamp_score(parsed.get("answer_relevancy"), None),
        "judge_used": "custom_llm_judge",
    }


# from src.reranker import BGEReranker, QwenReranker
# from src.rag_agent import _hyde

EVAL_DIR = REPO_ROOT / "eval"
QA_PAIRS_DIR = EVAL_DIR / "data/qa_pairs"
RESULTS_DIR = EVAL_DIR / "results"
ANSWER_RESULTS_PATH = RESULTS_DIR / "answer_results.jsonl"
RETRIEVAL_RESULTS_PATH = RESULTS_DIR / "retrieval_results.jsonl"
SUMMARY_PATH = RESULTS_DIR / "summary.json"


def _strip_doc_ids(question: str) -> str:
    """Remove inline doc_XXX references so the agent resolves documents by title."""
    # "In doc_001, ..." → "..."
    q = re.sub(r"\bIn doc_\d{3},\s*", "", question, flags=re.IGNORECASE)
    # " (doc_001)" parentheticals
    q = re.sub(r"\s*\(doc_\d{3}\)", "", q)
    # ": doc_009 or doc_013" trailing qualifier after colon
    q = re.sub(r":\s*doc_\d{3}(?:\s+or\s+doc_\d{3})?", "", q)
    # any remaining bare doc_XXX
    q = re.sub(r"\bdoc_\d{3}\b", "", q)
    return re.sub(r"\s{2,}", " ", q).strip()


def _normalise_question(
    item: dict[str, Any], file_stem: str, idx: int
) -> dict[str, Any]:
    """Normalise a qa_pairs JSON item to the format expected by the eval runner.

    Handles:
    - qa_id: prefix with file_stem to avoid collisions across files
    - evidence: renamed from gold_evidence, keeping only doc_id and quote
    - question: doc_XXX inline references stripped so the agent resolves by title
    """
    qa_id = f"{file_stem}__{item.get('qa_id') or f'q{idx}'}"
    raw_evidence = item.get("gold_evidence") or item.get("evidence") or []
    evidence = [
        {
            "doc_id": ev.get("doc_id", ""),
            "quote": ev.get("quote", ""),
            "evidence_type": ev.get("evidence_type", ""),
        }
        for ev in raw_evidence
    ]
    question = _strip_doc_ids(item.get("question", ""))
    return {**item, "qa_id": qa_id, "evidence": evidence, "question": question}


def load_questions(qa_dir: Path) -> list[dict[str, Any]]:
    """Load all QA pairs from *.json files in qa_dir, normalising schemas."""
    questions: list[dict[str, Any]] = []
    for path in sorted(qa_dir.glob("*.json")):
        items = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(items, list):
            items = [items]
        stem = path.stem
        for idx, item in enumerate(items, start=1):
            questions.append(_normalise_question(item, stem, idx))
    return questions


def strip_think_blocks(text: str) -> str:
    """Remove <think>…</think> reasoning blocks and keep only the final answer."""
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"<think>[\s\S]*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^\s*answer\s*:\s*", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


_RETRY_INSTRUCTION = (
    "\n\nIMPORTANT: This is a retry. The previous attempt returned Unsupported. "
    "You MUST follow the doc-routing protocol strictly: "
    "(1) call search_knowledge_base with the topic words alone to identify the relevant "
    "document_summary chunk and read its Document ID (doc_XXX). "
    "(2) call search_knowledge_base again with the original question, passing that doc_id "
    "as the doc_id argument to scope the search to that specific document."
)


def normalize_unsupported(text: str) -> str:
    """Accept only the literal word 'Unsupported' (case-insensitive) as a valid refusal.

    The agent prompt already mandates this exact output. Hedging phrases are intentionally
    NOT converted here — they count as wrong answers so we can measure whether the agent
    actually follows its own instructions.
    """
    if text.strip().lower() == "unsupported":
        return "Unsupported"
    return text


def _exact_match_score(predicted: str, gold: str) -> float:
    """Heuristic correctness fallback when the LLM judge is unavailable.

    Returns 1.0 for an exact or substring match, or when all numeric tokens from
    gold appear in predicted AND token overlap is ≥70% (strong factual match).
    Returns 0.5 for partial overlap (≥50%) or when all numeric tokens are present.
    Returns 0.0 otherwise.
    """
    p = predicted.lower().strip()
    g = gold.lower().strip()
    if not g:
        return 1.0 if not p else 0.0
    if p == g or g in p or p in g:
        return 1.0
    gold_tokens = set(re.findall(r"\w+", g))
    pred_tokens = set(re.findall(r"\w+", p))
    overlap_ratio = (
        len(gold_tokens & pred_tokens) / len(gold_tokens) if gold_tokens else 0.0
    )
    # Key-fact check: numeric/dollar tokens from gold that appear in predicted.
    key_tokens = set(re.findall(r"\d[\d,./]*", g))
    pred_numeric = set(re.findall(r"\d[\d,./]*", p))
    all_numeric_present = bool(key_tokens) and key_tokens <= pred_numeric
    # Strong match: all numeric facts present AND high word overlap → 1.0
    if all_numeric_present and overlap_ratio >= 0.70:
        return 1.0
    if overlap_ratio >= 0.5:
        return 0.5
    if all_numeric_present:
        return 0.5
    return 0.0


def _normalize_text(text: str) -> str:
    """Normalize evidence/result text for robust substring and token matching."""
    text = text.lower().replace("|", " ")
    text = re.sub(r"[\u2018\u2019]", "'", text)
    text = re.sub(r"[\u201c\u201d]", '"', text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _evidence_matches_hit(evidence_quote: str, hit_content: str) -> bool:
    """Return True when a retrieved hit contains the gold evidence closely enough."""
    quote = _normalize_text(evidence_quote)
    content = _normalize_text(hit_content)
    if not quote:
        return False
    if quote in content:
        return True

    def _token_variants(token: str) -> set[str]:
        variants = {token}
        compact = token.replace(",", "").replace("$", "")
        variants.add(compact)
        if re.fullmatch(r"-?\d+\.0+", compact):
            variants.add(compact.split(".", 1)[0])
        if re.fullmatch(r"-?\d+\.\d+", compact):
            variants.add(compact.rstrip("0").rstrip("."))
        return {v for v in variants if v}

    quote_tokens = [t for t in re.findall(r"[a-z0-9$.,¼½'-]+", quote) if len(t) > 1]
    if not quote_tokens:
        return False
    overlap = sum(
        1 for token in quote_tokens if any(v in content for v in _token_variants(token))
    )
    return overlap / len(quote_tokens) >= 0.75


def _hit_doc_id(hit: dict[str, Any]) -> str | None:
    meta = hit.get("metadata", {}) or {}
    if meta.get("doc_id"):
        return str(meta["doc_id"])
    source = str(meta.get("source_file") or meta.get("file_name") or "")
    match = re.search(r"\bdoc_\d{3}\b", source)
    return match.group(0) if match else None


def _summarize_hit(hit: dict[str, Any]) -> dict[str, Any]:
    meta = hit.get("metadata", {}) or {}
    return {
        "doc_id": _hit_doc_id(hit),
        "chunk_type": meta.get("chunk_type"),
        "page": meta.get("page"),
        "sheet_name": meta.get("sheet_name"),
        "row_ref": meta.get("row_ref"),
        "content": (hit.get("content") or "")[:1200],
        "score": hit.get("score"),
    }


_EXCEL_EVIDENCE_TYPES = {"sheet_row", "sheet_table"}


def _is_excel_question(question: dict[str, Any]) -> bool:
    """True when all gold evidence comes from structured Excel/CSV sources."""
    evidence = question.get("evidence") or []
    return bool(evidence) and all(
        ev.get("evidence_type", "") in _EXCEL_EVIDENCE_TYPES for ev in evidence
    )


def evaluate_retrieval(question: dict[str, Any], top_k: int = 20) -> dict[str, Any]:
    """Evaluate evidence retrieval, routing by evidence modality.

    - Unanswerable (no evidence): method=none, all metrics None.
    - Excel questions (sheet_row/sheet_table evidence): method=structured, skip Qdrant.
      Accuracy is derived later from agent answer correctness.
    - PDF questions: method=vector, Qdrant hit@K with no chunk-type routing filter
      (force_chunk_types=[]) so PDF table/figure questions are not zero-hit.
    """
    evidence = question.get("evidence") or []
    base = {
        "qa_id": question["qa_id"],
        "question_type": question.get("question_type"),
        "evidence_count": len(evidence),
        "hits": [],
        "hit_at_5": None,
        "hit_at_10": None,
        "mrr": None,
        "recall_at_10": None,
        "recall_at_20": None,
        "matched_evidence": [],
    }
    if not evidence:
        return {**base, "retrieval_method": "none"}
    if _is_excel_question(question):
        return {**base, "retrieval_method": "structured"}

    hits = retrieve(
        question["question"],
        top_k=top_k,
        qdrant_url=QDRANT_URL,
        collection=QDRANT_COLLECTION,
        use_qdrant=True,
        force_chunk_types=[],  # bypass routing filter so PDF table questions are not zero-hit
    )

    all_hits: list[dict[str, Any]] = list(hits)
    matched_evidence: list[dict[str, Any]] = []
    ranks: list[int | None] = []

    for ev_idx, ev in enumerate(evidence):
        doc_id = ev.get("doc_id") or ""
        rank = None
        for idx, hit in enumerate(hits, start=1):
            if _evidence_matches_hit(ev.get("quote", ""), hit.get("content", "")):
                rank = idx
                break
        ranks.append(rank)
        if rank is not None:
            matched_evidence.append(
                {"evidence_index": ev_idx, "doc_id": doc_id, "rank": rank}
            )

    matched = [rank for rank in ranks if rank is not None]
    return {
        "qa_id": question["qa_id"],
        "question_type": question.get("question_type"),
        "evidence_count": len(evidence),
        "retrieval_method": "vector",
        "hits": [_summarize_hit(hit) for hit in all_hits[:top_k]],
        "hit_at_5": 1.0
        if any(rank is not None and rank <= 5 for rank in ranks)
        else 0.0,
        "hit_at_10": 1.0
        if any(rank is not None and rank <= 10 for rank in ranks)
        else 0.0,
        "mrr": max((1.0 / rank for rank in matched), default=0.0),
        "recall_at_10": sum(1 for rank in ranks if rank is not None and rank <= 10)
        / len(evidence),
        "recall_at_20": sum(1 for rank in ranks if rank is not None and rank <= 20)
        / len(evidence),
        "matched_evidence": matched_evidence,
    }


def evaluate_answer(
    question: dict[str, Any], answer: str, retrieved_contexts: list[str]
) -> dict[str, Any]:
    """Compute answer metrics (correctness, faithfulness, answer_relevancy).

    Defaults to a compact custom JSON judge. DeepEval can still be selected with
    EVAL_JUDGE_MODE=deepeval, but its multi-step faithfulness/relevancy metrics
    are slow and brittle for this demo benchmark.
    """
    gold_answer = (question.get("gold_answer") or "").strip()
    cleaned = strip_think_blocks(answer)
    # Only normalize hedging/refusal patterns for unanswerable questions — applying
    # this globally would convert retrieval failures on answerable questions to "Unsupported",
    # replacing a partially-correct answer with a fully-wrong one.
    normalized_answer = (
        normalize_unsupported(cleaned)
        if question.get("question_type") == "unanswerable"
        else cleaned
    )

    correctness = None
    faithfulness = None
    answer_relevancy = None
    judge_used = "custom_llm_judge"

    # Short-circuit for exact match — avoids DeepEval non-determinism on trivial cases
    # (e.g. normalized_answer == gold_answer == "Unsupported" scoring 0.0).
    is_unanswerable = question.get("question_type") == "unanswerable"
    # Faithfulness (groundedness in retrieved TEXT) is undefined for text-to-SQL
    # Excel answers — there is no retrieved context, the value comes from DuckDB.
    # Score them None (excluded from the average), same as unanswerable questions.
    no_faithfulness = is_unanswerable or _is_excel_question(question)

    if normalized_answer.strip().lower() == gold_answer.strip().lower():
        return {
            "correctness": 1.0,
            "faithfulness": None if no_faithfulness else 1.0,
            "answer_relevancy": 1.0,
            "judge_used": "exact_match_shortcircuit",
            "clean_answer": normalized_answer,
        }

    if os.getenv("EVAL_JUDGE_MODE", "custom").lower() != "deepeval":
        try:
            judged = _custom_judge_answer(
                question, normalized_answer, gold_answer, retrieved_contexts
            )
            if no_faithfulness:
                judged["faithfulness"] = None
            judged["clean_answer"] = normalized_answer
            return judged
        except BaseException as exc:
            logging.warning(
                "  [WARN] custom judge failed (%s) — using exact-match fallback.", exc
            )
            return {
                "correctness": _exact_match_score(normalized_answer, gold_answer),
                "faithfulness": None,
                "answer_relevancy": None,
                "judge_used": "exact_match_fallback",
                "clean_answer": normalized_answer,
            }

    try:
        from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, GEval
        from deepeval.models import GPTModel
        from deepeval.test_case import LLMTestCase, LLMTestCaseParams

        judge_model_name, judge_base_url, judge_api_key = _judge_config()
        judge_model = GPTModel(
            model=judge_model_name,
            base_url=judge_base_url,
            api_key=judge_api_key,
        )
        test_case = LLMTestCase(
            input=question["question"],
            actual_output=normalized_answer,
            expected_output=gold_answer,
            retrieval_context=_trim_judge_context(retrieved_contexts),
        )
        correctness_metric = GEval(
            name="Correctness",
            model=judge_model,
            async_mode=False,
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT,
            ],
            criteria=(
                "Score whether the answer correctly answers the question and is semantically "
                "consistent with the expected answer. Accept paraphrases and formatting differences."
            ),
        )
        faithfulness_metric = FaithfulnessMetric(model=judge_model, async_mode=False)
        answer_relevancy_metric = AnswerRelevancyMetric(
            model=judge_model, async_mode=False
        )
        correctness_metric.measure(test_case)
        faithfulness_metric.measure(test_case)
        answer_relevancy_metric.measure(test_case)
        correctness = correctness_metric.score
        faithfulness = faithfulness_metric.score
        answer_relevancy = answer_relevancy_metric.score
        if correctness is None:
            correctness = _exact_match_score(normalized_answer, gold_answer)
            judge_used = "deepeval_partial_exact_match_fallback"
    except BaseException as _judge_exc:
        import time

        logging.warning(
            "  [WARN] deepeval judge failed (%s) — retrying once after 5s.", _judge_exc
        )
        time.sleep(5)
        try:
            from deepeval.metrics import (
                AnswerRelevancyMetric,
                FaithfulnessMetric,
                GEval,
            )
            from deepeval.models import GPTModel
            from deepeval.test_case import LLMTestCase, LLMTestCaseParams

            judge_model_name, judge_base_url, judge_api_key = _judge_config()
            judge_model2 = GPTModel(
                model=judge_model_name,
                base_url=judge_base_url,
                api_key=judge_api_key,
            )
            test_case2 = LLMTestCase(
                input=question["question"],
                actual_output=normalized_answer,
                expected_output=gold_answer,
                retrieval_context=_trim_judge_context(retrieved_contexts),
            )
            cm2 = GEval(
                name="Correctness",
                model=judge_model2,
                async_mode=False,
                evaluation_params=[
                    LLMTestCaseParams.INPUT,
                    LLMTestCaseParams.ACTUAL_OUTPUT,
                    LLMTestCaseParams.EXPECTED_OUTPUT,
                ],
                criteria="Score whether the answer correctly answers the question and is semantically consistent with the expected answer. Accept paraphrases and formatting differences.",
            )
            fm2 = FaithfulnessMetric(model=judge_model2, async_mode=False)
            arm2 = AnswerRelevancyMetric(model=judge_model2, async_mode=False)
            cm2.measure(test_case2)
            fm2.measure(test_case2)
            arm2.measure(test_case2)
            correctness = cm2.score
            faithfulness = fm2.score
            answer_relevancy = arm2.score
            if correctness is None:
                correctness = _exact_match_score(normalized_answer, gold_answer)
                judge_used = "deepeval_partial_exact_match_fallback"
        except BaseException:
            logging.exception(
                "  [WARN] deepeval retry also failed — using exact-match fallback."
            )
            correctness = _exact_match_score(normalized_answer, gold_answer)
            judge_used = "exact_match_fallback"

    return {
        "correctness": correctness,
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "judge_used": judge_used,
        "clean_answer": normalized_answer,
    }


def run(
    category_filter: str | None = None,
    qa_files: list[str] | None = None,
) -> dict[str, Any]:
    """Run the full-agent benchmark and write per-question results plus a summary."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if qa_files:
        questions = []
        for f in qa_files:
            items = json.loads(Path(f).read_text(encoding="utf-8"))
            if not isinstance(items, list):
                items = [items]
            stem = Path(f).stem
            for idx, item in enumerate(items, start=1):
                questions.append(_normalise_question(item, stem, idx))
    else:
        questions = load_questions(QA_PAIRS_DIR)
    if category_filter:
        questions = [q for q in questions if q.get("question_type") == category_filter]

    retrieval_rows = [evaluate_retrieval(question) for question in questions]
    RETRIEVAL_RESULTS_PATH.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in retrieval_rows),
        encoding="utf-8",
    )

    agent = build_rag_agent()
    reflection_pipeline = build_reflection_pipeline(agent)
    decomposition_pipeline = build_decomposition_pipeline(agent)

    answer_rows: list[dict[str, Any]] = []

    for idx, question in enumerate(questions, start=1):
        query = question["question"]
        print(f"[{idx}/{len(questions)}] {question['qa_id']} — {query[:80]}")

        # Multi-hop PDF/cross-doc questions go through decomposition so the agent gets
        # an explicit sub-question plan. Excel cross-doc questions are excluded — the
        # SQL sub-agent handles multiple tables natively in one call, and decomposition
        # would just add an extra gpt-4o-mini call that competes with the eval judge TPM.
        evidence_doc_ids = {
            ev.get("doc_id", "")
            for ev in question.get("evidence", [])
            if ev.get("doc_id")
        }
        is_multihop = not _looks_excel(query) and (
            len(evidence_doc_ids) > 1
            or question.get("question_type", "").startswith("cross")
        )

        retrieved_contexts: list[str] = []
        if is_multihop:
            try:
                answer = ask_with_decomposition(
                    decomposition_pipeline, query, collected_chunks=retrieved_contexts
                )
            except Exception as exc:
                print(f"  [WARN] decomposition pipeline failed: {exc}")
                answer = "Unsupported"
        else:
            # stream_agent collects tool-call chunks for FaithfulnessMetric.
            # Falls back to reflection on failure.
            try:
                answer_tokens: list[str] = []
                for token in stream_agent(
                    agent, query, collected_chunks=retrieved_contexts
                ):
                    answer_tokens.append(token)
                answer = "".join(answer_tokens)
            except Exception as exc:
                print(
                    f"  [WARN] stream_agent failed ({type(exc).__name__}): {exc}. Retrying with reflection pipeline."
                )
                try:
                    answer = ask_with_reflection(reflection_pipeline, query)
                except Exception as exc2:
                    print(f"  [ERROR] reflection pipeline also failed: {exc2}")
                    answer = "Unsupported"

        # If answer is still Unsupported, give reflection a chance regardless of question type.
        if answer.strip().lower() == "unsupported":
            try:
                reflected = ask_with_reflection(reflection_pipeline, query)
                if reflected.strip().lower() != "unsupported":
                    answer = reflected
            except Exception as exc:
                print(f"  [WARN] reflection retry failed: {exc}")

        # Final fallback: forced retry with explicit doc-routing instruction.
        # Mirrors api.py — Groq nondeterminism at temp=0 occasionally causes the
        # agent to skip stage-1 routing on first attempt. Only fires for non-multihop
        # paths (decomposition has its own per-sub-question retry).
        if not is_multihop and answer.strip().lower() == "unsupported":
            try:
                retry_chunks: list[str] = []
                retry_tokens: list[str] = []
                for token in stream_agent(
                    agent, query + _RETRY_INSTRUCTION, collected_chunks=retry_chunks
                ):
                    retry_tokens.append(token)
                retry_answer = "".join(retry_tokens).strip()
                if retry_answer and retry_answer.lower() != "unsupported":
                    answer = retry_answer
                    retrieved_contexts = retry_chunks
            except Exception as exc:
                print(f"  [WARN] forced retry failed: {exc}")

        answer_eval = evaluate_answer(question, answer, retrieved_contexts)
        answer_rows.append(
            {
                "qa_id": question["qa_id"],
                "question": query,
                "gold_answer": question.get("gold_answer"),
                "predicted_answer": answer_eval.pop("clean_answer", answer),
                **answer_eval,
            }
        )

        # Pace requests to stay under the eval judge TPM limit (200k/min on gpt-4o-mini).
        time.sleep(6)

    ANSWER_RESULTS_PATH.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in answer_rows),
        encoding="utf-8",
    )

    correctness = [
        r["correctness"] for r in answer_rows if r.get("correctness") is not None
    ]
    faithfulness = [
        r["faithfulness"] for r in answer_rows if r.get("faithfulness") is not None
    ]
    answer_relevancy = [
        r["answer_relevancy"]
        for r in answer_rows
        if r.get("answer_relevancy") is not None
    ]
    judge_breakdown = dict(Counter(r.get("judge_used", "unknown") for r in answer_rows))

    # Split retrieval rows by modality
    vector_rows = [r for r in retrieval_rows if r.get("retrieval_method") == "vector"]
    structured_qids = {
        r["qa_id"] for r in retrieval_rows if r.get("retrieval_method") == "structured"
    }
    unanswerable_qids = {
        r["qa_id"] for r in retrieval_rows if r.get("retrieval_method") == "none"
    }

    hit_at_5 = [r["hit_at_5"] for r in vector_rows if r.get("hit_at_5") is not None]
    hit_at_10 = [r["hit_at_10"] for r in vector_rows if r.get("hit_at_10") is not None]
    mrr = [r["mrr"] for r in vector_rows if r.get("mrr") is not None]
    recall_at_10 = [
        r["recall_at_10"] for r in vector_rows if r.get("recall_at_10") is not None
    ]
    recall_at_20 = [
        r["recall_at_20"] for r in vector_rows if r.get("recall_at_20") is not None
    ]

    # Structured accuracy: token-overlap between agent answer and gold answer
    answer_by_qid = {r["qa_id"]: r for r in answer_rows}
    excel_rows = [answer_by_qid[qid] for qid in structured_qids if qid in answer_by_qid]
    excel_correct = sum(1 for r in excel_rows if (r.get("correctness") or 0) >= 1.0)

    # Unanswerable: fraction where agent correctly refused
    unanswerable_rows = [
        answer_by_qid[qid] for qid in unanswerable_qids if qid in answer_by_qid
    ]
    unanswerable_correct = sum(
        1 for r in unanswerable_rows if (r.get("correctness") or 0) >= 1.0
    )

    summary = {
        "question_count": len(questions),
        "vector_retrieval_metrics": {
            "scope": "PDF/OCR questions only (Qdrant dense search)",
            "question_count": len(vector_rows),
            "hit_at_5": sum(hit_at_5) / len(hit_at_5) if hit_at_5 else None,
            "hit_at_10": sum(hit_at_10) / len(hit_at_10) if hit_at_10 else None,
            "mrr": sum(mrr) / len(mrr) if mrr else None,
            "evidence_recall_at_10": sum(recall_at_10) / len(recall_at_10)
            if recall_at_10
            else None,
            "evidence_recall_at_20": sum(recall_at_20) / len(recall_at_20)
            if recall_at_20
            else None,
        },
        "structured_retrieval_metrics": {
            "scope": "Excel/CSV questions (DuckDB-served, not Qdrant)",
            "question_count": len(excel_rows),
            "answer_accuracy": excel_correct / len(excel_rows) if excel_rows else None,
        },
        "unanswerable_metrics": {
            "question_count": len(unanswerable_rows),
            "correct_refusal_rate": unanswerable_correct / len(unanswerable_rows)
            if unanswerable_rows
            else None,
        },
        "agent_answer_metrics": {
            "correctness": sum(correctness) / len(correctness) if correctness else None,
            "faithfulness": sum(faithfulness) / len(faithfulness)
            if faithfulness
            else None,
            "answer_relevancy": sum(answer_relevancy) / len(answer_relevancy)
            if answer_relevancy
            else None,
        },
        "judge_breakdown": judge_breakdown,
        "question_breakdown": dict(Counter(q["question_type"] for q in questions)),
        "answer_result_file": str(ANSWER_RESULTS_PATH),
        "retrieval_result_file": str(RETRIEVAL_RESULTS_PATH),
    }
    SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--category",
        default=None,
        help="Filter to a single question_type (e.g. table_lookup)",
    )
    parser.add_argument(
        "--qa-files", nargs="+", default=None, help="Run only these QA JSON files"
    )
    args = parser.parse_args()
    print(
        json.dumps(
            run(category_filter=args.category, qa_files=args.qa_files),
            ensure_ascii=False,
            indent=2,
        )
    )
