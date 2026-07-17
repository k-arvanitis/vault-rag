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

from src.answer_pipeline import answer_query  # noqa: E402
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
from src.rag_agent import build_rag_agent  # noqa: E402
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
        "- HEDGED/INFERRED CLAIMS ARE UNFAITHFUL: if the answer justifies a claim with language like "
        "'implied by', 'the absence of X suggests', 'typically involves', 'is likely', or otherwise "
        "reasons from general/domain knowledge rather than quoting or restating something actually in "
        "the RETRIEVED CONTEXT, that claim is NOT supported — score it unfaithful (low score) even if "
        "the conclusion happens to match the expected answer. A guess that turns out right is still a "
        "guess. Only score a comparative claim faithful when BOTH compared facts are explicitly present "
        "in the RETRIEVED CONTEXT (directly stated or a clear paraphrase), not when one side is inferred.\n"
        "- answer_relevancy: score whether ACTUAL ANSWER directly addresses the QUESTION. "
        "Concise direct answers are relevant.\n\n"
        f"QUESTION:\n{question['question']}\n\n"
        f"EXPECTED ANSWER:\n{gold_answer}\n\n"
        f"ACTUAL ANSWER:\n{normalized_answer}\n\n"
        f"RETRIEVED CONTEXT:\n{context}\n"
    )

    client = openai.OpenAI(base_url=judge_base_url, api_key=judge_api_key)
    # Retry transient rate limits AND malformed-JSON responses (the shared free-tier
    # judge endpoint has sustained exhaustion windows, and separately sometimes
    # returns a truncated/non-JSON body despite response_format=json_object) before
    # letting the caller fall back to the much cruder exact-match scorer — either
    # failure mode silently degrades a whole question's score to token-overlap
    # matching instead of the semantic judge, for reasons unrelated to the actual
    # answer quality.
    last_exc: Exception | None = None
    parsed: dict[str, Any] | None = None
    for attempt, delay in enumerate((0, 3, 8)):
        if delay:
            import time

            time.sleep(delay)
        try:
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
                # Reasoning models (gpt-oss-120b and similar) spend completion tokens
                # on a reasoning trace BEFORE emitting the JSON answer — 220 tokens
                # was consumed entirely by reasoning, so content came back null every
                # time (finish_reason="length"), not because of a malformed response.
                # Verified directly: the same prompt with max_tokens=220 produced
                # content=null after 317 reasoning tokens; this budget covers both.
                max_tokens=900,
                response_format={"type": "json_object"},
            )
        except openai.RateLimitError as exc:
            last_exc = exc
            continue
        parsed = _extract_json_scores(response.choices[0].message.content or "")
        if parsed:
            break
        last_exc = RuntimeError("custom judge returned invalid JSON")
    if not parsed:
        raise last_exc  # noqa: B904 — re-raise after exhausting retries

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
RAW_ANSWERS_PATH = RESULTS_DIR / "raw_answers.jsonl"
SUMMARY_PATH = RESULTS_DIR / "summary.json"
FAILURES_PATH = RESULTS_DIR / "failures.md"

# Failure-category heuristics keyed on question_type, used only to give a human
# a starting point in failures.md — not a scoring signal.
_FAILURE_CATEGORY_BY_TYPE = {
    "cross_document_compare": "cross-document reasoning",
    "numeric_lookup": "numeric extraction",
    "table_lookup": "table/spreadsheet lookup",
    "table_grounding": "table/spreadsheet lookup",
    "ocr_extraction": "OCR/scanned-page extraction",
    "figure_grounding": "figure/chart grounding",
    "negation_check": "negation handling",
    "single_doc_factoid": "single-document factoid",
    "unanswerable": "false positive (should have refused)",
}


def _accuracy_by_type(answer_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per-question-type count + mean correctness, for the eval breakdown table."""
    by_type: dict[str, list[float]] = {}
    for row in answer_rows:
        qtype = row.get("question_type") or "unknown"
        score = row.get("correctness")
        if score is not None:
            by_type.setdefault(qtype, []).append(score)
    return {
        qtype: {"count": len(scores), "correctness": sum(scores) / len(scores)}
        for qtype, scores in sorted(by_type.items())
    }


def _render_failures_md(
    answer_rows: list[dict[str, Any]], threshold: float = 0.5
) -> str:
    """Render a markdown table of every question scoring below threshold, for manual triage."""
    failures = [
        r
        for r in answer_rows
        if (r.get("correctness") if r.get("correctness") is not None else 1.0)
        < threshold
    ]
    lines = [
        "# Eval failures — auto-generated by `make eval`, do not hand-edit\n",
        f"{len(failures)} of {len(answer_rows)} questions scored below {threshold} correctness.\n",
        "| qa_id | type | category | correctness | question | gold | predicted |",
        "|---|---|---|---:|---|---|---|",
    ]
    for r in failures:
        qtype = r.get("question_type") or "unknown"
        category = _FAILURE_CATEGORY_BY_TYPE.get(qtype, qtype)
        question = (
            (r.get("question") or "").replace("|", "\\|").replace("\n", " ")[:120]
        )
        gold = (r.get("gold_answer") or "").replace("|", "\\|").replace("\n", " ")[:100]
        predicted = (
            (r.get("predicted_answer") or "")
            .replace("|", "\\|")
            .replace("\n", " ")[:100]
        )
        score = r.get("correctness")
        lines.append(
            f"| {r['qa_id']} | {qtype} | {category} | {score:.2f} | {question} | {gold} | {predicted} |"
        )
    return "\n".join(lines) + "\n"


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


def normalize_unsupported(text: str) -> str:
    """Accept only the literal word 'Unsupported' (case-insensitive) as a valid refusal.

    The agent prompt already mandates this exact output. Hedging phrases are intentionally
    NOT converted here — they count as wrong answers so we can measure whether the agent
    actually follows its own instructions.
    """
    if text.strip().lower() == "unsupported":
        return "Unsupported"
    return text


_DATE_SLASH_DMY = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b")
_DATE_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")


def _canonicalize_dates(text: str) -> str:
    """Rewrite DD/MM/YYYY and YYYY-MM-DD dates to a single ISO-without-separators
    form (YYYYMMDD) so the same date in either format tokenizes identically for
    _exact_match_score's numeric-token comparison. Without this, '29/04/2025' and
    '2025-04-29' are the same date but compare as different token sets — a real
    false-negative found evaluating table_lookup questions (source data stores
    dates as ISO, but gold answers were authored in DD/MM/YYYY)."""
    text = _DATE_SLASH_DMY.sub(lambda m: f"{m.group(3)}{m.group(2)}{m.group(1)}", text)
    text = _DATE_ISO.sub(lambda m: f"{m.group(1)}{m.group(2)}{m.group(3)}", text)
    return text


def _exact_match_score(predicted: str, gold: str) -> float:
    """Heuristic correctness fallback when the LLM judge is unavailable.

    Returns 1.0 for an exact or substring match, or when all numeric tokens from
    gold appear in predicted AND token overlap is ≥70% (strong factual match).
    Returns 0.5 for partial overlap (≥50%) or when all numeric tokens are present.
    Returns 0.0 otherwise.
    """
    p = _canonicalize_dates(predicted.lower().strip())
    g = _canonicalize_dates(gold.lower().strip())
    if not g:
        return 1.0 if not p else 0.0
    # A "Clarify: ..." response is a deferral, not a commitment to an answer — even
    # when the right value happens to be one of several candidates it lists, that's
    # not the same as confidently answering. None of this corpus's gold answers are
    # themselves a Clarify response, so any Clarify prediction is a miss here.
    if p.startswith("clarify:"):
        return 0.0
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


def _load_questions_for_run(
    category_filter: str | None, qa_files: list[str] | None
) -> list[dict[str, Any]]:
    """Shared question-loading logic for generate_answers() and run()."""
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
        available = sorted({q.get("question_type") for q in questions})
        questions = [q for q in questions if q.get("question_type") == category_filter]
        # An exact-match typo here (e.g. "cross_document" instead of the real
        # "cross_document_compare") used to silently produce 0 questions --
        # generate_answers() would then write nothing, and judge_answers()
        # (which doesn't check freshness) would silently re-score whatever
        # stale raw_answers.jsonl was already on disk, with zero warning that
        # nothing had actually been re-run. Reproduced live (2026-07-15): an
        # entire session's worth of "before/after" eval comparisons were
        # unknowingly judging the same frozen answers from days earlier.
        if not questions:
            raise ValueError(
                f"--category {category_filter!r} matched 0 questions. "
                f"Available question_type values: {available}"
            )
    return questions


def generate_answers(
    category_filter: str | None = None,
    qa_files: list[str] | None = None,
    resume: bool = False,
) -> list[dict[str, Any]]:
    """Phase 1: run the full agent on every question and save raw predicted answers.

    No judge calls here — this only exercises retrieval + generation, so it can
    be run and inspected independently of whether the judge is working. Writes
    eval/results/retrieval_results.jsonl (no LLM involved) and
    eval/results/raw_answers.jsonl (qa_id, question, gold_answer, predicted
    answer, retrieved context — everything judge_answers() needs).

    resume: skip qa_ids already present in an existing raw_answers.jsonl
    instead of regenerating them. The full corpus is ~109 questions at
    ~1-1.5 min each (no parallelism) -- a killed run previously meant losing
    everything and restarting from zero. With resume=True, re-running the
    same command after an interruption only pays for the remaining questions.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    questions = _load_questions_for_run(category_filter, qa_files)

    retrieval_rows = [evaluate_retrieval(question) for question in questions]
    RETRIEVAL_RESULTS_PATH.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in retrieval_rows),
        encoding="utf-8",
    )

    total_questions = len(questions)
    done_rows: list[dict[str, Any]] = []
    if resume and RAW_ANSWERS_PATH.exists():
        done_rows = [
            json.loads(line)
            for line in RAW_ANSWERS_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        done_ids = {row["qa_id"] for row in done_rows}
        questions = [q for q in questions if q["qa_id"] not in done_ids]
        if done_rows:
            print(
                f"Resuming: {len(done_rows)} already-generated answers found, {len(questions)} remaining."
            )

    if not questions:
        print(f"\nAll {len(done_rows)} answers already present -> {RAW_ANSWERS_PATH}")
        return done_rows

    agent = build_rag_agent()
    reflection_pipeline = build_reflection_pipeline(agent)
    decomposition_pipeline = build_decomposition_pipeline(agent)

    raw_rows: list[dict[str, Any]] = list(done_rows)

    for idx, question in enumerate(questions, start=1):
        query = question["question"]
        print(f"[{idx}/{len(questions)}] {question['qa_id']} — {query[:80]}")

        # Decomposition is opt-in, off by default. api.py never uses it — production
        # traffic always goes through the plain ReAct agent (stream_agent) below. An
        # on/off ablation on the cross-document bucket (same judge, same code, only
        # this flag differed) showed decomposition net-hurts: it tends to answer only
        # one half of two-part comparison questions, where the plain agent's own
        # multi-tool-call loop naturally covers both. Set EVAL_ENABLE_DECOMPOSITION=1
        # to re-run that experiment; default matches what real users get.
        evidence_doc_ids = {
            ev.get("doc_id", "")
            for ev in question.get("evidence", [])
            if ev.get("doc_id")
        }
        is_multihop = (
            bool(os.getenv("EVAL_ENABLE_DECOMPOSITION"))
            and not _looks_excel(query)
            and (
                len(evidence_doc_ids) > 1
                or question.get("question_type", "").startswith("cross")
            )
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
            # answer_query is the same routing/retry/multi-part-split pipeline
            # api.py's /query endpoint uses — sharing it here means a fix
            # verified live in the app (e.g. the qa_4 pronoun-split fix) also
            # shows up in this eval run, instead of eval silently measuring a
            # weaker, unshared code path. Its own routing + unsupported-retry
            # + comparison-retry replace the old standalone forced-retry block
            # below (removed: same instruction, but without the routing
            # directive answer_query's internal retry already applies).
            try:
                result = answer_query(agent, query)
                answer = result["answer"]
                retrieved_contexts = result["collected"]
            except Exception as exc:
                print(
                    f"  [WARN] answer_query failed ({type(exc).__name__}): {exc}. Retrying with reflection pipeline."
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

        raw_rows.append(
            {
                "qa_id": question["qa_id"],
                "question_type": question.get("question_type"),
                "question": query,
                "gold_answer": question.get("gold_answer"),
                "evidence": question.get("evidence") or [],
                "predicted_answer": answer,
                "retrieved_contexts": retrieved_contexts,
            }
        )

        RAW_ANSWERS_PATH.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in raw_rows),
            encoding="utf-8",
        )

    print(
        f"\nGenerated {len(raw_rows)}/{total_questions} answers -> {RAW_ANSWERS_PATH}"
    )
    print("Run judge_answers() (or `--phase judge`) next to score them.")
    return raw_rows


def judge_answers(raw_answers_path: Path = RAW_ANSWERS_PATH) -> dict[str, Any]:
    """Phase 2: judge previously-generated answers and write the final summary.

    Reads raw_answers.jsonl (from generate_answers()) and retrieval_results.jsonl
    (written alongside it, no judge needed for that file) — does not touch the
    agent or Qdrant at all, only the judge model.
    """
    raw_rows = [
        json.loads(line)
        for line in raw_answers_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    retrieval_rows = [
        json.loads(line)
        for line in RETRIEVAL_RESULTS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    answer_rows: list[dict[str, Any]] = []
    for idx, raw in enumerate(raw_rows, start=1):
        print(f"[{idx}/{len(raw_rows)}] judging {raw['qa_id']}")
        question = {
            "question": raw["question"],
            "question_type": raw.get("question_type"),
            "gold_answer": raw.get("gold_answer"),
            "evidence": raw.get("evidence") or [],
        }
        answer_eval = evaluate_answer(
            question, raw["predicted_answer"], raw.get("retrieved_contexts") or []
        )
        answer_rows.append(
            {
                "qa_id": raw["qa_id"],
                "question_type": raw.get("question_type"),
                "question": raw["question"],
                "gold_answer": raw.get("gold_answer"),
                "gold_evidence": raw.get("evidence") or [],
                "predicted_answer": answer_eval.pop(
                    "clean_answer", raw["predicted_answer"]
                ),
                **answer_eval,
            }
        )
        # Pace requests to stay under the eval judge TPM limit.
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

    judge_model_name, _judge_base, _judge_key = _judge_config()
    summary = {
        "benchmark_date": time.strftime("%Y-%m-%d"),
        "answer_model": GENERATION_MODEL,
        "judge_model": judge_model_name,
        "document_count": len(
            json.loads((REPO_ROOT / "eval" / "document_manifest.json").read_text())
        ),
        "question_count": len(raw_rows),
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
        "question_breakdown": dict(Counter(r.get("question_type") for r in raw_rows)),
        "correctness_by_question_type": _accuracy_by_type(answer_rows),
        "answer_result_file": str(ANSWER_RESULTS_PATH),
        "retrieval_result_file": str(RETRIEVAL_RESULTS_PATH),
    }
    SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    FAILURES_PATH.write_text(_render_failures_md(answer_rows), encoding="utf-8")
    return summary


def run(
    category_filter: str | None = None,
    qa_files: list[str] | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Run the full benchmark end-to-end: generate every answer, then judge all of them.

    Prefer generate_answers() + judge_answers() separately when you want to
    inspect raw answers before spending judge-model calls on them.
    """
    generate_answers(category_filter=category_filter, qa_files=qa_files, resume=resume)
    return judge_answers()


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
    parser.add_argument(
        "--phase",
        choices=["generate", "judge", "full"],
        default="full",
        help=(
            "generate: run the agent and save raw answers only (no judge calls). "
            "judge: score an existing raw_answers.jsonl. "
            "full: both, in sequence (default)."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Skip qa_ids already present in raw_answers.jsonl instead of "
            "regenerating them -- re-run the same command after an "
            "interrupted long run to only pay for what's left."
        ),
    )
    args = parser.parse_args()

    if args.phase == "generate":
        generate_answers(
            category_filter=args.category, qa_files=args.qa_files, resume=args.resume
        )
    elif args.phase == "judge":
        print(json.dumps(judge_answers(), ensure_ascii=False, indent=2))
    else:
        print(
            json.dumps(
                run(
                    category_filter=args.category,
                    qa_files=args.qa_files,
                    resume=args.resume,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
