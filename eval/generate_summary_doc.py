"""Regenerate docs/EVAL_SUMMARY.md from eval/results/summary.json.

summary.json is the one canonical, machine-readable eval result -- this
script renders its numbers into the portfolio-facing markdown doc instead of
anyone hand-copying figures, which is exactly how the doc and the real
numbers drifted apart before (see PROGRESS.md's 2026-07-16 eval entry).

Usage:
    uv run python eval/generate_summary_doc.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = REPO_ROOT / "eval" / "results" / "summary.json"
OUT_PATH = REPO_ROOT / "docs" / "EVAL_SUMMARY.md"


def _pct(x: float | None) -> str:
    return f"{x * 100:.1f}%" if x is not None else "n/a"


def render(summary: dict) -> str:
    vec = summary["vector_retrieval_metrics"]
    struct = summary["structured_retrieval_metrics"]
    unans = summary["unanswerable_metrics"]
    agent = summary["agent_answer_metrics"]
    by_type = summary["correctness_by_question_type"]

    rows = "\n".join(
        f"| {qtype} | {v['count']} | {_pct(v['correctness'])} |"
        for qtype, v in sorted(by_type.items(), key=lambda kv: kv[1]["correctness"], reverse=True)
    )

    return f"""# Vault RAG — Evaluation Summary

*Generated from `eval/results/summary.json` by `eval/generate_summary_doc.py` --
do not hand-edit the numbers below; re-run the script after a new `make eval`.*

**Benchmark date:** {summary.get("benchmark_date", "unknown")}
**Answer model:** `{summary.get("answer_model", "unknown")}`
**Judge model:** `{summary.get("judge_model", "unknown")}`
**Documents:** {summary.get("document_count", "unknown")}
**Questions:** {summary["question_count"]}

## Headline

| What it measures | Result |
|---|---|
| Overall answer correctness (10 question types) | **{_pct(agent["correctness"])}** |
| Grounded answers (faithfulness) | **{_pct(agent["faithfulness"])}** |
| Answers address the question (relevancy) | **{_pct(agent["answer_relevancy"])}** |
| Finds the right source (retrieval Hit@5) | **{_pct(vec["hit_at_5"])}** |
| Refuses to invent answers (unanswerable questions) | **{_pct(unans["correct_refusal_rate"])}** |
| Structured data (Excel/CSV) answer accuracy | **{_pct(struct["answer_accuracy"])}** |

## Retrieval metrics ({vec["question_count"]} PDF/OCR questions, Qdrant dense search)

| Metric | Value |
|---|---|
| Hit@5 | {_pct(vec["hit_at_5"])} |
| Hit@10 | {_pct(vec["hit_at_10"])} |
| MRR | {_pct(vec["mrr"])} |
| Evidence recall@10 | {_pct(vec["evidence_recall_at_10"])} |
| Evidence recall@20 | {_pct(vec["evidence_recall_at_20"])} |

## Structured data ({struct["question_count"]} Excel/CSV questions, DuckDB-served)

| Metric | Value |
|---|---|
| Answer accuracy | {_pct(struct["answer_accuracy"])} |

## Refusal / abstention ({unans["question_count"]} unanswerable questions)

| Metric | Value |
|---|---|
| Correct refusal rate | {_pct(unans["correct_refusal_rate"])} |

## Correctness by question type

| Question type | Count | Correctness |
|---|---|---|
{rows}

## Judge breakdown

Scoring path used per question: {json.dumps(summary["judge_breakdown"])}

## Known gap

**Multi-document evidence coverage** is not yet a formal benchmark metric in
this run -- this run predates the deterministic comparison path
(`answer_comparison_deterministic` in `src/answer_pipeline.py`). A manual
spot-check against the live API (5 repeated runs of a real two-document
comparison question) showed 5/5 returning evidence from both requested
documents; this is not a substitute for a real benchmark number and is not
reported as one here. Measuring it formally requires a fresh `make eval`
run with cross_document_compare questions specifically checked for
per-document source coverage, not just answer correctness.

## Regenerating

```bash
uv run python eval/run_eval.py       # full run: generate + judge (real LLM calls)
uv run python eval/generate_summary_doc.py   # re-render this doc from summary.json
```
"""


def main() -> None:
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    OUT_PATH.write_text(render(summary), encoding="utf-8")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
