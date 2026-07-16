# Vault RAG — Evaluation Summary

*Generated from `eval/results/summary.json` by `eval/generate_summary_doc.py` --
do not hand-edit the numbers below; re-run the script after a new `make eval`.*

**Benchmark date:** 2026-07-16
**Answer model:** `openai/gpt-oss-120b`
**Judge model:** `gpt-4o-mini`
**Documents:** 18
**Questions:** 109

## Headline

| What it measures | Result |
|---|---|
| Overall answer correctness (10 question types) | **83.8%** |
| Grounded answers (faithfulness) | **86.1%** |
| Answers address the question (relevancy) | **86.9%** |
| Finds the right source (retrieval Hit@5) | **98.6%** |
| Refuses to invent answers (unanswerable questions) | **78.6%** |
| Structured data (Excel/CSV) answer accuracy | **76.2%** |

## Retrieval metrics (74 PDF/OCR questions, Qdrant dense search)

| Metric | Value |
|---|---|
| Hit@5 | 98.6% |
| Hit@10 | 98.6% |
| MRR | 85.4% |
| Evidence recall@10 | 94.4% |
| Evidence recall@20 | 96.8% |

## Structured data (21 Excel/CSV questions, DuckDB-served)

| Metric | Value |
|---|---|
| Answer accuracy | 76.2% |

## Refusal / abstention (14 unanswerable questions)

| Metric | Value |
|---|---|
| Correct refusal rate | 78.6% |

## Correctness by question type

| Question type | Count | Correctness |
|---|---|---|
| table_grounding | 3 | 100.0% |
| table_lookup | 16 | 93.8% |
| numeric_lookup | 6 | 91.7% |
| ocr_extraction | 25 | 86.0% |
| single_doc_factoid | 17 | 81.2% |
| negation_check | 5 | 80.0% |
| unanswerable | 10 | 80.0% |
| cross_document_compare | 20 | 77.5% |
| numeric_reasoning | 4 | 75.0% |
| figure_grounding | 3 | 66.7% |

## Judge breakdown

Scoring path used per question: {"custom_llm_judge": 77, "exact_match_shortcircuit": 26, "exact_match_fallback": 6}

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
