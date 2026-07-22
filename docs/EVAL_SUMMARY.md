# Vault RAG — Evaluation Summary

*Generated from `eval/results/summary.json` by `eval/generate_summary_doc.py` --
do not hand-edit the numbers below; re-run the script after a new `make eval`.*

**Benchmark date:** 2026-07-22
**Answer model:** `openai/gpt-oss-120b`
**Judge model:** `gpt-4o-mini`
**Documents:** 18
**Questions:** 109

## Headline

| What it measures | Result |
|---|---|
| Overall answer correctness (10 question types) | **90.6%** |
| Grounded answers (faithfulness) | **90.4%** |
| Answers address the question (relevancy) | **94.0%** |
| Finds the right source (retrieval Hit@5) | **98.6%** |
| Refuses to invent answers (unanswerable questions) | **92.9%** |
| Structured data (Excel/CSV) answer accuracy | **95.2%** |

## Retrieval metrics (74 PDF/OCR questions, Qdrant dense search)

| Metric | Value |
|---|---|
| Hit@5 | 98.6% |
| Hit@10 | 98.6% |
| MRR | 85.1% |
| Evidence recall@10 | 95.0% |
| Evidence recall@20 | 97.5% |

## Structured data (21 Excel/CSV questions, DuckDB-served)

| Metric | Value |
|---|---|
| Answer accuracy | 95.2% |

## Refusal / abstention (14 unanswerable questions)

| Metric | Value |
|---|---|
| Correct refusal rate | 92.9% |

## Correctness by question type

| Question type | Count | Correctness |
|---|---|---|
| figure_grounding | 3 | 100.0% |
| negation_check | 5 | 100.0% |
| numeric_reasoning | 4 | 100.0% |
| table_grounding | 3 | 100.0% |
| table_lookup | 16 | 100.0% |
| unanswerable | 10 | 90.0% |
| single_doc_factoid | 17 | 88.2% |
| ocr_extraction | 25 | 88.0% |
| cross_document_compare | 20 | 86.5% |
| numeric_lookup | 6 | 75.0% |

## Judge breakdown

Scoring path used per question: {"custom_llm_judge": 94, "exact_match_shortcircuit": 15}

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
