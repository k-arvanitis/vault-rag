# Vault RAG Case Study

## Problem

Most business document search systems fail when the corpus stops being clean: contracts sit next to spreadsheets, scanned invoices, public reports, FOIA packets, and tables with inconsistent formatting. The goal of Vault RAG is to answer questions over that kind of mixed collection while keeping the pipeline inspectable.

## Constraints

- Documents can be born-digital, scanned, tabular, or mixed within the same file.
- Retrieval must work across PDFs, Excel, CSV, Markdown, Word, and images.
- Operators need to inspect parsed content, retrieved chunks, and evaluation failures.
- The system should support privacy-conscious deployment: local parsing and embeddings, with only retrieved context sent to the generation model unless configured otherwise.

## Design

Vault RAG uses a staged document intelligence pipeline:

1. Per-file and per-page parsing routes born-digital PDF pages to text-layer extraction and scanned pages to OCR.
2. Structure-aware chunking preserves headings, tables, row context, and document summaries.
3. Dense and sparse vectors are stored together in Qdrant and fused with reciprocal rank fusion.
4. A reranker improves precision before generation.
5. A LangGraph ReAct agent can issue multiple searches for multi-document questions.
6. Streamlit exposes ingestion, chat, document inspection, retrieved chunks, and evaluation results.
7. Slack provides a query-only surface for end users.

## Evaluation

The benchmark contains 56 questions over 8 public documents:

- 32 single-document factoid questions
- 8 table lookup questions
- 10 cross-document comparison questions
- 6 unanswerable questions

Latest full run:

| Metric | Score |
|---|---:|
| Correctness | 96.4% |
| Faithfulness | 97.3% |
| Answer relevancy | 98.2% |
| Hit@10 | 98.0% |
| Evidence recall@20 | 98.0% |

DeepEval remains available as an ablation mode, but the primary benchmark uses a custom JSON-only LLM judge. DeepEval’s multi-step metrics were useful during development but unstable for this demo corpus: Qwen judges returned invalid JSON, while GPT-mini judges timed out or under-scored faithfulness when context was trimmed too aggressively.

## Debugging Lessons

The most important cross-document failure was not retrieval. The evidence was usually present, but the agent sometimes answered after only one tool call. The fix was to detect one-source answers to multi-part questions and force a decomposed second retrieval pass before final synthesis.

Other concrete fixes included:

- better snippet window selection for answer-dense passages
- query enrichment for qualifiers such as `new topic areas`, `closed-implemented`, and `since June 2024`
- deterministic repairs for common table/numeric synthesis slips
- an evaluation dashboard that shows gold answers, generated answers, scores, and retrieved evidence together

## Remaining Limits

- Low-quality OCR can still produce wrong values in scanned invoice packets.
- Complex table layouts remain parser-sensitive.
- The benchmark is intentionally small enough to inspect by hand; it is a portfolio benchmark, not a broad academic leaderboard.

## Why It Matters

The project demonstrates the work behind production RAG: parsing choices, metadata design, retrieval diagnostics, cross-document tool use, evaluation design, and operator workflows. The final metric is useful, but the more important result is that each failure can be traced to a specific stage of the pipeline.
