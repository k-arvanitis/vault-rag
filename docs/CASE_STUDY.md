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
6. A Next.js chat UI exposes ingestion, chat, document inspection, retrieved chunks, and evaluation results — talking to the agent over a FastAPI backend.
7. Slack provides a query-only surface for end users.

## Demo

| Chat with citations | Trace panel — retrieved vs. rejected |
|---|---|
| ![Chat UI](../assets/chat-ui.png) | ![Retrieved and rejected chunks](../assets/trace-rejected.png) |

| Document inspector | Evaluation dashboard |
|---|---|
| ![Document inspector](../assets/document-inspector.png) | ![Evaluation dashboard](../assets/eval-panel.png) |

## Evaluation

The benchmark now contains 109 questions over 18 real public documents (grown from 82/14 with the addition of an SOP manual, a lease + amendment package, and 4 targeted refusal-style questions) spanning ten question types: OCR extraction, table lookup, numeric lookup, numeric reasoning, figure grounding, table grounding, negation check, cross-document comparison, single-doc factoid, and unanswerable. These are the current, full numbers — see the [Evaluation section of the main README](../README.md#evaluation) for the complete breakdown and [Recent fixes](../README.md#recent-fixes) for what changed since the previous numbers.

**Agent answer metrics** (all 109 questions)

| Metric | Score |
|---|---:|
| Correctness | **81.9%** |
| Faithfulness | **79.5%** |
| Answer relevancy | **83.9%** |

**Vector retrieval** (74 PDF/OCR questions, Qdrant)

| Metric | Score |
|---|---:|
| Hit@5 | **95.9%** |
| Hit@10 | **97.3%** |
| MRR | **85.9%** |
| Evidence recall@10 | **93.5%** |

**Structured retrieval** (21 Excel/CSV questions, DuckDB)

| Metric | Score |
|---|---:|
| Answer accuracy | **76.2%** |

**Unanswerable questions** (14 questions)

| Metric | Score |
|---|---:|
| Correct refusal rate | **78.6%** |

The primary benchmark uses a custom JSON-only LLM judge (`gpt-4o-mini`, OpenAI). DeepEval remains available as an ablation mode but was removed from the primary path due to instability on this corpus. The judge itself was found and fixed this session — it had been scoring some correct, fully-grounded cross-document answers as 0% faithful; see [Recent fixes](../README.md#recent-fixes) for the reproduced case and the fix.

## Debugging Lessons

The most important cross-document failure was not retrieval. The evidence was usually present, but the agent sometimes answered after only one tool call. The fix was to detect one-source answers to multi-part questions and force a decomposed second retrieval pass before final synthesis.

Other concrete fixes included:

- better snippet window selection for answer-dense passages
- DuckDB routing restricted to flat-structure files only — adding multi-sheet workbooks bloated the system prompt and collapsed structured accuracy from 90% to 57%
- stream_agent normalization to ensure hedging phrases (`I cannot determine`) map to the canonical `Unsupported` token on both streaming and non-streaming paths
- deterministic repairs for common table/numeric synthesis slips
- an evaluation dashboard that shows gold answers, generated answers, scores, and retrieved evidence together

## Remaining Limits

- Multi-hop cross-document questions where evidence is split across 100+ page documents and the reranker doesn't surface both chunks together.
- Questions that require arithmetic the agent is explicitly instructed to refuse (by design).
- LLM non-determinism on a small number of numeric lookups where the correct row is retrieved but the wrong value is extracted.

## Why It Matters

The project demonstrates the work behind production RAG: parsing choices, metadata design, retrieval diagnostics, cross-document tool use, evaluation design, and operator workflows. The final metric is useful, but the more important result is that each failure can be traced to a specific stage of the pipeline.
