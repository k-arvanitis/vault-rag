# Vault RAG evaluation benchmark

This directory contains the portfolio-grade benchmark scaffold for evaluating Vault RAG on a small mixed-format business-document corpus.

## Goals

- Evaluate retrieval quality separately from answer quality
- Cover born-digital PDFs, scanned PDFs, spreadsheets, CSVs, and report PDFs with tables/figures
- Keep the corpus small enough to inspect manually but strong enough to demonstrate engineering judgment
- Freeze all benchmark inputs with stable `doc_id`s, source metadata, and SHA256 hashes

## Judge design

The primary answer evaluator is a compact custom LLM judge:

- exact-match short-circuiting for deterministic answers
- one JSON-only `gpt-4o-mini` call for non-exact answers
- retrieved-context selection around the answer and gold evidence terms
- deterministic retrieval metrics against gold evidence annotations
- **claim-level faithfulness** (RAGAS-style): a claim is supported if it can be *inferred* from the retrieved context, not only if it appears verbatim — so a cross-document conclusion is faithful when its component facts are present in the chunks; only contradictions, absent facts, or wrong-source mixing are penalised

DeepEval remains available with `EVAL_JUDGE_MODE=deepeval`, but it is not the default reported run. In practice, its multi-step metrics were brittle for this corpus: Qwen judges returned invalid JSON, and GPT-mini judges timed out or under-scored faithfulness when context was trimmed too aggressively.

Vault RAG also needs file-type-aware retrieval scoring:
- page-level retrieval on PDFs
- page-level OCR evidence checks on scanned PDFs
- row-level retrieval on XLSX / CSV inputs
- multi-document evidence coverage for cross-document questions
- abstention scoring for unsupported questions

The benchmark therefore combines custom retrieval/evidence metrics with a controlled answer-side judge.

## Recommended corpus shape

Target corpus size: **8 documents**

Current QA size: **70 questions** across 13 files in `data/qa_pairs/`
- Question types: `ocr_extraction`, `numeric_lookup`, `table_lookup`, `table_grounding`, `figure_grounding`, `cross_document_compare`, `unanswerable`

## Files in this directory

- `document_manifest.json` — frozen corpus metadata
- `data/qa_pairs/` — one JSON file per document (or document pair), loaded at eval time
- `run_eval.py` — benchmark runner
- `schema/manifest.schema.json` — manifest schema
- `schema/questions.schema.json` — QA schema
- `data/raw/` — downloaded public source documents
- `results/` — saved evaluation outputs

## Benchmark rules

1. Every document gets a stable `doc_id`
2. Every downloaded file records `source_url`, `accessed_at`, and `sha256`
3. Every QA item stores gold evidence, not just a gold answer
4. Cross-document questions must require evidence from 2+ documents
5. Unanswerable questions must be truly unsupported, not merely difficult

## Proposed metrics

### Retrieval metrics
- Hit@5
- Hit@10
- MRR
- Recall@k for multi-evidence questions

### Answer metrics
- Exact / normalized match for objective answers
- Grounded correctness for free-text answers
- Abstention accuracy for unsupported questions

## Latest result

Pending re-run after retrieval de-scoping (no `scope_doc_id`) and qa_pairs migration.
Previous scoped-retrieval numbers (for reference): Correctness 96.4%, Hit@10 98.0%.
