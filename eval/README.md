# Vault RAG evaluation benchmark

This directory contains the portfolio-grade benchmark scaffold for evaluating Vault RAG on a small mixed-format business-document corpus.

## Goals

- Evaluate retrieval quality separately from answer quality
- Cover born-digital PDFs, scanned PDFs, spreadsheets, CSVs, and report PDFs with tables/figures
- Keep the corpus small enough to inspect manually but strong enough to demonstrate engineering judgment
- Freeze all benchmark inputs with stable `doc_id`s, source metadata, and SHA256 hashes

## Judge design

The primary answer evaluator is a compact custom LLM judge, configured (via `EVAL_JUDGE_MODEL` in `.env`) to run on `gpt-oss-120b` — distinct from the `qwen/qwen3-32b` answer model, so there's no self-grading bias. (The code's built-in default when no judge model/key is configured is `gpt-4o-mini` via the OpenAI API — see `_judge_config()` in `run_eval.py` — but the published numbers use the `gpt-oss-120b` override.)

- exact-match short-circuiting for deterministic answers
- one JSON-only judge call for non-exact answers
- retrieved-context selection around the answer and gold evidence terms
- deterministic retrieval metrics against gold evidence annotations
- **claim-level faithfulness** (RAGAS-style): a claim is supported if it can be *inferred* from the retrieved context, not only if it appears verbatim — so a cross-document conclusion is faithful when its component facts are present in the chunks; only contradictions, absent facts, or wrong-source mixing are penalised

DeepEval remains available with `EVAL_JUDGE_MODE=deepeval`, but it is not the default reported run. In practice, its multi-step metrics were brittle for this corpus: Qwen judges returned invalid JSON, and GPT-mini judges timed out or under-scored faithfulness when context was trimmed too aggressively.

### Judge prompt

The exact prompt sent to the judge model for every non-exact-match answer (`_custom_judge_answer()` in `run_eval.py`):

```
You are grading a RAG system. Return ONLY valid JSON with numeric scores from 0 to 1.
Use this schema exactly: {"correctness": number, "faithfulness": number, "answer_relevancy": number, "reason": string}

Scoring rules:
- correctness: compare ACTUAL ANSWER to EXPECTED ANSWER for the facts requested. Accept
  paraphrases, concise answers, source labels, currency symbols, and formatting differences.
  If all requested values/facts are present, score 1.0 even if wording differs.
- faithfulness: judge at the CLAIM level (RAGAS-style) — a claim is supported if it can be
  inferred from the RETRIEVED CONTEXT, not only if it appears verbatim. A comparison, ranking,
  or conclusion that follows from facts that ARE present in the context IS supported — score
  it faithful. Do not require exact wording; values under matching field labels count as
  support. Do not penalize missing citations. Penalize only claims that CONTRADICT the
  context, introduce facts ABSENT from the context, or mix values across the wrong sources.
- answer_relevancy: score whether ACTUAL ANSWER directly addresses the QUESTION. Concise
  direct answers are relevant.

QUESTION / EXPECTED ANSWER / ACTUAL ANSWER / RETRIEVED CONTEXT are then interpolated in.
System message: "You are a strict but fair evaluation judge. Output valid JSON only."
```

Exact string matches (`normalized_answer == gold_answer`) short-circuit this call entirely and score 1.0 without an LLM round-trip — see `judge_used: "exact_match_shortcircuit"` in `answer_results.jsonl`.

Vault RAG also needs file-type-aware retrieval scoring:
- page-level retrieval on PDFs
- page-level OCR evidence checks on scanned PDFs
- row-level retrieval on XLSX / CSV inputs
- multi-document evidence coverage for cross-document questions
- abstention scoring for unsupported questions

The benchmark therefore combines custom retrieval/evidence metrics with a controlled answer-side judge.

## Corpus shape

Corpus size: **18 documents** (full list + sources in the main [README](../README.md#benchmark-corpus))

Current QA size: **93 questions** across the files in `data/qa_pairs/`
- Question types: `single_doc_factoid`, `ocr_extraction`, `table_lookup`, `numeric_lookup`, `figure_grounding`, `table_grounding`, `negation_check`, `cross_document_compare`, `unanswerable`
- `eval/data/qa_pairs_unused/` holds 133 auto-generated questions from a parked scale-up experiment — intentionally excluded from `data/qa_pairs/` so they don't get silently swept into a run (see that directory's README before moving anything back)

## Files in this directory

- `document_manifest.json` — frozen corpus metadata
- `data/qa_pairs/` — one JSON file per document (or document pair), loaded at eval time — the actual benchmark questions + gold answers
- `run_eval.py` — benchmark runner
- `schema/manifest.schema.json` — manifest schema
- `schema/questions.schema.json` — QA schema
- `data/raw/` — downloaded public source documents
- `results/` — saved evaluation outputs, all regenerated by `make eval`:
  - `results/summary.json` — aggregate metrics (Hit@K, MRR, correctness, faithfulness, refusal rate) plus `correctness_by_question_type` — count and mean correctness per question type, so the headline number is never just one blended figure
  - `results/answer_results.jsonl` — every question's predicted answer, gold answer, gold evidence (`doc_id` + quote), question type, and judge verdict — nothing is summarized away
  - `results/retrieval_results.jsonl` — per-question retrieval hits and evidence recall
  - `results/failures.md` — every question that scored below 0.5 correctness, with a heuristic failure category (cross-document reasoning, numeric extraction, OCR, etc.), gold vs. predicted answer — auto-generated each run, not hand-maintained

## Reproducing the benchmark

```bash
make seed        # optional — ingest a starter subset if the collection is empty
make eval        # full 93-question run → writes eval/results/summary.json
make eval-cross  # cross-document questions only
```

Or via the API: `POST /eval/run` kicks off the same full run in the background (see [API endpoints](../README.md#api-endpoints)); poll `GET /eval/status/{job_id}`, then read `GET /eval/summary`.

## Benchmark rules

1. Every document gets a stable `doc_id`
2. Every downloaded file records `source_url`, `accessed_at`, and `sha256`
3. Every QA item stores gold evidence, not just a gold answer
4. Cross-document questions must require evidence from 2+ documents
5. Unanswerable questions must be truly unsupported, not merely difficult

## Metrics computed

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

See the [Evaluation](../README.md#evaluation) section of the main README for the current headline numbers and the full metric breakdown by modality. Raw per-question results are in `results/answer_results.jsonl` and `results/retrieval_results.jsonl`.

Known issue (tracked in `TODO.md`): `results/summary.json` is a stale run and does not match the numbers published in the main README — re-run `make eval` before treating it as current.
