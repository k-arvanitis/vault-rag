# Backend investigation — 2026-07-22 (deep pass, full architecture)

Investigation-only session. No fixes implemented, nothing wired into the live pipeline.
Method: deterministic replay against `raw_answers_baseline_no_gate_20260722.jsonl` (109 rows),
direct DuckDB/Qdrant/chunk-file inspection, local reranker runs (CPU, free), and a small number
of targeted live `query_excel` replays (gpt-4o-mini, cents — no full eval run, no OpenRouter
generation calls). Every finding is labeled **CONFIRMED** (reproduced directly) or
**HYPOTHESIS** (consistent with evidence, not reproduced).

## Executive summary

The core unresolved problem — *the model answers from general knowledge and staples a real but
non-supporting citation on* — is **substantially more deterministic than previously believed**.
Two previously unknown backend bugs (F1, F2) jointly explain the flagship Finding-5 cases
(doc_015 handwashing b1, doc_001 qa_3/qa_4) as *hard retrieval exclusions*, not retrieval
nondeterminism: the answer-bearing chunk was **provably removed from the candidate pool before
the model ever saw it**. Fixing F1+F2 attacks Finding 5's dominant sub-mechanism (retrieval
miss → parametric fallback) at the root, which no grounding gate can do. Separately, a
fabrication point on the Excel path was caught red-handed via an instrumented live trace (F3),
and the eval harness's blind reflection override was re-confirmed as the source of three
"fabricated instead of refusing" rows (F4) — the production pipeline refuses those correctly.

---

## F1 (CONFIRMED — systemic ingest corruption): 133/862 chunks embedded with a literal rate-limit error string

**Evidence.** `contextualize_chunk` (`src/chunker.py:211`) returns `f"Error: {e}"` on any LLM
failure; `chunk_markdown` then unconditionally bakes that string into the embedded text:
`chunk.vector_text = f"CONTEXT: {context}\n\nCONTENT:..."` (`src/chunker.py:442`). Counted
across `data/output/chunks/`: **133 of 862 chunks (15.4%)** carry
`CONTEXT: Error: Error code: 429 - {'error': {'message': 'All models exhausted...'}}` as their
embedding prefix. Verified live in Qdrant (doc_001 chunk 4's stored `vector_text` starts with
the error string). Both dense (embedder embeds `vector_text`) and sparse
(`src/vector_store.py:100` embeds `vector_text`) vectors are polluted.

**Affected documents map exactly onto the failing questions:**
- `doc_001_procurement_policy` — **69 of 80 chunks** (qa_3, qa_4 failures)
- `doc_015_food_sop_manual` — 52 chunks, including chunk 13, the handwashing chunk (Finding 5's
  flagship case b1, and the Finding-3 misattribution doc)
- `doc_016a_original_lease`, `doc_016c_second_amendment` — 6 chunks each (lease QA flagged as
  possible false positives)

All 133 chunks share an identical ~150-char prefix, which pulls their dense vectors toward each
other and away from their content.

**Solutions:**
1. **(Recommended) Patch-in-place**: change `contextualize_chunk` to return `""` on failure (and
   `chunk_markdown` to skip the CONTEXT prefix when empty), re-enrich just the 133 chunks,
   re-embed them via Ollama, upsert into Qdrant by `_stable_id` — the exact procedure already
   proven on doc_008 (2026-07-06). Effort: small script + ~133 enrichment/embed calls. Risk:
   low; deterministic point IDs overwrite in place. Also add an ingest-time guard (reject a
   context line matching `^Error`) so this class can't silently recur.
2. Full re-ingest of the 4 documents. Cleaner, but re-runs OCR/VLM (GPU + Groq cost) for no
   extra benefit over (1).
3. Fix-forward only (guard, no re-embed). Cheapest, but leaves 15% of the corpus degraded —
   not acceptable given the affected docs are the failing ones.

**Priority: #1.** Cheap, deterministic, directly on the failure path, and it invalidates prior
conclusions about doc_015/doc_001 "retrieval quality".

## F2 (CONFIRMED — deterministic retrieval exclusion): `filter_token` fires on prose queries and hard-excludes answer chunks

**Evidence.** `_resolve_scope` (`src/tools/retrieval_tool.py:558-568`) gives *every* query a
`filter_token` via `_extract_table_filter_token` — a mechanism designed for table row lookups —
whenever the query has any ≥5-char candidate word. The token becomes a **hard Qdrant content
must-match** (`src/retriever.py:122`), silently removing every chunk that doesn't contain it
verbatim. Reproduced end-to-end:

- Query *"Who is the authorizing manager listed in the LACERA procurement policy?"* →
  `filter_token="LACERA"`. The answer chunk (doc_001 chunk 4: "Authorizing Manager: Ricki
  Contreras") does **not** contain "LACERA" — its logo text was OCR'd as **"L/CERA"**. With the
  token: chunk 4 absent from all 37 hits. Without it: chunk 4 is in the pool (rank 22/79) and
  the BGE reranker ranks it **#3** — it would have been shown to the model. Same story for
  qa_4 (both "Original Issue Date" chunks, 4 and 78, lack the literal token "LACERA").
- The agent's actual b1 query *"food safety SOP"* → `filter_token="safety"`; doc_015 chunk 13
  (handwashing, "10 to 15 seconds") contains no "safety" → deterministically excluded. **This
  is the mechanism behind Finding 5's flagship case**, previously attributed to "query too
  generic" nondeterminism.
- Other derivations confirmed: "washing", "Bensenville", "services", "Airport" — essentially
  every prose question is one-word hard-filtered.

**Why eval never saw it:** `evaluate_retrieval` (`eval/run_eval.py:604`) calls `retrieve()`
directly with **no filter_token and no reranker** — hit@5 = 0.986 measures a retrieval path the
agent does not use. The metric is structurally blind to this bug.

**Why it usually works anyway:** the answer chunk usually *does* contain the distinctive entity
word. It fails precisely when the answer sentence doesn't repeat the entity token (vocab
mismatch — the known-hard class in `project_retrieval_quality_issues`), and then it fails
deterministically, producing the exact Finding-5 signature: a topical-but-answerless pool the
model answers over anyway, citing its top chunk.

**Solutions:**
1. **(Recommended) Union retrieval**: run the filtered and unfiltered searches and merge
   (filtered hits first — preserves the exact-match boost for table lookups), letting the
   reranker arbitrate. Recall can only go up; cost is one extra Qdrant query (~ms). Effort:
   ~10 lines in `_fetch_docs`/`retrieve`. Risk: slightly noisier pools for genuine ID lookups;
   the reranker + existing scroll-boost keep those on top.
2. Scope the token: only set `filter_token` when the query looks like a table/ID lookup
   (numeric ID ≥5 digits, or sheet-routing vocabulary). Smaller diff, but the boundary
   ("which queries are table-ish") is a heuristic that will misfire both ways.
3. Fall back automatically: if the filtered search's reranked top-N is weak, retry unfiltered.
   Adds a score-threshold knob F10 shows is unreliable — avoid.

**Priority: #2** (tied to #1 — both must land before re-measuring Finding 5's frequency).
Also fix the eval blind spot: add an agent-path retrieval metric (hit@N over `_fetch_docs`
output) so this can't regress invisibly.

## F3 (CONFIRMED via instrumented live trace): Excel FORMAT step fabricates values not present in its input

**Evidence.** Instrumented `_llm_chat` and re-ran
*"For Supplier Name 'Screwfix Direct' on 2025-04-03 in PLACE / STREET SCENE what is the
Purchase of Expenditure and NET Amount?"* (the real defect inside
`doc_006_doc_007_cross_document_qa__qa_2`). In the failing run the SQL was **correct**, the
result was a single clean row `Purchase of Expenditure: MATERIALS`, and the FORMAT call
(gpt-4o-mini, `FORMAT_PROMPT`) output **"150.00"** — a number that exists elsewhere in the
table but nowhere in the FORMAT input. Reproduction rate ≈ 1/3 across 7 full-graph runs
(2×"150.00", 5×"MATERIALS"); the isolated inner graph also showed FORMAT punting "Unsupported"
on the same clean input once (recovered only by the `single_col_first_value` fallback).

This is a fabrication point *downstream of every gate* — `_column_matches_question`,
SQL execution, and the ambiguity guard were all correct.

**Solutions:**
1. **(Recommended) Extractive verification**: the FORMAT step is copy-only by contract (rule 5).
   After formatting, require every output value to appear in the SQL result text (normalize
   whitespace/case; treat numerics as equal if float-equal, so `150.0` vs `150.00` compares
   correctly). On failure: use `single_col_first_value` when available, else one re-ask, else
   Unsupported. Pure Python, no added latency in the good case, fully unit-testable offline
   against the exact reproduced transcript. Effort: small. Risk: low — multi-field
   `Field=value; Field=value` outputs need value-wise splitting, covered by tests.
2. Replace the FORMAT LLM with deterministic extraction for single-column/single-row results
   (the majority case) and keep the LLM only for multi-column phrasing. Even safer for the
   common case; slightly more code.
3. Prompt tightening only — not recommended; this project has 5+ logged "prompt-only fix
   verified not working" precedents, and the failing call already had explicit rules.

Cosmetic but worth fixing alongside: `synthesize_node`'s label heuristic
(`sq.split(' in ')[-1]`, `src/tools/excel.py:872`) produces garbage prefixes like
"PLACE / STREET SCENE what is the Purchase of Expenditure: …" on every multi-part answer.

**Priority: #3.** Fabrication-class, deterministic guard available, cheap to validate.

## F4 (CONFIRMED by replay): the eval harness's blind reflection override — not the product pipeline — produced three of the "fabricated instead of refusing" rows

**Evidence.** Replayed all three through the real production path (`query_excel` directly,
against a copy of the live DuckDB):

| qa_id | baseline run recorded | production path today |
|---|---|---|
| `doc_006..._qa_9` (gold Unsupported) | "Invoice Number" | **Unsupported** (gate rejected all SQL, `SQL: []`) |
| `doc_014..._qa_2` (gold Unsupported) | "None" | **Unsupported** (same) |
| `unanswerable_qa__qa_5` (gold Unsupported) | "doc_005_fueling_records_invoice" | (see F5) |

The recorded wrong answers match the already-documented mechanism (`TODO.md`, 2026-07-21): when
`answer_query` correctly returns Unsupported, `eval/run_eval.py:969-976` re-runs the question
through `ask_with_reflection` → `ask_agent` — which appends *"Retry with broader search — do
not restrict to exact text matches"* (`src/pipeline.py:98-101`) and **blindly accepts** any
non-Unsupported output, bypassing `answer_query`'s malformed-generation/no-sources/citation
guards entirely. The recorded row keeps the *original* run's contexts (or none), so the jsonl
is also misleading about what evidence the recorded answer saw.

**This corrupts the refusal metric downward and hides the fact that refusal behavior in
production is currently good.** The standing TODO items remain correct:
1. **(Recommended)** Controlled ablation (override on/off, same slice, compare correctness AND
   refusal), then remove or re-route through `answer_query` so it inherits the guards.
2. Minimum interim step (no behavior change): record `"override_fired": true` + the override's
   own contexts in the raw row, so future analysis can separate pipeline failures from
   harness artifacts. Trivial effort, zero risk.

**Priority: #4** — it's measurement corruption, and it taints every conclusion drawn from
refusal rows until resolved.

## F5 (CONFIRMED offline): `_is_bare_filename_answer` misses extension-less filename stems

`unanswerable_qa__qa_5` answered the bare stem `doc_005_fueling_records_invoice`.
`_BARE_FILENAME_RE` (`src/answer_quality.py:176`) requires a file extension, so the guard
returns False for the stem (verified directly: stem → not detected; same string + ".pdf" →
detected). Whichever path emitted it (main or override — indistinguishable from the row), the
last-line guard that exists precisely for this case has a one-regex gap.

**Fix:** extend the pattern with an extension-less alternative anchored to the `doc_\d+_`
prefix, e.g. `\bdoc_\d+(?:_[a-z0-9-]+)+\b` OR'd into `_BARE_FILENAME_RE`, + 2 unit tests.
Effort: trivial. Risk: near zero (the stem shape can't appear in legitimate answer values —
the ABSTENTION_BLOCK already declares filenames non-answers). **Priority: #5** (cheapest real
fix in this report).

## F6 (CONFIRMED not-reproducible): doc_006 qa_6 false refusal is SQL-generation nondeterminism

The row exists (verified by direct DuckDB query: exactly one row, `STAFF TRAVEL EXPENSES`), and
`query_excel` answers it correctly today with clean first-shot SQL. The baseline run's
Unsupported was a one-off gpt-4o-mini SQL/decompose variance. No code defect found; do not
patch. (Same verdict class as the prior session's "mechanism B refuted" trace.) If it recurs
in a future run, trace live at that time.

## F7 (CONFIRMED — ingestion data quality, n=1): doc_004 invoice number 1↔I confusion

The question's date qualifier ("September 1, 2023 invoice") points at a VLM figure description
that reads *"The invoice number is I30114, dated September 1, 2023"* — the VLM misread the
scanned invoice image (1→I). The correct "130114" also exists in the same chunk (cover-letter
text). The model obeyed VERBATIM VALUES and copied the corrupted passage — the pipeline worked;
the data is wrong. The PDF router itself routed correctly (`pymupdf4llm` label; image → VLM).

**Options:** (a) targeted VLM prompt line for digit fidelity in IDs (cheap, unmeasurable
benefit at n=1); (b) OCR post-correction heuristics (I/l→1, O→0 in digit contexts) — risky,
can corrupt legitimate text; (c) accept and document. **Recommend (c)** unless more OCR-digit
failures accumulate. Priority: low.

## F8 (eval artifact): doc_007 qa_2 date-format mismatch is a scoring false positive

`2025-04-29` vs gold `29/04/2025` — same date. `_normalize_dates` converts all ingested dates
to ISO at ingest time, so the DuckDB path *cannot* reproduce the source's original format; the
VERBATIM VALUES prompt rule is unsatisfiable for SQL-answered dates. Fix belongs in the judge
(date-equivalence normalization in the exact-match shortcircuit), not the pipeline.

## F9 (false positives, verified grounded): heuristic-scan noise

- `doc_003_doc_008_cross_document_qa__qa_2`: both gold facts ("sound and resilient", "$667.5
  billion") verified present in the retrieved contexts; the answer attributes them to the
  correct documents by title. Correct and grounded; wording-mismatch scan noise.
- `doc_006_doc_007` qa_1/qa_3/qa_4: predicted values match gold exactly (5239.0 / RPS BUSINESS
  HEALTHCARE / Retail / 209.32 / GENERAL OFFICE EXPENSES / 289.46 / 3.45 / GLOBAL GARDEN /
  500.00). Only qa_2 contained a real defect (F3).

## F10 (NEGATIVE result, verified offline): a weak-retrieval score gate is not viable

Computed max rerank score per baseline row: gold-matching rows range down to **−7.3** while
doc_001 qa_3's *poisoned* pool peaks at **+7.8** (the reranker is confidently wrong when the
true chunk is excluded). At any threshold τ∈{0,1,2}, the gate flags as many correct rows as
wrong ones (e.g. τ=1: 5/44 correct vs 8/32 non-matching). This kills the "refuse when all
retrieved scores are weak" idea from the 2026-07-12 RFQ TODO item as a general mechanism.
Reason: score magnitude measures match-to-pool, not pool-contains-answer.

---

## Architecture-area notes (per the investigation brief)

1. **PDF router** (`src/parser/pdf_parser.py`): logic read in full; routing is per-page,
   threshold 50 chars, force-overrides correct. doc_001/doc_004 both correctly labeled
   `pymupdf4llm`. No misrouting found. Real ingestion-quality issues found are F1 (enrichment)
   and F7 (VLM digit misread), not router bugs. *Code-read + spot-verified, no systematic
   per-page audit run.*
2. **Chunker** (`src/chunker.py`): `_has_section_header` guard present and correct (per the
   standing feedback memory). The known title/short-chunk weakness is already mitigated by
   `_title_shortcut_answer`. The new, real chunker-adjacent finding is F1.
3. **Retrieval**: F2 (filter_token) is the headline. Additional (code-read, minor): when
   `scope_doc_id` is set, `retrieve()` discards the primary search and re-runs
   `_search_with_scope` **three times with identical arguments** — `scope_doc_key` is dead
   (`_metadata_filter` ignores it, docstring admits it) — 3 redundant Qdrant round-trips per
   scoped call. Pure waste, not a correctness bug; one-line cleanup when convenient.
4. **Agent orchestration — the "second targeted query" question**: characterized from code +
   replay. Nothing in the loop *forces* a follow-up query except: bare-Unsupported retry,
   comparison-incompleteness retry, and multi-part split. A first query that returns a
   topical-but-answerless pool (exactly what F1/F2 produce) triggers **no** retry — the model
   answers over the bad pool, and the ANSWERING/CROSS-DOCUMENT prompt rules actively push it
   to produce a value. b2's pass/fail flip (prior session) is the model *happening* to issue a
   refinement. **HYPOTHESIS (consistent, not re-traced live): fixing F1+F2 removes most of the
   pressure; the residual "model doesn't refine" variance is real but secondary.** A
   forced-refinement mechanism (e.g. always issue a second, entity-specific query when the
   first call's pool lacks any sentence-level term overlap with the question focus) is
   possible but should wait for post-F1/F2 re-measurement.
5. **Excel/DuckDB**: F3 (FORMAT fabrication), F6 (nondeterminism), F4 (harness, not pipeline).
   Gates (`_column_matches_question`, readonly-SQL, ambiguity Clarify) all verified working in
   live replays. `_relax_by_dropping_predicate` was *suspected* for wrong-row answers but the
   instrumented trace exonerated it here — the wrong value came from FORMAT.
6. **Citation/grounding pipeline**: both prior gate attempts remain correctly reverted. The
   viable middle ground, in order: (a) land F1+F2 and re-measure Finding 5's frequency — the
   confirmed cases were retrieval exclusions, which no gate can fix and which gates would
   merely mask; (b) if residual ungrounded-answer cases remain, the entailment judge with a
   mandatory quote-span ("quote the exact supporting sentence before YES/NO") is the only
   direction with evidence behind it, and it must be validated on the existing 109-row replay
   harness first (the reusable asset from 2026-07-22). F10 removes the score-threshold option
   from the table. The `parse_sources` 8-cap vs 12-shown marker loss (b3 note) is still open
   but low-severity.
7. **Prompts** (`src/prompts.py`): ABSTENTION_BLOCK is confirmed violated under retrieval
   pressure (b1 — parametric answer despite the explicit rule), consistent with the project's
   long record of prompt-only fixes failing. No prompt changes recommended; the leverage is in
   retrieval (F1/F2) and code-level guards (F3/F5).

## Priority ranking

| # | Finding | Class | Effort | Risk | Status |
|---|---|---|---|---|---|
| 1 | F1 — 429-polluted embeddings (133 chunks) | corpus corruption | S (patch-in-place script) | low | CONFIRMED |
| 2 | F2 — filter_token prose exclusion (+ eval blind spot) | retrieval correctness | S–M | low-med | CONFIRMED |
| 3 | F3 — Excel FORMAT fabrication guard | fabrication | S | low | CONFIRMED |
| 4 | F4 — reflection override ablation/removal | eval integrity | M (needs ablation run) | med | CONFIRMED mechanism |
| 5 | F5 — extension-less bare-filename regex | guard gap | XS | ~0 | CONFIRMED |
| 6 | F8 — judge date-format normalization | eval scoring | XS | ~0 | CONFIRMED |
| 7 | Entailment gate v2 (quote-span), offline-validated | grounding | M | med | direction only — after 1+2 re-measure |
| 8 | F7 — OCR digit confusion | data quality | — | — | accept/document |

**Sequencing note:** 1→2 first, then a full eval re-run (budget-check with user) to re-baseline
Finding 5's frequency before deciding on item 7. Items 3/5/6 are independent and can ship with
unit tests alone. Item 4's ablation shares the same eval run as the re-baseline — do them
together to save one full run.
