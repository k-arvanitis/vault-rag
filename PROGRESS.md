# vault-rag — Progress & Plan

Single source of truth. Update this at the start of every session.
Last updated: 2026-07-08

---

## Session 2026-07-08, part 2 — deep-dive on why gains were small; honest answer

User pushback after the full-run report: +2.3 correctness / +0.9 faithfulness is a small
gain for the time spent — asked whether the model is the bottleneck, whether the eval set
is representative, why refusal isn't 100%, why structured accuracy sits at 76%. Investigated
every non-1.0 answer in the four weakest question types by hand (zero new OpenRouter spend,
used data already collected). Honest findings:

**The model is not the bottleneck.** Every failure traced to a root cause landed on: broken
eval ground truth, judge-scoring noise, or a specific retrieval/indexing gap — never bad
reasoning from Qwen3-32B. In two structured-bucket cases the agent's "which transaction —
give me a date or number" response was the *correct* behavior for a genuinely ambiguous
question and got marked wrong for it.

**Structured/Excel 76.2% (n=21, 5 wrong) — mostly broken eval labels, verified against the
raw spreadsheet directly:**
- `doc_006_...qa_1` — "Google Ads2372193163" matches **7 rows** with different NET amounts;
  question gives no disambiguating date/transaction-number. Gold picked row 2 arbitrarily.
- `doc_006_...qa_3` — "Asda Groceries Online" matches **363 rows**. Same problem.
- `doc_006_...qa_6` — gold's own evidence quote is for supplier "687 - Wilmington", not
  "Trainline" as the question asks. The gold label doesn't match its own question.
- `doc_006_doc_007_...qa_5` — judge-scoring noise: identical predicted text to a prior run,
  scored 1.0 then 0.5. Not a real difference.
- `doc_006_doc_007_...qa_2` — the one genuine pipeline failure in this bucket (real
  "Unsupported" when doc_006 data likely exists) — not investigated further this session.

True ceiling here, net of bad labels and noise, is closer to ~90-95%, not 76%.

**Real bug found and NOT yet fixed — refusal isn't 100% because of a document-scope
violation.** `doc_015_food_sop_manual_qa__qa_5` asks "the SOP manual's policy on vacation"
(doc_015 = a food-safety SOP, correctly has zero vacation content, gold=Unsupported).
Retrieval correctly found nothing in doc_015 — but instead of refusing, the agent answered
using doc_010's (an unrelated HR handbook) real vacation-policy content and labeled it
"The SOP manual outlines...". Confident cross-document misattribution, not a hallucination
from model knowledge. No check currently verifies that an answer's cited source actually
matches the document a question names. **Not fixed this session** — flagged as higher-risk
than the front-matter fix below (harder to reliably detect "which doc did the question
name"); do only if the front-matter fix pattern is cheap to extend.

**doc_006 eval labels — fixed (2026-07-08, later same session).** Per user request, fixed
rather than removed the 3 broken structured-bucket questions found earlier:
- `qa_1`: added `on 2025-04-01` to disambiguate "Google Ads2372193163" (was 7 matching rows).
- `qa_3`: added date + department to disambiguate "Asda Groceries Online" (was 363 rows).
- `qa_6`: was completely broken — gold evidence was for supplier "687 - Wilmington", not
  "Trainline" as the question asked. Replaced with a real Trainline row (verified unique via
  date+department), fixed `row_ref` (29 → 28) and the evidence quote to match.
Not re-run against the full corpus (per user: no more full-eval reruns needed right now) —
these are label corrections, verifiable independently whenever the corpus is next scored.

**doc_015 refusal bug — fix attempted, verified NOT working (2026-07-08, later same
session).** Strengthened the existing `DOCUMENT IDENTITY CHECK` rule in `ABSTENTION_BLOCK`
(`src/prompts.py`) to explicitly forbid answering with another document's real content while
describing it as the named document's, and to require outputting `Unsupported` if a second
search still finds nothing in the named document. Verified scoped against the full
`doc_015_food_sop_manual_qa.json` (13 questions, no regressions on the other 12) —
**qa_5 still fails identically**, word-for-word the same wrong answer as before the prompt
change. Confirms the pattern from every other fix attempt this session (C2/A/B, the
front-matter fix): **prompt-only rules do not reliably override retrieval bringing back
wrong-document content.** The Excel hallucination bug earlier in the project only got fixed
with a programmatic hard gate (`_column_matches_question` in `src/tools/excel.py`), not
prompt text alone — this bug needs the same category of fix: a code-level check comparing a
retrieved chunk's `file=`/doc_id against the question's named document before allowing an
answer to use it, not another prompt instruction. **Not implemented this session** — three
consecutive prompt-only attempts failing identically is a strong enough signal to stop
trying that approach here; a real fix is a small code change in the retrieval/answering
path, tracked as a TODO, not a further prompt patch.

**Real bug found, root-caused, fix attempted, fix verified NOT working —
`doc_001`'s entire front-matter (title, authorizing manager, issue date, review date) is
invisible to retrieval, costing 4 of 5 questions on that document.** Root cause (confirmed
directly against Qdrant, not guessed): the chunks (index 2 and 4) **are** correctly indexed
— this is not a missing-embedding bug. Their auto-generated `CHUNK_CONTEXT_PROMPT` context
line was dominated by describing a decorative LACERA logo image (from the VLM figure
description) instead of naming the actual facts sitting right next to it in the same chunk
("Authorizing Manager: Ricki Contreras", "Original Issue Date: December 15, 2005"). Added a
rule to `CHUNK_CONTEXT_PROMPT` (`src/prompts.py`) instructing it to name textual facts over
describing decorative images when both appear in one chunk. Regenerated context for chunks
2/4, re-embedded, re-upserted in place (same pattern as the C2/A2 patches). **Verified
scoped against all 5 doc_001 questions — no change**: qa_1/3/4/5 still fail identically,
chunk 4 still never enters the retrieval candidate pool at all. The context-text rewrite
improved wording but didn't move dense-embedding similarity enough to matter. **The real fix
needs to be retrieval-side** (e.g. forced-include of a document's first N chunks when a
question asks about the document itself/its metadata — already flagged as a TODO item from
2026-07-03, now with much stronger evidence: costs 4 full questions on one document, not
just "title is hard"). Not attempted this session — scope/risk tradeoff, same caution as
A2's rollout.

**Reframing the session's own headline honestly, per the same self-correction:** the
decompose-off default (this session's main lever) was a *non-lever* — decomposition was
never in production and was already effectively off via the API-key bug, so disabling it
just confirmed the existing baseline behavior. The +2.3 correctness / +0.9 faithfulness came
from *last* session's C1/Excel/judge fixes finally landing in a clean measured run, not from
anything new done today. The front-matter retrieval gap found just now is plausibly a bigger
single lever (+~4 points, 4 questions on one document alone) than everything else touched
this session — but it needs a retrieval-architecture fix, not a prompt patch, and is not
done.

---

## Session 2026-07-08 — decomposition was eval-only, silently broken, and net-harmful

**Headline finding: the decomposition pipeline (`build_decomposition_pipeline` in
`src/pipeline.py`) has never been used by the live app.** `api.py` (and by extension
the Slack bot) only ever calls `build_rag_agent()` + `stream_agent()` — the plain
ReAct agent. Decomposition only ever existed inside `eval/run_eval.py`, gated on
`question_type.startswith("cross")`.

**Real bug found and fixed**: `build_decomposition_pipeline()` hardcoded
`api_key=GROQ_API_KEY` for its OpenAI client regardless of `generation_api_base`.
With `GENERATION_API_BASE` pointed at OpenRouter (current config), every decompose
call 401'd — silently, because it sat inside a bare `except Exception: sub_questions
= [question]`. This has been true ever since the OpenRouter switch, meaning every
previously-published eval number (79.6%/78.6% etc.) was **already measuring
decomposition silently falling back to single-hop** — i.e., already equivalent to
"decomposition off." Fixed the key resolution (mirrors the identical fix already in
`build_rag_agent()`/`_llm_call()`) and split the swallow into two paths: real API
failure now logs a warning instead of silently masquerading as a routing decision.

**Ran a clean on/off ablation, same judge (gpt-4o-mini), same code, same 6
cross-doc qa files (28 questions) — only the decompose flag differed:**

| | ON (decompose, key fixed) | OFF (decompose disabled) |
|---|---|---|
| cross_document_compare correctness (n=20) | 0.60 | **0.70** |
| Overall correctness (28q) | 0.707 | **0.779** |
| Overall faithfulness | 0.727 | **0.795** |

Spot-checked every changed answer: 3 genuine wins for OFF (decompose consistently
answers only one half of two-part comparisons — same failure mode across different
questions, not a fluke), 1 genuine loss for OFF, 1 pure judge noise (identical
predicted text, different score). Net signal matches the aggregate.

**Conclusion: decomposition is net-harmful, at least as currently prompted, and
was never in the production path anyway.** `eval/run_eval.py`'s decompose gate is
now **opt-in** (`EVAL_ENABLE_DECOMPOSITION=1` to re-enable for future experiments),
default off — this makes the eval harness match what real users actually get,
instead of measuring a code path `api.py` never calls.

**A2 (doc_014 sheet-summary `Description:` enrichment) — patch confirmed still
live in Qdrant, but not yet proven effective.** In the off-arm run,
`cross_document_added_qa__qa_2` ("which document refers to supplier payments")
still misattributed to doc_002 instead of gold doc_014 — the exact query A2 was
meant to help win. **Do not roll A2 out to the other ~20 structured docs until the
upcoming full 109q eval shows doc_014's own structured questions actually improved**
— rolling out an unverified technique to 20 docs on a loaded box would repeat the
same mistake as the C2/A/B prompt patches earlier this session (shipped as "fixed"
without live re-verification).

**Full 109-question eval — completed.** Batched (4 files via `--qa-files`, ~24-30q
each) inside a tmux session (`vault_eval_full`) to survive both mid-run restarts and
box disconnects. Took ~3.5h wall clock under heavy shared-box load (11-13 users,
load average swung 10→57 during the run) — not a code problem, pure contention.
Decomposition off (new default). Results:

| Metric | Old baseline | This run | Δ |
|---|---|---|---|
| Correctness | 79.6% | **81.9%** | +2.3 |
| Faithfulness | 78.6% | **79.5%** | +0.9 |
| Answer relevancy | 84.4% | 83.9% | -0.5 |
| Hit@5 (PDF/OCR, n=74) | 95.9% | 95.9% | unchanged (no re-ingestion) |
| Structured/Excel accuracy (n=21) | 76.2% | 76.2% | unchanged |
| Unanswerable refusal (n=14) | 78.6% | 78.6% | unchanged |

Correctness/faithfulness lift is real — driven by the C1 (cumulative-vs-incremental)
fix and earlier-session Excel/judge fixes, finally measured cleanly. Full breakdown
by question type in `eval/results/summary.json`.

**A2 verdict: dropped, per the pre-registered gate.**
`cross_document_added_qa__qa_2` ("which document refers to supplier payments")
still misattributes to doc_002 instead of gold doc_014 in the full run — identical
failure to the ablation test. The `Description:` enrichment is confirmed live in
Qdrant (checked directly) but does not fix the disambiguation problem it targeted.
**Not rolling out to the other ~20 structured docs** — document as a known
limitation in the README instead of chasing further prompt/data patches.

README.md, docs/CASE_STUDY.md, eval/results/summary.json all updated to the new
numbers as of this session.

---

## Status right now (stale as of 2026-07-06, see above for 2026-07-08 update)

**Eval numbers on disk (`eval/results/summary.json`, README, CASE_STUDY) are the PRE-fix baseline**
(109 questions, correctness 79.6% / faithfulness 78.6% / structured 76.2% / refusal 78.6%).
Five more root-cause bugs were found and fixed today (all prompt-only or targeted Qdrant patches,
no reingestion) — see "Completed this session (continued)" below. **None of these five are
reflected in the numbers above yet** — a fresh full 109-question eval to measure their combined
effect has failed to complete three times in a row, every time from the Claude Code session
itself restarting mid-run (background job gets torn down), not from a code bug or box load.
Partial run artifacts were discarded and `eval/results/*.jsonl` were restored to the consistent
pre-fix 109-line baseline so the repo isn't left in a half-overwritten state. Nothing from either
session is committed to git yet. See `TODO.md` for full narrative detail.

---

## Completed this session (2026-07-06)

- [x] **Corpus grew 93 → 109 questions** (still 18 documents). Added 3 targeted "field
      genuinely doesn't exist" refusal questions (`doc_006`/`doc_007`/`doc_014`) to test the
      Excel hallucination pattern found below.
- [x] **Eval judge fixed** — was `gpt-oss-120b` via OpenRouter, scoring some correct,
      fully-grounded cross-document answers as 0% faithful (reproduced directly). Switched to
      `gpt-4o-mini` (OpenAI) with a stricter judging prompt penalizing hedged/inferred claims.
      Verified fix against the reproduced case: faithfulness now scores 1.0 as expected.
- [x] **Excel/SQL hallucination path fixed** — agent was answering with a wrong-but-real column
      value (e.g. a supplier name for "VAT registration number") instead of refusing when a
      requested field didn't exist in a table. Fixed via: an explicit refusal instruction in
      `SQL_PROMPT_HEADER`, an anti-column-aliasing rule (model was caught disguising a wrong
      column via `SELECT x AS "<concept asked about>"`), a code-level fix in
      `src/tools/excel.py`'s `evaluate()` for a separate bug where the formatting step's own
      "no rows" phrasing could leak out as a fake answer, and a **programmatic hard gate**
      (`_column_matches_question`) checking real vocabulary overlap between the SQL's selected
      column and the question — added because prompt instructions alone were verified to still
      fail sometimes. Structured accuracy: 28.6% (baseline) → 42.9% → 61.9% → **76.2%** (current).
      One originally-failing case now 4/4 reliable; a second much improved (~80%) but not airtight.
- [x] **Figure-grounding root cause found and fix implemented** — was wrongly assumed to need a
      new vision model; actually a retrieval-disambiguation problem (see TODO.md for full
      detail). Fix in `src/parser/pdf_parser.py` (`_nearby_figure_label`), applied retroactively
      to `doc_008` via direct chunk/embedding patch + re-embed + Qdrant upsert (no full
      re-ingestion needed). Verified: the exact target question now answers correctly, citing
      the patched chunk. One question flipped the other way in the same test — open, see below.
- [x] Ruff format cleaned up (5 files were failing `ruff format --check`, now 0).
- [x] README.md, docs/CASE_STUDY.md, TODO.md fully synced to current numbers — the "three docs
      cite three different eval numbers" problem tracked lower in TODO.md is resolved.

---

## Completed this session (continued — same day, later)

Deep-dived the four lowest metrics (correctness, faithfulness, structured, refusal — everything
except hit@5, which was already good) by bucketing every non-perfect answer in the 109-question
run against its retrieved context, to separate real retrieval misses from generation/reasoning
bugs. Parent-child chunking was considered and rejected — none of the diagnosed failures were a
chunk-granularity problem (see TODO.md for the full per-case evidence). Five root causes found,
all fixed:

- [x] **C1 — cumulative-vs-incremental number confusion** (`doc_008_qa__qa_3`: model reported
      the $667.5B headline total instead of the $71.3B "additional... in 2024" figure the question
      asked for, even though the correct chunk was retrieved with faithfulness 1.0). Added a
      `CUMULATIVE VS. INCREMENTAL` rule to `ANSWERING_BLOCK` in `src/prompts.py`. **Verified**:
      scoped eval confirmed correctness 0.0 → 1.0.
- [x] **C2 — numeric-deadline ambiguity in chunk context sentences** (`doc_010_...qa_1`: gold
      "thirty (30) days" harassment-investigation deadline lost to an unrelated "three (3) months"
      administrative-leave chunk elsewhere in the same 100+-page handbook). Root cause: the
      auto-generated one-sentence `CONTEXT:` line embedded per chunk didn't name the specific
      number, so two unrelated numeric-deadline chunks embedded too similarly. Strengthened
      `CHUNK_CONTEXT_PROMPT` to require naming specific numeric deadlines/thresholds; patched and
      re-embedded the live `doc_010` chunk directly in Qdrant (same no-reingest pattern as the
      earlier figure-label fix). **Patched, not yet re-verified against the live agent** (ran out
      of clean box/session time).
- [x] **A — document title vs. section heading** (`doc_001_...qa_1`: asked for the document's
      title, model returned "V. Purchasing and Contracting Policy" — a numbered section heading —
      instead of the actual cover-page title "POLICY FOR THE PROCUREMENT OF GOODS AND SERVICES
      (PGS)"). Added a `DOCUMENT TITLE VS. SECTION HEADING` rule to `ANSWERING_BLOCK`.
      **Not yet verified.**
- [x] **A2 — structured/Excel sheet-summary chunks lack natural-language framing**
      (`cross_document_added_qa__qa_2`: query "supplier payments" — with the doc_id deliberately
      stripped by the eval harness's `_normalise_question`, to test natural-language resolution —
      matched `doc_002`, a prose PDF contract mentioning "Charges, Payment..." literally, at a very
      weak score of 0.0311, instead of `doc_014`, the actual Bristol supplier-spend dataset, whose
      only chunk is a terse columnar `Sheet summary: 6727 rows. Columns: Body, Body Name, Name,
      amount...` with zero natural-language phrasing). Patched both `doc_014` Qdrant points
      (`document_summary` + `sheet_summary`) with an added `Description:` line in plain language,
      re-embedded. Same pattern as C2 but for the Excel-ingestion path (`src/ingest_table_rows.py`'s
      `sheet_summary_text()`/`_build_sheet_summary_point()` have no LLM-written description at
      all — pure string concatenation from column headers/samples). User explicitly scoped this to
      a single-doc proof patch, not a full-corpus rollout (~21 structured docs) — that rollout is
      now a backlog item. **Patched, live-agent re-verify blocked twice by shared-box contention,
      not re-attempted after that.**
- [x] **B — false refusal despite complete evidence** (`doc_003_doc_008_cross_document_qa__qa_3`:
      confirmed via the raw agent tool-call trace that BOTH pieces of gold evidence were literally
      present in what the agent retrieved — doc_003's "four key vulnerabilities" figure text, and
      doc_008's "112 matters... across 42 new topic areas" sentence verbatim — yet the agent's
      final answer was a bare `Unsupported`). Root cause: the strict `ABSTENTION_RULE` combined
      with no explicit instruction for synthesizing a "compare X and Y" answer from two
      separately-retrieved facts, so the model treated "I can't state this in one verbatim
      sentence" as "nothing is supported." Added a `CROSS-DOCUMENT COMPARE — NEVER BLANKET-REFUSE`
      rule to `ANSWERING_BLOCK`. **Not yet verified.**

All five fixes are prompt-only (`src/prompts.py`) or live Qdrant patches — no code architecture
changes, no reingestion required. A fresh full 109-question eval to measure their combined effect
has been attempted three times and failed three times, every time because the Claude Code CLI
session itself restarted mid-run (background shell job gets torn down when the parent process
exits — confirmed via "no completion record found... may have been running when the previous
Claude Code process exited"), not because of a code bug, a hang, or box contention. Partial output
was discarded each time and `eval/results/*.jsonl` restored to the consistent pre-fix baseline.

---

## Tests run this session (what's actually verified vs. still open)

| Test | Result | Confidence |
|---|---|---|
| Full 109-question eval (OpenRouter `qwen/qwen3-32b` generation, `gpt-4o-mini` judge) | Correctness 79.6%, faithfulness 78.6%, hit@5 95.9%, structured 76.2%, refusal 78.6% | High — clean run, zero errors, cost $0.12 |
| Excel hard-gate: `doc_006` invoice-number question | 4/4 correctly refuses | High — repeated, consistent |
| Excel hard-gate: `doc_007` payment-method question | ~4/5 correctly refuses (was 0/4 before any fix) | Medium — real improvement, not airtight; occasional LLM nondeterminism still slips a wrong column past the gate |
| Excel hard-gate regression check (16 previously-working Excel questions) | Zero new regressions | High — diffed directly against pre-fix answers |
| Figure-grounding fix: `doc_008_qa__qa_3` (Figure 3 dollar amount) | **Now correct**, citing the patched chunk directly | High — direct causal proof, not coincidence |
| Figure-grounding fix: `doc_008_qa__qa_4` (Figure 4 mission) | Still wrong (`Unsupported`) both before and after | Confirmed unresolved, separate cause |
| Figure-grounding fix: `doc_008_qa__qa_8` (Figure 5 count) | **Re-verified 2026-07-08: correct** (99) | Confirmed the earlier "flip to wrong" was nondeterminism/box contention, not a real regression |
| C1 fix: `doc_008_qa__qa_3` cumulative-vs-incremental | **Now correct** (0.0 → 1.0) | High — scoped eval, direct before/after |
| C2 fix: `doc_010_...qa_1` numeric-deadline context patch | **Re-verified 2026-07-08: still wrong** — model answers "three months" (a different, unrelated chunk), not the patched "30 days" chunk. Root cause is retrieval ranking (distractor chunks outscore the patched one), not fixable by a context-string patch alone. Documented as a known limitation. | Confirmed still broken |
| A fix: `doc_001_...qa_1` title-vs-section-heading prompt rule | **Re-verified 2026-07-08: still wrong** — the title/cover-page chunk never makes it into retrieval's top-5 at all, so the prompt rule has nothing to apply to. Retrieval-side gap (needs forced-include of a doc's first chunks for "about this document" questions), not a prompt problem. Documented as a known limitation. | Confirmed still broken |
| A2 fix: `doc_014` sheet-summary description patch | **Re-verified 2026-07-08: patch confirmed live in Qdrant, but still doesn't fix the disambiguation it targeted** — `cross_document_added_qa__qa_2` still misattributes "supplier payments" to doc_002 instead of doc_014, in both a scoped ablation and the full 109q run. **Not rolling out to other structured docs.** | Confirmed ineffective, dropped |
| B fix: `doc_003_doc_008_...qa_3` false-refusal prompt rule | **Re-verified 2026-07-08: still wrong** for this specific question (bare `Unsupported`), but the underlying decompose API-key bug (see top of file) was the bigger issue — fixing it lifted the wider cross_document_compare bucket from ~0.53-0.74 (varies by baseline snapshot) to 87.5% in the full run. This one question's residual failure looks like a narrower evidence-retrieval gap, not a prompt problem. | Bucket-level: fixed. This question: still open |
| Full 109-question eval with all fixes combined | **Completed 2026-07-08**, batched in tmux, decomposition off. Correctness 79.6%→81.9%, faithfulness 78.6%→79.5%. See top of file. | Done |

---

## Next up (in order)

1. **Get a clean full 109-question eval run** with all 5 new fixes + the earlier session's fixes
   (Excel gate, judge, figure-label) combined — this is the single most important open item.
   3 attempts today all died from the Claude Code CLI session itself restarting mid-run (~20-30 min
   between restarts observed), not from box load or a code bug. Consider chunking into 4 batches of
   ~27 questions via `--qa-files` (each finishes in a few minutes, survives a mid-session restart)
   and merging results, rather than one long single run.
2. Once the full run lands, re-run the 3 figure_grounding questions (`doc_008_qa` qa_3/qa_4/qa_8)
   specifically to settle whether qa_8's earlier flip was a real regression or nondeterminism.
3. Investigate `qa_4` (Figure 4/$197B) separately — retrieval still isn't surfacing that chunk
   at all.
4. Commit everything (nothing is committed yet across either session) — `src/rag_agent.py`,
   `src/tools/excel.py`, `src/tools/retrieval_tool.py`, `src/prompts.py`, `src/config.py`,
   `src/parser/pdf_parser.py`, `eval/run_eval.py`, the qa_pairs files, README/CASE_STUDY/TODO.
5. `git push -u origin main`.
6. **Demo assets** — with backend on `:8001` + `npm run dev`:
   - Screenshot: cited answer with `[Source N]` citations + sources panel
   - Screenshot: document inspector (PDF page vs extracted markdown)
   - Short recording: ingest PDF → ask question → cited answer
7. **Sample corpus** — add 2–3 docs to `samples/` + `scripts/demo.py` so a client can run it in 5 min without their own files

---

## Backlog (nice-to-have)

- [ ] Pipeline improvement story in README: real ablation (baseline dense → +hybrid → +rerank →
      +force-include) — never actually run as a controlled experiment; the "Recent fixes" table
      in README is a real before/after story but for the Excel path specifically, not a
      retrieval-technique ablation.
- [ ] Add more figure-grounding questions to the corpus (n=3 is too small to trust on its own,
      especially now that there's a real fix to validate against).
- [ ] The "agent skips a required retrieval and guesses instead" gap (found during the judge
      investigation, documented in README's Known limitations) — not yet fixed.
- [x] ~~Roll the natural-language `Description:` enrichment out to all ~21 structured/Excel docs~~
      **Dropped 2026-07-08** — re-verified twice (scoped ablation + full 109q run) and the
      `doc_014` proof-of-concept still doesn't fix the disambiguation query it targeted
      (`cross_document_added_qa__qa_2` still picks doc_002 over doc_014). Not worth rolling out
      an unproven technique to 20 more docs. If revisited, the real fix is likely retrieval-side
      (reranker/scoring), not another natural-language description string.
- [ ] LiteLLM semantic cache (3 open blockers in `TODO_LITELLM.md`) — optional, not blocking
- [ ] Langfuse cost logging — optional

---

## Key numbers (current, full 109-question run)

| Metric | Value |
|---|---|
| Correctness | 79.6% |
| Faithfulness | 78.6% |
| Answer relevancy | 84.4% |
| Hit@5 (PDF/OCR, n=74) | 95.9% |
| Structured/Excel accuracy (n=21) | 76.2% |
| Unanswerable refusal rate (n=14) | 78.6% |
| Eval dataset | 109 questions (custom), 18 documents |
| Judge | `gpt-4o-mini` (OpenAI) |
