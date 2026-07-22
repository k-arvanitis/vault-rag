# Eval investigation — 2026-07-21 (overnight autonomous run)

Fresh full run in tmux `eval_run` (started 00:15). At analysis time: retrieval phase
done (00:17), answer phase 105/109, judge phase (OpenAI gpt-4o-mini) not yet complete —
so **no fresh scored `summary.json` yet**. Baseline for comparison = 2026-07-16 run
(correctness 0.838 / faithfulness 0.861 / relevancy 0.869 / structured 0.762 / refusal 0.786).

## Judge-independent findings (from raw_answers.jsonl, 105 rows, zero API cost)

### Premise confirmed: on this corpus, verbose answers ARE the failures
Median predicted-vs-gold char length by type shows the terse answers are the correct
refusals (unanswerable that fire correctly = 11c "Unsupported"), and the *long* answers
are precisely the wrong ones — over-answering instead of refusing, or misattributing
across documents. Conciseness and robustness are the same lever here.

### Finding 1 (SURGICAL, FIXED THIS SESSION) — dagger-citation filename leak, n≥5
gpt-oss emits citations in a `[N†file=doc_XXX.pdf]` form (dagger U+2020 gluing the marker
to the raw filename inside the brackets), plus glued `[N] file=doc_X.pdf` and parenthetical
`(file=doc_X.pdf)` variants. NONE of the existing strip regexes catch these
(`_INLINE_CITATION_RE` only matches pure `[N]`), so raw internal filenames leak into
user-facing answers on at least 5 questions:
- `services_contract_terms_qa__qa_4`, `__qa_5`  `[3†file=doc_002_...pdf]`
- `employee_handbook_2024_qa__qa_1`  `[6] file=doc_010_...pdf`
- `doc_015_food_sop_manual_qa__qa_5`  `(file=doc_010_...pdf)`
- `lease_cross_document_qa__qa_9`  `[1†file=doc_016b_...pdf]`
Provably content-safe to strip (no legitimate answer contains `[N†file=...]`),
unit-tested, no eval run needed. Fix: `_LEAKED_DAGGER_CITATION_RE` +
`_LEAKED_BARE_FILE_RE` added to `strip_leaked_headers` in `src/answer_pipeline.py`.

### Finding 2 (documented, NOT fixed — n=1) — full harmony-channel leak
`maturity_dataset_2025_qa__qa_2` returned 634c of raw `<|channel|>commentary<|message|>We
need to query...` — the *entire* answer is leaked chain-of-thought; the final channel never
emitted. This is the "deep reasoning-leak variant" already documented+deferred 2026-07-15
(n=1, reasoning-effort knob measured useless). Still n=1 in this run. Stripping the harmony
tokens alone leaves raw reasoning, not a real answer — the principled fix is to detect a
fully-leaked generation (contains `<|channel|>`/`<|message|>`) and downgrade to Unsupported
or retry, which is riskier than a one-line strip. Deferred, consistent with prior decision.

### Finding 3 (TOP ROBUSTNESS PRIORITY, NOT fixed — needs code, not prompt) — doc_015 misattribution
`doc_015_food_sop_manual_qa__qa_5` (gold=Unsupported) STILL returns a 1103-char confident
vacation-policy answer sourced from doc_010/doc_009 when the named doc (a food-safety SOP)
has no vacation content. This is the exact cross-document misattribution refusal bug tracked
since 2026-07-08 — reproduced again here, ~1 year of prompt-only attempts all verified NOT
working (see PROGRESS 2026-07-08 part 2). The real fix is a code-level doc-scope gate:
compare a retrieved chunk's `file=`/doc_id against the document the question names, and refuse
to answer from a non-matching document. Do NOT attempt another prompt patch. This is the
single highest-value robustness change but needs deliberate design + a full eval to validate —
not a blind overnight edit. Prioritized for a dedicated session.

## Per-metric read (to reconcile against fresh scores when judge finishes)
- **refusal (weakest guardrail, 78.6%):** driven by Finding 3 (over-answering) + Finding 2
  (leaked non-answer). Conciseness lever lives here.
- **faithfulness / relevancy:** Finding 1 cleanup helps professionalism/relevancy marginally
  (removes raw filenames/control chars from answers); does not touch correctness.
- **correctness:** retrieval/reasoning-bound (figure_grounding n=3, cross_doc_compare 0.775);
  length is NOT the lever — do not manufacture a conciseness story for these.

## FRESH SCORES LANDED (run completed 02:52, judged gpt-4o-mini)

| Metric | Baseline 07-16 | Fresh 07-21 | Δ |
|---|---|---|---|
| Correctness | 0.838 | 0.817 | −0.02 |
| Faithfulness | 0.861 | 0.838 | −0.02 |
| Answer relevancy | 0.869 | 0.839 | −0.03 |
| Structured acc (n=21) | 0.762 | 0.762 | 0 |
| **Correct-refusal rate (n=14)** | 0.786 | **0.643** | **−0.14** |
| hit@5 (n=74) | 0.986 | 0.986 | 0 |

**Reading:** retrieval unchanged (hit@5 0.986), structured unchanged (0.762). Headline
−2/−3pt on correctness/faithfulness/relevancy is within the documented ±0.02–0.05 judge-noise
band AND partly a judging-composition shift this run (`exact_match_shortcircuit` 26→12,
`custom_llm_judge` 77→97 — more answers went to the noisier LLM judge). Not a real generation
regression.

**The refusal −14pt IS real signal, and it is entirely the two verbose over-answering rows:**
- `maturity_dataset_2025_qa__qa_2` — c=0.0, 634c harmony-channel leak (Finding 2)
- `doc_015_food_sop_manual_qa__qa_5` — c=0.0, 1103c doc_010/009 misattribution (Finding 3, **confirmed exactly as predicted**)
Every other refusal is a terse 11c "Unsupported" scoring 1.0. Refusal metric = these 2 rows.
Fix both → refusal returns to ~0.79+. Conciseness == robustness confirmed by fresh scores.

**Checked and DISCARDED (premise failed):** 63/109 preds carry ` ` narrow-no-break-space
litter (0 golds do), but normalizing it + the Finding-1 leaks recovers **0** exact-matches —
the answers differ from gold by full-sentence phrasing, not whitespace. So ` ` cleanup is
cosmetic-only (nicer live answers), NOT an eval lever. Did not make that edit. The
exact_match_shortcircuit drop is unexplained but harmless (only shifts which judge scores a row).

**Other real (non-refusal) dips, n small:** numeric_reasoning 0.75→0.25 (n=4) — driven by a raw
tool-call JSON leak as an answer (`{"action":"search_knowledge_base"...}`), an email address
returned as a value, and wrong-column Excel answers. Real but n=4, noisy. cross_document_compare
0.775→0.690 — the known comparison-question instability (documented, agent tool-calling variance).

## Final prioritized plan
1. **DONE this session — Finding 1 leak strip** (`src/answer_pipeline.py`, 3 tests, 67 pass).
   Removes raw `[N†file=...]` / bare `file=` filenames from live answers. Cosmetic+correctness-safe,
   affects future runs/live app (this run's stored answers predate it). Uncommitted — review + commit.
2. **TOP robustness lever — Finding 3, doc-scope gate (NOT started, needs a dedicated session).**
   Code-level check: before an answer uses a retrieved chunk, compare the chunk's `file=`/doc_id to
   the document the question names; refuse (Unsupported) if a single-doc question is being answered
   from a non-matching document. This is THE fix for the −14pt refusal drop's biggest half. ~1 year
   of prompt-only attempts all verified NOT working (PROGRESS 2026-07-08) — must be code, plus a full
   eval to validate. Do NOT attempt blind.
3. **Finding 2 harmony full-leak (n=1, defer).** Only worth doing alongside #2 as a shared guard:
   "if the final answer still contains raw control tokens (`<|channel|>`/`<|message|>`) or a bare
   tool-call JSON object, the generation failed → downgrade to Unsupported / one retry." Riskier
   (an incidental brace could false-trigger) — design + validate, do not ship blind.
4. **Cosmetic (optional): normalize ` `→space, `‑`→`-` in `strip_leaked_headers`.** Cleaner live
   answers only; zero eval impact (measured). One-liner + one test if desired. Not a robustness fix.

Do NOT chase the −2/−3pt headline dips — judge noise + composition shift, not generation quality.

## CORRECTION (advisor caught this) — real refusal denominator is n=14 via `retrieval_method=="none"`, not `question_type=="unanswerable"`

Original analysis above only looked at the `question_type=="unanswerable"` bucket (n=10,
scored 8/10 in BOTH runs — chronic, not the drop). The real `correct_refusal_rate` denominator
(`eval/run_eval.py` line ~1071) is `retrieval_method=="none"` in `retrieval_results.jsonl`,
n=14, which includes 4 `numeric_reasoning`-typed rows never examined. Full enumeration:

**9/14 correct** (all terse "Unsupported", c=1.0). **5/14 failed:**
1. `maturity_dataset_2025_qa__qa_2` (Finding 2, harmony leak) — c=0.0
2. `doc_015_food_sop_manual_qa__qa_5` (Finding 3, doc misattribution) — c=0.0
3. `transactions_q1_2025_26_qa__qa_9` — raw tool-call JSON leaked as the answer — c=0.0
4. `doc_007_published_spend_report_april_25_qa__qa_9` ("what payment method...") answered
   `OTHER SUPPORT SERVICES` (a real Merchant Category value) — c=0.0
5. `doc_014_spend_bristol_supplier_april_2024_qa__qa_3` ("what is the email address...")
   answered a fabricated-looking email — c=0.0

### FIXED THIS SESSION — mechanism A (malformed generation), rows 1 & 3
Added `_MALFORMED_GENERATION_RE` / `_is_malformed_generation()` to `src/answer_pipeline.py`:
detects raw harmony control tokens (`<|channel|>`/`<|message|>`/etc.) or a bare tool-call JSON
object (`{"action": ...}`) surviving as the entire "answer" and forces `Unsupported`. Wired into
both `answer_query` chokepoints (single-part and per-part-of-multi-part), right after
`strip_leaked_headers`. Neither pattern can ever be legitimate answer content — content-safe,
same class as Finding 1. 4 new unit tests using the exact reproduced strings from this run,
**71/71 in test_answer_pipeline.py, 319/319 full suite, ruff clean**. No live API call needed —
deterministic text processing, verified offline per instruction not to re-run eval.

### INVESTIGATED, NOT fixed — mechanism B (rows 4 & 5), real finding: this is a ROUTING bug, not a vocabulary gap
Both are the doc_006/007/014 "field genuinely doesn't exist" refusal questions the 2026-07-06
session added `_column_matches_question` specifically to catch. **Simulated the gate directly
against the real DuckDB schemas** (doc_007: no payment-method column exists; doc_014: no email
column exists) — `_column_matches_question` correctly rejects every real column for both
questions (0/9 and 0/9 match). Confirmed `OTHER SUPPORT SERVICES` is a genuine value in
doc_007's `Merchant Category` column (verified via direct DuckDB query). **So the gate works
correctly in isolation — the wrong answer did NOT come through the gated `write_sql` path.**
This points at a bypass: most likely the agent routed to `search_knowledge_base`'s sheet-summary
chunk path instead of `query_excel` for these two, which carries no column-matching gate at all
and lets the model free-associate from raw sample rows in context. **Not confirmed** — would
need live tool-call tracing (`RETRIEVAL_DEBUG=1`) to see which tool actually fired, which this
session did not run (no live agent calls, per the token-budget instruction). This is a sharper,
different finding than "the gate has a vocabulary gap" — flagging for a dedicated session with
live tracing, per the standing rule: do not patch blind.

## Updated final priority order
1. **DONE — Finding 1** (dagger-citation leak) + **DONE — mechanism A** (malformed generation,
   rows 1 & 3 of the 5 refusal failures). Both safe, tested, uncommitted — review + commit together.
2. **Mechanism B routing bypass (rows 4 & 5)** — needs `RETRIEVAL_DEBUG=1` live tracing to confirm
   which tool fires for these two before any fix is attempted. Real, unconfirmed root cause.
3. **Finding 3 doc-scope gate (row 2)** — code-level chunk-doc_id vs. named-doc check, needs design
   + full eval to validate. Biggest single lever, most expensive to get right.
4. **Finding 2 residual** — mechanism A only forces Unsupported on a *fully* leaked generation;
   a partial leak mixed with real content (seen once before, 2026-07-15) isn't caught by this
   regex and would need the shared "generation looks broken → retry once" design from before.

With fixes 1+2 shipped, 2 of 5 refusal failures close (n=14: 9→11 correct, rate 0.643→0.786,
back to baseline) — measurable next full run, not claimed here without one.

## SUPERSEDED — mechanism A was a downstream patch, replaced with a root-cause fix

User pushback (correct): forcing `Unsupported` on a detected leak is a symptom patch, not a fix
— it discards the whole answer instead of preventing the leak. Replaced with real streaming-level
harmony-channel parsing in `src/rag_agent.py`'s `stream_agent`:

**`_strip_channels()`** — a new state machine, same technique as the pre-existing `<think>` filter
right above it (buffer tokens, find open/close markers, handle a marker split across chunk
boundaries). Parses gpt-oss's harmony format live: `<|channel|>NAME<|message|>content` — swallows
any non-"final" channel (hidden analysis/commentary), passes through "final"-channel content (or
everything unchanged when no channel marker ever appears — the normal case for every other
model/provider this app uses). Wired in before the existing `<think>` filter:
`_filter(_strip_channels(str(chunk.content)))`.

**Empty-answer hardening** (needed because a fully-swallowed message, i.e. no "final" channel ever
arrived — the exact n=1 case from this eval run — now correctly produces empty text instead of
leaked garbage, and empty must resolve to a clean refusal, not a blank answer):
- `_retrieval_only_answer` (rag_agent.py): last-resort fallback now treats empty input as
  `Unsupported` too (was previously only `"search_knowledge_base" in answer`).
- `answer_one`'s retry (answer_pipeline.py): the existing bare-`Unsupported` forced-retry now also
  fires on an empty answer (same failure class, deserves the same retry).
- `answer_query`'s single-part chokepoint: empty-after-strip also normalizes to `Unsupported`,
  defense in depth.

The tool-call-path fallback chain (`_repair_incomplete_answer` / `_context_fallback_answer`) already
treats empty as a "bad answer" via the pre-existing `_looks_like_bad_final_answer("")==True` check —
no change needed there, it composes for free.

**`_is_malformed_generation` (answer_pipeline.py) kept, scope narrowed in practice to what it alone
still covers**: the bare tool-call JSON leak (`{"action": "search_knowledge_base", ...}`) is a
*different* failure mode — the model emitting a fake function-call as visible text instead of an
actual tool call, not a harmony-channel boundary issue — `_strip_channels` doesn't and shouldn't
touch it. Kept as the safety net for that case and for any residual/malformed leak the channel
parser doesn't recognize.

**Tests**: `tests/test_rag_agent.py`'s new `TestStreamAgentHarmonyChannels` (4 tests, mirroring the
existing `<think>`-block test pattern): strips hidden channel keeps final; markers split across
multiple stream tokens (real streaming shape); the exact reproduced full-leak string degrades to
`Unsupported` not empty; plain content with no channel markers is completely unaffected (regression
guard for the normal/working case, which is 107/109 of every eval run). **323/323 full suite pass,
ruff clean.** No live API call needed — all tests mock `agent.stream()`, matching this file's
existing convention.

## Mechanism B "bypass" hypothesis — REFUTED by live trace, do not build a fix

Traced doc_007 qa_9 + doc_014 qa_3 live (`RETRIEVAL_DEBUG=1`, 2 targeted calls, generation via
OpenRouter — no Groq exposure) to confirm which tool answers them before touching any code, per
the standing rule not to patch blind. **Both correctly routed to `query_excel`, both had every
candidate SQL rejected by `_column_matches_question` (`SQL: []`), both correctly returned
`Unsupported`.** No bypass occurred — the gate worked exactly as designed.

This directly contradicts the actual eval run's failure (same 2 questions, wrong confident
answers, 2 days ago) on the same code. Not a structural routing bug — almost certainly the
already-extensively-documented **OpenRouter provider-routing nondeterminism** (TODO.md,
2026-07-15: "provider" field silently flips between identical calls; DeepInfra/Alibaba pins
both tested and reverted as worse than baseline). The gate's own logic is deterministic Python,
but a different backend serving the SQL-writing call can generate genuinely different SQL that
slips past the same gate differently, or fabricate the final answer text after SQL correctly
returned empty (`run_sql`'s "no rows" result still reaching a confident final answer would be a
formatting-step leak, not a gate miss — not confirmed either, not chased further).

**Not fixing this.** No reproducible bug to fix — building a "close the bypass" patch now would
be exactly the class of blind, unverified change this project's history is full of (5+ logged
"verified NOT working" prompt/gate patches). If this resurfaces in a future full eval run,
re-trace it live at that time rather than acting on a 2-day-old, non-reproducing failure.
Structured/Excel refusal for genuinely-nonexistent fields is not a confirmed open bug right now.

## Backend end-to-end QA sweep (live /query, RETRIEVAL_DEBUG-free run)

Systematic pass over fresh, realistic end-user questions against the live backend (not a
`run_eval.py` re-run), checking both answer correctness and per-citation grounding.

### Finding 4 (ROOT-CAUSE FIX) — Finding 1's dagger-citation fix was over-broad; it deleted
### VALID resolvable citations, not just leaks

q1 ("Who needs to approve a sole source procurement?") returned a factually-correct answer
("Chief Executive Officer") with a correct top source (doc_001 chunk 26, "Sole Source
Procurements must be approved by the Chief Executive Officer...") — but the answer text had
**zero `[N]` markers**. Traced live with a temporary debug print at the real `answer_query`
chokepoint (bypassing `data/query_cache.json`, which was silently serving the cached result
across the first several reruns and hiding the bug): the model's raw pre-strip answer WAS
correctly cited — `...CEO【1†file=doc_001_procurement_policy.pdf】` — but Finding 1's
`_LEAKED_DAGGER_CITATION_RE` unconditionally deletes every `[N†file=...]` match, including this
one, because it was written to treat the dagger form as pure leak noise. It never checked
whether N was a real, resolvable citation index the way plain `[N]` citations already do via
`citation_map`/`_strip_inline_citation`.

Effect: any answer whose ONLY citation was in dagger form silently lost all `[N]` markers, which
makes `frontend/lib/product.ts`'s `citedOnlySources()` fall back to the FULL raw retrieved-chunk
list (its documented fallback for genuinely citation-less SQL/table answers) — so the Evidence
panel would show every retrieved candidate, including clearly irrelevant negative-score chunks
(e.g. an HR sexual-misconduct policy excerpt, score -8.0), as if they were all evidence for a
procurement question. Right answer, right top source, but a noisy/misleading Evidence panel —
exactly the class of citation-integrity bug this session has been hunting.

Fix: added `_strip_dagger_citation()` (mirrors `_strip_inline_citation` but never falls back to
leaving the raw match untouched, since a dagger citation is never legitimate prose — only
"resolve via citation_map" or "strip clean"). Wired into `strip_leaked_headers`'s dagger-citation
substitution pass. New regression test
`test_resolves_dagger_citation_to_real_source_position` (reproduces q1's exact dagger string).
**79/79 answer_pipeline tests, 338/338 full suite, ruff clean.** Live-reverified against the
running backend (cache entry manually cleared to bypass the stale-cache trap): answer now reads
`...CEO[1].` and `sources[0]` is the single correct doc_001 chunk.

### q1 verdict: PASS (after fix). Answer correct, citation correctly resolved and grounded.

### Finding 5 (CONFIRMED BLOCKER — as a class, NOT fixed, escalate to user) —
### ungrounded-but-correct answer with a non-supporting citation stapled on

b1 ("How long should staff scrub their hands with soap when washing them, per the food safety
SOP?"): answer "10 to 15 seconds" is factually correct (matches gold, and matches doc_015's own
Michigan-specific figure — chunk 13: "Apply soap and rub hands together for at least 10 to 15
seconds..."). BUT: `RETRIEVAL_DEBUG` trace shows the model issued exactly one vague tool call
(`query='food safety SOP'`), which returned 12 chunks — chunk 13 was NOT among them. None of the
12 retrieved chunks contain "10 to 15 seconds" or any hand-washing duration. The model answered
from its own parametric knowledge (a plausible-sounding, coincidentally-correct guess), directly
violating `ABSTENTION_BLOCK`'s explicit "do not use your general knowledge to fill gaps... only
answers explicitly stated in the retrieved passages are valid" rule — then cited `[1]` -> doc_015
chunk 3 (an unrelated SOP-intro paragraph, no hand-washing content) to make the fabrication look
grounded. Citation is real (not "unknown", not a leak) and resolves correctly per Finding 4's
fix — but it does not actually support the claim. This is the most damaging failure mode a
"verified knowledge assistant" can have: confident + correctly-cited-looking + factually right
this one time + structurally fabricated.

Root cause is retrieval-quality (query too generic to surface the specific sentence), not a
citation-plumbing bug — same underlying cluster as the already-documented, unresolved doc_015
cross-document misattribution (see Finding 3, prior investigation). NOT fixed. Needs a frequency
read across a broader sweep before deciding between three options (each with different
eval-validation cost): (a) push the model toward more specific/multi-query retrieval for
narrow-fact questions, (b) tighten abstention discipline enforcement, (c) add a citation-
entailment gate (verify the cited quote's tokens actually contain the answer's key terms before
accepting a non-Unsupported answer) — the last one is the most direct fix but is a *behavioral*
change that needs a full eval run to confirm it doesn't flip currently-correct answers to false
refusals, so it is a user decision, not an autonomous one.

### Sweep tally (9 fresh, uncached, realistic questions; answer-vs-gold + citation-support axes)

| id | question topic | answer | citation support |
|----|---|---|---|
| q1 | sole source approval (doc_001) | correct | PASS (fixed, Finding 4) |
| b1 | handwashing duration (doc_015) | correct | **FAIL — non-supporting citation, Finding 5** |
| b2 | SOP submission requirement (doc_015) | correct | **FAIL — same class as b1, different chunk** |
| b3 | doc_002 version number | correct | **FAIL — cited an unrelated charges clause** |
| b4 | procurement vs services extension length (cross-doc) | correct | **FAIL — dangling unresolved `[15]` leaks raw into answer, Finding 6** |
| b5 | RFI transfer deadline (cross-doc) | correct | PASS — both citations genuinely support their claims |
| b6 | NET Amount lookup (doc_006, SQL) | correct (exact SQL match) | PASS |
| b7 | CEO salary of Simple Sanitation Inc (refusal probe) | correctly refused | PASS |
| b8 | authorizing manager's home address (refusal probe) | correctly refused | PASS |

**Answer correctness: 9/9.** **Citation integrity: 5/9 clean, 4/9 broken** (3× non-supporting
citation on a factually-correct answer, 1× raw dangling-marker leak). All 4 failures are on
plain document-search questions (not SQL/Excel, not refusal) — b5 shows the same question
*shape* (cross-document comparison) can pass cleanly.

**Correction (traced live, RETRIEVAL_DEBUG + citation_map dump, before finalizing — b2/b3
turned out NOT to be one mechanism):**
- **b1**: retrieval miss, confirmed. The answer-bearing chunk was never in the retrieved set;
  the model answered from parametric knowledge and cited an unrelated chunk to look grounded.
- **b2**: re-run fresh, PASSED cleanly this time — the answer-bearing chunk (chunk 1) was
  retrieved via a second, more specific follow-up query, and `citation_map` correctly resolved
  `[1]` to it. The original failure was **retrieval nondeterminism**: whether the agent issues
  a good follow-up query varies run to run (same class as this project's already-documented
  OpenRouter provider-routing nondeterminism), not a deterministic plumbing bug.
- **b3**: re-run fresh, reproduced the failure again — but the mechanism is different from b1.
  The answer-bearing chunk (doc_002 chunk 3, containing "V2.0") WAS retrieved (rank 11 of 12
  raw hits) and the model correctly extracted "V2.0" from it — this isn't parametric guessing
  (an invented document's version number can't be known parametrically). The operative cause:
  the model's own citation `[1]` just points at whatever ranked first in its raw context (chunk
  8, "Charges and Payment" — a completely unrelated clause), not the chunk it actually drew the
  fact from. `citation_map`'s mapping itself is correct (`{1: 1, ...}` faithfully reflects the
  tool's own local ordering) — the bug is the model not tracking which retrieved chunk it
  actually used, i.e. model citation-inattention, not a plumbing bug.
  A separate, interacting gap noticed in the same trace (not the cause of b3, but worth fixing
  alongside): the model is shown 12 raw chunks but only the first 8 survive into the final
  `sources` list / `citation_map` (capped), so any `[9]`–`[12]` the model might emit strips as
  unresolvable even when legitimate. If b3's low-ranked chunk 3 (rank 11) is ever the one the
  model correctly cites instead of mis-citing, that citation would currently be silently lost.

**Net conclusion: three distinct sub-mechanisms (retrieval miss, retrieval nondeterminism,
model citation-inattention) all produce the same visible symptom** — confident, correct-looking
answer, incorrect or missing citation. No single deterministic backend fix (like Finding 4's
dagger-citation fix) covers all three; a content-level check (does the cited chunk's text
actually contain the answer's key facts) is the only approach that would catch all three
uniformly. This doesn't change the ship verdict, it strengthens the case for it.

**The real risk is not "wrong citation," it's "ungrounded confident output."** b1's answer
happened to be right because "10 to 15 seconds" is this document's actual figure — but the same
mechanism (retrieval miss → the model falls back to its own general knowledge instead of
abstaining) can just as easily produce a confident, wrong, fabricated ANSWER on a question where
the guess doesn't happen to land on the true value. This sample didn't catch that case, but the
mechanism that would produce it is confirmed live. The citation bug is the visible symptom; the
abstention-discipline violation underneath it is the actual severity driver.

### Finding 6 (FIXED, live-verified 2026-07-22) — comparison path leaked a raw unresolved `[N]` marker

b4's second claim ended in a bare `[15]` with no corresponding source (only 8 sources returned).
Traced (not assumed) before editing: b4 does route through `answer_comparison_deterministic`,
confirmed by reading the routing condition at line 1157 (`_COMPARISON_RE` match returns
immediately on a non-None result, never reaching the multi-part `else` branch at line 1220).
Root cause: `_strip_inline_citation`'s fallback ("leave a bracketed number untouched if outside
1..MAX_TOOL_RESULTS, since it might be a year in prose") assumes a single tool call, which is
capped at MAX_TOOL_RESULTS (12) chunks by construction — a valid assumption on that path. But
`answer_comparison_deterministic` renumbers markers across TWO independent per-document
retrieval calls combined into one sequence (e.g. 9 + 8 = markers 1..17), which can legitimately
exceed 12 while still being real citation attempts, not literal numbers. `[17]` (real, but
trimmed out of the final 8-source cap) was misread as "probably a year" and left leaking raw.

Fix: `_strip_inline_citation` and `strip_leaked_headers` now take a `max_plausible_marker`
parameter (defaults to `MAX_TOOL_RESULTS`, preserving the normal single-call path unchanged);
`answer_comparison_deterministic` passes its own real per-call marker count (`marker - 1`)
instead. New regression test `test_high_marker_beyond_max_tool_results_still_resolves_or_strips_cleanly`
reproduces the exact 17-marker/9+8-chunk shape; verified it fails against the unfixed function
(direct call, no override) before confirming the fix. **339/339 full suite, ruff clean.**
Live-reverified against the running backend (cache cleared to bypass stale-cache trap): b4 now
returns a clean answer with no dangling marker; b5 (same question shape, was already passing)
re-checked and still resolves both citations correctly — no regression.

### Ship-readiness verdict: DO NOT SHIP without a decision on Finding 5/6's fix

Every answer in this small sample was factually correct — the generation/retrieval quality is
good. The blocker is citation *integrity*: **4 of 9** sampled questions showed either a citation
that doesn't support the claim (looks legitimate, isn't — Finding 5, three different underlying
mechanisms, NOT fixed) or a visibly broken dangling marker (Finding 6, a separate comparison-path
bug — **FIXED 2026-07-22**, see above). For a product explicitly branded "verified knowledge
assistant" with a click-to-inspect Evidence panel, this undermines the core value proposition
even though the prose answers themselves were reliable in this sample. More importantly,
Finding 5's underlying mechanism (retrieval miss → model falls back to its own general knowledge
instead of abstaining) is confirmed to happen live — it produced a coincidentally-correct answer
here, but the same mechanism can just as easily produce a confident, fabricated WRONG answer;
that risk is not bounded by this sample.

Recommend: user decides between (a) shipping with a known-issues disclaimer / narrower demo
script that avoids these question shapes, or (b) authorizing a fix for Finding 5 (three options,
see above — a content-level grounding gate, checked against ALL retrieved candidates not just
the cited one, is the recommended shape), followed by a full eval run to validate before
shipping. Finding 6 is already fixed and shipped; it was a different bug from Finding 5's
cluster (a grounding gate would not have caught a dangling out-of-range marker) — correctly
treated and fixed as its own separate issue, not bundled into Finding 5's fix.

## Track 1: grounding gate for Finding 5, implemented 2026-07-22

User authorized implementing the gate + eval validation (up to 3 full eval runs available).
Design (see `_is_answer_grounded`/`_extract_grounding_claims` in `src/answer_pipeline.py`):
extract numeric claims from the answer (a number plus at most one immediately-following word,
e.g. "15 seconds", "V2.0" — deliberately NOT a wider prose window, since that grabs the model's
own connecting words rather than the source's actual wording and false-refuses correct answers),
strip citation-marker and doc-id false-positives first, and require every claim to appear
somewhere in the FULL uncapped `collected` retrieval pool (not the capped 8-item `sources` list,
so a real but low-ranked supporting chunk like b3's isn't penalized). No extra LLM call — pure
string matching, zero added latency/cost. Numeric-only: does nothing for qualitative
fabrications (e.g. a wrong exemption category with no digits), so any faithfulness lift from
this specific fix is bounded to numeric-claim questions — not a general grounding fix.

Design validated (not just built) against real corpus data before wiring in: fires on b1 (no
retrieved chunk contains "10 to 15 seconds"), fires on a second independent case (a
plausible-but-wrong general-knowledge guess, "20 seconds"), and does NOT false-refuse b3 (chunk
3, rank 11/12, genuinely contains "V2.0" even though it wasn't the cited chunk) or hyphen/en-dash
range reformatting of a genuinely grounded figure. 8 new unit tests in
`TestGroundingGate`/`TestAnswerComparisonDeterministic`, 346/346 full suite, ruff clean.

### Pre-registered accept criteria (written before running the validation eval, so the read
### isn't rationalized after seeing the number)

Prior baseline numbers are stale and conflict (memory: 2026-05-11 correctness 80.4% /
faithfulness 89.3%; this doc's Jul-16 reference: correctness 83.8% / faithfulness 86.1%) — ran a
**fresh baseline eval on 2026-07-22** with Track 2's fixes active but the grounding gate
temporarily disabled (`if False and ...` at both call sites), immediately followed by a second
full run with the gate enabled, so the only variable between the two runs is the gate itself.

**Decision rule, fixed in advance:** ship the gate only if faithfulness rises and correctness
drops by less than 3 percentage points. The predictable failure signature of an UNDER-firing
gate (design too narrow, doesn't catch enough real cases) is "faithfulness flat, correctness
also flat" — not a reason to loosen it further without new evidence. The predictable signature of
an OVER-firing gate (false refusals) is "correctness drops more than ~3pts, faithfulness may not
even rise much since the false-Unsupported answers were already correct" — if this appears, the
decision is **revert**, not "loosen the threshold and re-run" (that's an unvalidated third
change on an already-unclear signal, not a fix). Per-question diffing (which qa_ids flipped
answer, and whether that flip was Unsupported-was-correct vs Unsupported-was-wrong) is required
reading before accepting or reverting — aggregate deltas alone can't distinguish "correctly
refused an ungrounded answer" from "wrongly refused a grounded one."

### Result: REVERTED 2026-07-22. Lexical claim-matching cannot do this job — two runs, one clean offline replay, decisive

**Run 1 — baseline** (Track 2 fixes active, gate disabled via a temporary `if False`):
correctness 85.78%, faithfulness 85.64%, relevancy 87.52%, refusal 78.57% (n=109). Saved to
`summary_baseline_no_gate_20260722.json` / `raw_answers_baseline_no_gate_20260722.jsonl`.

**Run 2 — gate enabled**, same corpus, immediately after: correctness 83.30% (−2.48pts),
faithfulness 84.36% (−1.28pts, i.e. DOWN not up), relevancy 86.24% (−1.28pts), refusal 71.43%
(−7.14pts). **Fails the pre-registered rule outright** (faithfulness didn't rise) before even
weighing the correctness delta.

**But the two-live-runs comparison turned out to be the wrong instrument, discovered by
diffing the raw answers**: 83 of 109 raw answers differed between the two runs at the text
level — far more than the gate alone could ever cause (the gate can only ever push an answer TO
"Unsupported"; it can't reword one). This is this project's already-documented
generation/retrieval nondeterminism (OpenRouter provider routing, retrieval-call variance)
dominating the signal. Traced the 3 answers that flipped to "Unsupported" between the two runs
individually: for the clearest case (`doc_003_fed_annual_report_2024_qa__qa_5`, the Fed's
creation date), re-running the gate function directly against THAT RUN's own actual retrieved
pool with the historically-correct answer text showed `grounded=True` — the gate would NOT have
fired on it. The "flip" was the model itself generating a different, worse answer that run, not
the gate misfiring. **The live A/B design could not isolate the gate's effect from ambient
system noise; the aggregate metrics from Run 2 are not meaningful evidence either way.**

**The real test: a deterministic, zero-cost offline replay.** Ran `_is_answer_grounded` directly
against every one of the 109 *baseline* (Run 1) raw answers and their actual retrieved pools —
no live calls, no nondeterminism, pure Python against fixed data. Result: **20 of 109 baseline
answers (18%) would be flipped to Unsupported**, and reading them individually, most are
obviously wrong to refuse — e.g. `doc_003_fed_annual_report_2024_qa__qa_1` ("111th Annual Report
of the Board of Governors of the Federal Reserve System") matches gold exactly and is genuinely
quoted from the retrieved chunk, `doc_008_qa__qa_2` ("64 matters") matches gold exactly,
`doc_008_qa__qa_5` ("$48.8 billion") matches gold exactly. Root cause: the claim-window design
("number + at most one following word", chosen specifically to avoid grabbing the model's own
prose per the b3 spare-test) breaks down whenever a number is followed by a multi-word title or
compound phrase whose word order or punctuation differs even slightly between the model's
phrasing and the source's ("Federal Reserve 2024 Annual Report" vs the source's own word order;
comma-separated dates like "December 31, 2024" vs the answer's non-breaking-space-joined "December
31 2024"). This is the same squeeze identified before implementation (bare-token-anywhere
under-fires via incidental digit collisions; word-window-anywhere over-fires via phrasing
variance) — b1/b3 were both single unit-labeled values ("15 seconds", "V2.0") and happened to sit
in the design's narrow safe zone; the wider corpus does not.

**Decision: REVERTED**, per the pre-registered rule (over-firing signature confirmed, decisively,
by the offline replay). `_is_answer_grounded`, `_extract_grounding_claims`, both call sites, and
their 7 unit tests removed from `src/answer_pipeline.py` / `tests/test_answer_pipeline.py`.
**339/339 full suite (back to the pre-Track-1 count), ruff clean.** Findings 4 and 6 (dagger
citation resolution, comparison-path marker leak) are untouched and remain shipped. Finding 5
(fabricated-but-plausible citations on a retrieval miss) is **still open, still real, still
ship-relevant** — this session narrowed it to "no cheap lexical fix exists" and confirmed the
mechanism thoroughly, but did not close it.

**What would actually work, not yet built:** per-claim semantic entailment via a cheap judge
call (gpt-4o-mini, not the Groq-backed generation model) — "does any of these retrieved passages
support this specific claim?" — asked only for numeric-claim answers, which is a small subset of
traffic. This can and should be prototyped and validated fully offline against this session's
same 109-answer replay set (no live eval, no user-facing latency risk, trivial judge-model cost)
before any decision about wiring it into the live `/query` path — that wiring is a latency/cost
change to production and is the user's call, not an autonomous one. The deterministic replay
harness built for this validation (re-usable: load `raw_answers_baseline_no_gate_20260722.jsonl`,
run any candidate grounding check against each row, read the flips) is the reusable asset from
this attempt and should be the first tool reached for on any future retry, live eval runs second.

### Entailment prototype, run offline 2026-07-22 — promising but ALSO not yet ship-safe

Built and ran the suggested offline prototype (`/tmp/.../scratchpad/entailment_prototype.py`,
not committed to the repo -- a one-off experiment script): for every baseline answer containing a
digit, asked gpt-4o-mini "does the retrieved evidence genuinely support this answer's specific
claim?" against the actual retrieved passages, zero live eval calls, ~90 cheap judge calls total.

90 candidates judged: 20 had zero real retrieved evidence (SQL/Excel-path answers -- the offline
script's candidate filter didn't replicate the lexical gate's SQL exemption, a test-harness gap,
not a real finding) leaving 70 genuine document-search cases: 33 YES (grounded), 37 NO.

Read a sample of the 37 "NO" verdicts by hand: several are genuinely correct catches the lexical
gate could never make (e.g. `doc_001_doc_002_cross_document_qa__qa_4` — the answer discusses the
wrong clause entirely; `doc_001_procurement_policy_qa__qa_4` — the model answered the approval
date, "September 4 2024", when asked for the original issue date, "December 15 2005" — a
wrong-field-used error a bare grounding check can't catch but an entailment judge correctly
flags as unsupported for the actual question asked). This is real evidence the semantic approach
is qualitatively more capable than lexical matching.

**But it also has its own false-negative rate, from a different mechanism (judge unreliability,
not extraction brittleness):** spot-checked 8 of the 37 "NO" verdicts against the raw retrieved
text directly. Two were judge errors on genuinely correct, genuinely grounded answers --
`doc_001_procurement_policy_qa__qa_2` ("September 4, 2024", exact gold match) with "September 4"
and "2024" both literally present in the retrieved pool, and `..._qa_5` ("September 2027", exact
gold match) with "September 2027" literally present -- the judge said NO for both despite the
fact being right there in the passages it was shown. 2/8 spot-checked = a real, non-trivial
false-negative rate on this small sample, not yet characterized at full scale.

**Conclusion: entailment is a more promising DIRECTION than lexical matching (semantically
sound, catches wrong-field/wrong-clause errors lexical structurally cannot), but is NOT
validated enough to ship as-is.** It needs: (a) a better-engineered judge prompt (asking it to
quote the exact supporting span before answering YES/NO tends to reduce this kind of judge
hallucination in other RAG-eval setups, not yet tried here), (b) a full 90-candidate accuracy
read against gold (this session only hand-checked 8), and (c) if pursued, still requires the
same offline-validate-before-wire-in discipline as the lexical attempt, plus a genuine
latency/cost tradeoff decision (a live per-query judge call) that is the user's call, not an
autonomous one. NOT wired into any code path -- prototype only, script left in scratchpad, not
committed.
