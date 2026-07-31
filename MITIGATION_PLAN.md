# Cross-document evidence pipeline — investigation + mitigation plan

**Date:** 2026-07-30 (investigation session, no code changed)
**Scope:** Phase 1 (root-cause investigation), Phase 2 (ranked mitigations), Phase 3 (per-item measurement).
**Nothing in this file has been implemented.** No commits made.

Live state this was measured against: Qdrant `documents_chunks` on `:7333`, demo-scoped corpus
(doc_001 79 content chunks, doc_002 29, doc_006, doc_016a), `BAAI/bge-reranker-v2-m3` on cuda,
HyDE on (OpenRouter reachable), working tree = the uncommitted 2026-07-30 state.

Repro scripts used (throwaway, in the session scratchpad, not added to the repo):
`repro_crossdoc.py`, `repro2_rank.py`, `repro3_goldrank.py`.

---

## 0. Headline: the logged diagnosis was wrong, and the eval cannot see the real bug

Two corrections to what `PROGRESS.md` / `TODO.md` currently record as open:

1. **"The reranker overrides the correct dense-score result."** Reproduced directly and it is the
   opposite. On the demo question's doc_002 clause query, the cross-encoder ranks the correct
   chunk **#1 by a wide margin** (`chunk=6`, score −0.73, next best −5.82). The component that
   demotes it is `_merge_hits(raw_hits[:3], reranked_hits)` in `retrieval_tool.py` — the top 3
   *pre-rerank* hits are pinned ahead of the reranker's own #1. The reranker is the part of this
   pipeline that gets it right; it is being overridden, not overriding.
2. **The correct chunk is retrieved and returned by the tool** (position 2 of 12). It is then
   dropped by `parse_sources`' 8-source cap, which gives the second document exactly one slot.
   That is why the model writes the right clause text and it comes back uncited.

And the measurement problem behind both: **`eval/` scores answer text, never citation-to-evidence
alignment.** An answer whose prose is right while its citation points at a wrong-document chunk
scores 1.0. So `cross_document_compare = 0.850` neither ranks nor detects the flagship failure —
it is not the right instrument for prioritising this work, and Phase 3 has to add one.
(It is also understated: `doc_003_doc_008_cross_document_qa__qa_1` scores 0.0 with a manifestly
correct answer, penalised only for naming the documents by title instead of `doc_003`/`doc_008`.)

Secondary numbers, for the record:

- `faithfulness` on `cross_document_compare` is **0.833 (n=15)** — the lowest bucket of any
  category, below its own correctness (0.850). It is the closest existing proxy for the evidence
  problem, but it is judged against *retrieved context*, not against *which source a claim is
  pinned to*, so it is not a substitute for the new metric.
- The two refusal numbers differ **by population, not by definition**, both thresholded at
  `correctness >= 1.0` (`eval/run_eval.py:1133-1157`): `unanswerable_metrics` (n=14) selects rows
  with `retrieval_method == "none"` — every should-refuse question, including 4 Excel
  "field doesn't exist" rows typed `numeric_reasoning`; `correctness_by_question_type["unanswerable"]`
  (n=10) selects `question_type == "unanswerable"` only. **0.857 (12/14) is the honest refusal
  number**; 0.900 is the mean correctness of a subset of it. Two failures total:
  `unanswerable_qa__qa_5`, `doc_014_..._qa_3`.

---

## Phase 1 — ranked, root-caused failure modes

Ranked by (evidence quality × breadth of categories affected × distance from the product promise).

### FM-1 — `parse_sources`' 8-cap gives the second document exactly one citation slot
**Confidence: reproduced directly.** Structural, not intermittent.

`parse_sources` (`src/answer_pipeline.py:1044-1060`) ends with a diversity pass that guarantees
each distinct filename **one** slot, then fills the remainder in list order:
`(diverse + rest)[:8]`. The deterministic comparison path emits doc_001's 12 blocks then
doc_002's 12 blocks as one call (no `_CALL_BOUNDARY`), so:

```
diverse = [doc_001_c1, doc_002_c1]
rest    = [doc_001_c2..c12, doc_002_c2..c12]
result  = [doc_001_c1, doc_002_c1, doc_001_c2..c7]   → 7 doc_001, 1 doc_002
```

Measured output of the real path on the demo question (`repro_crossdoc.py`):

```
[1] doc_001 chunk 52   [2] doc_002 chunk 11   [3] doc_001 chunk 59  ...  [8] doc_001 chunk 40
correct doc_001 evidence survives parse_sources: True
correct doc_002 evidence survives parse_sources: False
```

`7 of 8 sources are doc_001` — the exact symptom recorded on 2026-07-30, now explained. doc_002's
one slot is spent on its position-1 chunk (the Staff/notice clause, FM-2), and the correct
"3 Supply of Services" chunk at position 2 is discarded. The synthesis model still *sees* it (the
prompt gets all 24 blocks), writes the correct clause, emits a marker for it — and
`build_citation_map` can't resolve a marker whose chunk is no longer in `sources`, so
`strip_leaked_headers` strips it. **Right answer, no citation** is a deterministic consequence.

Explains: the demo cross-doc question; the "doc_002's fact came back uncited entirely" note;
plausibly `cross_document_added_qa__qa_2`-style rows where one document's evidence is absent.
Applies to *any* answer whose evidence spans a document with many chunks and one with few —
not comparison-only. It is the shortest path between "the pipeline retrieved the truth" and
"the user cannot verify it."

### FM-2 — the `raw_hits[:3]` head-insert overrides the cross-encoder for the top 3 slots
**Confidence: reproduced directly, and generalised over 18 gold evidence quotes.**

`retrieval_tool.py:1246-1248` (the non-stage1 / doc-scoped branch — the branch every
`_retrieve_for_doc` call takes):

```python
top_other = _merge_hits(raw_hits[: min(3, _rerank_top_n)], reranked_hits)[:_rerank_top_n]
```

The first three positions the agent sees are pre-rerank order. Measured on the doc_002 clause
query (`repro2_rank.py`, same pool for all three orderings):

| ordering | rank of the correct chunk (`chunk=6`) |
|---|---|
| dense/hybrid `retrieve()` order, with `filter_token` | 2 (chunk 11 first) |
| dense/hybrid `retrieve()` order, without `filter_token` | 2 (identical — the token isn't the cause) |
| cross-encoder reranker | **1** (−0.73 vs −5.82 for chunk 11) |

Generality (`repro3_goldrank.py`, all 18 gold evidence quotes across the 14 gold questions whose
documents are live — doc_001, doc_002, doc_001×doc_002 cross-doc; gold chunk located by quote
match, ranked in the doc-scoped pool):

```
gold chunk at rank 1 under retrieve() order : 4/18
gold chunk at rank 1 under reranker order   : 10/18
reranker strictly better than retrieve()    : 11/18
retrieve() strictly better than reranker    : 2/18
```

So the head-insert is systematically harmful, not a one-off, and not comparison-specific — it
degrades the top of *every* scoped retrieval, which is where `single_doc_factoid`,
`ocr_extraction` and `figure_grounding` all live. It also inverts the standard published pattern
(fuse broadly, then let a cross-encoder arbitrate the final order — see §Phase 2 research notes).

This is also the correct re-reading of the older `_attach_excel_marker_if_missing` docstring note
("the reranker parking the real evidence at rank 3, not rank 1") and of the 2026-07-09
figure-grounding investigation: the reranker was blamed for orderings it did not produce.

### FM-3 — `retrieve()` sorts *every* query's hits by term-occurrence count, not by relevance
**Confidence: reproduced directly.** Root cause underneath FM-2.

`src/retriever.py:506` calls `_extract_table_filter_terms(query)`, which has **no gate at all** —
it returns every token ≥3 chars that isn't a stop word, for any query. When it returns anything
(i.e. essentially always), `retrieve()` finishes with:

```python
scored_from_qdrant.sort(key=lambda h: (_table_term_score(h), h.get("score", 0.0)), reverse=True)
```

Primary key = *how many query words literally appear in the chunk*; the hybrid fusion score is
only a tie-break. Measured on the doc_002 clause query:

```
filter_terms: ['focuses','instead','customer','issued','notices','varying','service','scope']
chunk 11  termcount 3  score 0.5000     <- ranked 1
chunk  6  termcount 3  score 0.3929     <- ranked 2 (the correct chunk)
chunk 23  termcount 2  score 0.5833     <- ranked 4, despite the highest fusion score
```

A heuristic built for spreadsheet row lookups ("does this row contain the supplier name") is
deciding the order of prose retrieval, corpus-wide. It is invisible wherever a reranker runs
afterwards and dominant wherever one doesn't: the `raw_hits[:3]` head-insert (FM-2), stage-1 doc
routing, `_per_doc_retrieval_queries`' own scoring probe, and
`answer_quality._direct_retrieval_answer`. Fixing FM-2 alone leaves those consumers on it.

This is a strong candidate explanation for several long-standing "the chunk is indexed, correct
and never ranks" mysteries in `PROGRESS.md` (doc_001 front matter, C2's numeric-deadline
distractor, `doc_008_qa__qa_4`), all of which were re-attributed to embeddings, chunk context or
the reranker and repeatedly failed to be fixed there. **Inferred for those specific cases, not
re-reproduced** — the documents are not in the live demo corpus.

### FM-4 — citation-evidence quality is unmeasured; the eval judge can't see it
**Confidence: reproduced (judge artifact confirmed on a specific row); structural for the rest.**

`_custom_judge_answer` scores predicted answer text against the gold answer (+ faithfulness
against retrieved context for non-Excel rows). Nothing compares *the source attached to a claim*
with *the evidence that claim came from*. Consequences:

- The whole FM-1/FM-2 failure class is invisible to eval — an uncited-but-correct answer and a
  wrong-document-cited answer both score correctness 1.0.
- Prioritisation of this work cannot be justified by expected eval-point movement, and shouldn't
  be attempted. Say so rather than implying `0.850` ranks it.
- `doc_003_doc_008_cross_document_qa__qa_1` shows the judge marking a correct comparison 0.0 for
  a naming convention, so the bucket is understated by roughly one question (≈5pts) on top.

### FM-5 — `build_citation_map` maps only the last tool call, so multi-call answers collide
**Confidence: root-caused in the 2026-07-30 session, routed around rather than fixed.**

Each `search_knowledge_base` call restarts `[N]` at 1. `build_citation_map`
(`src/answer_pipeline.py:671`) therefore only maps `calls[-1]`, by its own docstring. Any answer
built from two calls (every agent-path comparison, every retry that fires a second search) emits
two different facts both labelled `[1]`, of which only one resolves. The 2026-07-30 fix routed the
demo question around this via `answer_comparison_deterministic` — correct for that question, but
the agent path is still the fallback for 12 of 15 gold cross-doc questions (FM-6), and it is still
broken there.

### FM-6 — `_resolve_comparison_doc_ids` resolves 3 of 15, so most comparisons take the broken path
**Confidence: swept in the 2026-07-30 session (11 regex-matched, 3 resolvable); unchanged.**

Resolution today is: explicit `doc_XXX` in the question → UI multi-select scope → filename-stem
token overlap. Real users type none of those. Everything else falls back to the agent path, which
carries FM-5. The content-based retrieval fallback was tried and reverted (3 confidently wrong
pairs); do not retry that signal.

### FM-8 — the corrective loop is disarmed on exactly the questions it was built for
**Confidence: read directly in code this session; not separately reproduced live.**

`answer_one` sets `skip_grounding_check=is_comparison` (~1476), turning `_verify_grounded` off for
the whole comparison class — justified in the code by "comparisons already get their own
doc-coverage retry." But that retry's precise branch, `_missing_mentioned_docs` (111-119), opens
with `if len(mentioned) < 2: return []` and only counts literal `doc_XXX` tokens in the question.
Real users type none, so on a natural-language comparison **both** mechanisms are off at once:
the groundedness check by construction, the targeted coverage check by an empty trigger.

What survives is the coarse `n_sources < 2` fallback (1491) — a count of distinct filenames,
satisfiable by two *wrong* files, and computed from `parse_sources`, i.e. through FM-1's cap.
Since that cap guarantees each file one slot, `n_sources >= 2` is true whenever two documents
were retrieved at all, including when one document's single slot holds the wrong chunk. The
retry then fires only on the crudest signals (flat `Unsupported`, or a partial "Unsupported"
fragment) and never on "this document's evidence is wrong."

This is the concrete content of "the corrective loop is mis-wired, not missing" — see the
research table below, and M-7.

### FM-7 — `_best_snippet` can keep the right chunk and drop the answer sentence
**Confidence: inferred from code, not reproduced.** Listed so it isn't mistaken for a new bug later.

`_format_hits` replaces any prose chunk longer than `MAX_CHUNK_CHARS` with
`_best_snippet(content, retrieval_query, max_chunk_chars)`. On the comparison path the
`retrieval_query` is one clause of the question, so a long chunk can be truncated around the
wrong sentence even when the chunk itself is correct. Not observed in this session's repros
(the relevant chunks were under the limit). Check before spending effort on it.

---

## Phase 2 — mitigations, in execution order

Ordering principle: everything that makes the *existing* correct retrieval survive to the user
comes before anything that changes what is retrieved, and both come before any new architecture.
Two of this repo's own dead ends (context regeneration ×2, the content-based doc resolver) were
attempts to change retrieval when the evidence was already being thrown away downstream.

### M-1 — allocate the source cap fairly across documents (fixes FM-1)
**Scope: narrow, low-risk patch.** Same class as the shipped citation-numbering fix.
**Files:** `src/answer_pipeline.py`, `parse_sources` (the `(diverse + rest)[:8]` tail, ~1044-1060).
**Architectural direction: neither** — this is plumbing, and no amount of agentic or corrective
machinery upstream helps while the cap discards the result.

Replace "one guaranteed slot per file, then fill in list order" with a round-robin interleave by
filename (take each file's rank-1, then each file's rank-2, …, until the cap). Two files → 4 slots
each instead of 7/1. Single-document answers are bit-for-bit unchanged (one file → identical
order), which bounds the blast radius to exactly the multi-document case that is broken.

Do **not** raise the cap from 8 as the fix. 8 is a UI/context bound; raising it hides the
imbalance for two documents and returns for three.

**Already verified against the live path before proposing it** (`repro4_roundrobin.py`, same
question, same retrieval): round-robin gives `{'doc_001': 4, 'doc_002': 4}` and **both** documents'
correct chunks survive (doc_001 `chunk 59` at slot 3, doc_002 `chunk 6` at slot 4), versus
doc_002's being dropped today. The proposal is measured, not assumed.

**Sequencing warning — do not let M-2 mask this.** If M-2 lands first, doc_002's correct chunk
moves to tool position 1, so its single slot happens to hold the right chunk and *the demo question
passes without M-1*. That is the "smaller fix looked sufficient" trap this repo has hit repeatedly.
M-1's justification is the general case: the reranker puts gold at rank 1 only 10/18 of the time,
so the second document's one slot is wrong roughly half the time. Land M-1 first, and judge it by
the unit test below, never by the demo question.

**Measurement (M-1):** `repro_crossdoc.py`'s `correct doc_002 evidence survives parse_sources`
must go False → True with a 4/4 split (confirmed achievable above). Then a unit test, no live
services (repo convention): synthetic `collected` of 12 blocks for file A + 12 for file B, assert
each file gets `floor(8/n_files)` slots and that B's **rank-2** chunk survives — rank-2
specifically, so the test still fails if only M-2 is applied. This test fails today.

### M-2 — stop pre-rerank order from overriding the cross-encoder (fixes FM-2)
**Scope: real behaviour change on every retrieval — ship-gated on a full eval replay**, per this
repo's own rule for retrieval changes (there is a documented 12-point regression from exactly this
class of change).
**Files:** `src/tools/retrieval_tool.py`, `_fetch_docs`'s non-stage1 branch (~1246-1248).
**Architectural direction: #2, narrowly** — this *is* the "retrieval confidence gating" idea from
the CRAG family, except the mechanism already exists (a trained cross-encoder) and is being
overridden. Fix the wiring rather than adding an evaluator on top.

Preferred form: let reranker order stand, and keep the head-insert only as a *guard* — merge in
`raw_hits[:1]` only when the reranker's top score is below a confidence floor (all-negative logits
= the reranker itself is unsure). This preserves the original intent (don't let a bad reranker
call lose a strong lexical match) without spending 3 of the top slots on it unconditionally.

Simplest form, if the guard proves hard to tune: drop the head-insert entirely in the scoped
branch. Measured evidence says this is a net win 11:2 on gold rank-1 hits.

**Measurement (M-2):** `repro3_goldrank.py` generalised into a checked-in offline script —
"gold-evidence rank@1 across every gold question whose document is in the live collection",
currently **4/18 under `retrieve()` order vs 10/18 under reranker order**. After the fix, the
*tool's own returned order* (not the reranker in isolation) must reach the reranker's 10/18 or
better. This is the "chunk relevance @ rank 1 across N questions" metric the brief asked for, and
it covers every category, not the one demo question. Then the ship gate: a full 109-question eval
replay with `hit@5`, `ocr_extraction`, `single_doc_factoid` and `figure_grounding` all not
regressed.

### M-3 — gate the term-count reordering to actual table queries (fixes FM-3)
**Scope: real behaviour change, corpus-wide; ship-gated on the same eval replay.** Sequence it
*after* M-2 and measure it separately — this repo has repeatedly lost the ability to attribute a
change by landing two retrieval edits together.
**Files:** `src/retriever.py` — `_extract_table_filter_terms` (242-261) and its use at 506 / the
final sort at ~691-703.
**Architectural direction: neither** — removing a mis-scoped heuristic, not adding a pattern.

Fire the term-count sort only when the query is genuinely a structured lookup: the extracted token
is **ID-shaped** (`\d{5,}`, or mixed alphanumeric — `_extract_table_filter_token`'s own first
branch), or the candidate pool contains `sheet_row`/`sheet_table` chunk types. Prose queries keep
hybrid fusion order.

"A `filter_token` exists" is **not** a usable gate: this session's own `RETRIEVAL_DEBUG` output
shows `filter_token='explicitly'` and `filter_token='customer'` extracted from the two prose
clause queries — any word ≥5 chars qualifies, so that condition is always true. (Those garbage
tokens are the same root shape as this failure mode and are already largely defused by the
2026-07-22 union-retrieval fix: in `repro2_rank.py` the filtered and unfiltered orders were
byte-identical, confirming the token no longer decides recall. Noting it here so it isn't
re-opened as a new bug.) Note the interaction with `qdrant_top_k = max(top_k, 100)`
on the same flag — gating shrinks the candidate pool for prose queries, which is a *second*
behaviour change riding along; keep the over-fetch unconditional and gate only the sort.

**Measurement (M-3):** the same rank@1 harness as M-2, run with M-2 already in place, so the
delta is attributable to this change alone. Expect movement on the direct-`retrieve()` consumers
specifically: stage-1 doc routing and `_per_doc_retrieval_queries`' clause-scoring probe (whose
chosen clause per doc should not change for the demo question — a regression signal if it does).
Plus a unit test asserting a prose query no longer triggers the sort and a supplier/ID query
still does.

### M-4 — measure citation-evidence precision (fixes FM-4; unblocks judging M-1/M-2/M-5)
**Scope: eval-harness addition, no pipeline risk.** Do this **first in wall-clock terms** if
anything is to be run overnight — it is what makes the other items provable.
**Files:** `eval/run_eval.py` (new metric alongside `unanswerable_metrics`), reading the existing
`gold_evidence` (`doc_id` + `quote`) already present on every row of `answer_results.jsonl`.
**Architectural direction: #2** — this is the "citation-verification / groundedness re-checking"
family, applied as *measurement* rather than as an inline generation gate.

Per answered question, compute:
- **citation coverage** — fraction of gold-evidence documents that have at least one cited source
  from that document (catches FM-1 directly: the demo question scores 0.5 today);
- **citation precision** — fraction of cited sources whose excerpt actually contains the gold quote
  (or ≥8 consecutive gold words, tolerating markdown/OCR drift — the matcher from
  `repro3_goldrank.py`);
- **uncited-answer rate** — answers that resolved zero markers while not being `Unsupported`.

All three are computable offline against data already on disk, with no regeneration spend.

**Measurement (M-4):** the metric itself is the deliverable; validate it by hand-checking it
against the four known cross-doc rows in the current `answer_results.jsonl` (it must score
`doc_001_doc_002_..._qa_1` low on coverage and `doc_003_doc_008_..._qa_1` high, i.e. disagree with
the judge exactly where the judge is known to be wrong).

### M-5 — make marker numbering globally unique across tool calls (fixes FM-5)
**Scope: narrow patch, but it touches every answer's citation rendering** — unit-test-heavy.
**Files:** `src/answer_pipeline.py` (`build_citation_map`, and wherever `collected` blocks are
accumulated per call), `src/rag_agent.py`'s `stream_agent` where the `[N]` blocks are emitted.
**Architectural direction: neither.**

Offset each call's markers by the running total (call 2 starts at `[13]`, not `[1]`), exactly as
`answer_comparison_deterministic` already renumbers, then let `build_citation_map` map *all* calls
instead of only the last. This removes the reason `build_citation_map` is restricted, and fixes
the agent fallback path that 12 of 15 gold cross-doc questions still take. It also makes the
deterministic path's own renumbering redundant rather than special.

**Measurement (M-5):** unit test with a two-call `collected` fixture where both calls contain a
`[1]`, asserting both resolve to distinct positions (fails today); plus live re-verification of
the demo cross-doc question forced down the *agent* path (`_resolve_comparison_doc_ids` returning
None), which currently produces the duplicate `[1]`. Same "reproduce live, fix, re-verify against
the exact reproduced case" pattern the repo uses throughout.

### M-6 — resolve comparison documents by LLM against the registry, with abstention (fixes FM-6)
**Scope: real addition — one extra LLM call on comparison questions only.** Do it **last**: with
M-1/M-2/M-5 landed, both the deterministic and the agent paths produce correct citations, so this
becomes an improvement rather than a prerequisite.
**Files:** `src/answer_pipeline.py`, `_resolve_comparison_doc_ids`.
**Architectural direction: #1 (more agentic), in its cheapest form** — a single constrained
resolution call, not a per-document sub-agent.

Give the model the document registry (id, filename, title, `document_summary` text) and the
question, and ask it to return exactly two ids or `NONE`. This is a **materially different signal**
from the reverted fallback: that one ranked documents by how much they dominated a broad
corpus-wide retrieve, which is precisely the signal doc_001 corrupts by volume (the same root
cause as the "unscoped retrieval reliability" item). Registry-based resolution reads document
*identity*, never retrieval scores, so doc_001's chunk count cannot influence it.

Non-negotiable gate, learned from that revert: validate offline against all 15 gold cross-document
questions **before wiring it into the pipeline**. Ship only if it resolves ≥12/15 correctly with
**zero** confidently-wrong pairs — a wrong pair is worse than the fallback, since the deterministic
path then guarantees evidence from two wrong documents.

Also worth doing regardless of M-6, cheap: the UI's multi-select source scope already hits the
precise `forced_doc_id` branch with no doc ids in the question text. Surfacing it as the
recommended flow for cross-document questions is a product answer to the same problem and needs
no backend change.

### M-7 — re-arm the corrective loop for natural-language comparisons (fixes FM-8)
**Scope: narrow patch on the trigger, plus one deliberate re-enable that must be measured.**
**Files:** `src/answer_pipeline.py` — `_missing_mentioned_docs` (111-119), `answer_one`'s
`skip_grounding_check=is_comparison` (~1476), `_comparison_incompleteness` (122-130).
**Architectural direction: #2** — this is CRAG's corrective action, already built here, currently
disarmed on the questions it was built for. Repairing it is strictly cheaper than adding a
CRAG layer on top of it.

Two changes, in this order:

1. **Widen the coverage trigger past literal doc ids.** Feed `_missing_mentioned_docs` the doc
   ids resolved for the question (the UI's `forced_doc_id` scope, and M-6's resolver once it
   exists) rather than only ids the user typed. This is the same resolution problem as M-6 and
   should reuse its output — a shared resolver, not a second heuristic.
2. **Only then reconsider `skip_grounding_check`.** The skip exists for a measured reason (the
   grounding judge flipped ~1/3 of correct comparison answers to `Unsupported`, 2026-07-15), so
   do not simply switch it back on. The defensible version: keep the skip while the coverage
   check is blind, and once (1) lands, re-test whether a coverage-verified comparison answer
   still gets falsely downgraded. If it does, the skip stays and that is a documented, measured
   trade-off rather than the current accidental one.

Note the interaction with M-1: `n_sources` is computed from `parse_sources`, so the one-slot-
per-file guarantee makes `n_sources >= 2` true whenever two documents were retrieved at all,
including when one document's only slot holds the wrong chunk. The coarse fallback trigger is
therefore weaker than it looks, and M-1 changes what it sees.

**Measurement (M-7):** for each of the 15 gold cross-document questions, log whether the
comparison retry fires and on which branch (`missing` / `n_sources < 2` / `has_partial` / none).
Today the precise `missing` branch fires on at most 3. Target: it fires on every question where
M-4's citation-coverage metric is < 1.0, and does not fire where coverage is already 1.0 (a
retry on a complete answer is a latency cost and a chance to make it worse). Unit test with a
resolved-but-unnamed doc pair asserting `_missing_mentioned_docs` now returns the uncovered id —
fails today.

### M-8 — decompose-then-retrieve, and delete the document resolver (supersedes M-6)
**Scope: architecture change that is a net deletion.** Added 2026-07-31 after the M-6 work
showed the problem was framed wrong.
**Architectural direction: #1 (more agentic), in the form the current literature actually uses.**

M-6 asks "which two documents is this question about?" *before* retrieving — route-then-retrieve.
That forces an up-front classification, which is why it is irreducibly probabilistic (measured
8/9/12/9 correct across four sweeps) no matter how the prompt or gate is tuned.

The established pattern for multi-document comparison is the inverse: decompose into
sub-questions, retrieve each **globally**, and let document identity fall out of the evidence.
LlamaIndex's multi-document agents do exactly this ("compare documents without needing to know
in advance which specific documents are relevant"), and 2026 agentic-RAG surveys list
decomposition + adaptive routing as the recurring production pattern — not document
pre-selection.

**Measured on the live corpus before proposing it** (`test_unscoped_clauses.py`): split each
comparison into its clauses, retrieve each clause unscoped over the whole collection, and read
off which document the top evidence came from — **3/3 correct**, deterministic, zero LLM calls:

```
"requires legal review or approval before contracts proceed"  -> doc_001 (59, 30, 62)
"customer-issued notices for varying the service scope"       -> doc_002 (23, ..., 11)
```

The same run exposes where the real gap is: **2 of 4 questions produced NO CLAUSE SPLIT**,
because `_split_comparison_clauses` only matches the literal connector "while the other". The
missing capability is decomposition coverage, not document resolution.

Every piece already exists here: `_llm_split_subqueries` (`src/answer_quality.py`, used on other
paths) for decomposition, `search_knowledge_base` for global retrieval,
`_comparison_incompleteness` for the >=2-distinct-documents coverage check, and
`answer_comparison_deterministic` for synthesis with globally-unique markers (post-M-5).

**Proposed change:** in `answer_comparison_deterministic`, stop requiring resolved doc_ids to
engage. Decompose the question, retrieve each sub-question unscoped, keep the union as evidence,
and require >=2 distinct documents in it before synthesising. `_retrieve_for_doc`'s forced
`FORCED_DOC_ID` scoping (the one line that creates the resolver requirement) goes away; so does
`_resolve_comparison_doc_ids_llm`, `_build_doc_catalogue`, `_entry_matches_question`, the
retry-on-abstain, and M-6's whole catalogue path. The precise `mentioned`/`forced_doc_id`
branches stay — when the user *has* named documents or scoped them in the UI, honouring that is
still correct and still deterministic.

**Risk to check before shipping:** unscoped retrieval is what the reverted 2026-07-30
content-based resolver got burned by (doc_001 dominates unrelated queries by chunk volume). The
material difference here is that this never *ranks documents* — each clause keeps its own top
evidence, and a clause is a specific topical query rather than a whole diluted question. The
3/3 above includes the case where doc_001 has 79 chunks against doc_002's 29 and still did not
crowd out clause B.

**Measurement (M-8):** the existing `test_unscoped_clauses.py` extended to every gold
cross-document question, reporting (a) how many decompose into >=2 sub-questions and (b) how
many produce evidence spanning the correct document pair — replacing M-6's resolver accuracy
entirely. Target: beat M-6's ceiling (12/17) with no nondeterminism, i.e. the same number twice
in a row. Plus M-4's citation coverage on the same questions, which is the metric that actually
expresses the product promise.

### Considered and rejected

- **Bypass the reranker inside `_retrieve_for_doc`** (one of the two options logged on
  2026-07-30). Measurement kills it: the reranker is the accurate ranker here (10/18 vs 4/18).
  Falling back to dense order would make things worse in the general case, and would have
  "worked" on the demo question only by accident.
- **Raising the 8-source cap.** Hides FM-1 for two documents, returns for three.
- **A full CRAG-style corrective loop, Self-RAG, or per-document retrieval sub-agents.** See below
  — the corrective machinery already exists here; adding another layer on top of a pipeline that
  discards correct evidence downstream would be building on the bug.
- **Any fix that synthesises a citation when none resolves** (e.g. attaching the top-ranked chunk).
  Explicitly rejected on 2026-07-23 with evidence, and this session's data reinforces it: the
  tool's rank-1 chunk was the *wrong* one for doc_002.

### Research notes: CRAG / Self-RAG / RAG-Fusion mapped onto this codebase

The honest mapping is **this codebase already has a corrective-RAG loop; it is mis-wired, not
missing.** Concretely:

| Published pattern | What already exists here | Verdict |
|---|---|---|
| CRAG's retrieval-relevance evaluator (lightweight model scores retrieved docs; confidence buckets) | `BAAI/bge-reranker-v2-m3` produces exactly this signal, per chunk, already | Present. FM-2 discards it for the top 3 slots. Use it (M-2) instead of adding a second evaluator. |
| CRAG's corrective action on low confidence (discard + re-retrieve / web search) | `_comparison_incompleteness` + `_COMPARISON_RETRY_INSTRUCTION`; `answer_quality._looks_like_bad_final_answer` → `_direct_retrieval_answer` | Present, and per-document coverage (`_missing_mentioned_docs`) is a *better* trigger for this corpus than a generic relevance threshold. No web-search fallback and none wanted — the promise is "verify against *your* documents". |
| Self-RAG's `IsSup` / groundedness reflection token | `_verify_grounded` / `_apply_grounding_check` post-generation | Present, and its history here is instructive: it was flipping *correct* comparison answers to `Unsupported` ~1/3 of the time and is now skipped for comparisons. Adding more self-critique on this axis has a measured record of net harm in this repo. |
| Self-RAG's trained reflection tokens generally | — | Rejected: requires instruction fine-tuning of the generation model. Disproportionate to the failure modes found, none of which are generation-quality failures. |
| RAG-Fusion (multi-query generation + RRF) | Hybrid dense+sparse RRF exists; multi-query fan-out does not | The one genuinely missing piece, and it is the already-logged candidate fix for the separate "unscoped/open-corpus reliability 13/20" item. Note it would be fused by RRF — the same order FM-3 currently corrupts, so M-3 is a prerequisite for it to be worth trying. |
| Citation-verification / attribution checking | `_narrow_quotes_to_answer`, `_CITATION_OVERLAP_FLOOR`, `build_citation_map` | Present at display time, absent at *measurement* time — hence M-4. |

The canonical published arrangement is: retrieve broadly → fuse → let a cross-encoder decide the
final order. This pipeline fuses, reranks, and then **partially un-does the rerank** (FM-2) on top
of an order that was itself sorted by literal term counts (FM-3). Both fixes move it toward the
standard arrangement by deletion, not addition — which is why they rank above every new pattern.

---

## Phase 3 — measurement summary (one line per mitigation)

| # | Mitigation | Proof it worked | Fails today? |
|---|---|---|---|
| M-1 | Fair source-cap allocation | Unit test on a synthetic 12+12 `collected` asserting file B's **rank-2** chunk survives (must not be judged by the demo question — M-2 would mask it); `repro_crossdoc.py` False → True, 7/1 → 4/4, already confirmed reachable | Yes (measured) |
| M-2 | Reranker order wins over pre-rerank head-insert | Gold-evidence **rank@1 across 18 gold quotes**: tool-returned order 4/18 → ≥10/18; full 109q eval replay with hit@5 / ocr_extraction / single_doc_factoid / figure_grounding not regressed | Yes (measured) |
| M-3 | Term-count sort gated to table queries | Same rank@1 harness re-run with M-2 already in; clause-selection for the demo question unchanged; unit tests for prose-query (no sort) vs supplier-ID query (sort) | Yes (measured) |
| M-4 | Citation coverage / precision / uncited-rate metric | Metric validated by hand against the 4 known cross-doc rows; must disagree with the judge exactly on `doc_003_doc_008_..._qa_1` | n/a (new instrument) |
| M-5 | Globally unique markers across tool calls | Two-call fixture where both calls emit `[1]` → both resolve distinctly; live re-run of the demo question forced down the agent path shows no duplicate `[1]` | Yes (root-caused 2026-07-30) |
| M-6 | Registry-based LLM doc resolution | Offline sweep of all 15 gold cross-doc questions: ≥12/15 correct pairs, **0 confidently-wrong pairs**, before any wiring | Yes (3/15 today) |
| M-7 | Re-arm the corrective loop (coverage trigger, then reconsider the grounding skip) | Per-question log of which retry branch fires across the 15 gold cross-doc questions: the precise `missing` branch must fire wherever M-4 coverage < 1.0 and not fire where it is 1.0; unit test on a resolved-but-unnamed doc pair | Yes (fires on ≤3/15 today) |

Overall gate for "cross-document evidence is fixed", replacing the single demo question as the
acceptance test: on the 4 `doc_001`×`doc_002` gold cross-doc questions **plus** the demo question,
every answer cites ≥1 source per document, and every cited source's excerpt contains the gold
quote for the claim it is attached to (M-4's coverage = 1.0, precision = 1.0). That is the
product promise stated as a measurement, and no current metric expresses it.
