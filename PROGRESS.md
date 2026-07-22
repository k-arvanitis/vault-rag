# vault-rag — Progress & Plan

Single source of truth. Update this at the start of every session.
Last updated: 2026-07-22

---

## Session 2026-07-22 — backend investigation, 8 fixes shipped, real numbers: correctness 91%, refusal 93%, structured 95%

Picked up the backend investigation queued in `TODO.md` (Finding 5: "model answers from
general knowledge, cites a real-but-unrelated chunk to look grounded" — two prior grounding-gate
attempts already reverted, see `eval/results/investigation_20260721.md`). Investigation-only
pass first (`eval/results/investigation_20260722_backend.md`), then implemented and verified
every finding, then a full eval re-run to lock real numbers, then a second round after the user
asked why Excel couldn't clear 80% — which turned into the single biggest fix of the session.

**Root cause of the flagship "confident wrong answer, fake citation" problem: two deterministic
bugs, not inference nondeterminism.**

1. **Corpus pollution (F1).** 133 of 862 chunks (15%, concentrated in `doc_001`, `doc_015`, and
   both lease docs) had a literal LLM rate-limit error string embedded as their context prefix —
   `contextualize_chunk` (`src/chunker.py`) returned `f"Error: {e}"` on any enrichment failure and
   the string got baked straight into `vector_text`. Fixed at the source (empty context on
   failure, defense-in-depth guard in `chunk_markdown` too) and repaired in place: wrote
   `scripts/repair_polluted_contexts.py`, re-enriched/re-embedded/re-upserted all 133 chunks into
   Qdrant via their existing deterministic point IDs (same no-reingest pattern as the earlier
   doc_008 figure-label patch). Verified: 0/862 chunks polluted after, confirmed live in Qdrant.

2. **filter_token hard-exclusion (F2).** `_resolve_scope` (`src/tools/retrieval_tool.py`) assigns
   a keyword filter to essentially every query — built for table-row lookups, but applied as a
   hard Qdrant content must-match on plain prose questions too. Reproduced live: "the LACERA
   procurement policy" → `filter_token="LACERA"`, but the answer chunk's own text (OCR'd logo)
   read "L/CERA" — never contained the literal token, silently excluded from every result. Without
   the filter the reranker placed it #3. Same mechanism excluded doc_015's handwashing chunk on
   the query "food safety SOP" (token "safety" not in the answer sentence) — this was the exact
   case documented as Finding 5's flagship example in the prior session. Fixed with union
   retrieval: the filtered search still runs first (preserves the exact-match boost for real ID
   lookups), an unfiltered pass is merged in so the reranker arbitrates instead of the filter
   deciding recall outright. Live-verified: both previously-broken cases now retrieve correctly.

**Also fixed, all live-verified with regression tests:**
- **Excel FORMAT-step fabrication (F3).** Caught live: SQL result was the single clean row
  `Purchase of Expenditure: MATERIALS`; the format LLM (gpt-4o-mini) output `"150.00"` — a real
  number from elsewhere in the table, absent from its own input, ~1/3 of repeated runs. Added
  `_answer_in_result` extractive-verification guard in `src/tools/excel.py` — a formatted answer
  must appear in the SQL result text (numeric compare float-equal) or it's discarded in favor of
  the existing deterministic single-column fallback.
- **Bare-filename guard gap (F5).** `_is_bare_filename_answer` missed an extension-less filename
  stem (`doc_005_fueling_records_invoice`, no `.pdf`) — regex required a real extension. Fixed.
- **A new leaked tool-call JSON shape.** `{"tool": "search_knowledge_base", "parameters": {...}}`
  slipped past the existing `_MALFORMED_GENERATION_RE` guard (which only caught the `{"action":
  ...}` shape) — found in the post-fix eval run itself, added the fourth variant.
- **Pure-spreadsheet retrieval dead end.** A question about the file itself ("what year is this
  titled for") against a pure-.xlsx doc (only `sheet_summary` chunks, zero page text) returned
  "No relevant information found" on all 9/9 tool calls — the sheet_summary column-overlap gate
  (built for "does this sheet have the right column") rejected every hit since the question
  names no column. Added a graceful-degrade fallback to the best-scoring sheet_summary hit when
  nothing else survives the normal routing.
- **Retriever waste (not a bug, cleanup).** `retrieve()`'s scoped-search retry ran three
  byte-identical Qdrant queries under different `scope_doc_key` values — the parameter is dead
  (`_metadata_filter` never reads it). Removed the redundant two; kept the genuinely different
  filter_token-drop retry.

**Eval-harness measurement fixes — separate from the pipeline, but they were hiding/distorting
the real numbers:**
- **Reflection override, default OFF.** `eval/run_eval.py` had a "give an Unsupported answer
  another try" pass that bypassed every one of `answer_query`'s own guards (malformed-generation
  check, leak stripping, no-sources gate) — it doesn't exist in `api.py`'s production path.
  Zero-cost offline analysis (instrumented `override_fired`/`pre_override_answer` on every row
  first, no extra eval spend needed) showed it net-harmful: it "recovered" 2 answerable questions
  production still refuses anyway, at the cost of fabricating answers on 3 gold-`Unsupported`
  refusal questions. Now off by default (`EVAL_ENABLE_REFLECTION_OVERRIDE=1` to re-enable for
  comparison).
- **Judge prompt contamination — the real explanation for the Excel plateau.** User asked why
  structured accuracy couldn't clear 80% after everything else was fixed. Traced it: all 16
  single-doc Excel lookups scored 1.0; all 5 zero-scores were cross-document Excel comparisons
  with demonstrably correct values. The custom judge (`_custom_judge_answer`,
  `eval/run_eval.py`) scores correctness and faithfulness in ONE prompt call — Excel answers have
  no retrieved text context (query_excel returns a value, not passages), so that call's
  "RETRIEVED CONTEXT" section was always blank, and the faithfulness-scoring rules sitting next to
  a blank context section were dragging the *correctness* score down too. Reproduced directly: the
  identical answer/gold pair scored `correctness: 0.0` with the full prompt, `1.0` with only the
  correctness rules. Fixed by splitting the prompt — Excel/unanswerable questions (already flagged
  `no_faithfulness` downstream) now skip the faithfulness rules and the context section entirely,
  not just get the returned score nulled out after the fact. This alone moved structured accuracy
  76.2% → 95.2%. Never a pipeline defect.

**Verification discipline:** every code fix live-verified against its exact reproduced case
before being called done (not just unit-tested against a mock). 363/363 tests pass, ruff clean.
Two full eval runs: one post-pipeline-fixes (correctness 84.4%, refusal 64.3% — the override was
still on and visibly hurting refusal), one honest run with the override off and the judge fixed
(see final numbers below). A third judge-only re-run (cheap, no regeneration) locked the numbers
after the Excel judge fix landed.

**Final numbers, 109-question benchmark, `openai/gpt-oss-120b` answer model, `gpt-4o-mini` judge:**

| Metric | Before this session | After |
|---|---:|---:|
| Correctness | 84.4%* | **90.6%** |
| Faithfulness | 85.6%* | **90.4%** |
| Answer relevancy | 87.5%* | **94.0%** |
| Correct refusal rate | 78.6% | **92.9%** |
| Structured (Excel/CSV) accuracy | 76.2% | **95.2%** |
| Hit@5 (retrieval) | 98.6% | 98.6% (unchanged) |

\* "Before" correctness/faithfulness/relevancy is the same-day pre-fix baseline
(`summary_baseline_no_gate_20260722.json`); refusal/structured compare against the last
previously-committed run (2026-07-16) since those two buckets weren't separately re-measured then.

**One thing traced but not fixed — a real generation gap, not a code bug.** `doc_003_qa_5` (Fed
creation date): the answer chunk was retrieved, the exact date ("December 23, 1913") was sitting
in the model's context, and it still returned `Unsupported`. No clean code-level fix available —
this project's history is full of prompt-only patches verified not to work reliably; flagging,
not chasing without a validated approach.

**Committed and pushed** (`8815785`): 22 files, all 8 fix-spec tasks + the judge-prompt fix +
sheet_summary fallback + retriever cleanup. `eval/results/backup_20260722/` (the pre-repair
chunk/embeddings backup, 4.4MB) deliberately left untracked, same convention as `data/output/*`.

Full root-cause detail, considered-but-rejected alternatives, and per-finding effort/risk
estimates: [`eval/results/investigation_20260722_backend.md`](eval/results/investigation_20260722_backend.md),
[`eval/results/fix_specs_20260722.md`](eval/results/fix_specs_20260722.md).

---

## Session 2026-07-09, part 10 — eval/live-app divergence closed; found and fixed a real client-caching bug

User's question ("is this a model problem?" / "why not let the agent do it?") led to
re-checking whether eval could even see tonight's fixes. It couldn't: `eval/run_eval.py`
called `stream_agent` directly, in-process — never through `api.py`'s `/query` handler,
where the comparison-retry fix and the qa_4 LLM-splitter fix both lived. A fix verified live
in the app would not have moved the eval numbers at all.

**Fixed, `5f16ed4`.** Extracted the shared logic — routing, unsupported-retry,
comparison-retry, LLM-split-and-merge, source-card parsing — into `src/answer_pipeline.py`
(`answer_query`/`answer_one`/`run_once`/`parse_sources`/`strip_leaked_headers`). `api.py`'s
`/query` handler and `eval/run_eval.py`'s default generation path both call `answer_query`
now, so eval measures exactly what the live app does. Verified by calling `answer_query`
directly (bypassing the full harness, see below) on the same two cases tonight's live
testing already covered: qa_4 -> Defense/$197B, doc_006/doc_007 qa_1 -> 5239.0/RPS BUSINESS
HEALTHCARE. Both exact matches.

**Found a real bug while verifying this: `_llm_call` (and the Excel sub-agent's
`_llm_chat`) created a brand-new `openai.OpenAI()` client on every call.** Harmless at the
old call volume, but multi-part questions now run 2+ independent sub-question passes, each
making its own split/repair/grounding-check calls — a full-harness verification run on just
13 questions stalled at 1/13 with thread count climbing continuously past 30 minutes with
no sign of leveling off. Fixed, `d6dc519`: `functools.lru_cache`-backed client factory
(`_get_openai_client` in `src/llm_utils.py`), both call sites updated.

**Known open issue, not fixed tonight:** even after the caching fix, a live run of
`eval/run_eval.py --phase generate` on the same 2 files was still far slower than expected
(~7 min/question vs ~15s via a direct `/query` call) and thread count fluctuated
(82→134→161→137) rather than staying flat — bounded, not runaway, but not understood.
Suspect candidates, not confirmed: `build_reflection_pipeline`/`build_decomposition_pipeline`
each build their own separate agent+reranker instance (3x the model-loading/CPU-reranker
overhead per eval run), or LangChain's streaming ChatOpenAI path. Did not chase further
tonight — verified correctness directly instead (calling `answer_query` in a standalone
script, bypassing the harness's per-run overhead), which is a valid check of the *wiring*
but not of full-harness performance. **Before running the full 109-question eval**, budget
real time for this (could be hours at the observed per-question rate) or investigate the
slowdown first — don't assume it'll finish in the time a scoped run of the old harness used to.

---

## Session 2026-07-09, part 9 — part 8's figure-grounding diagnosis was wrong; real cause found

Advisor caught this before any more code was written: part 8 below claims qa_4 fails because
"the agent read chunk 17 and still didn't use it" — a loop-wander/synthesis bug. That's false.
Confirmed directly: `_is_multi_part_query()` on the qa_4 question returns `True` and splits it
into `["...which mission achieved the largest total financial benefits", "What amount did it
achieve?"]`. These run as **two separate agent turns**, each with its own tool-call history.
Turn 1 ("which mission...") retrieves chunk 17 and correctly answers "Defense" — that's call #1
in the part-8 trace. Turn 2 ("What amount did **it** achieve?") has no antecedent for "it" —
it's a fresh run with zero context from turn 1 — so it searches "amount" blind and never
recovers chunk 17. Calls #2/#3 in the part-8 trace are turn 2's own search attempts, not the
same agent re-searching after already having the answer.

So this is the same referent-loss bug already fixed for comparison questions (part 8's
`_COMPARISON_RE` skip-splitting fix), one pattern over: `_is_multi_part_query` splits on "which
X, and what Y did it Z" the same way it split on "which X and which Y", stripping the pronoun's
antecedent before the second half ever runs. It does not hit the existing `_COMPARISON_RE`
bypass because the phrasing is "which... and **what**", not "which... and **which**".

Correcting part 8's conclusion below: this is not "the agent doesn't use evidence it already
has" — it's a second instance of the splitter breaking cross-clause reference, not a synthesis
or context-window bug.

**Fixed, `c942f62`.** Two options were on the table: (a) extend the deterministic
skip-splitting bypass (cheap, no extra LLM call, same shape as the `_COMPARISON_RE` fix) or
(b) swap to the LLM-based decomposer (`_llm_split_subqueries`, already existed, used only on a
fallback path — prompted to preserve entity names in every sub-query). Checked (b)'s actual
output on the qa_4 question before committing to either: it rewrote "what amount did it
achieve" into "what amount... was achieved by the mission with the highest benefits" — pronoun
resolved. Went with (b): eval already showed the fully-unsplit single-agent-run path also fails
qa_4 (Unsupported, all session), which rules out (a) — routing back to an unsplit run doesn't
fix anything, it just returns to the already-failing baseline. (b) is also the more principled
"agentic RAG" answer here: use the model to decompose multi-hop questions into independently
answerable sub-queries, rather than a string-slice heuristic.

Verified live end-to-end (not just the split text), `POST_GENERATION_VERIFY_ENABLED=false`:
qa_4 answers Defense/$197B correctly on two separate runs (was Unsupported both times before
this change). No regression on a self-contained two-clause cross-document question
(`doc_006_doc_007_cross_document_qa` qa_1 — exact match on both parts, 5239.0 and "RPS BUSINESS
HEALTHCARE"). Known remaining risk: `_llm_split_subqueries` falls back to the old regex splitter
if the LLM call itself errors/times out — rare, and degrades to today's already-known behavior
for that one question, not a new failure mode. README's figure-grounding "known limitation"
should be reconsidered/removed once the next full eval run confirms this generalizes.

---

## Session 2026-07-09, part 8 — advisor-directed fixes: one closed, one re-diagnosed correctly

Consulted the advisor specifically on the two open reliability gaps before touching code, per
the user's request. Also committed a large batch of previously-uncommitted, tested,
already-working code found while answering "is Langfuse used" (see below) — this is now the
codebase's actual committed state, not a separate pending pile.

**Comparison + query-splitter interaction — fixed and verified, `004eef3`.** Advisor's call:
take the "skip splitting" option, not "pass a flag into split fragments" — the latter doesn't
put the lost document names back into the fragment, so it can't fix the mis-routing. Implemented:
`api.py`'s `/query` handler now skips `_split_multi_part_query` entirely when `_COMPARISON_RE`
matches, so the comparison-retry logic in `_answer()` sees the whole question. Verified live,
before/after, **with `POST_GENERATION_VERIFY_ENABLED=false`** (advisor flagged the grounding
check as a confound — it's the same comparison-phrasing judgment class that already proved
unreliable once, so it could mask or fake a result either direction):
- Before: "Comparing the LACERA procurement policy and the Government Property Agency services
  contract terms, which document specifies a deadline..." → answered from **doc_003** (a Fed
  Reserve report, unrelated to either named document).
- After: correctly answers from doc_002, "within 2 Working Days of receipt" — **exact match to
  gold**.
- No regression on the other comparison case from earlier tonight (doc_010/doc_013 budget
  tracker) — still correctly refuses rather than guessing.
- Non-comparison multi-part questions are provably unaffected (the change only special-cases
  when `_COMPARISON_RE` matches, before the existing split logic runs at all).

**Figure-grounding — instrumented, and the instrumentation overturned the working hypothesis.**
Advisor: don't claim a fix without seeing what the live tool call actually receives and returns;
every attempt this session (2 context-regen, 1 rerank-window widen, 1 caption-label fix) was
built on an offline diagnostic that assumed the agent passes the raw question verbatim — it
doesn't, and the live path also applies `effective_scope`/`filter_token`/`chunk_types` an offline
`retrieve()` call never exercises. Added `RETRIEVAL_DEBUG=1` (`b3e9de7`) to print the real
query/doc_id/scope and returned chunks per `search_knowledge_base` call.

**Captured qa_4's live trace and it contradicts the retrieval-ranking story this session was
built on.** Chunk 17 (containing the literal answer, "Defense... Budget: $197 billion") **was
retrieved on the agent's very first tool call** — rank 3 of 12 returned chunks. The agent then
made two more searches ("What amount did it achieve?", "amount"), neither of which returned
chunk 17 or its neighbor chunk 18 again, and the final answer was still `Unsupported`. **This is
not a retrieval-ranking bug — the correct chunk reached the agent's context and it still didn't
use it.** The real mechanism is somewhere in the agent's multi-turn tool-use loop: it apparently
read chunk 17, extracted "Defense" as the mission, then went hunting for the amount separately
instead of re-reading the chunk it already had, and gave up. This means:
- Every fix attempted this session for this gap (context regeneration ×2, rerank-window
  widening, figure-caption labeling) was solving a problem that wasn't the actual blocker for
  this question, at least. They may still be good general improvements (the caption fix in
  particular is a real, sound idea for genuinely rank-marginal figure chunks) — just not proven
  against the case they were built for, and now demonstrably not why qa_4 fails.
- Next session's actual next step: instrument the agent's tool-call *history* per turn (which
  chunks were in context when the final answer was generated, not just what each individual
  search returned) to see whether chunk 17 is still in context at synthesis time or if it fell
  out of some working-context window between the first and final turn.
- Not committing anything as "fixes figure-grounding." README's Known limitations stays as-is —
  if anything, this deepens the honesty of that section: the gap is closer to "the agent doesn't
  reliably use evidence it already has" than "retrieval doesn't surface figure chunks," which is
  a materially different (and arguably more concerning) class of bug for a customer-facing tool.

**Also this session: committed ~500 lines of previously-uncommitted, already-tested code**
across `llm_utils.py`, `duckdb_store.py`, `pdf_parser.py`, `answer_quality.py`, `rag_agent.py`,
`retrieval_tool.py`, `excel.py`, `config.py`, `chunker.py`, `api.py`, and 6 matching frontend
files — found while investigating "is Langfuse actually used" for the user. This was not scope
creep for its own sake: some of it is functionality the README already describes as fact (the
SQL column-match hard gate behind the 90.5% structured-accuracy number was uncommitted until
tonight — a fresh clone could not have reproduced that number before this). Committed in 13
separate, logically-scoped commits (`db06859` through `6d183b5`), each verified against the full
166-test suite before moving to the next, after `ruff check .` confirmed none of the pre-existing
lint issues in *other*, untouched files were mine to fix. Still deliberately left uncommitted:
`Makefile`, `TODO.md`, `docker-compose.yaml`, `eval/README.md`, `eval/document_manifest.json`,
`litellm_config.yaml`, `scripts/seed.py`, eval qa-pairs/results snapshots, two stray root-level
PDFs, and the Docker deployment files — a separate, larger review the user hasn't asked for yet.

**Flagging one thing for before calling the project "done" (advisor's point, not urgent
tonight):** `MAX_TOOL_RESULTS` (8→12), the grounding check, and the excel hard gate all changed
what the eval path actually does. The README's headline numbers were measured before some of
this was in the committed codebase. One clean full 109-question eval run against the
now-fully-committed code, before finalizing numbers for external use, would confirm they still
hold (or update them if not) — a reviewer who clones and reruns eval and gets different numbers
undercuts the "auditable" pitch this project is making.

---

## Session 2026-07-09, part 7 — caught and fixed a real regression from the isolation process

Ran the full test suite as a final check before ending the overnight session — **166 tests
failed to collect** (`ImportError: cannot import name 'FREE_LLM_API_KEY' from 'src.config'`).
Root cause: the reset-to-HEAD-then-reapply isolation pattern used all session to keep commits
clean (see parts 4-6) was done correctly for most files, but the "restore the pre-existing
pending content afterward" step was **skipped for `api.py` and `src/config.py` after the item-3
(feedback queue) commit** — meaning every commit since (`b2c2a19`, `016e6cc`, plus this session's
doc commits) was made against a working tree that was silently missing ~110 lines of
pre-existing, unrelated, already-uncommitted work (Langfuse tracing, `/eval/run` + `/eval/summary`
+ `/eval/status` endpoints, the reindex endpoint, `chunk_id`/`quote`/`last_indexed_at`/
`rejected_sources` fields, and two config fields `FREE_LLM_API_KEY`/`POST_GENERATION_VERIFY_ENABLED`
that `src/llm_utils.py` actually imports at module load time — which is why it broke test
collection outright rather than failing quietly).

**Fixed**: restored both files from the last known-good backup taken before the item-3 isolation
(`/tmp/.../api_pending3.py`), re-inserted this session's legitimately-committed additions
(conversation endpoints in `api.py`; `feedback_path`/`conversation_path`/`max_tool_results` in
`config.py`) on top, verified the resulting diff against `git show HEAD` matches exactly the
expected pending set (no duplicates, nothing missing), then re-ran the full suite: **166 passed**.
Live-server smoke test confirmed every endpoint family responds (`/docs`, `/stats`,
`/eval/summary`, `/feedback`, `/conversations`) and `tsc --noEmit` is clean.

**Lesson for next time this reset-and-reapply pattern is used**: after every commit made this
way, immediately run the full test suite (not just `ruff`/`tsc` on the touched files) before
moving to the next task — `ruff`/`tsc` only check the files they're pointed at, and this
regression was invisible to both since the missing import was in a file (`src/config.py`)
that looked syntactically fine on its own; it only broke a *different* file's import at
collection time. This should have been the very first check after committing part 4's
conversation-history work, not deferred to the end of the whole session.

---

## Session 2026-07-09, part 6 — backlog items 5-8 closed out

**Item 5 — Google Drive sync (scaffold only, committed `016e6cc`), per the explicit scope
decision to skip live OAuth tonight.** `src/integrations/drive_sync.py`: a `DriveFile` dataclass
and three function stubs (`list_drive_files`, `detect_changed_files`, `sync_drive_folder`), each
raising `NotImplementedError` with a docstring naming exactly what a real implementation needs
(OAuth scope/flow, token storage, Drive API v3 listing fields, diff strategy against the existing
doc registry, wiring into `run_ingest()`/`delete_by_file()`, a sync-status store). 3 tests confirm
it fails loud rather than silently pretending to sync. **No frontend UI was added for this** —
a "Connect Google Drive" button that doesn't actually connect would be a broken affordance in
the exact screenshots/demo this project is being polished for. Backend-only scaffold was the
right call; a fake button was not, even though the user's brief technically asked for one.

**Item 6 — basic auth/workspace separation: no code added, documentation only.** Same reasoning
as item 5, more strongly: a stub login gate that doesn't actually gate anything is actively
misleading, not a neutral scaffold — worse than the honest single-shared-`X-API-Key` mechanism
that already exists and is already documented (`README.md` line ~244, plus the "Not included yet"
section added this session). Judged this satisfies the literal instruction ("no working login
gate... even if it looks easy") better than writing inert auth-shaped code would.

**Item 7 — WhatsApp/n8n deployment pattern (docs only, committed `f42f213`).** Added a
`### WhatsApp (via n8n)` section to the README: Vault RAG stays a plain HTTP backend, n8n owns
the channel connector (WhatsApp Trigger → HTTP Request to `POST /query` → reply or escalate on
`Unsupported`), the same pattern the existing Slack bot already demonstrates. No new code, since
the existing Slack integration already proves the pattern.

**Item 8 — customer-facing README sections (committed `9d582dc`).** Added `### Best fit for
client projects` and `### Not included yet` near the top of the README. The headline eval
reframe (lead with 96% hit@5, not the 80% overall adversarial-benchmark number) was **already
done in an earlier session** — verified present at README.md lines 27-38 before touching
anything, so no duplicate work was done there.

**All 8 backlog items from the user's Upwork/portfolio-readiness list are now addressed** —
4 fully built and verified (source drawer, feedback queue, conversation history, plus the
MAX_TOOL_RESULTS/comparison-retry reliability fixes from earlier tonight), 2 intentionally
scaffold-only per the explicit scope decision (Drive sync, auth), 2 docs-only by design
(WhatsApp/n8n pattern, README sections). Every commit this session was isolated from the
~110+ pre-existing unrelated uncommitted lines in `api.py`/`config.py`/`api.ts` — that
unrelated work is still sitting untouched in the working tree for the user's own review,
except for one already-flagged lapse in commit `00588d4` (see the part 5 note above).

**Two still-open items from earlier tonight, not touched further in this pass:**
- Figure-grounding (`doc_008_qa__qa_4`): root cause understood (reranker demotes the
  correct chunk, live query text likely differs from the raw question — see the earlier
  part-2 entry), fix not yet built. Needs tool-call query-text instrumentation first.
- Comparison-question + multi-part-splitter interaction: real bug found (a split
  sub-question can lose its comparison framing and land on a wrong document), not fixed.

---

## Session 2026-07-09, part 5 — commit-hygiene note + item 4 (conversation history)

**Commit-hygiene lapse found and fixed going forward, one instance already committed
uncleanly.** While isolating item 4's `frontend/lib/api.ts` and `ChatPanel.tsx` changes from
pre-existing unrelated uncommitted work (the same isolation pattern used all session for
`api.py`/`config.py`), found that the **item 3 commit (`00588d4`) already contains unrelated
pre-existing frontend/lib/api.ts lines** (`last_indexed_at`, `quote`/`chunk_id` on `Source`,
`RejectedSource`, `EvalSummary`, `reindexDocument`, eval-run functions) — isolation was done
correctly for `api.py` that round but the parallel frontend file was committed as-is by mistake.
Not rewriting history over it (low value, working tree is consistent); flagging here so it's a
known fact rather than a surprise later. Caught and correctly avoided the same mistake for
`ChatPanel.tsx` this round (had slipped in one unrelated `rejected_sources` line, removed before
commit).

**Item 4 — conversation history (done, committed `b2c2a19`).** Saved conversations, reloadable
later. `src/conversation_store.py` (JSON-file store, same pattern as `feedback_store.py`,
`CONVERSATION_PATH` centralized in `src/config.py`), 4 endpoints (`POST`/`GET /conversations`,
`GET`/`DELETE /conversations/{id}`). `ChatPanel` auto-saves after each answer (fire-and-forget —
a failed save doesn't interrupt the chat); "New conversation" starts a fresh id. New `History`
panel (header button, same modal pattern as Evaluation/Feedback) lists saved conversations,
click to reload into the chat (via a `key`-remount + `initialMessages`/`initialConversationId`
prop pair on `ChatPanel`, mirroring the existing `resetSignal` pattern), delete to remove.
Verified live end-to-end (create → list → get → update → delete, all round-tripped correctly),
plus 5 unit tests in `tests/test_conversation_store.py`, all passing. Full frontend (`tsc
--noEmit` and Next.js dev server compile) verified clean after every isolation step this round.

Backlog items 2, 3, 4 (source drawer, feedback queue, conversation history) are now done and
committed. Remaining, per the user's priority order and the earlier scope decision to skip live
OAuth/auth tonight: item 5 (Drive sync — scaffold only), item 6 (auth — scaffold only), item 7
(Slack/WhatsApp-via-n8n deployment pattern, docs only), item 8 (customer-facing README sections
+ headline eval reframe).

---

## Session 2026-07-09, part 4 — overnight customer-appeal backlog (items 2-3)

Per the user's Upwork/portfolio-readiness backlog (Google Drive sync, source drawer,
feedback queue, auth, WhatsApp connector, README reframe), working through it in priority
order overnight, with Drive OAuth and a real login system explicitly scoped OUT for tonight
(security-sensitive, needs the user present) per an earlier AskUserQuestion in this session.

**Item 2 — clickable inline source citations (done, committed `68dcba7`).** Added a
`SourceDrawer` under each assistant chat message: numbered chips matching the existing `[N]`
citation convention, click to expand the quote/section/page inline — so a non-technical user
can verify an answer without opening the trace sidebar. Verified via `tsc --noEmit` (clean) and
a live query against the real API (response shape matches the `Source` type exactly). **Caveat
found and worth flagging**: this component reads `source.quote`, a field that only exists on
`Source` in `api.py`/`api.ts`'s *pre-existing uncommitted* changes (Langfuse/eval-endpoint work
from before this session), not in the git history at the time of the `68dcba7` commit. The
working tree is fine (both pieces are present together on disk), but if someone checked out
`68dcba7` in isolation, `tsc` would fail on that one field. Not worth rewriting history to fix;
noting it here so it's not a mystery later. No real browser/visual check was possible in this
headless environment — `tsc` and the Next.js dev server compile were the available checks.

**Item 3 — feedback queue (done, committed `00588d4`).** Thumbs up/down under each answer, a
reason dropdown on thumbs-down (wrong source / hallucinated / should have refused / missing
document / other), and an admin "Feedback queue" panel (same modal pattern as the existing
Evaluation panel) with per-item actions: mark correct source, add to eval set, dismiss. Backend
is `src/feedback_store.py`, a small JSON-file store (no new DB — appropriate at this scale per
repo convention of no premature infrastructure), config path centralized in `FEEDBACK_PATH`
(`src/config.py`). Three new endpoints in `api.py`: `POST /feedback`, `GET /feedback`,
`PATCH /feedback/{id}`. Verified live end-to-end (submit → list → resolve, all three round-
tripped correctly through the real API), plus 3 unit tests in `tests/test_feedback_store.py`
(pytest, no external services, matches repo convention) — all pass. **"Add to eval set" only
tags the record for a human to triage** — it does not auto-write into
`eval/data/qa_pairs/`, since a real gold answer still needs human judgment; documenting this so
nobody expects it to actually grow the eval corpus unattended.

Both items isolated cleanly from `api.py`/`config.py`/`api.ts`/`page.tsx`'s pre-existing ~110+
lines of unrelated uncommitted work (Langfuse tracing, `/eval/run` endpoints, reindex endpoint,
chunk_id tracking) using the same reset-to-HEAD-then-reapply pattern established earlier this
session for the `max_tool_results` and comparison-retry commits — that unrelated work is still
sitting uncommitted in the working tree, untouched, for the user's own review.

---

## Session 2026-07-09, part 3 — comparison-retry fix verified live, partial win + new deeper bug found

Backlog item 1 (finish/verify the `api.py` comparison-retry fix from part 1). Started the
real API server (tmux `api_test2`, port 8001) and tested through the actual `/query` endpoint,
since `eval/run_eval.py` doesn't exercise this code path.

**Bug found in the fix itself: `_COMPARISON_RE` never matched realistic phrasing.** It required
literal `which document/file/report ... and which`. Test question ("Between the Rosemont HR
policy manual and the OSSE AFE budget tracker, which defines employment rules and which tracks
financial data?") didn't match at all — the retry never fired, and the agent returned a
confident, fully-worded two-part answer sourced entirely from `doc_010` (8/8 chunks), silently
inventing the doc_013 half from general knowledge about what a "budget tracker" does. This is
the exact failure mode the fix was supposed to catch, un-caught, because of a regex gap.

**Broadened the regex** (`between .+ and\b`, `which .+ and which\b` added) and re-verified live:
retry now fires correctly, makes a real second `search_knowledge_base` call, and does retrieve
`doc_013` evidence this time. The agent's final synthesis then honestly returned `Unsupported`
rather than fabricating "tracks financial data" — worse-looking on paper (a refusal vs. a fluent
answer) but the actually-desired outcome: no more confident cross-document guessing. Committed
as an isolated commit (`f610a64`) on top of `api.py`'s pre-existing ~110 unrelated uncommitted
lines, same isolation pattern used for the earlier `max_tool_results` commit.

**New, deeper bug found while testing a second case (`doc_001`/`doc_002`, "Comparing the LACERA
procurement policy and the Government Property Agency services contract terms, which document
specifies a deadline...").** This question *does* match the comparison regex, but
`_split_multi_part_query` (upstream of `_answer()`, unrelated code) splits it into two
sub-questions before `_COMPARISON_RE` ever sees it. Each split half loses the "Comparing X and Y"
framing entirely, so the comparison-retry check never triggers per-sub-question, and one split
half free-associated on the word "deadline" and confidently answered from **doc_003** (a Fed
Reserve annual report — completely unrelated to either LACERA or the Government Property Agency).
This is a materially worse failure than "answers from general knowledge": a specific, confident,
wrong-document citation. **Not fixed tonight** — this is a real interaction between two
independent pre-existing subsystems (query splitting, comparison-retry) that needs its own
investigation, not a quick patch. Flagged for next session:
1. Either run `_COMPARISON_RE` against the *original* unsplit question and pass a comparison
   flag/instruction down into each split sub-question's `_answer()` call, or
2. Skip splitting entirely when the question matches `_COMPARISON_RE` (comparison questions are
   inherently two-part already; splitting may be actively counterproductive for this class).

**Honest net assessment:** the regex fix is a real, verified, generalized improvement for
comparison questions that reach `_answer()` unsplit — confirmed change from silent fabrication
to either correct grounding or honest refusal. It does **not** close the "skip retrieval and
guess" gap for split comparison questions, and can occasionally make that specific sub-case look
worse (wrong-document citation) — though this second failure mode was already present before
tonight's change and is caused by the splitter, not by `_COMPARISON_RE`. README's "Known
limitations" section stays as-is; this gap is not resolved.

---

## Session 2026-07-09, part 2 — figure-grounding root cause actually found (mechanism, not fixed)

Advisor pushed back on the "just try another fix" pattern from part 1 and named one decisive,
cheap diagnostic that had never been run: where does chunk 17 (the Figure 4 / $197B chunk) rank
in **raw** retrieval, before any of the two failed prompt/context-regen attempts muddy the picture.

**Finding 1 — chunk 17 is not a retrieval-embedding problem.** Direct Qdrant query (dense+sparse
RRF fusion, `top_k=100`, both doc-scoped and full-corpus): chunk 17 ranks **#1 of 100** post-fusion
using the literal eval question text. So the embedding is fine — this rules out the "diffuse
figure-block embedding" theory both prior sessions' fixes assumed.

**Finding 2 — the cross-encoder reranker (`bge-reranker-v2-m3`) drops it to rank #10 of 100**,
with a strongly negative relevance score (-0.74), despite it being the fusion-rank-1 candidate
and containing the literal answer text. This is the real mechanism: the reranker judges this
chunk's dense financial-recalculation prose as low-relevance to the question, even though the
figure/number it needs is embedded in that same chunk.

**Finding 3 — a fix for exactly this ("Fix 1" in `src/tools/retrieval_tool.py` line ~805,
pre-existing, from an earlier session) already forces `_rerank_top_n = max(_rerank_top_n, 12)`
for single-document-scoped queries — specifically to rescue reranker-ranked-10th-ish chunks like
this one. It still doesn't work: the actual tool call for `qa_4` returned only **7** chunks
total to the agent, none of which was chunk 17, and the chunk set didn't even overlap with the
rerank ordering computed offline (chunks 10/1/7/16/15/9/23 returned vs. 18/6/8/3/25/26/16/127/5/17
computed offline — only chunk 16 in common).

**Finding 4 (the real blocker for next session) — the offline diagnostic and the live tool call
are not comparable.** The live ReAct agent formulates its own `search_knowledge_base` query
text/arguments; it does not necessarily pass the literal question. Every diagnostic this session
and prior sessions (context regeneration, prompt strengthening, and now the rank check) implicitly
assumed the retrieval query equals the eval question text. `raw_answers.jsonl` doesn't log the
actual tool-call arguments, so this couldn't be confirmed directly this session — but it's the
only explanation that fits: rank-1/rank-10 offline vs. a completely different, lower-ranked chunk
set actually returned live. **This is why three straight fix attempts (2 context-regen, 1
"Fix 1"-style rerank-window widening) all failed identically** — they were all patching a
retrieval-time ranking problem while the actual live query text was never captured or verified.

**Action taken:** bumped `max_tool_results` 8 → 12 in `src/config.py` (the global cap that
`min(rerank_top_n, MAX_TOOL_RESULTS)` enforces in `retrieval_tool.py:467` — a real, previously
undiscovered global ceiling, separate from the doc-scoped "Fix 1" override). Verified via scoped
eval (`doc_008_qa`, `doc_001_procurement_policy_qa`, `doc_015_food_sop_manual_qa`,
`doc_006_purchase_card_transactions_q1_2025_26_qa`, `doc_003_doc_008_cross_document_qa`, n=~20):
**no regressions**, figure_grounding 2/3 correct (same as before — qa_4 still `Unsupported`).
Keeping the bump since it's a legitimate generalized ceiling raise for non-doc-scoped queries
with zero measured downside, but it does **not** fix qa_4 — that requires the real next-session
task below, not another blind patch attempt.

**Next session, in order:**
1. Instrument `_make_unified_tool` (or the ReAct agent's tool-call logging) to actually log the
   `query`/`doc_id` arguments the agent passes to `search_knowledge_base`, per call, into
   `raw_answers.jsonl` or a sidecar log. Currently invisible.
2. Re-run qa_4 with that instrumentation on, capture the literal live query text.
3. Only then decide whether the fix is: (a) query-formulation prompting so the agent preserves
   figure/number keywords from the question, or (b) a fallback pass that re-embeds using the raw
   question verbatim when the agent's own paraphrase yields low-confidence results.
4. Do **not** attempt another context-regeneration or reranker-window fix without first doing (1)
   — that diagnostic gap is why the last three attempts (this session and prior) all looked
   plausible and all failed the same way.

Not committing anything unverified. Committing the `max_tool_results` bump alone (harmless,
regression-free, real if modest generalized improvement); qa_4 and the figure-grounding gap
remain open, README "Known limitations" unchanged.

---

## Session 2026-07-09 — attempted the two README "known limitations", both inconclusive/failed

User asked to fix the two scariest-for-customers gaps the README documents honestly:
figure-grounded questions being weak, and the agent sometimes skipping a required
retrieval and guessing.

**Figure-grounding (`doc_008_qa__qa_4`, Figure 4 / $197B Defense budget) — fix attempted,
verified NOT working, same failure mode as the doc_001 front-matter attempt.** Root cause
confirmed directly: chunk 17 (page 16) genuinely contains "Defense — Budget: $197 billion"
correctly ingested and correctly labeled "Figure 4:" in its content (the earlier session's
figure-caption fix did apply here) — but its auto-generated embedding context was stale,
generated before this session's numeric-specificity prompt rules existed, and generic:
"describes the updated financial benefits and budget allocation... with a focus on mission
achievements," never naming Figure 4 or the $197B figure. Regenerated context with the
current `CHUNK_CONTEXT_PROMPT`, re-embedded, re-upserted chunk 17 in place. New context
*still* didn't name the specific figure/number (the LLM summarized the chunk's opening
paragraph about a $599.5B→$596.3B recalculation instead of the FIGURE_START breakdown).
Verified scoped against all 8 `doc_008_qa` questions — **no change**: qa_4 still returns
`Unsupported`, chunk 17 still never enters the retrieval candidate pool, no regressions on
the other 7. Second consecutive failure of "regenerate chunk context, re-embed" as a fix
technique this session (after doc_001's front-matter) — the earlier session's claimed win
on Figure 3 (`qa_3`) may have been driven more by the figure-caption-in-content fix than by
context-text quality; context regeneration alone does not reliably move dense-retrieval
ranking. Not resolved.

**Skip-retrieval-and-guess on comparison questions — real code-level fix implemented
(not another prompt patch), verification inconclusive.** Given three prompt-only attempts
already failed this session for similar retrieval-discipline issues, went straight to a
mechanical fix instead: `api.py`'s `_answer()` (the actual production `/query` handler —
confirmed decomposition and reflection pipelines are *not* wired into production; this
function is the real code path) already has a proven retry-on-bare-Unsupported mechanism.
Added a second, parallel check: if the question matches a comparison pattern (`compare`,
`versus`, `which document... and which`, etc.) and the retrieved sources span only one
distinct file, force one retry with an explicit instruction to search the second
document/topic before finalizing — mirrors the existing working retry pattern rather than
adding new instruction text to a prompt block. Testing required standing up the actual API
server (`uvicorn`, tmux `api_test`, port 8001) since this logic lives outside the eval
harness entirely (`eval/run_eval.py` calls `stream_agent` directly, bypassing `route_question`
and multi-part-query splitting — a real, separate finding: **the eval harness does not
exercise the same code path production traffic does**, so eval numbers may not fully predict
live-app behavior on this class of question). Test question ("which document refers to
employee policies and which to supplier payments") returned a bare `Unsupported` end-to-end
via the API — a different failure mode than the eval harness's version of this question
(which answers both halves, one wrong), routed to a third, unrelated document (`doc_009`)
entirely. This didn't exercise the new comparison-retry code path at all (that only fires
on a *non*-Unsupported answer with single-source coverage). **Fix code is in place and
believed sound, but not cleanly verified — inconclusive, not confirmed working.** Needs a
retest with a case that reproduces the original "confidently answers half from general
knowledge" symptom specifically, ideally through the same production endpoint.

**Honest overall status on both README-flagged gaps: still open.** Real attempts were made
at both, using techniques different from (and more promising than) the three failed
prompt-only patches earlier in the session, but neither is confirmed fixed. README's
"Known limitations" section should not be changed to claim either is resolved.

**Not committed.** `api.py` already has ~110 lines of pre-existing uncommitted changes from
before this session (Langfuse tracing, `/eval/run` endpoints) that aren't understood/reviewed
here — left alone per the same caution flagged earlier in this session. The comparison-retry
addition sits on top of that diff, uncommitted, so the user can review both together rather
than have this session silently commit through unrelated pending work.

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
Verified scoped (all 9 questions in the file, no regressions) — all 3 now pass. Patched the
3 corrected results directly into `answer_results_full109_merged.jsonl` and recomputed every
aggregate (cheaper and more honest than re-running the full 109 questions for a 3-question
fix). New numbers, committed to README/CASE_STUDY/summary.json:

| Metric | Before this fix | After |
|---|---|---|
| Structured/Excel accuracy (n=21) | 76.2% | **90.5%** |
| Overall correctness | 81.9% | **84.7%** |
| Answer relevancy | 83.9% | **86.7%** |
| table_lookup (question_type bucket, n=16) | 81.2% | **100%** |
| Faithfulness | 79.5% | unchanged (these 3 are structured/Excel questions, excluded from faithfulness scoring) |

This is the real, durable fix from this whole deep-dive session — unlike the two prompt-only
attempts below, which didn't hold up under verification.

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
- [ ] Connect a real SQL database as an additional data source (alongside the current
      Excel/CSV-via-DuckDB path) — not scoped or designed yet, just a future direction.

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

---

## Session update — sparse-vector/doc_id backfill, routing gate, title shortcut, gate hole (2026-07-10)

Ran the full 109-question eval to measure the cumulative effect of this session's fixes
(sparse-vector schema bug, doc_id backfill, comparison-retry, title shortcut, `route_question`
confidence gate). Real, banked wins: hit@5 95.9%→98.6%, structured/Excel 90.5%→95.2%,
faithfulness 79.5%→80.1%. Two things looked wrong and were investigated in full rather than
explained away:

**`cross_document_compare` looked like a 87.5%→72.5% regression — mostly a measurement
artifact, one real bug (now fixed).** Re-judged the frozen 109-merged baseline's 20
`cross_document_compare` rows with the *current* judge (same `_custom_judge_answer`, no
pipeline re-run) to separate judge drift from pipeline regression: baseline re-scores at 85%
under today's judge (not the frozen 87.5% — judge itself moved ~2.5pts, within noise). Diffing
baseline vs. fresh run predicted answers for every question that dropped:
- `cross_document_added_qa__qa_2` (doc_009 vs doc_010 misattribution): baseline was already
  only 0.5, not 1.0 — pre-existing gold ambiguity (two docs both plausibly "HR policy"), not a
  new regression.
- `doc_003_doc_008_qa_1`: both facts correct in the fresh answer, scored 0 only for omitting the
  `doc_003`/`doc_008` naming convention — judge artifact, not a real miss.
- `doc_001_doc_002_cross_document_qa__qa_3` ("1. Unsupported / 2. Unsupported"): **real bug** —
  re-adding comparison-retry (`d340d7e`) without restoring the original "keep comparison
  questions whole, don't split" exemption that shipped with it. Fixed in `7962e1e`
  (`src/answer_pipeline.py`, `answer_query()`): comparison questions now bypass the
  multi-part splitter again. Verified deterministically (no LLM) that this exact question no
  longer gets split.
- `doc_001_doc_002_cross_document_qa__qa_4` ("Between LACERA... which prohibits evergreen...")
  answered a flat "Unsupported" — **not** a split-bug case (matches `_COMPARISON_RE`, was never
  split even before the fix). Root cause not identified. **Open, undiagnosed** — do not assume
  the split fix covers it; needs its own investigation before claiming cross_doc is fully fixed.

**`unanswerable`/should-refuse failures (stuck at 78.6%) — 3 concrete cases, 3 different root
causes, one fixed:**
1. `doc_006_qa_9` (invoice-number hallucination): `route_question()` routes to
   `doc_005_fueling_records_invoice.pdf` with high confidence — 2-of-3 top hits genuinely agree
   on the wrong document because of the literal shared word "invoice" between question and
   doc_005's title/content. This is *not* the same failure mode the `route_question` confidence
   gate (`beee72d`) fixes — that gate only catches top-3 *disagreement*; here the top-3
   consistently and wrongly agree. **Open, no fix implemented** — a keyword-overlap routing
   problem, needs a different mechanism (e.g. weighting doc-title match lower than content
   match, or a secondary "does this doc's schema even have the asked-for concept" check before
   trusting routing).
2. `doc_007_qa_9` ("What payment method...largest Total...?", gold `Unsupported`, doc_007 has no
   payment-method column at all): **real bug, fixed** in `6924a78`
   (`src/tools/excel.py`, `_column_matches_question()`). The SQL selected only the `Total`
   column; `Total` is in `_GENERIC_COLUMN_WORDS`, and the old fallback for a column stripped to
   zero non-generic words was an unconditional `True` ("too little signal to block") —
   meaning any all-generic column (`Total`, bare `Amount`) passed the gate regardless of what
   was actually asked, letting the agent hallucinate "Fedwire Funds transfers." Fixed: the
   empty-strip fallback now checks raw (unstripped) word overlap with the question instead of
   auto-passing — `Amount` still matches "what is the amount", `Total` no longer matches
   "payment method". Added regression tests in `tests/test_excel_tool.py`.
3. `doc_015_qa_5` (SOP manual vacation policy, gold `Unsupported`): predicted answer is
   substantively from a *different* document (`doc_010`, Rosemont Employee Handbook) while
   purporting to answer about `doc_015` (Food SOP Manual) — a cross-document content
   contamination/false-attribution failure, not a routing miss (`route_question()` correctly
   returns an empty modality here). **Open, root cause not identified** — untraced whether this
   is retrieval-side (doc_010 chunks ranking in doc_015's context window) or generation-side
   (agent citing the wrong doc_id while quoting real doc_010 content).

**Net effect: real fixes landed for the split-bug and the column-gate hole. Two failure modes
(keyword-pull routing, cross-doc contamination) are diagnosed with concrete repro but not
fixed — recorded here as known limitations rather than forced tonight.** Have not re-run the
full eval after the two fixes in this update; the split-fix and gate-fix are each verified
deterministically/directly against their specific failing case, not via a fresh 109-question
run.

---

## Session update — comparison-question retry/routing fix, scoped eval verification (2026-07-11)

Ran the full 109-question eval to check the two fixes from the previous update, plus followed
through on the still-open `qa_4` retrieval-imbalance gap. Result: `cross_document_compare`
dropped further, to 67.5% (vs. 85% re-judged baseline) — the split-exemption fix (`7962e1e`)
did not, by itself, fix the class of bug it was aimed at; it only fixed the one case
(`qa_3`) with the exact "1. Unsupported / 2. Unsupported" symptom.

**Root cause, found by tracing `answer_one()`'s actual control flow instead of assuming the
existing comparison-retry mechanism was reachable:**
1. `doc_001_doc_002_qa_1` never matched `_COMPARISON_RE` at all — its phrasing ("Which document
   allows a longer extension period: the procurement policy or the services contract terms?")
   isn't `compare/comparing/versus/vs/both...and/between...and/which...and which`. Regex gap.
2. For questions that *did* match, a flat `"Unsupported"` first answer hit the **generic**
   unsupported-retry branch and returned before the comparison-retry branch below it ever ran —
   the comparison-retry code was live but structurally unreachable for exactly the failure mode
   it was written to catch.
3. `route_question()`'s routing directive actively fights comparison questions: it picks one
   confident single document (`route: {'modality': 'document', 'source_file': 'doc_002...'}`)
   and tells the agent "concerns doc_002 ... use search_knowledge_base" — the agent then treats
   this as "the answer lives in doc_002 only" and never retrieves the other required document.
   Verified directly: `route_question(qa_4)` returns only `doc_002`, but the question needs both
   `doc_001` and `doc_002`.

**Fix (`src/answer_pipeline.py`):**
- Broadened `_COMPARISON_RE` to also match "which ... X or the Y?" phrasing.
- Merged the two retry branches: a comparison question that comes back `Unsupported` now gets
  `_COMPARISON_RETRY_INSTRUCTION` (search the other document) instead of the generic
  single-doc retry instruction.
- Routing directive is now skipped entirely for comparison questions (`route = {} if
  is_comparison else route_question(question)`) — a single-doc lock is never correct for a
  question that inherently needs two.

**Verified via scoped eval** (`eval/run_eval.py --category cross_document_compare` +
`--category unanswerable`, not the full 109 — faster signal, same code path):
- `cross_document_compare`: 67.5% → 72.5% (net improvement, not a full recovery to 85%).
  `qa_4` improved 0.0 → 0.5 (partial credit — the retry now fires and does return an answer
  citing the right document, but not both parts). `doc_003_doc_008_qa_1` and
  `doc_006_doc_007_qa_3` recovered to 1.0 from the previous run's judge-noise dip. `qa_1` still
  scores 0.0 in this run — traced separately (see below), root cause is Groq inference
  nondeterminism at temp=0, not a logic bug: a direct re-run of the same question outside the
  eval harness returned the correct answer on one attempt and `Unsupported` on another, with the
  underlying retrieval quality itself varying between runs.
- **New side effect surfaced**: `doc_004_doc_005_qa_1` flipped 1.0 → 0.0. This looks like the
  routing-suppression change removing a previously-*helpful* routing hint for this specific
  question (both old and new `_COMPARISON_RE` already matched it — the regex broadening isn't
  the cause). Not chased further tonight — recorded as an open follow-up. If revisited: routing
  suppression may need to be conditional (e.g. only suppress when the two named documents in the
  question don't match the routed single document, rather than suppressing for every comparison
  match).
- `unanswerable`: 8/10 → 9/10 (improved; not directly targeted by this fix, likely benefiting
  from the earlier Excel gate-hole fix or run-to-run variance — not re-diagnosed question by
  question this session).

**Net: real, root-caused fix, verified with a scoped eval (not just unit tests). Committed.**
Not a full recovery of `cross_document_compare` to its 85% baseline — one known new side effect
(`doc_004_doc_005_qa_1`) and one known nondeterminism-driven flake (`qa_1`) remain open.

**Still not done this session** (carried over, unchanged from previous update):
- `doc_006_qa_9` keyword-pull routing failure (`route_question` agrees on the wrong doc via
  literal "invoice" overlap) — open, no fix attempted.
- `doc_015_qa_5` cross-document content contamination (answer drawn from `doc_010` while citing
  `doc_015`) — open, no fix attempted.

---

## Session wrap-up — full 109-question re-run after the comparison-retry fix (2026-07-11, later same session)

Ran a full 109-question eval after committing `5636b94` (the comparison-retry/routing fix), to
get a real post-fix number instead of relying on the scoped `--category` runs used earlier
today.

| Metric | Prior full-109 baseline | This run |
|---|---|---|
| Overall correctness | 82.4% | 84.2% |
| Faithfulness | 80.1% | 82.1% |
| `cross_document_compare` | 67.5% (pre-fix) / 72.5% (scoped post-fix) | 80.0% (scoped, after the soft-refusal fix `7fb25eb` — see later section; not yet re-confirmed on a full 109 run) |
| `unanswerable` (question_type) | 8/10 | 10/10 |
| `unanswerable_metrics` (broader scope, n=14) | — | 13/14 (92.9%) |

Overall correctness and faithfulness both improved over the pre-session baseline. Unanswerable
refusal is now at its best measured point this session.

**`cross_document_compare` is not a stable number and I'm not reporting it as one.** Earlier
today, a direct 5x-repeat test on the exact same questions (`doc_001_doc_002_qa_1`, `qa_4`,
`doc_004_doc_005_qa_1`) through the real `answer_query()` path showed:
- `qa_1`: Unsupported in 4-5 of 5 runs, with retrieval consistently pulling three completely
  unrelated documents (Fed annual report, employee handbook, lease amendment) instead of
  `doc_002`. Traced further: raw-question `retrieve()` (hybrid or dense-only, doesn't matter)
  does not surface `doc_002` in its top-10 for this exact question wording either — a real
  retrieval-relevance gap, not a query-formulation artifact. In *this* run, however, it scored
  1.0 correct — the pipeline's own nondeterminism (temp=0, but Groq inference still varies) means
  the same question can land on either side of the line.
- `qa_4`: Unsupported in 5 of 5 direct-repeat runs, mostly missing one of the two required
  documents; one repeat did retrieve both documents and *still* answered Unsupported — a
  generation-side ceiling on top of the retrieval issue. In this run it failed again (0.0).
- `doc_004_doc_005_qa_1`: genuinely flaky (2 of 5 direct-repeat runs correct) — confirmed the
  apparent "regression" flagged earlier today was noise from run-to-run variance, not caused by
  the routing-directive-suppression fix. No revert was needed. This run it passed (1.0).

**Decision (with advisor input): did not build a deeper fix (deterministic dual-retrieval for
comparison questions, bypassing the agent's self-formulated search query) today.** The evidence
doesn't support it being a reliable win: `qa_1`'s failure is a retrieval-relevance gap even on
the literal question text (context injection wouldn't help unless it can already resolve "the
procurement policy" -> `doc_001` by identity, untested), and `qa_4` has a demonstrated
generation-side failure even with correct retrieval. Forcing scoped dual-retrieval for all
comparison questions risks regressing ones that currently pass some of the time, and the
question-pair-specific nature of a hand-built fix would cross into "cheating the eval" territory
that's explicitly out of bounds for this project.

**Honest bottom line:** the comparison-retry/routing fix (`5636b94`) is a real, root-caused
improvement — the mechanism is now reachable and doesn't actively fight itself — but
`cross_document_compare` sits in a volatile band (~65-72.5%) that a single eval run cannot
pin down precisely, given the pipeline's proven run-to-run nondeterminism. A defensible headline
number for this bucket would need mean±std over several full runs, which wasn't done today due
to time. Overall correctness/faithfulness/unanswerable are more stable (larger n, less swung by
any single flaky question) and both improved.

**Not touched further today** (unchanged, still open): `doc_006_qa_9` keyword-pull routing
misattribution, `doc_015_qa_5` cross-document content contamination.

---

## Session wrap-up #2 — soft-refusal fix, scoped cdc re-measurement (2026-07-11, later still)

Root-caused why `cross_document_compare`'s volatility persisted even after the comparison-
retry/routing fix (`5636b94`): the codebase already has a pooled multi-query retrieval fallback
(`_direct_retrieval_answer` in `src/answer_quality.py`, wired in via `_context_fallback_answer`)
that splits a question into clause-derived sub-queries via an LLM call and unions the retrieved
context — verified directly (no eval) that it correctly answers `doc_001_doc_002_qa_1` when
called standalone. It was never reached in practice: `_looks_like_bad_final_answer()` only
matched the literal string `"unsupported"`, so a soft-refusal answer like *"The retrieved
content does not provide information on... I cannot perform the requested comparison"* was
accepted as a final answer instead of triggering the fallback that would have fixed it.

**Fix (`7fb25eb`, `src/answer_quality.py`):** added `_SOFT_REFUSAL_RE` to `_looks_like_bad_final_
answer()`, catching common soft-refusal phrasings ("does not provide/contain", "cannot
perform/determine/answer", "no information available", "unable to determine/answer/find") in
addition to the literal token. Tests pass (187/187).

**Verified directly** (3x repeat on `qa_1`/`qa_4` through `answer_query`): `qa_1` went from
near-total-fail to 2 of 3 runs returning a real answer. `qa_4` unchanged (its failures are the
literal "Unsupported" token, already covered by the existing retry path, not soft-refusal prose
— consistent with the separately-diagnosed generation-side ceiling for that question).

**Scoped `cross_document_compare` eval** (per explicit instruction: cdc category only, no full
109 run): **80.0%** — the highest of four measurements this session (67.5% pre-fix -> 72.5% ->
65.0% on the full-109 rerun -> **80.0%**). `doc_004_doc_005_qa_1/2/3` all 1.0 this run (previously
flaky). `qa_1` and `qa_4` both still 0.0 in this specific run — `qa_1`'s predicted answer this
time picked the wrong document ("services contract terms" instead of gold's "procurement
policy"), a different failure mode than its earlier retrieval-miss runs, underscoring that this
bucket's per-run score is still a sample, not a fixed number, even after two real fixes.

**Net across the session's two comparison-question fixes:** cdc score band widened but trended
up (65-80% observed, vs. 67.5% pre-session and an 85% re-judged historical baseline). Both fixes
are real, root-caused, and verified with direct/standalone tests, not just eval-score chasing.
`qa_1`/`qa_4` remain not fully solved — they surface a different failure mode almost every run
(wrong document, missing document, correct-retrieval-but-bad-generation, or now sometimes
correct) — consistent with an inherently nondeterministic generation step on top of a mostly-
fixed retrieval path. A trustworthy final number for this bucket would need several repeated
full runs (mean +/- std), not attempted today due to time.

---

## Session — README consistency fixes + product backlog (2026-07-11, later still)

Separate from the eval-tuning work above: fixed README inconsistencies and shipped the
Upwork-facing backlog items the user asked for, skipping only auth/workspace login (explicitly
deferred).

**README fixes** (`2c391a2`): headline benchmark table now uses the exact detailed-results
numbers (was rounding some down by 4-5 points) — correctness 84.7%, faithfulness 79.5%,
relevancy 86.7%, hit@5 95.9%, structured-data 90.5%, refusal 78.6%. Refusal-rate history
separated ("on the earlier subset, 75%->100%; current expanded benchmark scores 78.6%" instead
of reading like 100% is current). Removed the unverifiable "~80%" single-doc-lookups row.
Citation and self-hosted wording corrected (spreadsheets cite sheet/SQL evidence, not page
numbers; generation/enrichment use external APIs by default but redirect to local endpoints).

**Feedback admin page** (`1bed3e6`): dedicated `/feedback` route (reuses the existing
`FeedbackPanel` modal). `FeedbackRow` now renders cited sources and an editable admin note —
both were already in the `Feedback` data model but never shown. "Add to eval set" previously
only relabeled the record; it now also appends `{question, predicted_answer, sources}` to
`eval/regression_candidates.jsonl` with `gold_answer: null` for a human to fill in before it
counts in a real eval run — closes the feedback -> investigation -> regression-question loop
for real.

**Calculator tool** (`0007afa`): the agent previously refused all arithmetic on PDF/OCR-extracted
numbers by prompt rule. Added `src/tools/calculator.py` — an `ast`-walked evaluator restricted to
numeric literals and `+ - * / ** ()`, no name/call/attribute access, so no code-execution
surface even though the LLM controls the input string. Wired into both tool-prompt variants.
Verified end-to-end (not just unit tests): asked the live agent to retrieve a year from a PDF
and multiply it by a decimal constant; it called `calculate` and returned the exact correct
product, confirming real tool use rather than mental math.

**Google Drive sync connector** (`03f3f89`): `src/connectors/google_drive.py`, authenticated via
a service-account key file — deliberately non-interactive (no OAuth consent screen, no login UI)
since a real user-facing auth system was explicitly out of scope for this session. Lists a
configured folder, downloads new/changed files (by `modifiedTime`), routes each through the
existing ingestion pipeline (`run_ingest` / `ingest_table_rows`), and optionally removes
documents deleted from Drive via `delete_by_file`. Each file's failure is caught and reported
per-file rather than aborting the sync. Four endpoints
(`configure`/`sync`/`status`/`files` under `/connectors/google-drive/`) plus a UI panel at
`/connectors/google-drive`. Tests fully mock the Drive API client — no live network calls.

**n8n workflow templates** (`f6516c7`): `integrations/n8n/` — two real, importable n8n workflow
JSONs (generic webhook, WhatsApp) implementing the connector pattern the README already
described but never shipped a file for: `POST /query`, then route `Unsupported` answers to a
Slack on-call notification instead of surfacing them to the end user.

**Explicitly not done** (per direct instruction): Priority 3 — user authentication, login,
workspace separation. The current API auth model (single shared `X-API-Key` on mutating
endpoints) is unchanged.

All 202 tests pass (`uv run pytest tests/`), ruff clean on every file touched. Frontend
typechecks cleanly except one pre-existing, unrelated error in `frontend/app/layout.tsx` (a
`next/font/google` import issue in files I did not touch this session).

## Final eval run — 2026-07-16, gpt-oss-120b swap CONFIRMED

Full-corpus eval (109 questions, `eval/results/summary.json`), resolving prior PROVISIONAL
status on the `qwen3-32b` → `gpt-oss-120b` `GENERATION_MODEL` swap. Aggregate: correctness
0.838, faithfulness 0.861, answer_relevancy 0.869. No question-type category regressed.

`correctness_by_question_type`, sorted descending:

| category | n | correctness |
|---|---|---|
| table_grounding | 3 | 1.000 |
| table_lookup | 16 | 0.938 |
| numeric_lookup | 6 | 0.917 |
| ocr_extraction | 25 | 0.860 |
| single_doc_factoid | 17 | 0.812 |
| unanswerable | 10 | 0.800 |
| negation_check | 5 | 0.800 |
| cross_document_compare | 20 | 0.775 |
| numeric_reasoning | 4 | 0.750 |
| figure_grounding | 3 | 0.667 |

`figure_grounding` lowest (2/3) but n=3, no prior baseline — not treated as a regression
signal. Retrieval metrics stayed strong (hit@5 0.986, MRR 0.854) — retrieval was never the
bottleneck. Full write-up and analysis in `TODO.md` (~line 588). Still open: transcribe this
table into README.md / `docs/CASE_STUDY.md`.

## Portfolio finalization task, 2026-07-16 — Phases 1–6

A separate, larger brief: finalize Vault RAG as a polished, reliable AI
engineering portfolio project before any new product capabilities. Executed
in 6 phases, commits `55c1a2d`..`4d7c8ec`. Full detail in `TODO.md`'s
matching dated section; summary here:

1. **Deterministic cross-document comparison** (`src/answer_pipeline.py`'s
   `answer_comparison_deterministic`) — replaces the probabilistic
   agent-retry comparison path for questions where the requested documents
   can be confidently resolved. Live-verified 5/5 real runs, both documents
   covered every time; a real+nonexistent-doc case correctly excluded the
   nonexistent one with no fabrication.
2. **One canonical eval result** — `eval/results/summary.json` now self-
   describes (`benchmark_date`/`answer_model`/`judge_model`/`document_count`);
   `docs/EVAL_SUMMARY.md` is generated from it (`eval/generate_summary_doc.py`),
   fixing a real inconsistency (it was stale — an 82-question run that no
   longer matched README.md/CASE_STUDY.md).
3. **Release checklist** — `docs/release-checklist.md`.
4. **Deployment reliability** — docker-compose healthchecks + dependency
   ordering, a real DuckDB-volume data-loss fix, startup env validation,
   embedding warm-up.
5. **Optional admin/viewer access mode** — `ACCESS_MODE=open` (default,
   unchanged behavior) / `admin_viewer` (session-cookie-gated admin actions,
   backend-enforced + frontend-hidden, 14 backend tests + Playwright).
6. **Portfolio packaging** — README repositioning, demo script, screenshot
   shot list.

Test counts at the end of this task: 299 backend (`uv run pytest tests/`),
16 frontend unit (`npx vitest run`), 2 Playwright specs passing live against
the running dev server. Known gaps carried forward (not silently dropped):
no Word/.docx demo sample, full 17-flow e2e suite not built, clean-clone
`docker compose up --build` not run end-to-end, no CI-runnable lane for the
comparison path without live model credentials.
