# vault-rag — TODO to portfolio-ready

## RESOLVED (2026-07-22) — Finding 5 root-caused and fixed; reflection override defaulted off

The backend investigation queued below (Finding 5: "model answers from general knowledge, cites
a real-but-unrelated chunk to look grounded") turned out to be two deterministic bugs, not
inference nondeterminism — a corpus-embedding pollution (133/862 chunks had a literal rate-limit
error string baked into their embedded context) and a table-lookup keyword filter hard-excluding
answer chunks on ordinary prose questions. Both fixed and live-verified against their exact
reproduced cases, along with an Excel answer-fabrication guard, a bare-filename gap, a new
leaked-JSON variant, a pure-spreadsheet retrieval dead end, and — separately — an eval-judge
prompt bug that was the real explanation for the Excel accuracy plateau (contaminated correctness
scores on context-free questions). The reflection override below is now off by default.

**Full detail, numbers, and the two originally-queued sections (investigation scope, known-wrong
answers list, override evidence) — kept in full in:**
- [`eval/results/investigation_20260722_backend.md`](eval/results/investigation_20260722_backend.md) — root-cause findings
- [`eval/results/fix_specs_20260722.md`](eval/results/fix_specs_20260722.md) — per-fix implementation specs
- `PROGRESS.md`'s 2026-07-22 entry — narrative + before/after numbers
- README.md's [Recent fixes](README.md#recent-fixes) — user-facing summary

**Final numbers (109-question benchmark):** correctness 90.6%, faithfulness 90.4%, answer
relevancy 94.0%, correct refusal rate 92.9%, structured (Excel/CSV) accuracy 95.2%, hit@5 98.6%
(unchanged). Committed and pushed (`8815785`).

**One item traced but genuinely still open** (not a code bug — no clean fix available):
`doc_003_qa_5` (Federal Reserve creation date) — the answer chunk was retrieved with the exact
date in context, and the model still returned `Unsupported`. This project's history has several
prompt-only patches for similar issues verified not to hold; not chasing without a validated
approach. Low priority, n=1.

## UI product refinement (2026-07-14 spec) — full requirements + phased execution plan

Full spec given verbatim by the user; condensed here into a checklist so it survives
across sessions. Product promise: "Ask questions across your organisation's documents
and verify every answer directly against the original source." Normal users should never
need to understand chunks/Qdrant/embeddings/reranker/tool-calls/SQL — that lives under
Technical details / Advanced settings only.

**Before starting real work:** GENERATION_API_BASE pointed at OpenRouter was causing
intermittent empty answers (see the retrieval-flakiness item below) — this blocks
verifying any of the citation/evidence UI. Fix in progress: bringing up local vLLM
(Qwen3-32B-AWQ, `:8005`) as `.env`'s `GENERATION_API_BASE`, per the documented fallback
in TODO item 2 and CLAUDE.md's vLLM cheatsheet. Config-only change, not backend code.

### Already implemented (do not rebuild) — confirmed by inspection
- Evidence tab (default) / Technical trace tab split — `RightPanelTabs.tsx` (§5, §6).
  Just needs the label rename "Technical trace" → "Technical details".
- Tools / Generated SQL / Retrieved chunks / Rejected accordions — `TraceSidebar.tsx`,
  already Collapsible-based, already hides raw internals behind an expand (§6).
- Live PDF bbox highlighting + figure crop (shipped this session) — `EvidencePanel.tsx` (§5B, §5C).
- Numbered citation chips below the answer, click-to-expand quote — `MessageList.tsx`'s
  `SourceDrawer` (§4, partial — see gaps below).
- Feedback thumbs up/down + reason dropdown — `MessageList.tsx`'s `FeedbackWidget` (§10A, partial).
- Reranker-score three-tier badges (no rainbow palette), already restrained — `SourceCard.tsx`.

### Real backend/product gaps — flag, do not fake with fragile frontend hacks
- **Inline citations after each claim (§4).** Backend's `_INLINE_CITATION_RE` in
  `src/answer_pipeline.py` deliberately strips `[N]` markers from the answer text —
  code comment explains the model's own numbering doesn't correspond to the source
  list, so leaving them in would mislead. No claim→source span mapping exists anywhere
  in the pipeline. Implementing true inline citation requires a real backend change
  (have the model emit citation markers that are then validated/renumbered against the
  actual source list) — out of scope for a frontend refinement pass. Ship the "Sources
  used · N" compact list + Popover per claim instead (spec explicitly allows this as
  the fallback pattern) and flag inline placement as backend-blocked in the final report.
- ~~**"Preserve streaming answers / stop generation" (§3B).** There is no token
  streaming today...~~ **Done, 2026-07-16**: `POST /query/stream` (api.py) is a
  real SSE endpoint — `src.answer_pipeline.stream_answer` streams tokens live
  for single-part, non-comparison questions (the common case); comparison and
  multi-part questions still need the complete answer before their
  retry/merge logic runs, so those arrive as one lump token event, not faked
  as live streaming. The streamed tokens are the model's raw, un-renumbered
  output — the final `done` event carries the clean, citation-renumbered
  answer, and the client replaces the raw concatenation with it, not appends.
  `ChatPanel.tsx`'s `send()`/`retry()` now write into the message incrementally;
  `Stop` (the existing Square button) aborts the fetch mid-stream via the same
  `AbortController` as before. Verified live via curl against a running
  `uvicorn api:app` — token event, then a `done` event with a correctly
  resolved `[3]` marker and full sources. **Known limitation, documented in
  the endpoint's docstring**: on client disconnect, the executor thread
  keeps running stream_answer to completion (a sync generator in a worker
  thread can't be cancelled from the async side without extra plumbing) —
  wasted compute, not a correctness bug.
  **Caught in review before this was called done**: the first cut of the
  single-part live path silently dropped answer_one's Unsupported-retry
  (Groq/temp=0 nondeterminism occasionally returns Unsupported despite the
  answer existing) — exactly the question type that retry exists for, since
  ChatPanel now uses `/query/stream` exclusively. Fixed: a first-attempt
  "Unsupported" now re-runs once via `run_once` (non-streamed, same
  `_RETRY_INSTRUCTION` as answer_one) and the retry's result, if it improved,
  replaces the streamed answer in the final `done` event. Covered by a new
  test (`test_retries_first_attempt_unsupported_like_answer_one`) and
  re-verified live via curl after the fix.
  **User tested #13/#14/#15 live in a browser, 2026-07-16 — found a real bug
  the curl smoke test couldn't surface**: `stream_agent`'s tool-use path (the
  common case — almost every real question) buffered the *entire* answer
  internally (generation → repair-pass LLM call → grounding-check LLM call,
  all sequential) and only `yield`ed once at the very end. The SSE plumbing
  was correct but there was nothing to stream — reported live as "a question
  takes about 2 minutes to respond" with no progressive text, matching what
  a single-lump curl response had already hinted at. This was pre-existing
  total latency (3 sequential LLM round-trips), not introduced this session,
  but the streaming feature delivered zero perceived-latency benefit for the
  case it mattered most for.
  **Fixed**: `stream_agent` gained a `live_tokens` parameter (default False,
  so `run_once`/`ask_agent`/eval are completely unaffected — verified, they
  just concatenate every yielded item regardless of timing). When True,
  tokens after the first tool call are yielded live as they arrive; if the
  repair pass or grounding check changes the text afterward, the *complete*
  corrected answer is yielded once more wrapped in a new `FinalCorrection`
  marker so callers know to replace, not append. `stream_answer` (the only
  other caller, in `answer_pipeline.py`) opts in with `live_tokens=True` and
  treats a `FinalCorrection` as the authoritative final answer, not another
  SSE token event. 5 new tests (`test_rag_agent.py`'s `TestStreamAgentLiveTokens`,
  `test_answer_pipeline.py`'s correction-handling tests). **Verified live**:
  curl against a running `uvicorn api:app` on a real multi-part contract
  question showed tokens arriving continuously at ~20ms intervals for
  ~13.5s, `done` event ~6s after that (repair + grounding check) — genuine
  incremental streaming, not a single lump.
- ~~**Retry** on a message doesn't exist either — only new-conversation. Same category.~~
  **Done, 2026-07-16**: turned out to need no backend change at all — `/query`
  is already stateless per-question (no history is threaded today), so retry
  is just re-asking the same question and replacing that assistant message's
  content/sources in place. `ChatPanel.tsx`'s `send()` refactored to share an
  `ask()` helper with a new `retry()`; `MessageList.tsx` shows a retry icon
  next to the feedback widget on every assistant turn.

### Phased execution order (highest product-value / lowest risk first)
1. [x] ~~Backend generation reliability fix (local vLLM)~~ — not done as planned (GPU
       budget conflict with the OCR model, user redirected focus back to UI work). But
       found and fixed a bigger, unrelated issue instead: **the API worker process had
       been running since 2026-07-12 17:57, completely stale** — `uvicorn --reload`
       silently stopped picking up file changes at some point this session (root cause
       not diagnosed; a leaked `multiprocessing` orphan child was also found still
       holding the port after a `kill -9` of the parent, which is why the first restart
       attempt failed with "Address already in use"). Every backend test earlier in this
       session ran against **old code** — the doc_001 empty-answer flakiness observed
       and logged below may or may not reproduce against a properly-reloading server;
       not re-verified. Killed both the stale worker and the orphaned child, restarted
       clean. **If backend changes stop showing up in live testing, check process start
       time (`ps -o lstart=`) against file mtimes before assuming the code is wrong.**
2. [x] Product-level data contracts + adapters (§14) — `frontend/lib/product.ts`
       (`Answer`/`Citation`/`Source`/`AnswerTrace`/`SourceLibraryItem` + adapters).
3. [x] Terminology pass — `Sidebar.tsx` status badges (Ready/Processing/Failed),
       `UploadZone.tsx` stages (Uploading/Processing/Ready, dropped the granular
       Parsing/Chunking/Embedding breadcrumb), "Technical trace"→"Technical details".
       Citation chips now open a Popover (hover/focus) with quote/section/Open source;
       "Sources used · N" compact summary replaces the always-expanded drawer.
4. [x] Navigation restructure (§2) — `AppHeader.tsx`: Feedback + Evaluation grouped
       under a "Quality" DropdownMenu, Google Drive under "Integrations"; primary
       History button renamed "Conversations". Note: still opens the same overlay
       panels (`FeedbackPanel`/`EvalPanel`/`GoogleDrivePanel`), not routed to
       `/feedback` etc. — those routes exist (`app/feedback/page.tsx`,
       `app/connectors/google-drive/page.tsx`) but are unused by the header; wiring
       them up properly is still open (see phase 6).
5. [x] Source-scope control (§3A) — "Ask across: All sources / one or more documents"
       shipped (`SourceScope.tsx` checkbox multi-select, `QueryRequest.doc_id: str |
       list[str] | None`, `FORCED_DOC_ID` contextvar in `retrieval_tool.py` accepting
       a list and bypassing single-doc inference to OR across all selected ids — same
       hard-override precedent as the single-doc case; the prompt directive alone was
       verified unreliable, see the commit message). query_excel still has no
       per-source scoping param, so a forced spreadsheet document relies on the soft
       prompt directive only (unaffected by this change).
6. [x] Sources screen (§7) — `app/sources/page.tsx`, a real table (name, type,
       status, last updated, actions: Open/Ask about this source/Reprocess/Delete)
       complementing the persistent sidebar list. Wired header's Quality/Integrations
       items to the existing `/feedback` and `/connectors/google-drive` routes
       (previously unused by the header despite existing) and added
       `/quality/evaluation` (had no route at all). "Replace" action from spec §7
       not implemented — no backend endpoint for replacing a source in place, only
       delete + re-upload. Conversations screen (§9) and Settings screen (§12) —
       NOT done. Conversations: `HistoryPanel` already covers title/updated/count/
       search/delete reasonably well as an overlay (kept as overlay deliberately —
       selecting a conversation must return to "/" to load it into the active chat,
       so a separate route doesn't add much); missing "source scope" and "preview of
       last question" columns, which need small backend additions to
       `ConversationSummary` (not done). Settings: not started at all — no route, no
       component; would need new backend config-read endpoints for the
       privacy/processing-location section (§12) since nothing currently exposes
       where OCR/embeddings/generation run.
7. [x] Spreadsheet evidence (§5D) — `EvidencePanel.tsx`'s `SpreadsheetEvidence`
       renders the real sheet (via `/table-sheet`) with best-effort row highlighting
       (a row lights up when one of its cells appears verbatim in the citation
       quote); falls back to first rows + disclaimer when nothing matches, same
       pattern as the PDF bbox highlight. **Real gap found, not fixed**: `query_excel`
       (the actual SQL Q&A tool) never produces citable sources at all — its
       artifact only carries `{sql, subquestions}`, no row/cell identity. Verified
       live: an excel-scoped question routed to `query_excel` returns `sources: []`
       every time. `SpreadsheetEvidence` only lights up for the `search_knowledge_base`
       → `sheet_summary` citation path, which does track a source. True cell-level
       evidence for SQL-answered questions needs `src/tools/excel.py`'s LangGraph
       pipeline to track which DuckDB rows it matched — real SQL-generation work,
       explicitly out of bounds for a UI-only pass.
8. [x] Responsive passes (§16), terminology sweep — **reviewed 2026-07-16, no
       code changes made** (see note below); remaining tests (§17) split out
       as its own item, see the frontend-test-coverage entry further down.

**Responsive review (2026-07-16), inspection only — no browser on this box
to visually confirm, so this is "checked by reading the code," not "checked
by looking at it":**
- Main layout (`app/page.tsx`) already does the right thing: `RightPanelTabs`
  is `hidden ... lg:flex` with a floating button + `Sheet` slide-over
  fallback below `lg`; `Sidebar`/`SidebarInset` are shadcn's own responsive
  primitives. `EvidencePanel`/`TraceSidebar` are `w-full` on mobile,
  `lg:w-[320px]` on desktop — mobile-first, not a fixed desktop width leaking
  onto small screens.
- `MessageList`'s bubble widths (`max-w-[70%]`/`max-w-[85%]`) are
  percentage-based, so they scale with viewport by construction.
- `app/sources/page.tsx`'s `Table` has no explicit overflow wrapper, but
  shadcn's own `Table` component (`components/ui/table.tsx`) already wraps
  itself in `overflow-x-auto` — no page-level gap.
- No genuine responsive gap surfaced under inspection. Not rewriting working
  code just to have a diff — didn't touch any layout files this pass.

**Terminology sweep, grep-verified (not guessing at the original §19 spec
text, which isn't preserved anywhere in this repo — see the phased-plan
intro's note that the spec was given verbatim by the user and only
condensed here):**
- `grep -rn "Technical trace"` across `frontend/`: zero matches — the
  earlier rename to "Technical details" is complete everywhere.
- `grep -rn "Parsing\|Chunking\|Embedding"`: zero matches — `UploadZone.tsx`
  confirmed to only ever show "Uploading/Processing/Ready" to the user,
  internal stage keys (`parsing`/`chunking`/`embedding`) stay backend-only.
- The one remaining `History` identifier (`AppHeader.tsx`'s lucide icon
  import and `onShowHistory` prop) is an internal name, not user-facing text
  — the visible label already reads "Conversations".

### Known backend gaps (flagged, not faked with fragile frontend hacks)
- **Inline citations after each claim (§4).** `_INLINE_CITATION_RE` in
  `src/answer_pipeline.py` deliberately strips `[N]` markers — no claim→source span
  mapping exists. Shipped the "Sources used · N" + Popover fallback instead (spec
  explicitly allows this). True inline citation needs the model to emit markers that
  get validated/renumbered against the real source list — a real backend change.
- **No token streaming.** `ChatPanel.tsx`'s `send()` awaits the full `/query` response;
  "stop generation" and "retry" don't exist. Needs an SSE/chunked backend endpoint.

## Session summary (2026-07-14) — evidence panel shipped, open items before wrap-up

Committed: PDF bbox highlighting + evidence panel, figure-image return (crop endpoint,
see "Return the actual figure image" below — now implemented), removed unused
`src/integrations/drive_sync.py` stub, added frontend typecheck to CI, added one
Playwright e2e smoke test (`frontend/e2e/golden-path.spec.ts`, local-only, not in CI).

- [ ] **Re-run the full 109-question eval** and reconcile `eval/results/summary.json` —
      ingest/chunker/vlm changed since the last committed run (hit@5 0.959, correctness
      0.847). A partial 20-question `cross_document_compare`-only run was discarded
      uncommitted during this session; do not let a partial run overwrite the real numbers.
- [ ] **Push to origin** — ~19 commits ahead of `origin/master`, all unpushed.
- [x] **Retrieval flakiness, investigated 2026-07-15 — narrower than this note assumed.**
      Re-tested the exact single-doc procurement-approval question 5x live: identical
      sources every time, no flakiness reproduced. The instability is real but scoped
      to **cross-document comparison questions** specifically ("compare X and Y"), not
      single-doc Q&A generally — this note's original framing was too broad. Fixed
      along the way: a `parse_sources()` truncation bug that could silently drop a
      document's genuinely-retrieved chunks from the source list (commit `10a5678`),
      and unified the comparison-retry logic so a retry's own answer gets the same
      completeness check as the initial attempt (same commit). Root cause of the
      remaining instability is NOT `GENERATION_API_BASE`/OpenRouter rate limiting as
      guessed here, and not model quantization either — tested both:
  - [x] Pinned `OPENROUTER_PROVIDER_PIN=DeepInfra` (fp8-quantized), ran the same
        comparison question 5x: 5/5 identical, but 5/5 wrong (flat "Unsupported"
        despite both docs correctly retrieved). Commit `e54b079`.
  - [x] Pinned `OPENROUTER_PROVIDER_PIN=Alibaba` (likely full precision, the model's
        publisher): *more* variable, not less — 3/5 dropped doc_009, 2/5 dropped
        doc_010, with different response styles. Rules out quantization as the cause
        (a full-precision provider should be at least as stable). Commit `31f77ee`.
  - [ ] **Not yet investigated: the agent's own tool-calling decision on comparison
        questions** (how many `search_knowledge_base` calls it makes, and for which
        document) appears to be the actual unstable part, independent of which
        provider generates the final text. Would need tracing actual tool-call
        sequences across repeated identical runs to confirm (start in
        `src/rag_agent.py`'s ReAct loop / `create_react_agent` config).
      Current state: accepted as a known, documented limitation. The retry-logic
      fixes above reduce how often it happens but don't eliminate it. Single-document
      Q&A (the product's core flow) is unaffected and verified deterministic.
- [ ] `TODO_LITELLM.md`/older sections of this file reference Postgres and `app.py` —
      stale, current `docker-compose.yaml` uses Redis and the entrypoint is `api.py`.
      Cosmetic cleanup, low priority.
- [x] **Frontend test runner added, 2026-07-16**: no component/unit test runner
      existed at all (only Playwright e2e). Added `vitest` + `@testing-library/react`
      (`npm run test`, `frontend/vitest.config.ts`/`vitest.setup.ts`) and targeted the
      two pieces of untested new logic from this session's #13/#15 work, since those
      never ran in a real browser (typecheck only — see the streaming section above):
      `AnswerContent`'s `[N]`-marker-to-CitationChip splitting (`MessageList.test.tsx`,
      4 tests: resolved marker renders as a real chip, click selects the right source,
      no-sources fallback stays plain text, an out-of-range marker degrades to plain
      text instead of crashing) and `streamQueryDocuments`'s SSE parsing
      (`lib/api.test.ts`, 4 tests: token+done events, an SSE frame split across two
      stream chunks, a `done` event carrying `error`, a non-OK HTTP response).
      **Still open**: `SourceCard`, `EvidencePanel` bbox math, and most other
      components still have no unit tests — the runner now exists, so this is
      "not started yet" rather than "no way to start it."

See also `TODO_LITELLM.md` for the three open LiteLLM semantic-cache + Langfuse blockers
(tracked separately — those are optional enhancements, not blockers for publishing).

## Weak retrieval / ungrounded answer on RFQ-definition question (found 2026-07-12, not fixed)

Asked `doc_001_procurement_policy.pdf` (LACERA procurement policy) "what is RFQ in Lacera doc?"
against the live API directly (`curl /query`, bypassing the frontend entirely — confirmed not
a UI bug). Two separate runs of essentially the same question both showed the same failure
shape:
- All 8 retrieved chunks scored **-2.19 to -7.6** (reranker scores) — nothing retrieved is a
  strong match.
- The generated answer confidently defines RFQ ("itemized list of prices for Goods or
  Services... hardware") but **no retrieved chunk contains that text** — the model appears to
  be answering from its own training knowledge of what RFQ generically means in procurement,
  not from the retrieved context.
- The cited source chunk (e.g. "Master Agreements... RFSQ") is real and was genuinely
  retrieved, but doesn't actually support the answer's specific claim either.
- [ ] Check whether the actual RFQ definition chunk exists in `doc_001_procurement_policy.pdf`
      at all and why retrieval didn't surface it (embedding quality? chunk boundaries splitting
      the definition away from the "RFQ" heading? reranker threshold too permissive, letting
      weak matches through instead of refusing?).
- [ ] Consider whether the agent should refuse / hedge when all retrieved scores are this weak,
      instead of answering fluently from parametric knowledge.

## Session summary (2026-07-06) — eval numbers reconciled, two real bugs fixed

**The eval numbers are now real, current, and synced.** README.md, docs/CASE_STUDY.md, and
`eval/results/summary.json` all reflect the same full 109-question run (up from 93 — 3 new
refusal-style questions added, see below). The "three docs cite three different numbers"
problem tracked lower in this file is resolved as of this run; treat it as historical context,
not an open item.

**Corpus grew 93 → 109 questions** (still 18 documents). Added 3 targeted refusal questions
(`doc_006`/`doc_007`/`doc_014` — each asks for a field that provably doesn't exist in its
table, e.g. a VAT number in a dataset with no such column) to test a hallucination pattern
found this session.

**Bug #1 — Excel/SQL path hallucinated instead of refusing when a field didn't exist.**
Asked for a VAT registration number in a table with no such column, the agent returned a real
company name as if it were one. Root cause: the SQL-writing prompt never offered refusal as an
option, and the answer-formatting step was hard-coded to never abstain once any SQL rows came
back. Fixed with an explicit refusal instruction, an anti-column-aliasing rule (the model was
later caught disguising a wrong column via `SELECT x AS "<the concept asked about>"`), a
code-level fix for the formatting step leaking its own "no rows" phrasing out as a fake answer,
and — because prompt instructions alone were verified to still fail on one case even after
tightening — a programmatic hard gate in `src/tools/excel.py` (`_column_matches_question`) that
checks real vocabulary overlap between the SQL's selected column and the question before
trusting it. Structured accuracy: 28.6% (baseline) → 42.9% → 61.9% → **76.2%** (current, on a
harder test set than any of the earlier snapshots). One of the two originally-failing cases is
now 4/4 reliable; a second is much improved but not airtight — depends partly on the model
still following instructions on top of the hard gate.

**Bug #2 — the eval's faithfulness judge was scoring correct answers as 0% faithful.**
Reproduced directly: a correct, fully-cited cross-document comparison scored `faithfulness: 0.0`
even with both compared facts verifiably present in the retrieved context. Root cause was judge
unreliability on comparative phrasing (`gpt-oss-120b` via OpenRouter), not a generation problem.
Fixed by switching the judge to `gpt-4o-mini` (OpenAI) and tightening the judging prompt to
penalize hedged/inferred claims specifically. Verified against the same reproduced case:
faithfulness now scores 1.0 as expected.

**A real, separate, NOT-yet-fixed generation gap found during that same investigation:** on a
different comparison question, the agent skipped the second required retrieval entirely and
answered from general domain knowledge instead ("implied by... typically..."), phrased as an
inference. It happened to match the gold answer — that's luck, not evidence-backed reasoning.
Worth a dedicated look: the agent isn't reliably making both required tool calls on
two-document comparison questions.

**Figure-grounded questions (33% correctness, n=3) — root cause found, fix implemented and
partially verified, one open question remains (see below).**

Inspected the ingested chunks directly for both failing questions in `data/output/chunks/`:
- `doc_008_qa__qa_3` asks for the dollar amount in Figure 3. The VLM (`meta-llama/llama-4-scout-17b-16e-instruct`)
  correctly extracted it at ingest time — the chunk literally reads *"$596.3 billion identified
  between 2011 and 2023 and an additional $71.3 billion identified in 2024"* (gold: $71.3
  billion). The agent answered `$667.5 billion` instead — a different real number from a
  neighboring figure-description chunk (the report's grand total, one page over).
- `doc_008_qa__qa_4` asks which mission had the largest financial benefit. The chunk correctly
  contains *"Defense — Budget: $197 billion"* (the exact gold answer). The agent answered
  `Unsupported` — it didn't retrieve this chunk at all.

So the VLM ingestion pipeline was working well and did not need to be replaced — the numbers
were already correctly in the index. The actual gap was retrieval disambiguation: multiple
figure-description chunks live on nearby pages of the same report with overlapping vocabulary,
and nothing tied a query mentioning "Figure 3" specifically to the Figure 3 chunk over its
neighbors.

**Fix implemented in `src/parser/pdf_parser.py`:** a new `_nearby_figure_label()` helper
searches backward from each figure's insertion point (up to 800 chars) for the nearest
preceding `**Figure N: ...**` caption heading — confirmed via real `doc_008` text that
`pymupdf4llm` reliably extracts these as genuine page text, with the caption sometimes
separated from the image by an intervening paragraph (a "Note:" aside). The found label gets
baked directly into the figure's own `[FIGURE_START]` block, so both dense and sparse (BM42)
retrieval get a strong exact-token signal disambiguating it from neighboring figures. Wired
into all 3 marker-construction call sites (img_ref, picture_text/vector-graphic, and the fitz
fallback path). Lint/format clean.

**Applied retroactively to `doc_008` without a full re-ingestion** — confirmed captions and
their figure blocks already live in the same cached chunk (`data/output/chunks/*.json`), so a
full OCR/VLM re-run wasn't needed. Instead: patched the cached chunk + embeddings JSON files
directly (`data/output/chunks/doc_008_gao_24_106915_chunks.json` and
`data/output/embeddings/doc_008_gao_24_106915_chunks_embeddings.json`), re-embedded just the 9
affected chunks via Ollama (`bge-m3`), and upserted those 9 points into Qdrant using their
existing deterministic point IDs (`_stable_id`) so they overwrote in place. Across the whole
18-doc corpus, only `doc_008` has real numbered figures worth labeling (scanned 210 total
figure chunks corpus-wide; 9 got a label, the rest are genuinely uncaptioned decorative
photos with no numbered caption in the source PDF at all).

**Verification, isolated to the 3 figure_grounding questions (`doc_008_qa` qa_3/qa_4/qa_8):**
- Run 1 (pre-fix baseline): qa_3 wrong (`$667.5 billion`), qa_4 wrong (`Unsupported`), qa_8
  correct (`99`). Overall 1/3.
- Run 2 (post-fix): **qa_3 now correct** (`$71.3 billion`, explicitly citing `chunk=16` — the
  exact chunk patched with the `Figure 3:` label — direct proof the fix works for its target
  case). qa_4 unchanged (`Unsupported`, still wrong — different unresolved cause, retrieval
  still isn't surfacing that chunk at all). **qa_8 flipped from correct to `Unsupported`.**
  Overall still 1/3, but the composition changed.
- [ ] **Open question: is qa_8's flip a real regression from the fix, or LLM nondeterminism**
      (which has shown up repeatedly all session on repeated identical questions)? A repeat run
      to check consistency got stuck in D-state (uninterruptible disk sleep) for 15+ minutes —
      traced to genuine shared-machine contention (`load average: 9.88, 10.28, 11.90`, 11 users
      logged in at the time), not a bug — and was killed rather than left hanging. **Re-run the
      3 figure_grounding questions again when the box isn't under heavy load** to settle this.
- [ ] `qa_4` remains unresolved regardless — retrieval still isn't finding the Figure 4/$197B
      chunk. Worth its own investigation once qa_8 is settled.
- [ ] Only 3 questions test this — worth adding more figure-grounding questions to the corpus
      to get a number that isn't noise from n=3, especially now that there's a real fix to
      validate against.
- **The Excel hard-gate fix's effect on the full agent pipeline hasn't been confirmed by a
  fresh full `make eval` run** — only verified via isolated `query_excel` tool calls (which
  bypass the ReAct agent's own repair/retry logic). The numbers above (76.2%, current corpus)
  predate the hard-gate fix's landing; a fresh full run would likely show a small further
  improvement on the unanswerable/refusal-rate metric specifically, but this hasn't been
  measured yet.
- Nothing from this session is committed to git yet.

## Found overnight (2026-07-03), while getting a fresh eval run working

1. **Real code bug, now fixed**: `build_rag_agent()` in `src/rag_agent.py` wrapped the
   `ChatOpenAI` LLM in `.with_retry(stop_after_attempt=3)` before passing it to LangGraph's
   `create_react_agent()`. That wrapper (`RunnableRetry`) doesn't expose `.bind_tools()`, which
   `create_react_agent()` needs — every agent build was throwing
   `AttributeError: 'RunnableRetry' object has no attribute 'bind_tools'` the moment it tried a
   real tool-calling path. All 146 tests still passed because none of them exercise real
   `bind_tools()` (they mock around agent construction) — worth adding one non-mocked
   smoke-level check for this specific path so it can't regress silently again. Fixed by using
   `ChatOpenAI`'s native `max_retries=3` constructor param instead — same retry behavior, at
   the HTTP layer, no wrapper that breaks tool binding.

2. **LiteLLM fallback doesn't trigger on Groq's billing error.** `litellm_config.yaml` has 4
   deployments under `model_name: qwen/qwen3-32b` (local vLLM, Groq x2, OpenRouter) — the
   comment says "same name -> automatic fallback," but in practice `routing_strategy:
   simple-shuffle` picked the Groq deployment, it failed with Groq's billing-restricted error
   (HTTP 400 `BadRequestError`), and LiteLLM gave up entirely (`Available Model Group
   Fallbacks=None`) instead of trying the other 3 deployments. Looks like LiteLLM's
   same-model-name failover doesn't retry across deployments for 400-class errors by default —
   needs an explicit `retry_policy` / fallback-exception config to actually be reliable.
   - [ ] Investigate LiteLLM's `retry_policy` / `content_policy_fallbacks` /
         `fallbacks:` settings to make 400-class provider errors retry to the next deployment
   - [ ] Until fixed, `.env`'s `GENERATION_API_BASE` is pointed directly at the local vLLM
         server (`http://localhost:8005/v1`, model `qwen3-32b`) as a workaround — bypasses the
         proxy entirely, so Langfuse cost logging on generation calls is currently not
         happening. Revert to `http://localhost:4000/v1` + `qwen/qwen3-32b` once the fallback
         is actually reliable.

3. **`metadata.source_file` is inconsistent across ingestion pipelines.** Regular content
   chunks (from `src/chunker.py`) store the full absolute path
   (`/home/karvanitis/vault-rag/data/input/doc_001_procurement_policy.pdf`); document/sheet
   summary chunks (from the table ingestion path) store just the bare filename
   (`doc_001_procurement_policy.pdf`). Anyone filtering Qdrant directly by exact `source_file`
   match (as opposed to going through the app's own resolution helpers) will get inconsistent
   results depending on which chunk type they're querying. Worth normalizing to one format at
   ingest time.

4. **Title-lookup questions are a known-hard case for this retrieval setup.** Verified
   directly: asking for a document's literal title (e.g. "what is the title of doc_001")
   sometimes returns `Unsupported` even though the title is in the ingested content — the
   title lives in a short, low-content chunk near the start of the document that doesn't score
   well against a semantic query like "LACERA procurement policy." Substantive policy
   questions on the same documents answered correctly. Not a bug to "fix" blindly — but if the
   fresh eval run (started tonight) shows `ocr_extraction`/title-type questions scoring
   noticeably worse than other single-doc factoids, this is why, and a targeted fix (e.g.
   always including the earliest 1-2 chunks of a doc as retrieval candidates when the question
   asks about the document itself) would be the real fix, not a prompt tweak.

## Eval numbers are inconsistent across the repo — reconcile before next portfolio push

**RESOLVED as of the 2026-07-06 session summary at the top of this file** — README.md,
docs/CASE_STUDY.md, and eval/results/summary.json now all cite the same current 109-question
run. Kept below as historical context for how the numbers used to disagree.

Three different documents used to cite three different correctness numbers for "the"
82-question benchmark, and none of them agreed:
- `README.md` — Correctness 79.3%, Faithfulness 86.1%, Answer relevancy 92.1%
- `eval/results/summary.json` (the file the new `/eval/summary` API endpoint serves) — Correctness 70.7%, Faithfulness 100%, Answer relevancy 100%
- `docs/CASE_STUDY.md` — Correctness 84.6%, Faithfulness 86.7%, Answer relevancy 87.8%

This looks like results from different runs/judge configs never got fully synced back into
every doc that quotes them. What's already automated (`run_eval.py`, done):
- [x] `summary.json` now includes `correctness_by_question_type` — count + mean correctness
      per question type, computed fresh every run, not hand-written
- [x] `answer_results.jsonl` now carries `question_type` and `gold_evidence` (doc_id + quote)
      per row — no separate join needed to audit any question
- [x] `results/failures.md` is auto-generated every run — every question below 0.5
      correctness, with a heuristic failure category, gold vs. predicted answer
- [x] Judge prompt is quoted verbatim in `eval/README.md`

Also since this was written: the corpus grew from 82 questions/14 docs to 93 questions/18 docs
(added an SOP manual + a 3-file lease/amendment package, doc_015/doc_016a-c). The fresh run
below needs to happen on the *current* 93-question corpus, not the old 82 — so this reconciles
both problems in one pass. Also found and fixed: `eval/data/qa_pairs/` had two extra files
(`generated_verified_qa.json` + `generated_unanswerable_qa.json`, 133 questions from a parked
experiment) that `load_questions()` would have silently swept into any eval run — moved to
`eval/data/qa_pairs_unused/` so `make eval` now runs the intended 93, not 226.

LLM routing status as of this session — Groq's org is still billing-restricted.
`litellm_config.yaml`'s fallback chain includes a cross-provider tier (OpenRouter; the old
NVIDIA NIM fallback was dead/EOL'd), but LiteLLM doesn't actually fail over to it for Groq's
specific error (see item 2 above) — so `.env`'s `GENERATION_API_BASE` currently bypasses the
proxy entirely and points straight at OpenRouter (`https://openrouter.ai/api/v1`,
`qwen/qwen3-32b`), not the proxy. `EVAL_JUDGE_API_BASE`/`EVAL_JUDGE_API_KEY` also point
directly at OpenRouter (the eval judge client bypasses the proxy entirely regardless, see
`_judge_config()`). Langfuse cost logging on generation calls is not happening while this
bypass is in place.

**Comparison-question false-Unsupported, root-caused 2026-07-15.** The
`GENERATION_API_BASE` comment above (line ~15) assumed OpenRouter serves
`qwen/qwen3-32b` unquantized, escaping the ~19% false-"Unsupported" rate
found on the local AWQ-quantized vLLM. That assumption was never verified
and is likely wrong: `GET /models/qwen/qwen3-32b/endpoints` shows every
OpenRouter provider except Alibaba/Groq (both "unknown", not confirmed
full-precision) serving fp8 — DeepInfra, Nebius, SiliconFlow. OpenRouter
also silently varies WHICH provider serves a given call ("provider" field
in the response flipped between Nebius and DeepInfra across identical
requests), adding routing variance on top of quantization.
Experiment (`src/llm_utils._openrouter_provider_extra_body`,
`OPENROUTER_PROVIDER_PIN` env var, currently unset/off): pinned every
generation call to DeepInfra only and ran the same comparison question
("Compare the leave policies in doc_009 and doc_010") 5x. Result: 5/5
identical — but 5/5 wrong (flat "Unsupported" despite both documents
being correctly retrieved every time, confirmed via the `sources` field).
This rules out cross-provider routing as the dominant cause and points at
DeepInfra's fp8 quantization itself. Reverted the pin (strictly worse
than the unpinned mixed-but-sometimes-correct baseline).
**Alibaba tested too (2026-07-15), also inconclusive/worse.** Pinned to
Alibaba (the model's original publisher, best candidate for actual full
precision) and ran the identical 5x comparison-question test. Result:
*more* variable than DeepInfra, not less — 3/5 runs retrieved only
doc_009 (flat "Unsupported"), 2/5 retrieved only doc_010 (a real,
reasoned partial answer explicitly noting doc_009 "was not retrieved").
Neither which document gets dropped nor the response style was stable
across runs. Reverted the pin (same as DeepInfra — worse than baseline).

Conclusion: quantization was a reasonable hypothesis (DeepInfra's
clean-but-wrong 5/5 fit it) but Alibaba's noisier, differently-shaped
failures don't fit "compression is the cause" — a genuinely full-
precision provider should be at least as stable as a quantized one, not
less. The instability more likely sits in how the *agent's own tool-
calling* decides which document(s) to search on a comparison question
(sometimes both, sometimes only one, inconsistently) rather than purely
in generation quality. That's a LangGraph/tool-orchestration question,
out of scope for a provider pin. Accepting comparison-question
inconsistency as a known, documented limitation for now (see the
retry-logic mitigations already shipped — they reduce but don't
eliminate this) rather than continuing to chase it via provider choice.

**Deterministic comparison path built, tested, then reverted (2026-07-15) — dead end, but a useful one.**
Tried the "stop making the ReAct agent choose tools consistently, resolve the
comparison deterministically instead" approach: a two-document comparison
naming both documents by `doc_XXX` id would retrieve independently per
document (via the same `search_knowledge_base` tool, bypassing agent
tool-selection), guarantee both had evidence before synthesizing, and answer
with one direct LLM call. Fully implemented in `src/answer_pipeline.py`
(`_deterministic_comparison_answer`), unit-tested, no regression. Then
checked it against the real eval set and found it was **dead on arrival**:
zero of the 15 `cross_document` prose-comparison questions name a doc_id
literally — they're described by topic ("the procurement policy vs the
services contract terms"), which is exactly the kind of resolution a regex
can't do. The 4 questions that *do* name doc_ids literally are all
spreadsheet comparisons (`query_excel`'s job, not `search_knowledge_base`'s —
a guard against this was needed to avoid a regression there). Reverted the
whole path (2026-07-15) — real users never type document IDs, so this was
solving a problem that doesn't occur in practice. Lesson: don't build
deterministic *id-based* routing for something the input will never contain;
if a deterministic path is worth revisiting, it needs to resolve documents by
title/description (fuzzy matching against the doc registry), which is a much
harder and riskier problem than this one, not attempted.

**GENERATION_MODEL switched qwen3-32b → openai/gpt-oss-120b (2026-07-15), plus a real bug fixed.**
User's hypothesis ("maybe it's the model, not the agent") tested directly:
built two fresh agents (same code, different `model_name`) and ran the same
3 known-failing comparison questions through both. `qwen3-235b-a22b` fixed
2 of 3; `gpt-oss-120b` (smaller, already used elsewhere in this project as
`EVAL_JUDGE_MODEL`, so zero new integration risk) fixed all 3 in that
isolated test. Switched `.env`'s `GENERATION_MODEL` to `openai/gpt-oss-120b`.

Running the full `cross_document` eval slice (not just the 3 hand-picked
questions) surfaced a real, separate bug: `gpt-oss-120b` is a reasoning
model — it spends tokens on hidden chain-of-thought before any visible
content. `_verify_grounded` (`src/answer_quality.py`, runs on every generated
answer) called the LLM with `max_tokens=10` expecting a bare "YES"/"NO";
reasoning tokens alone consumed that budget, so `content` came back `None`
and `_llm_call` (`src/llm_utils.py`) crashed on an unguarded `re.sub()`. Fixed
both: `_llm_call` now returns `""` instead of crashing on `None` content
(makes `_verify_grounded`'s existing "fail open on any judge error" behavior
apply naturally instead of raising), and `_verify_grounded`'s `max_tokens`
bumped `10 → 128` for headroom. This bug would silently produce empty
answers with *any* reasoning-style generation model, not just this one —
worth remembering if the model is changed again.

**IMPORTANT correction, same session: the eval numbers above through this
point were bogus.** `eval/run_eval.py`'s `--category` filter does an exact
string match against `question_type`; `cross_document` (what I'd been
passing) never matches the real stored value `cross_document_compare`, so
`generate_answers()` silently produced 0 rows on *every* invocation this
session, and `judge_answers()` (which never checks whether the raw-answers
file is fresh) kept re-scoring the same stale `raw_answers.jsonl` left over
from 2026-07-11 — before the model swap, before any of today's fixes. Every
number quoted above (0.8, 0.775, 0.825, 0.8 again) was judge nondeterminism
on byte-identical frozen predictions, not a measurement of any code change.
The "same two questions fail identically across separate full runs, despite
passing 3/3 in isolation" mystery wasn't nondeterminism at all — it was the
same cached answer being re-judged with slightly different judge-model
noise each time. Fixed the trap itself: `_load_questions_for_run` now raises
if `--category` matches 0 questions, listing the valid `question_type`
values, instead of silently proceeding on stale data.

**Real, freshly-generated run (2026-07-15, `--category cross_document_compare`,
generation confirmed fresh via file mtime + "Generated 20/20 answers" log
line, not just judge output):** `cross_document_compare` **0.825** (15
perfect / 3 partial / 2 zero out of 20), faithfulness 1.0. All 5 questions
that were broken in the true baseline — the procurement-vs-services-contract
extension question, the evergreen-prohibition question, and the lease
3-way-amendment question — now score a clean **1.0**. The fix that actually
worked: `gpt-oss-120b`'s grounding judge (the post-generation check in
`_verify_grounded`) is stricter on comparative/inferential claims than the
generation model itself, and was flipping correct comparison answers to
"Unsupported" ~1/3 of the time even with full evidence coverage — root-caused
by diagnostic script (dumped `sources` + answer across repeated runs with
verify on/off: 6/6 correct with verify off, 4/6 with verify on, always both
docs present regardless). Fixed by skipping the grounding check specifically
for comparison questions (`stream_agent`'s new `skip_grounding_check` param,
threaded from `answer_pipeline.answer_one`) — comparisons already get their
own doc-coverage retry (`_comparison_incompleteness`), which checks the thing
that matters without the grounding judge's false-positive risk. Comparison-
question reliability is genuinely fixed for the class of failure this session
investigated (both-sides-retrieved-but-answer-still-wrong-or-refused).

**Screwfix-Direct-style multi-part-split bug — root-caused and fixed
(2026-07-15).** `_is_multi_part_query` (`src/answer_quality.py`) detects "?
And for" as a multi-part boundary (its own `\?\s+and\s+for\b` pattern), but
`_split_multi_part_query`'s actual split patterns had no matching rule for
it — detector and splitter were out of sync, so a detected multi-part
question silently fell through to "return the whole thing unsplit," and the
agent then answered only the second half. Fixed by adding "And for" to the
question-boundary split alternation; unit-tested
(`tests/test_answer_quality.py`). Confirmed live: the question now correctly
splits into two numbered sub-answers. This exposed a *different*, narrower,
still-open problem: 3 of the 5 Excel-comparison questions in this eval slice
now split correctly but get one sub-part's field wrong (e.g. answering with
the wrong column, or asking for clarification instead of using the row
already identified in the question) — a per-field retrieval-precision issue
on split spreadsheet sub-queries, not a split-detection bug. Not yet fixed.

**Excel per-field retrieval precision on split sub-questions — root-caused
and fixed (2026-07-15).** The Screwfix repro's SQL: `WHERE "Department"
ILIKE 'PLACE / STREET SCENE'`. But per the gold row data, Directorate="PLACE"
and Department="STREET SCENE" are two *separate* columns — the question's
"PLACE / STREET SCENE" phrasing is a display convention joining them, not a
literal stored value. Filtering the whole joined string against one column
matches zero rows, which is why the row lookup silently failed (or, worse,
one variant filtered the joined value against the wrong column entirely:
`"Purchase of Expenditure" ILIKE 'PLACE / STREET SCENE'`). Fixed by adding a
rule to `SQL_PROMPT_HEADER` (`src/prompts.py`): a "/"-joined value in the
question is often two separate field values that need two separate column
filters, with the existing Directorate/Department pair given as the worked
example. Confirmed live: Screwfix now returns the exact gold values
(MATERIALS, 39.54) for the previously-failing sub-part.

**New, separate bug found while re-verifying the SQL fix: gpt-oss-120b
leaks its "harmony" reasoning-channel markers into visible answers
(2026-07-15).** gpt-oss's response format names its hidden chain-of-thought
channel "analysis" and the real-answer channel "final"; reproduced live via
OpenRouter streaming that this boundary isn't always cleanly separated
before content reaches the client, leaking the bare channel-name word glued
directly onto real content with no space ("1. final5239.0", "finalJuly 1
2024 ..."). Intermittent, not deterministic — an isolated repro of the exact
same question came back clean on one run, leaked on another, consistent with
the OpenRouter per-call provider-routing variance already documented above.
Fixed the cosmetic case: `strip_leaked_headers` (`src/answer_pipeline.py`)
now strips a bare "analysis"/"final" only when glued with no space to an
uppercase letter or digit (legitimate prose use always has a space after,
so this can't strip real content) — unit-tested. A second, deeper variant of
the same root cause was also seen once: one row's *entire* answer was leaked
raw reasoning text, never reaching the final channel at all (same shape as
the earlier `_verify_grounded` max_tokens-starvation bug, but in the main
agent's own synthesis call). Not fixed — the real fix is either raising
`ChatOpenAI`'s `max_tokens=2048` (`src/rag_agent.py`'s `build_rag_agent`) or
setting OpenRouter's `reasoning: {effort: "low"}` for this model, both of
which are cost/latency tradeoffs that need a decision, not just a bug fix.

**Deep reasoning-leak variant — closed by decision, not fixed (2026-07-15).**
Tested `reasoning: {"effort": "low"}` via OpenRouter's `extra_body` directly
against the real endpoint: only ~10% reduction in reasoning-token count at
realistic budgets (70→63 tokens), useless at small budgets (still starves at
`max_tokens=10` even with `effort=low`). Not a real lever. Advisor call: this
variant was seen exactly once, cost 0.5pt not a full miss, and an isolated
repro of the same question came back clean on retry — an n=1 sighting, not a
measured pattern. `max_tokens`/reasoning-effort are global knobs that add
cost/latency to *every* call in the app; not worth spending on an unquantified
single sighting. Documented, not pursued further. Reopen only if the full
corpus run below shows this recurring at meaningful frequency.

**Finding 5 (feature sprawl) — closed, no code change.** Nav already demotes
secondary screens (Feedback, Quality Evaluation, Google Drive) into dropdown
groups, distinct from the two top-level actions (Sources, Conversations) that
support the core flow. Confirmed adequate by inspection; nothing to fix.

**Model swap validation status: CONFIRMED, full-corpus run landed
2026-07-16.** Clean 109/109 run (no resume, fresh `raw_answers.jsonl`),
generation + judging both completed with no crashes. Aggregate:
correctness 0.838, faithfulness 0.861, answer_relevancy 0.869.
`correctness_by_question_type`:

| category | n | correctness |
|---|---|---|
| table_grounding | 3 | 1.000 |
| numeric_lookup | 6 | 0.917 |
| table_lookup | 16 | 0.938 |
| ocr_extraction | 25 | 0.860 |
| single_doc_factoid | 17 | 0.812 |
| unanswerable | 10 | 0.800 |
| negation_check | 5 | 0.800 |
| cross_document_compare | 20 | 0.775 |
| numeric_reasoning | 4 | 0.750 |
| figure_grounding | 3 | 0.667 |

No category collapsed. `cross_document_compare` at 0.775 on the full 20 is
consistent with the earlier 0.815-0.825 slice estimate within known judge
noise (±0.02-0.05 observed on identical data earlier this session — see
above). `figure_grounding` is the lowest at 0.667 but n=3 (2/3 correct) —
too small to read as a regression signal one way or the other; this
category was never separately benchmarked pre-swap, so there's no prior
number to compare against. Structured (Excel/DuckDB) answer_accuracy
0.762 (n=21), unanswerable correct-refusal-rate 0.786 (n=14) — both
reasonable, no guardrail failure. Retrieval metrics (PDF/OCR side,
n=74): hit@5 0.986, MRR 0.854, evidence_recall@10 0.944 — retrieval was
never the bottleneck, consistent with earlier findings.

**Conclusion: gpt-oss-120b swap is a net win, no regression found. Closing
this as final.**

- [x] **Done, 2026-07-16**: `README.md` and `docs/CASE_STUDY.md` both updated to
      quote the 2026-07-16 run (`eval/results/summary.json`) — correctness 83.8%,
      faithfulness 86.1%, answer_relevancy 86.9%, retrieval/structured/unanswerable
      metrics, and the `correctness_by_question_type` breakdown table, all pasted
      into both docs' Evaluation sections. Model name updated from `qwen/qwen3-32b`
      to `openai/gpt-oss-120b` in the eval-section prose only — the Tech stack
      table further down README.md still says `qwen/qwen3-32b` for the generation
      LLM and was deliberately not touched (out of scope; a larger doc-consistency
      pass, not an eval-numbers transcription).
- [ ] The ablation table (baseline dense-only → +hybrid → +rerank → +doc-routing → final) is
      already tracked below under "Pipeline improvement story" — do it in the same pass since
      it requires the same kind of fresh eval run(s)

## More found later the same night (2026-07-03, after the above was written)

**Two more real API-key-resolution bugs, same root cause, now fixed:**
- `build_rag_agent()`'s key resolution (`src/rag_agent.py`) always picked `LITELLM_MASTER_KEY`
  whenever it was set, regardless of which `generation_api_base` was actually configured —
  silently sent the proxy's key to a real provider being hit directly (401 on OpenRouter).
  Fixed: resolution now keys off the actual base URL.
- `_llm_call()` in `src/llm_utils.py` (used by 5 call sites — the answer-repair/fallback paths
  in `answer_quality.py` and HyDE in `retrieval_tool.py`) had the identical bug, plus used
  `os.getenv()` for `GROQ_API_KEY`/`OPENAI_API_KEY`, which never actually contains `.env`
  values (pydantic-settings reads `.env` directly, does not populate `os.environ`). Fixed the
  same way, centrally, in the shared helper — covers all 5 callers at once.
- Both fixes added `OPENROUTER_API_KEY` as a proper exported constant in `src/config.py`
  (previously only an internal `Settings` field, unlike `GROQ_API_KEY` etc.).
- Caveat on the earlier "local vLLM has a ~19% false-Unsupported rate" finding: that was
  measured *before* the `_llm_call` fix. Local vLLM doesn't validate its API key at all, so the
  bug likely didn't affect that specific run's repair/fallback calls — but this hasn't been
  re-confirmed. Don't treat the 19% number as settled without re-testing now that the bug is
  fixed.

**New feature, implemented and tested, not yet validated against real data:**
- Post-generation grounding check (`_verify_grounded` in `answer_quality.py`,
  `_apply_grounding_check` in `rag_agent.py`) — one extra LLM call per answered (non-Unsupported)
  query, verifying the answer is supported by retrieved context; downgrades to `Unsupported` on
  a failed check. Toggle: `POST_GENERATION_VERIFY_ENABLED` in `config.py` (default on). 9 new
  mocked unit tests pass. **No eval run has completed since this was added** — its real-world
  effect on correctness/refusal-rate/latency is unmeasured. Check this specifically in the next
  full eval run's numbers vs. a run with the flag off.

**Still missing from the hallucination-prevention checklist (discussed at length, not built):**
- [ ] Citation-required generation as a hard gate — currently prompt-only, nothing checks "did
      this answer cite a source" and rejects/retries if not
- [ ] Conflict/contradiction handling — no instruction anywhere for "if two sources disagree,
      say so" instead of picking one silently
- [ ] Cross-document minimum-evidence rule as an explicit hard gate — decomposition covers part
      of this but it's not verified as a real enforced check

**Decomposition-pipeline call count — discussed, NOT implemented, needs an ablation first:**
Multi-hop questions currently cost ~8 LLM calls (1 decompose + ~4 ReAct reasoning/tool-call
turns + ~3 HyDE calls, one per sub-question). Two proposed cuts:
- Skip the ReAct "should I call a tool" reasoning turn per sub-question — low risk, the
  sub-question already implies retrieval is needed by construction (decomposition's whole job)
- Skip HyDE per sub-question — real, unmeasured risk (HyDE bridges question-vs-document
  vocabulary gaps; a decomposed sub-question isn't guaranteed to already use document wording)
Both cuts also lose the ReAct loop's ability to adaptively retry/redirect within a sub-question
if the first retrieval comes up short — a real capability, not just overhead, especially on
hard cross-document questions.
- [ ] Do not implement blind. Add as rows to the ablation table (already tracked above) — run
      the eval with and without each cut, compare Hit@5/correctness, decide from data

**OpenRouter budget is tight — check before the next full eval attempt:**
The OpenRouter account had ~$5 total credit; ~$3.55 (71%) was already used as of tonight,
~$1.45 remaining, and it's not possible to isolate how much of that was tonight's testing vs.
prior usage. The 93-question generate run was stopped mid-way (~17/93) for this reason —
`eval/results/raw_answers.jsonl` currently holds an incomplete/unusable partial set.
- [ ] Before attempting the full 93-question run again: either top up OpenRouter credit, or
      switch back to local vLLM now that the `_llm_call`/key-resolution bugs are fixed (the
      quality concern from the earlier vLLM run is a separate, still-open question — see above)

## Uncommitted changes
- [ ] Working tree has modified `.github/workflows/ci.yml`, `README.md`, and deleted `Dockerfile` + `app.py`. Review and commit:
  ```bash
  git diff                          # inspect changes
  git add .github/workflows/ci.yml README.md
  git rm Dockerfile                  # if intentionally deleted (app.py already removed)
  git commit -m "chore: clean up legacy files, update CI and README"
  # Do NOT commit CLAUDE.md — it is gitignored
  ```

## Secrets / keys
- [ ] Confirm `.env` has `GROQ_API_KEY`, `QDRANT_URL`, and `LANGFUSE_*` (optional) set
- [ ] (Optional) `LITELLM_MASTER_KEY=sk-vault-local-dev` is already in `.env` from the LiteLLM integration

## Live validation
- [ ] Bring up services: `docker compose up -d` (Qdrant + Postgres + LiteLLM proxy + Redis)
- [ ] Run ingestion smoke on a mixed-format document (a born-digital PDF, a scanned PDF, and an Excel file)
- [ ] Ask a question in the UI and confirm cited answer comes back
- [ ] Run the eval to confirm the shipped numbers reproduce:
  ```bash
  uv run python eval/run_eval.py      # 82-question benchmark
  ```
  Expected: hit@5 ~94%, faithfulness ~86%, relevancy ~92%

## Demo assets
- [ ] Create `assets/` if not already present
- [ ] Screenshot: a question answered with `[Source N]` citations and sources panel visible
- [ ] Screenshot: the Next.js document inspector showing retrieved chunks and parsed markdown
- [ ] Short screen recording: ingest a PDF → ask a factual question → cited answer
- [ ] Replace any `_pending_` placeholder in the README Demo section

## Publish
- [ ] Remote already configured (`github.com/k-arvanitis/vault-rag`). After committing above changes, push:
  ```bash
  git push -u origin main
  ```
- [ ] Verify the CI badge stays green

## LiteLLM integration (optional — do after publishing)
Three open blockers are documented in `TODO_LITELLM.md`:
1. App bypasses proxy — set `GENERATION_API_BASE=http://localhost:4000/v1` in `.env`
2. Semantic cache stores but doesn't hit — investigate similarity threshold / redisvl index
3. Langfuse cost logging — verify a trace appears at `localhost:3000` after a proxied call

## Product & Eval Gaps

### Make it product-like
Current demo is generic ("ask questions over your documents"). Reframe around a concrete client scenario:
- **Law firm:** "Upload 50 NDAs, ask 'which ones have a liability cap below $10k?' — cited answer with contract name and page."
- **Operations team:** "Upload SOPs and policy docs, ask 'what is the escalation procedure for a critical incident?' — cited answer from the exact SOP."

Add 2–3 sample documents to `samples/` (a mock SOP PDF, a short policy doc) and a `scripts/demo.py` that runs a canned Q&A session a client can run in 5 minutes.

### Missing measurements
- [ ] Run the 82-question benchmark and confirm numbers reproduce: `uv run python eval/run_eval.py`
- [ ] Document before/after per pipeline component — what did adding BM25 hybrid, RRF, force-include table chunks, and the reranker each contribute to hit@5? This is the "pipeline improvement story" that wins rank-5 Upwork jobs ("fix my broken RAG").

### Pipeline improvement story
Add a `## How each component improved accuracy` section to the README showing incremental score lift per technique (baseline dense-only → +hybrid → +rerank → +force-include). Clients hiring for RAG debugging want to see this reasoning.

### OCR quality eval
The README claims scanned-PDF support as a first-class capability (`src/ingestion/ocr.py`,
LightOn OCR via local vLLM) with its own page-level routing (`src/parser/pdf_parser.py`), but
there's no eval slice that isolates OCR quality from the rest of the pipeline.
- [ ] Add a real scanned PDF to `eval/pdfs/` — `data/ocr/Xactimate.pdf` in the ClaimFlow repo is
      a genuine noisy scan (17 pages, zero text layer, confirmed via PyMuPDF) and is a good
      candidate to copy over
- [ ] Add a handful of questions against it to `eval/data/` so the 82-question benchmark includes
      at least one scanned-only document (currently unclear how many of the 14 source docs are
      scanned vs. born-digital — audit this first)
- [ ] Report hit@5 / faithfulness broken out **scanned vs. born-digital** in the README eval
      table, not just as one blended number — this is the evidence for the "text-layer pages
      skip OCR, only scanned pages hit the GPU" claim actually paying off in answer quality
- [ ] Optional: pull a couple of real noisy scans from the FUNSD dataset
      (https://guillaumejaume.github.io/FUNSD/, 199 annotated scanned forms with word-level
      ground truth) as a raw OCR-accuracy check decoupled from retrieval — useful if you want a
      pure OCR-fidelity number independent of the RAG pipeline. Note: FUNSD ships PNG + JSON only,
      no PDF wrapper, so each image needs wrapping into a single-page PDF (or feeding straight to
      `src/ingestion/ocr.py`) to use it here.

### Return the actual figure image for figure-grounding questions (implemented 2026-07-14)
Right now a figure/chart question only ever gets a VLM-generated text description of the
figure (`src/ingestion/vlm.py`), never the image itself. When a user asks something like
"show me Figure 4" or the answer would be clearer as a picture, the agent has no path to
return the source image — it can only paraphrase what the VLM described at ingest time.
- [x] At ingest time, the figure bbox is captured (`src/parser/pdf_parser.py`) and embedded
      as an HTML comment in the `[FIGURE_START]` marker; `src/chunker.py` extracts it into
      `metadata["figure_bbox"]`.
- [x] `src/answer_pipeline.py`'s `parse_sources` surfaces `figure_bbox` on each source; new
      `GET /documents/{file}/pdf/{page}/crop?bbox=...` endpoint (`api.py`) crops and returns
      it as PNG. `EvidencePanel.tsx` renders it when present.
- [ ] **Existing ingested docs need reingestion** — old Qdrant points predate this field and
      have `figure_bbox: null`, so old figure sources still show no image until reingested.
- [ ] Slack bot doesn't attach the image yet — only the web evidence panel does.

## Session close-out — 2026-07-16, backend closed

**Backend audit + final eval: DONE, committed.**
- Commit `deb0d46`: SQL injection guard (DuckDB stacked statements in
  `query_excel`), agent doc_registry cache-invalidation timing bug (`api.py`),
  read-modify-write races in `feedback_store.py`/`conversation_store.py`
  (threading.Lock added), dead Groq dependency in `excel_cleaner` (silently
  broke every spreadsheet ingest — rewired to `EXCEL_AGENT_*`/OpenRouter),
  Makefile `eval-cross` category-name bug.
- Commit `3b2f23d`: final clean 109/109 full-corpus eval landed, PROVISIONAL
  flag on the `gpt-oss-120b` model swap resolved to CONFIRMED (see numbers
  above at line ~584). No category regressed. Full breakdown in
  `eval/results/summary.json`.
- `scripts/smoke_test.py` (`make smoke`) rerun 2026-07-16 on current code
  (after all audit fixes) — 8/8 checks pass: PDF reingest+query, Excel
  reingest+query, cross-doc comparison. Backend confirmed end-to-end.

**Not done, not started — explicitly deferred to today/tomorrow's frontend pass:**
- Frontend work of any kind — user's stated plan: backend closed today,
  frontend tomorrow. Nothing frontend was touched this session.
- README.md / docs/CASE_STUDY.md eval-numbers transcription (see checklist
  above right below "Conclusion: gpt-oss-120b swap is a net win").

**Two uncommitted things found in the working tree, not mine, not touched:**
- `pyproject.toml` / `uv.lock` have an uncommitted `markitdown` dependency
  (+ transitive `defusedxml`, `magika`) — didn't add it, unclear origin,
  left unstaged. Ask the user before committing or reverting.
- Untracked `REVIEW.md` and `Screenshot from 2026-07-12 13-11-21.png` in
  repo root — same, unclear origin, left untracked.

**Audit surface not exhaustively covered** (time-boxed, not a known gap):
`src/file_resolver.py`, `src/connectors/google_drive.py`, `src/reranker.py`,
ingestion pipeline modules (`src/ingest.py`, `src/parser/pdf_parser.py`,
`src/chunker.py`) got a lighter pass than `src/tools/excel.py` and the
store/cache files. No known bugs there — just not scrutinized as hard.

**Deliberately not pursued:** deep gpt-oss reasoning-channel leak (whole-
answer-as-raw-reasoning variant, n=1 sighting, ~line 570 above) — reopen
only if it recurs at meaningful frequency in a future eval run.

## Session continued, 2026-07-16 — UI-blocking backend gaps

Working through the 4 backend gaps flagged in the UI refinement section
(inline citations, excel citations, retry, streaming) plus phase 8/UI tests.

- [x] **Excel/SQL citations** (`9a72beb`): query_excel now threads the
  matched DuckDB row text through to a real citation, merged into
  `answer_query`'s `sources` list. Verified live: a table-lookup question
  now returns a real source with `sheet` + `quote` for SpreadsheetEvidence
  to highlight against — previously always `sources: []`.
- [x] **Inline citation markers**: `build_citation_map` in
  `src/answer_pipeline.py` maps the last tool call's raw `[N]` markers to
  their real 1-based position in the final (deduped/reordered) `sources`
  list; `strip_leaked_headers` renumbers resolvable markers instead of
  stripping them. Scoped to single-part questions only (multi-part answers
  merge several independent agent runs' unrelated numbering — deliberately
  not resolved, falls back to the old strip-on-sight behavior). Frontend:
  `MessageList.tsx`'s new `AnswerContent` renders a resolved `[N]` as a real
  `CitationChip` inline instead of dead text. 4 new unit tests including the
  diversity-cap reorder case (marker order and final position genuinely
  diverge). **Not verified live** — `GENERATION_API_BASE` (OpenRouter) is
  returning `400 no_db_connection` on every attempt right now (3 retries),
  OpenRouter's own backend, not this change (the same endpoint served the
  PDF smoke-test question fine earlier). Retry once the endpoint recovers;
  logic itself is covered by unit tests including a real divergence case.

## Live UI testing round, 2026-07-16 — 4 bugs/gaps found and fixed

User clicked through the app in a real browser (this box has none) after
#12–#17 landed. Found real issues a curl smoke test and typechecking
couldn't surface:

- [x] **"Original PDF not available locally."** `_resolve_pdf_path` (now
  `_resolve_source_file_path`, renamed since it's used for xlsx/csv too)
  only checked `data/input/` and repo root — most eval-corpus documents
  (e.g. doc_002, the one that triggered this) are never copied there by
  `make seed`, only `eval/data/raw/`. Added as a fallback for the PDF
  page-image/highlight endpoints *and* `/table-sheet`'s raw-rows lookup
  (same bug, same root cause, would have hit Excel/CSV eval docs too).
- [x] **`[FIGURE_END]` leaking into a citation quote**, alongside raw
  markdown table pipes. Root cause: a chunk boundary split a figure block so
  only the closing tag (no matching `[FIGURE_START]`) landed in this chunk's
  body — the paired strip regex doesn't match an orphan tag. Fixed by
  stripping any leftover lone tag too.
  - **Follow-up, still 2026-07-16**: user reported the same quote still
    showed raw `|---|---|` pipes and a literal `<br>` after this fix. The
    FIGURE_END fix only handled the figure-tag half — the markdown-table
    pipes were a separate, still-unfixed bug: a chunk that's itself a
    pymupdf4llm-rendered table (not wrapped in `TABLE_START`/`TABLE_END`)
    had no pipe-stripping at all. Now strips `<br>` tags, `---`-only
    separator rows, and all `|` characters from the quote/excerpt text.
    The "Exact region unavailable" *image* fallback is still correct as
    designed (a table has no bbox on the rendered page) — only the *quote
    text itself* needed the fix. Unit-tested, not yet re-seen live.
- [x] **Streaming didn't actually stream** (see the SSE section above) —
  `stream_agent`'s tool-use path buffered the whole answer and yielded once;
  fixed with the `live_tokens`/`FinalCorrection` mechanism, verified live
  (continuous ~20ms-interval tokens over ~13.5s, not a single lump).
- [x] **Inspector showed the SQL generator's internals, not a clean table**:
  the "cleaned" side of the raw/cleaned comparison dumped `cleaned_md`
  verbatim into a `<pre>` — that file starts with a `[File: ... | Sheet:
  ...]` header and a schema/sample-values summary before the actual
  markdown table. Now strips everything before the first table row and
  renders it as a real HTML table via ReactMarkdown+remarkGfm. Raw/cleaned
  comparison now open by default; the SQL-generator schema block collapsed
  by default (previously the reverse of what the user wanted to see first).
- [x] **Citation row click → inspector deep-link, new feature**: clicking a
  row in `SpreadsheetEvidence` (the Evidence panel's side-by-side table) now
  opens the full Document Inspector with that same row highlighted there.
  `InspectTarget` gained a `quote` field; a new `lib/tableMatch.ts` holds the
  shared "quote contains this row's cell value" heuristic so both views
  agree on which row is the match (previously duplicated inline in
  `EvidencePanel.tsx` only). 4 new unit tests for the shared helper.
- [x] **"Open document" gets stuck**, reported same round: no root cause
  confirmed live (couldn't reproduce without a browser), but a real gap
  found by code inspection — `PdfInspector`/`TableInspector`/`SheetCompare`'s
  fetches had no `.catch`/error path and no timeout, so a slow or hung
  backend call (e.g. a blocking Qdrant scroll inside an `async def` route)
  left the panel on its loading skeleton forever with no way out. Added a
  client-side 20s timeout (`INSPECTOR_TIMEOUT_MS` in `lib/api.ts`, scoped to
  inspector metadata calls only — NOT `/ingest`/`/query`/`/eval`, which
  legitimately run long) plus error states in all three components. **This
  round of the fix was wrong** — see below.
  - **Round 2**: user reported "same" after the timeout fix. Guessed the
    backend was blocking the event loop (`document_chunks`'s Qdrant scroll,
    `document_table_sheet`'s pandas/openpyxl parse both run synchronously
    inside `async def` routes) and wrapped both in `run_in_threadpool`.
    Still hadn't measured anything — motion without evidence.
  - **Round 3, actual root cause, found via advisor-directed `curl -w
    time`**: both endpoints were already fast (25ms, 105ms) — rounds 1–2
    were fixing real but irrelevant issues. The actual bug: `/table-sheet`
    returns `cleaned_md` uncapped — the full per-sheet markdown table
    (6701 rows, 920KB for `doc_006_purchase_card_transactions_q1_2025_26.xlsx`),
    unlike `raw_rows` which was already `nrows=60`-limited. The frontend
    renders `cleaned_md` whole through `ReactMarkdown`+`remarkGfm` — parsing
    a 6700-row markdown table client-side freezes the tab. No network call
    ever hung, so the round-1 timeout could never have caught it, and no
    concurrency was involved, so round 2's threadpool wrap did nothing for
    it either. Fixed by truncating `cleaned_md` server-side to 60 rows
    (`_truncate_markdown_table` in `api.py`, same bound as `raw_rows`) —
    confirmed via curl: response for the same sheet dropped from 920KB to
    10.9KB. 4 new unit tests (`tests/test_api.py`). Not yet re-clicked live
    in a browser, but this is the first round with an actual measured,
    reproduced cause rather than a guess.

Verification status, honestly split — not all of #12–#20 got eyes-on
confirmation this round:
- **Live-verified** (curl or browser): streaming (#18, curl-confirmed
  continuous token arrival). The user's own browser session surfaced the
  4 bugs/gaps above, so those symptoms are confirmed real.
- **Fix applied + unit/type-checked only, not yet re-seen live**: the PDF
  path-resolution fallback (never re-curled for doc_002 specifically), the
  `[FIGURE_END]` strip (one unit test, never re-rendered), the Inspector's
  cleaned-table rendering (#19 — `tsc` + `extractTableMarkdown` unit test,
  never opened in a browser), and the citation-row-click deep link (#20 —
  `tsc` + `findMatchedRowIndex` unit test + a static check that
  `RightPanelTabs` forwards `onOpenFullSource` into `EvidencePanel`'s
  `onInspect`; never actually clicked).
- Next real verification step: user re-runs `make api`/`make ui` and checks
  (1) a citation row click opens the inspector with the row highlighted,
  (2) the cleaned table renders as an actual table, not raw markdown text,
  (3) a PDF citation for an eval-corpus-only doc (e.g. doc_002) now loads.

## 2026-07-16, later same day — inspector cleanup, notes, file counts, ref bug

- [x] **Excel inspector simplified**: dropped the schema/sample-values
  section and the per-chunk "Indexed chunk" cards from the Excel/CSV
  inspector — user just wants the cleaned table, per feedback.
- [x] **Cleaned-table row highlight**: the citation row-click highlight only
  ever hit the raw table (`SheetCompare`'s cleaned pane rendered via
  `ReactMarkdown`, which can't be ref'd per-row). Now parses the cleaned
  markdown table manually (`parseMarkdownTable`) so both panes highlight
  and scroll to the same matched row.
- [x] **Extracted notes surfaced**: `excel_cleaner`'s LLM-identified notes
  (footnotes/titles stripped out of the cleaned data) were computed at
  ingestion time and silently discarded — `_load_into_duckdb` only ever
  returned column names. Now threaded through into the per-sheet `.md`
  file as a trailing `**Notes:**` section, rendered below the raw/cleaned
  tables. **Only applies to newly ingested files** — existing indexed
  Excel/CSV docs need a reingest (real LLM calls, not instant/free) to
  backfill notes. Not run automatically.
- [x] **Page/sheet/row counts in the sidebar**: derived entirely from
  metadata already on Qdrant payloads (page numbers, sheet_name, the
  document_summary chunk's own "N rows." lines) — no extra file I/O per
  document. Guard by file extension added after finding live that a
  stray duplicate PDF entry also carried sheet_name/table_N metadata
  from its own embedded-table extraction, which would have wrongly shown
  "N sheets" on a PDF without it.
- [x] **"Ask across" popover flickering open/closed** — this one got an
  actual confirmed root cause instead of another guess. User's own
  description ("popover flickers/jumps open-close") plus this session's
  track record of guessing wrong (the stuck-inspector saga above) meant
  it was worth spending the effort to observe it directly: wrote a
  throwaway headless Playwright probe (no real browser available on this
  box) that opened the app, clicked the button, and dumped console
  errors. Found: `Warning: Function components cannot be given refs...
  Check the render method of PopoverTrigger, at Button`. `components/
  ui/button.tsx`'s `Button` was a plain function component, not wrapped
  in `React.forwardRef` — every trigger that renders one via Base UI's
  `render` prop (this popover, AppHeader's dropdowns, Sidebar/SourceCard
  tooltips, MessageList's citation popover) had no real DOM ref for Base
  UI to anchor positioning/open-state to. Fixed by wrapping `Button` in
  `forwardRef`; re-ran the same probe and confirmed the popover now
  stays open steadily across a 1.5s sampling window (previously it
  wasn't opening at all in the probe, consistent with unstable
  open-state tracking). Found the identical bug in `Textarea` while
  investigating (ChatPanel's auto-focus `textareaRef.current?.focus()`
  was silently a no-op) and fixed it the same way.
- [ ] **Document search bar** (requested same round) — not yet started.

Both new asks (search bar, and further UI polish) are still open — see
the running TODO list below rather than assuming complete.

## 2026-07-16 (later) — Portfolio finalization task (Phases 1–6), search bar done

The "document search bar" item above is done — see the "Ask across" search input
entry further up. This section covers the separate, larger portfolio-finalization
brief (make the project a polished, reliable AI engineering portfolio piece before
any new product capabilities), executed in 6 phases across `55c1a2d`..`4d7c8ec`.

- [x] **Phase 2 — deterministic cross-document comparison** (`55c1a2d`). The
  ReAct-agent comparison path (`answer_one`'s retry logic) was probabilistic —
  it let the agent decide whether/how to make a second retrieval call.
  `answer_comparison_deterministic` in `src/answer_pipeline.py` instead resolves
  which documents a comparison question names (explicit doc_ids, the UI's
  multi-select scope, or a conservative two-way title match), retrieves
  independently per document via a direct `search_knowledge_base` call
  (bypassing the agent's own tool-call decision), and makes one synthesis call
  over the guaranteed-complete evidence. Falls back to the old agent path for
  ambiguous comparisons. **Live-verified 5/5** repeated real runs, all covering
  both requested documents; a real+nonexistent-doc adversarial run correctly
  excluded the nonexistent one with no fabrication. 15 new unit tests.
- [x] **Phase 3 — one canonical eval result** (`969b8c8`). `docs/EVAL_SUMMARY.md`
  was stale (an 82-question run, different judge) and disagreed with
  README.md/CASE_STUDY.md's numbers. `eval/results/summary.json` now carries
  `benchmark_date`/`answer_model`/`judge_model`/`document_count`;
  `eval/generate_summary_doc.py` renders `docs/EVAL_SUMMARY.md` straight from
  it — no more hand-copied numbers. Did **not** invent a "multi-document
  evidence coverage" metric (the canonical run predates Phase 2); documented
  as a known gap instead, separate from Phase 2's live spot-check.
- [x] **Phase 1 — release checklist** (`c0403c3`). `docs/release-checklist.md`:
  exact commands for stack-up, automated checks, smoke test, e2e, benchmark,
  plus a 12-step manual pass and an honest known-gaps section (no Word/.docx
  demo sample despite ingestion supporting it; full 17-flow e2e not built).
- [x] **Phase 4 — deployment/startup reliability** (`c02193e`, `9fce94f`).
  docker-compose.yaml: healthchecks (bash `/dev/tcp` for qdrant, `python
  urlopen` for litellm/api since curl/wget aren't in those images, `node
  http.get` for frontend) + `condition: service_healthy` on every
  `depends_on` — api could previously start querying qdrant/litellm before
  either had finished booting. Found and fixed a real data-loss gap: 
  `DUCKDB_PATH` (`.duckdb/vault.db`) was never mounted as a volume — every
  Excel/CSV table the SQL agent cleans (real LLM calls) would be silently
  lost on container recreate. Added `_validate_startup_env()` in `api.py`
  (clear error naming the exact missing key when `GENERATION_API_BASE` needs
  one) and an embedding warm-up at startup (reranker already warmed up;
  embedding never did).
- [x] **Phase 5 — optional admin/viewer access mode** (`57b0ad6`, `fb2b02d`,
  `4d7c8ec`). `ACCESS_MODE=open` (default) is bit-for-bit today's behavior.
  `ACCESS_MODE=admin_viewer` gates the existing admin-only endpoint set
  (upload/reprocess/delete/clear/eval-run/feedback-resolve/drive-config)
  behind an HMAC-signed session cookie set by `POST /admin/login` (or the
  existing `X-API-Key` header) — enforced in `require_admin()`, not just
  hidden buttons. Frontend: login page, Sidebar hides upload/reprocess/delete,
  AppHeader hides Integrations + Feedback nav + shows Login/Logout, eval
  panel hides Run eval. 14 backend authz tests + a live Playwright test
  (mocks `/admin/session` against the real running dev server, since it
  stays in `open` mode). Known deliberate trade-off: no real session store
  (no per-login random token, no server-side revocation) — documented in
  `_admin_session_token()`'s docstring, matches the brief's "small and
  optional," not full session management.
- [x] **Phase 6 — portfolio packaging, light-touch** (`52c1b6a`). README
  already had most of this (Who this is for / Best fit / architecture /
  privacy table / eval results) — added the one-line product-promise
  positioning under the title, an "Access modes" section, the 3 new
  `/admin/*` endpoints in the API table. New `docs/demo-script.md` (8-step,
  3–4 min) and `docs/screenshot-shot-list.md` (8 planned captures, no
  screenshots actually taken/fabricated).

**Known gaps, carried forward honestly** (see `docs/release-checklist.md`'s
own "Known gaps" section for the current list):
- No Word (`.docx`) sample in the demo/eval corpus, despite `.docx` ingestion
  being implemented in `src/ingest.py` — never exercised.
- Full 17-flow Playwright suite not built — only the core chat flow and the
  admin-viewer-access flow are automated end-to-end today.
- `docker compose up --build` was validated (`docker compose config`,
  individual healthcheck commands confirmed via `docker exec`) but never run
  end-to-end from a clean clone this session (time).
- No CI-runnable (non-live-credential) test lane for the deterministic
  comparison path — only unit tests (mocked) and the live spot-check exist.

## Open, not fixed — 2026-07-23 demo-prep session (both real, both scoped, neither started)

- [ ] **Unscoped/open-corpus retrieval reliability.** LACERA-style question
  measured at 13/20 (65%) correct when searching the full open corpus vs.
  20/20 (100%) when scoped to the source doc via the UI's document
  selector. Root cause: LLM query-planning nondeterminism changes which
  chunks land in the reranked pool run to run — not a filter/architecture
  defect (see the `filter_token` fix in the 2026-07-22 session above).
  Candidate fix: multi-query fan-out (issue a few query phrasings per
  retrieval call, merge results) to reduce dependence on any single
  LLM-generated query — a real architecture change, not a quick patch.
  Workaround for now: always scope to a document via the UI selector for
  anything that needs to be reliably right.
- [ ] **Excel routing flakiness.** The agent sometimes calls
  `search_knowledge_base` instead of `query_excel` for a table-value
  question (wrong tool choice), gets no evidence, returns "Unsupported."
  `route_question`'s classification isn't reliable enough on its own —
  same class of gap as the already-fixed "document modality" case, which
  is hard-enforced at the tool layer via `FORCED_DOC_ID` instead of
  trusting the prompt directive alone. `query_excel` has no equivalent
  per-source hard-enforcement today (see the older TODO note referenced at
  the "hard-enforce it at the tool layer" comment in
  `src/answer_pipeline.py::answer_one`). Workaround for now: scope to the
  Excel doc via the UI selector when precision matters.
- [ ] **LLM provider portability — bring-your-own key (OpenAI, or any
  OpenAI-compatible endpoint, including local).** Client-facing gap: right
  now the LLM side is a patchwork of separate provider configs per
  subsystem (`GROQ_API_KEY`, `NVIDIA_API_KEY`, `OPENROUTER_API_KEY`,
  `FREE_LLM_API_KEY`, `CHUNK_LLM_API_KEY`, `EXCEL_AGENT_API_KEY`, each with
  its own base URL/model — see `src/config.py`), not a single swappable
  provider setting. A client who wants to run this against their own
  OpenAI key, a different vendor, or a fully local model (e.g. the
  existing local vLLM setup, per the global CLAUDE.md GPU cheatsheet) can't
  just drop in one key today — needs a real audit of which subsystem talks
  to which provider and a unified, OpenAI-compatible-by-default
  configuration surface. Scope not yet sized.
