# vault-rag — TODO to portfolio-ready

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
- **"Preserve streaming answers / stop generation" (§3B).** There is no token streaming
  today — `ChatPanel.tsx`'s `send()` awaits the full `/query` response; the `streaming`
  state is really just a loading spinner. Nothing to "preserve" because it doesn't exist.
  A real fix means the backend needs an SSE/chunked response endpoint. Flag, don't fake.
- **Retry** on a message doesn't exist either — only new-conversation. Same category.

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
8. [ ] Responsive passes (§16), remaining tests (§17), final terminology sweep (§19).

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
- [ ] **Retrieval flakiness broader than the RFQ case documented below.** Observed during
      this session: the exact same question against the exact same document
      (`doc_001` procurement-approval question) returned wildly inconsistent results
      across repeated calls — 6 sources with a real answer, then empty answer/0 sources,
      then a real answer with 0 sources, then empty again. Not caused by this session's
      code changes (verified: unrelated questions answer fine and fast; no exception
      swallowing added). Looks related to `GENERATION_API_BASE` pointing directly at
      OpenRouter (rate limits?) combined with `stream_agent`/`run_once` in
      `src/answer_pipeline.py` never surfacing a failed/empty generation as an error —
      it just silently returns `answer: ""` with `200 OK`. Worth root-causing before
      calling the demo reliable: (1) add logging/error surfacing when the token stream
      produces nothing, (2) check whether this is OpenRouter rate-limiting.
- [ ] `TODO_LITELLM.md`/older sections of this file reference Postgres and `app.py` —
      stale, current `docker-compose.yaml` uses Redis and the entrypoint is `api.py`.
      Cosmetic cleanup, low priority.
- [ ] Frontend still has near-zero unit/component test coverage (one e2e smoke test
      now exists; `SourceCard`, `EvidencePanel` bbox math, etc. have no unit tests).
      Not blocking — common for a portfolio project to skip this — but note it if asked
      about test coverage in an interview.

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

Still open — needs an actual fresh eval run, can't be scripted around:
- [ ] Run one fresh, authoritative pass on the 93-question corpus: `make eval` (or
      `POST /eval/run` from the new eval dashboard) with `EVAL_JUDGE_MODEL=gpt-oss-120b` set
      (matches what the README claims)
- [ ] Update `README.md` and `docs/CASE_STUDY.md` so both quote the *same* run as
      `eval/results/summary.json`
- [ ] Paste the new `correctness_by_question_type` breakdown into the README/CASE_STUDY
      Evaluation sections (data will exist in summary.json after the run above — just needs
      transcribing, do not hand-type numbers before the run produces them)
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
