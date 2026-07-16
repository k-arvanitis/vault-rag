# Release checklist

Exact commands for running Vault RAG, its automated checks, and the manual
verification pass before a demo or release. Written for a clean clone on
this box's stack (Qdrant, DuckDB, FastAPI, Next.js, LangGraph) — see
`README.md` for the full setup story and `.env.example` for required config.

## 1. Bring the stack up

```bash
docker compose -f docker/ingestion-stack/docker-compose.yaml up -d   # Qdrant + LiteLLM + OCR
uv run uvicorn api:app --host :: --port 8001 --reload                # or: make api
cd frontend && npm run dev                                           # or: make ui
```

First run only:

```bash
uv run python scripts/seed.py     # make seed -- downloads + ingests 4 starter docs
```

## 2. Automated checks

```bash
uv run pytest tests/ -v --tb=short          # make test -- backend unit tests, all external services mocked
uv run ruff check src/ api.py slack_app.py eval/       # make lint
uv run ruff format --check src/ api.py slack_app.py eval/
cd frontend && npx tsc --noEmit             # frontend typecheck
cd frontend && npx vitest run               # frontend unit tests
cd frontend && npm run build                # production build
```

## 3. Live smoke test (real Qdrant/DuckDB/LLM — not mocked)

```bash
uv run python scripts/smoke_test.py         # make smoke
```

Run this before a release or after touching ingestion, retrieval, or
generation code — `make test` alone only proves the code paths are
internally consistent, not that the real services actually respond.

## 4. End-to-end (Playwright)

```bash
cd frontend && npx playwright test         # or: npm run test:e2e
```

Current coverage (`frontend/e2e/golden-path.spec.ts`) exercises the core
chat flow. Known gap: the fuller 17-flow checklist below (comparison,
retry, reprocess/delete, conversation persistence across a restart) is not
yet all automated — see "Known gaps" at the end of this doc.

## 5. Benchmark

```bash
uv run python eval/run_eval.py              # make eval -- full run, real LLM calls, several minutes
uv run python eval/run_eval.py --category cross_document_compare   # make eval-cross -- one category only
uv run python eval/generate_summary_doc.py  # re-render docs/EVAL_SUMMARY.md from summary.json
```

Requires live `GENERATION_API_BASE`/`EVAL_JUDGE_API_BASE` credentials (see
`.env.example`). Never hand-edit `docs/EVAL_SUMMARY.md` or
`eval/results/summary.json` — regenerate them from a real run.

## 6. Manual release pass

Walk the golden path once in a real browser before calling a build ready:

1. Upload a PDF and a spreadsheet — watch status go to Ready.
2. Ask a factual PDF question — confirm the answer streams token-by-token
   and citation chips appear.
3. Click a citation — Evidence panel shows the right document/page; PDF
   highlight either lands on the exact passage or shows the honest
   "Exact region unavailable" fallback (both are correct behavior).
4. Ask a spreadsheet aggregation question — Technical details shows the
   generated SQL; Evidence shows the matched sheet row.
5. Click that row — Document Inspector opens with the same row highlighted.
6. Scope "Ask across" to two documents, search inside the scope selector.
7. Ask a comparison question naming two real documents — confirm sources
   include both.
8. Ask something clearly out-of-corpus — confirm a plain refusal, not a
   guess.
9. Retry a failed/unsupported answer in place.
10. Reprocess, then delete, a document (through the confirmation dialog).
11. Save a conversation, reload the app, resume it.
12. Restart `make api`/`make ui` — confirm documents and saved
    conversations are still there (persistence isn't in-memory-only).

## Known gaps (honest, not silently deferred)

- **Demo corpus has no Word (.docx) sample.** `.docx`/`.doc`/`.ppt`/`.pptx`
  ingestion is implemented (`src/ingest.py`'s `run_ingest`/`--data-format`),
  but never exercised by the 18-document eval corpus or `make seed`. Add
  one permissively-licensed `.docx` and a matching manual/e2e check before
  claiming Word support is verified, not just implemented.
- **Full 17-flow Playwright suite is not yet built.** Only the core chat
  flow is automated end-to-end today. Flows most worth adding next:
  comparison-question source coverage, retry-in-place, reprocess/delete
  confirmation, save/reload/continue a conversation, and restart-persistence
  (item 12 above) — these are the ones a regression could silently break
  without a human in the loop.
- **A dedicated deterministic-config or mock-server test lane for the
  comparison path does not exist yet.** Phase 2's `answer_comparison_deterministic`
  has full unit coverage with a fake tool/LLM (`tests/test_answer_pipeline.py`)
  and was verified live 5/5 against the real running API — but there's no
  CI-runnable equivalent that doesn't require live model credentials.
