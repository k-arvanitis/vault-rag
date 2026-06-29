# vault-rag — TODO to portfolio-ready

See also `TODO_LITELLM.md` for the three open LiteLLM semantic-cache + Langfuse blockers
(tracked separately — those are optional enhancements, not blockers for publishing).

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
