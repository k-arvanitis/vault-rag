# vault-rag — Progress & Plan

Single source of truth. Update this at the start of every session.
Last updated: 2026-06-30

---

## Status right now

**Closest to done of all portfolio projects.** CI badge is green, eval numbers are strong, remote is configured. One commit needed to clean up the working tree, then push.

---

## Completed

- [x] LangGraph ReAct agent, hybrid retrieval (dense + BM25 + sparse RRF)
- [x] BGE cross-encoder reranker
- [x] Multi-format ingest: born-digital PDF, scanned PDF (OCR), Excel/CSV
- [x] Force-include table chunks (Excel rows always surface in results)
- [x] Next.js UI with document inspector (side-by-side PDF page vs extracted markdown)
- [x] FastAPI backend, Qdrant vector store, Docker Compose stack
- [x] CI: `.github/workflows/ci.yml` — green badge in README
- [x] Eval harness: 82-question benchmark, results in `eval/`
- [x] **Eval numbers: hit@5 ~94%, faithfulness ~86%, relevancy ~92%**
- [x] Remote configured: `github.com/k-arvanitis/vault-rag`

---

## Next up (in order)

1. Confirm `.env` has `GROQ_API_KEY` set (minimum to run), then `git push -u origin main`
2. **Demo assets** — with backend on `:8001` + `npm run dev`:
   - Screenshot: cited answer with `[Source N]` citations + sources panel
   - Screenshot: document inspector (PDF page vs extracted markdown)
   - Short recording: ingest PDF → ask question → cited answer
   - Replace `_pending_` placeholder in README Demo section
3. **Sample corpus** — add 2–3 docs to `samples/` + `scripts/demo.py` so a client can run it in 5 min without their own files

---

## Backlog (nice-to-have)

- [ ] Pipeline improvement story in README: show incremental hit@5 lift per technique (baseline dense → +hybrid → +rerank → +force-include). Clients hiring for RAG debugging want this.
- [ ] LiteLLM semantic cache (3 open blockers in `TODO_LITELLM.md`) — optional, not blocking
- [ ] Langfuse cost logging — optional

---

## Key numbers

| Metric | Value |
|---|---|
| hit@5 | ~94% |
| Faithfulness | ~86% |
| Answer relevancy | ~92% |
| Eval dataset | 82 questions (custom) |
