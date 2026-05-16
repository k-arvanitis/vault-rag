# Key engineering decisions

The decisions below are the ones that materially changed eval scores or removed entire failure modes. Not architectural overview — the [README architecture diagram](../README.md#architecture) covers that.

## Per-page PDF routing

Every PDF page is routed independently — not the whole document. Born-digital pages (≥50 chars of extractable text) go through pymupdf4llm on CPU with zero API calls. Scanned pages have no usable text layer and are sent to a local LightOn OCR vLLM server (or `unstructured` + tesseract on CPU when `PDF_PARSER=cpu`). Mixed documents — a contract where pages 1–10 are digital and page 11 is a faxed addendum — are handled correctly per page with no configuration.

Running OCR on born-digital pages is counterproductive: the vision model transcribes what it *sees* in a rendered image, introducing errors on numbers, equations, and code that are already perfect as text. The 50-char threshold keeps both paths completely separate.

## Three-pass deterministic table repair

Research papers and GHG inventory reports contain tables that OCR cannot handle correctly. Three specific failure modes are fixed without any LLM involvement:

1. **Two-row HTML headers** — OCR places sub-column names (`R@1 R@5 R@10`) in the first tbody row instead of the thead. Detected structurally; flattened to `KaLMv2 R@1` programmatically.
2. **LaTeX `$\text{...}$` tables** — arXiv papers embed raw LaTeX in OCR output. Header tokens are parsed and a correctly-aligned Markdown table is rebuilt.
3. **Missing rightmost column** — wide tables frequently lose the last column in vision-language OCR. `pypdf` falls back to the PDF text layer and recovers values positionally, including splitting concatenated decimals (`2.9524.141` → `2.952`, `4.141`).

All three repairs are structural — no hallucination risk; alignment is guaranteed by counting data-row tokens.

## Deterministic point IDs (idempotent re-ingestion)

Every Qdrant point is assigned a SHA-1 hash of `(file_name, chunk_index)` rather than a random UUID. Re-ingesting the same file overwrites vectors in place instead of appending duplicates. With random IDs, re-running ingestion silently doubles every chunk in the index — a subtle bug that degrades retrieval precision without any visible error.

## HyDE query expansion

The agent first generates a *hypothetical answer* using a fast LLM call and embeds that instead of the raw question. The embedded hypothetical is semantically much closer to real document text, improving recall for vague queries like "what were the sales figures" where the question shares few tokens with the answer.

## Sheet-summary chunks with ingest-time sample values

Finding the right spreadsheet sheet is a two-step problem. Step one is *discovery*: which of the 100+ indexed sheets actually contains the data the user is asking about? Step two is *retrieval*: run a SQL query against that specific DuckDB table.

Discovery fails when the query contains entity names — supplier names, beneficiary names, project codes — that do not appear in the sheet's column headers. A user asks *"what did WATES PROPERTY SERVICES LTD receive in April?"* but the doc_007 sheet_summary only contains column names like `Beneficiary, Total, Transaction Number`. The embedding of the query is semantically close to any spend-report-shaped sheet, not specifically to the one that contains WATES rows.

The fix mirrors the logic behind HyDE, but applied at ingest time instead of query time: each `sheet_summary` chunk is enriched with up to 5 real sample values per column, drawn directly from the data:

```
[File: doc_007_published_spend_report_april_25.csv | Sheet: ...]
Sheet summary: 240 rows.
Columns: Date, Transaction Number, Directorate, Local Authority Dept, Merchant Category, Beneficiary, Total
Sample values — Directorate: PLACE, CHILDREN YOUNG PEOPLE & FAMILIES | Local Authority Dept: STRATEGIC HOUSING, LEGAL SERVICES | Beneficiary: WATES PROPERTY SERVICES LTD, THORNE MOORENDS TOWN COUNCIL, DONCASTER SCHOOL SOLUTIONS LIMITED | Merchant Category: COUNCIL DWELLINGS, STRATEGIC HOUSING
```

The vector for this chunk now encodes both the *structure* (column names) and the *content* (actual entities). A query for "WATES PROPERTY SERVICES" lands near the doc_007 chunk because WATES is literally in the embedding, not because the model inferred that "beneficiary" and "property services company" might be related.

This is domain-agnostic: the same code works for spend reports, HR datasets, scientific measurements, or inventory tables — whichever columns contain non-numeric text values, their samples are included. No configuration or domain-specific tuning required.

## Dual-modality retrieval: vector search + DuckDB SQL

Vector search works well for prose, policies, and narrative reports — the kind of content where semantic similarity reliably surfaces the right passage. It breaks down for spreadsheet data.

When a user asks *"how much did supplier X pay in March?"* or *"what is the total spend for category Y?"*, the correct answer requires exact string matching, numeric filtering, and aggregation (`SUM`, `GROUP BY`). A dense vector for "how much did supplier X pay" is semantically similar to every row in a payment sheet — there is no useful distance signal. The right chunk is determined by column equality and arithmetic, not by proximity in embedding space.

To handle both, the system runs two independent retrieval paths:

- **Qdrant** for PDF and OCR documents — hybrid dense+sparse search, reranked by a cross-encoder.
- **DuckDB** for flat-structure Excel and CSV files — the agent writes a `SELECT` query; the in-process database executes it and returns an exact result.

The agent selects the path based on whether the target document appears in the DuckDB table list injected into its system prompt. PDF-format documents always go to Qdrant even when they contain numeric tables, because vector search on contextualised chunk summaries still outperforms SQL over unstructured OCR output.

DuckDB specifically — rather than Postgres or SQLite — because it is in-process (no server to run, no connection pool), columnar (aggregations over 10k-row CSV files return in milliseconds), and loads directly from a pandas DataFrame. The entire structured store is a single file at `DUCKDB_PATH`.

## Context-overflow retry

When the LLM returns a `400 Bad Request` due to token overflow, the agent retries with `rerank_top_n` halved. The reduction persists only for that single request. A shorter but valid answer is returned instead of a crash, without permanently setting a conservative chunk count.

## Contextual Retrieval (chunking)

For each chunk a fast LLM writes one sentence describing the topic, entities, and purpose; that sentence is prepended before embedding. Each vector therefore captures both what the chunk is *about* and what it *says*, improving recall for short or indirect queries. Implementation of [Anthropic Contextual Retrieval (2024)](https://www.anthropic.com/news/contextual-retrieval). Full chunking pipeline: [chunking.md](chunking.md).

## Retrieval-quality refinements

A second wave of changes after manual UI testing surfaced specific failure modes — answers that were `Unsupported` despite the data being present, source cards that showed irrelevant chunks, and broad queries dumping file lists instead of asking for clarification. Each refinement below is domain-agnostic; none targets a specific question or document.

### Filename resolution at the LLM layer

Chunks are stored with `file_name=foo.md` (post-parse) but the original source is `foo.pdf`. Without resolution, the LLM echoes `.md` in answers — a parsing artifact the user shouldn't see. `src/file_resolver.py` exposes `resolve_original_name(filename)` which maps stem → real filename by scanning `data/input/` and `eval/data/raw/`. Applied at every LLM-visible chunk header and in the FastAPI source parser. Same code, single source of truth.

### Stage-2 excludes summary chunks

Stage 1 explicitly retrieves `document_summary` chunks for doc routing (`force_chunk_types=["document_summary"]`). Stage 2 (the actual content fetch) was previously also returning these summaries through dense+sparse search, where they consistently scored at the top because of their dense topic coverage. They consumed rerank slots but never answer-bearing — by definition. All five stage-2 retrieve calls now pass `force_exclude_chunk_types=["sheet_summary", "document_summary"]`.

`sheet_summary` chunks are dual-purpose (they signal *"this sheet has these columns, route to the Excel tool"*) so they're fetched separately and capped at 2 with a stopword-filtered column-overlap threshold of `≥2`, not `>0` — generic words like `and/the/of` no longer count toward overlap.

### Stage-1 stem-overlap doc-routing boost

The stage-1 router uses dense similarity over `document_summary` chunks to identify likely source docs. This works well when the query is paraphrased prose; it fails when the query literally names the doc. Example: *"What does the procurement policy say about supplier selection?"* — the embedding is similar to multiple HR-policy summaries because they all discuss "policy" and "selection criteria", so HR docs win the top-5.

Fix: after dense routing, add any doc whose filename stem shares `≥2` tokens with the query (after stopword filter `{the, and, for, with, from}`) to `stage1_doc_ids`. For *"procurement policy"*, `doc_001_procurement_policy` matches on `{procurement, policy}` regardless of dense rank.

This is a strong signal — the user literally typed the title — so stem-matched docs are tracked separately as `stem_match_doc_ids` and treated with higher priority downstream.

### Force-inject missing stage-1 docs

The boost above only matters if the doc's chunks are in the candidate pool. With `retrieval_top_k=100`, popular docs can crowd a stage-1-identified doc out entirely. After the main hits assembly, missing stage-1 docs trigger a scoped `retrieve(scope_doc_id=missing_id, top_k=5)` which guarantees their chunks reach the reranker.

Presence detection uses a filename regex (`doc_\d+`) instead of `metadata.doc_id` because older ingestions only set `doc_id` on `document_summary` chunks — PDF chunks lacked it, so they appeared "missing" while actually being present.

### Per-doc slot reservation

The reranker often gives high logits to chunks with surface-form similarity to the query (e.g. an HR chunk with the phrase *"selection criteria"* outranks the actual procurement-policy text for *"supplier selection"*). A weak tiebreaker on stage-1 membership wasn't enough.

Replaced with explicit slot reservation: stem-matched docs and explicitly-mentioned doc_ids each get up to 2 guaranteed slots in the top-N. Rest of the slots fill from the reranker's order, skipping anything already reserved. Reranking now runs on **all** candidates (`top_n=len(docs)`) instead of cutting at `_rerank_top_n` first, so reserved chunks aren't silently dropped before reservation runs.

### Neighbor-chunk expansion

The classic chunker-boundary problem: section header lives in chunk N, the values table lives in N+1, the reranker scores N high (matches "vacation accrual schedule") but never sees N+1. Generic and content-agnostic fix: after the top-N is decided, batch-fetch `chunk_index ± 1` from each file via Qdrant scroll filter on `(source_file, chunk_index IN [...])` and append with `[prev chunk]` / `[next chunk]` separators bounded by `max_chunk_chars`. Skipped for `sheet_*` and `document_summary` chunks (they're standalone, no useful neighbors).

This single change fixes a class of misses — header-then-table, intro-then-list, definition-then-example — without re-ingesting or re-chunking.

### Tool-call boundary aware source display

For questions that name a specific document, the agent makes two calls per the system prompt: an unscoped `search_knowledge_base` to discover the doc_id from `document_summary`, then a scoped follow-up. The first call returns 8 noise chunks (matched on `filter_token` exact-match scroll); the second returns the answer-bearing chunks the LLM actually cites.

The previous source parser concatenated chunks from both calls and capped at 8 — call-1's noise filled all slots, call-2's real sources were never displayed. Fix: `stream_agent` and `ask_agent` insert a `---CALL_BOUNDARY---` marker between tool calls; `_parse_sources` groups by boundary and iterates in **reverse**, so the later (typically scoped) call's chunks display first. Falls back to single-call behavior when only one call happened.

### Prompt-driven clarification

A new `_CLARIFICATION_BLOCK` in the system prompt instructs the agent to respond with `Clarify: <one short question listing 2-4 specific topics>` when the question is too broad to answer with a specific value (e.g. *"what about HR policies?"*) and the retrieved chunks span 3+ unrelated docs or contain only summary chunks. Picks the topics from what was actually retrieved, so the clarification is grounded in available content.

Implemented as a prompt rule rather than a separate classifier node — zero infra change, easily tunable, and the LLM at temp=0 is consistent enough at evaluating the trigger condition.

### Deterministic HyDE

`_llm_call` default temperature was 0.5; only the main agent loop and the direct-answer paths explicitly passed `temperature=0`. HyDE expansion (used for query embedding) inherited the 0.5 default, which made retrieval results vary across identical queries. Default is now `0.0` — every LLM call in the pipeline is reproducible.

### Forced retry on bare Unsupported

Even at `temperature=0`, Groq inference is not perfectly deterministic — speculative decoding and other engine-side optimizations introduce small variation that can flip a tool-call decision. Observed symptom: identical query *"What does the procurement policy say about supplier selection?"* sometimes triggers the 2-step doc-routing protocol (correct answer with `doc_001` chunks) and sometimes skips it (bare `Unsupported`).

Mitigation at the API layer (`api.py /query`):

1. Run the agent normally.
2. If the final answer is exactly `"Unsupported"` (after `.strip().lower()`), re-run the agent **once** with the original question plus an explicit instruction:
   > *"This is a retry. The previous attempt returned Unsupported. You MUST follow the doc-routing protocol strictly: (1) call search_knowledge_base with the topic words to identify the relevant document_summary and read its doc_id; (2) call search_knowledge_base again with the original question scoped to that doc_id."*
3. If the retry produces a non-`Unsupported` answer, use it; otherwise keep the original abstention.

Trade-offs deliberately considered:

- **Cost** — adds at most one extra agent run per request, only on bare Unsupported. Real abstentions (e.g. *"home phone number of the CEO"*) return a longer abstention text containing `Unsupported` plus an explanation, so the exact-match check (`==`, not `in`) doesn't fire and the retry is skipped.
- **Hallucination risk** — the retry doesn't change retrieval logic; it just nudges the agent to use it more thoroughly. The reranker, abstention rule, and citation requirements still apply on the second pass.
- **Why not vLLM** — a self-hosted vLLM endpoint with a fixed seed would give true determinism, but the demo runs against Groq for cost and latency. The forced retry is the pragmatic substitute.

Validated: 5/5 runs of the previously-flaky procurement query now produce real answers with `doc_001_procurement_policy.pdf` in the source list.
