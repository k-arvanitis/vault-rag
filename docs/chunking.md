# Chunking pipeline

The chunker (`src/chunker.py`) runs in five passes plus a sixth pass that produces a per-document summary. Each pass operates on the output of the previous one — the goal is to land on chunks of `CHUNK_MIN_TOKENS=256` to `CHUNK_MAX_TOKENS=1024` tokens that respect semantic boundaries (page, section, paragraph) before falling back to character splits.

## Pipeline

1. **Page split** — splits the input markdown on `<!-- PAGE N | label -->` markers emitted by the parser. Each downstream chunk inherits its `page` field from this split, which keeps citation accuracy at the page level even after later merges and splits.

2. **Section split (`MarkdownHeaderTextSplitter`)** — splits on `#`, `##`, `###` headings. Section headings are surfaced as `metadata.section` so the agent can cite "Section 4.2 — Termination" instead of just a chunk index.

3. **Re-split oversized chunks** — any chunk over `CHUNK_MAX_TOKENS=1024` (tiktoken, `cl100k_base`) is re-split with a recursive character splitter so the embedding model never sees a truncated input.

4. **Merge tiny chunks** — adjacent chunks under `CHUNK_MIN_TOKENS=256` are merged within the same `(page, section)` group only. This avoids the common failure mode where a one-line paragraph below a heading becomes its own meaningless chunk. Named section headers (`## …`) are never merged across — `_has_section_header()` guards the merge step.

5. **Contextual enrichment** — the per-document summary (pass 6) is generated *first*, then for each chunk a fast LLM (`CHUNK_LLM_API_BASE`, default OpenRouter / Gemma) writes one sentence describing the topic, entities, and purpose of that chunk. The background shown to the LLM adapts to document length: if the whole document fits the token budget (`CONTEXT_ENRICH_DOC_BUDGET_TOKENS`, default 8000) the model sees the entire document — exactly Anthropic Contextual Retrieval; otherwise it sees the document summary plus a window of `CONTEXT_ENRICH_WINDOW_CHUNKS` (default 2) neighbouring chunks on each side, so the input stays bounded on a 150-page report. Either way the LLM is instructed to keep the sentence specific to that chunk, not to restate the background. The sentence is prepended before embedding:

    ```
    CONTEXT: This chunk describes payment terms under Section 4.2 of the supplier agreement.

    CONTENT:
    <original chunk text>
    ```

    Each vector therefore captures both what the chunk is *about* and what it *says*, improving recall for short or indirect queries. Implementation of [Anthropic Contextual Retrieval (2024)](https://www.anthropic.com/news/contextual-retrieval). The context sentence is also stored as `metadata.context` so the inspector can show the generated note next to the original text — the enrichment is auditable, not a black box.

6. **Document summary chunk** — the per-document summary (generated up front, before pass 5, so the enrichment step can use it) is also stored as one extra `chunk_type="document_summary"` chunk per file, of the form:

    ```
    Document ID: doc_017
    File: supplier_agreement_2024.pdf
    <one-paragraph summary>
    ```

    The agent retrieves these first to resolve a document title or alias to a `doc_id`, then scopes a follow-up retrieval to that `doc_id`. Without this two-step lookup the LLM has no reliable way to map "the supplier agreement" to the right document.

## Why these specific stages

- **Page first.** Citation accuracy stays at the page level even after merges and splits. Most failures during eval debugging trace back to a wrong page citation; protecting that boundary first pays off.
- **Section second.** Heading-aligned chunks group related sentences and let the agent cite by section name, not just chunk index.
- **Re-split before merge.** Splitting an over-budget chunk first means the merge step can opportunistically combine the tail with the next small chunk without going back over the limit.
- **Merge guarded by section.** Without the `_has_section_header()` guard, merging a small "Definitions" section into a large "Termination" section was the source of consistent eval misses — the prior tuning fix was wrongly attributed to `CHUNK_MIN_TOKENS`.

## Configuration

| Variable | Default | Effect |
|---|---|---|
| `CHUNK_MAX_TOKENS` | `1024` | Hard upper bound; pass 3 splits anything above this |
| `CHUNK_MIN_TOKENS` | `256` | Pass 4 merges anything below this within the same section |
| `CHUNK_LLM_API_BASE` | `https://openrouter.ai/api/v1` | Endpoint for contextual summaries; point at a local vLLM for fully air-gapped ingest |
| `CHUNK_LLM_MODEL` | `google/gemma-4-31b-it:free` | Cheap, fast model — only one sentence per chunk is needed |
| `CONTEXT_ENRICH_DOC_BUDGET_TOKENS` | `8000` | Pass 5: documents at or below this size are enriched against their full text; larger ones use the summary + neighbour window |
| `CONTEXT_ENRICH_WINDOW_CHUNKS` | `2` | Pass 5: neighbour chunks included on each side when the document exceeds the budget |

## Excel / CSV path

Excel and CSV files do not flow through this pipeline. They take a separate path through `src/ingest_table_rows.py`:

- Cleaned data lands in **DuckDB** (one table per sheet); the agent queries it via SQL.
- Only one `sheet_summary` chunk per sheet — and one `document_summary` chunk per file — go to Qdrant for discovery. Row data never enters Qdrant.

`sheet_summary` chunks are built deterministically — no LLM — from the column list plus up to 20 real sample values per text column (for up to 8 columns), drawn directly from the data. A query like `"WATES PROPERTY SERVICES"` lands on the right sheet because the entity is literally in the embedding, not because a model inferred a relationship from column names alone.
