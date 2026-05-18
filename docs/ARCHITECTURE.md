# Vault RAG — Architecture

How the system works end to end: ingestion, storage, and the live query path.
Companion docs: [engineering.md](engineering.md) (design decisions),
[CODEBASE_GUIDE.md](CODEBASE_GUIDE.md) (file-by-file), [chunking.md](chunking.md).

---

## 1. Overview

Vault RAG answers natural-language questions over a mixed document corpus —
born-digital PDFs, scanned PDFs, and spreadsheets — and returns grounded,
traceable answers. It has two halves:

- **Ingestion** — parses each file by modality, chunks it, embeds it, and writes
  it to the right store (Qdrant for text, DuckDB for spreadsheet rows).
- **Query** — a deterministic pre-routing stage followed by a ReAct agent that
  retrieves from Qdrant and/or runs SQL over DuckDB, then synthesises an answer.

Text content is searched semantically; spreadsheet content is queried with SQL.
Each modality uses the tool that actually fits it.

---

## 2. Ingestion pipeline

```
File (PDF / Excel / CSV) ─→ src/ingest.py  (file-type router)
     │
     ├──── PDF ───────────────────────────────┐   ├──── Excel / CSV ──────────────────────┐
     ▼                                         │   ▼                                       │
parser/pdf_parser.py — PER-PAGE ROUTER         │  src/ingest_tables.py                     │
 text layer ≥ 50 chars on the page?            │  • an LLM extracts the real schema        │
  YES → pymupdf4llm   (reads text layer; CPU)  │    (column names, data-start row,         │
        + figures → VLM  (llama-4-scout, Groq) │     footnote rows, short summary)         │
  NO  → LightOn OCR  (local vLLM, GPU)         │  • rows → DuckDB  (one table per sheet)   │
     │  markdown, one block per page           │  • document_summary + sheet_summary       │
     ▼                                         │    chunks → Qdrant  (discovery only;      │
src/chunker.py — 5 passes + 1 doc-level pass   │    row data never enters the vector store)│
  page split · section split · re-split >1024  │                                           │
  tokens · merge <256 tokens · contextual      └───────────────────────────────────────────┘
  enrichment (one LLM sentence per chunk)
  + document_summary chunk (doc_id + summary)
     │
     ▼
src/embedder.py + src/sparse_embedder.py — every chunk gets BOTH vectors
  Dense  — nomic-embed-text  (768-dim, cosine) — semantic similarity
  Sparse — BM42 (fastembed)  — exact-token recall for IDs / names / numbers
     │
     ▼
Qdrant — one hybrid collection (dense + sparse vector per point)
  Point ID = hash(file_name + chunk_index) → idempotent: re-ingest overwrites,
  never duplicates.
```

**PDF per-page routing.** Each page is classified independently. Pages with a
real text layer are read by `pymupdf4llm` (fast, CPU, no model); scanned pages
go to LightOn OCR on a local vLLM server. Figures on text-layer pages are sent
to a vision-language model for a text description. One PDF can mix both paths.

**Spreadsheets do not get chunked into Qdrant.** Their rows live in DuckDB (one
table per sheet); only a `document_summary` and per-sheet `sheet_summary` go
into Qdrant, so the system can *discover* a table without storing its rows as
vectors. The `sheet_summary` carries the DuckDB table name and column list.

---

## 3. Storage model

| Store | Holds | Used for |
|---|---|---|
| **Qdrant** | text chunks (PDF/OCR) + summary chunks for every file | hybrid dense+sparse semantic search |
| **DuckDB** | spreadsheet rows, one table per sheet | exact SQL lookups and aggregations |

The asymmetry is deliberate and is what makes deterministic routing possible
(§4): a question whose best matches are `.xlsx`/`.csv` chunks is structured-data;
one whose matches are `.pdf` chunks is a document question.

---

## 4. Query pipeline

```
User question
     │
     ▼
api.py /query — PRE-AGENT ROUTING  (plain functions, not a graph)
  • multi-part split — _split_multi_part_query fans a 2+-part question into
    separate sub-questions; each is answered by its own agent run and the
    results are merged in code (the agent's single-pass synthesis used to
    drop a part).
  • deterministic tool routing — route_question matches each (sub)question
    against the index: top-3 hits on .xlsx/.csv ⇒ query_excel, on .pdf ⇒
    search_knowledge_base. The resolved tool is prepended as a directive, so
    the agent follows an index-grounded instruction instead of guessing the
    tool from question wording.
     │
     ▼
GRAPH 1 — ReAct agent  (src/rag_agent.py, LangGraph create_react_agent)
  LLM: qwen/qwen3-32b (Groq, via LiteLLM proxy). One tool-calling loop, two tools:

   ┌── search_knowledge_base ─▶ Qdrant retrieval
   │     stage 1 — search document_summary chunks → resolve which doc(s)
   │     stage 2 — scoped hybrid search (dense + sparse, RRF-fused)
   │               + HyDE query expansion
   │               + cross-encoder rerank (top-100 → top-10)
   │               + neighbour-chunk expansion
   │
   └── query_excel ──────────▶ GRAPH 2 (Excel sub-agent)
     │
     ▼
GRAPH 2 — Excel sub-graph  (src/excel_agent.py, two LangGraph StateGraphs)
  LLM: gpt-4o-mini
  Outer: decompose per source → fan-out → synthesise per-part answers
  Inner: select_table → inspect schema + sample rows → write_sql →
         run_sql on DuckDB → evaluate (retry on SQL error, fall through on
         0 rows)
     │
     ▼
Post-processing — plain functions, NOT graphs
  • multi-part coverage check / repair retrieval
  • forced retry on a bare "Unsupported"
  • strip leaked chunk-header lines from the answer
     │
     ▼
Cited answer  (chunk + page for PDF · sheet + SQL trace for Excel/CSV)
```

**Deterministic tool routing** is the key reliability feature. The ReAct agent,
left alone, picks its tool by reading the question wording — which mis-routes a
scanned-invoice lookup to SQL because the words *sound* tabular. `route_question`
removes the guess: it retrieves the top hits and reads their file modality, then
prepends an explicit routing directive. See *Deterministic tool routing* in
[engineering.md](engineering.md).

---

## 5. Models

| Stage | Model | Where it runs |
|---|---|---|
| Born-digital PDF text | pymupdf4llm | local, CPU, no model |
| Figure description (VLM) | meta-llama/llama-4-scout-17b | Groq |
| Scanned-page OCR | lightonocr-2-1b-ocr-soup | local vLLM, GPU |
| Table schema extraction | llama-3.3-70b-versatile | Groq |
| Contextual chunk enrichment | gemma-4-31b | OpenRouter → Groq fallback |
| Dense embeddings | nomic-embed-text (768-dim) | Ollama |
| Sparse embeddings | BM42 | fastembed (CPU ONNX) |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 | local |
| Main agent LLM | qwen/qwen3-32b | Groq (via LiteLLM) |
| Excel sub-agent LLM | gpt-4o-mini | OpenAI |

---

## 6. Interfaces

- **FastAPI** (`api.py`) — `/ingest`, `/query`, document inspection endpoints.
  The single backend; the `/query` pre-routing stage lives here.
- **Next.js frontend** (`frontend/`) — upload, chat, trace sidebar (tools, SQL,
  retrieved chunks), document inspector (page image vs extracted content;
  spreadsheet schema vs raw/cleaned views).
- **Slack bot** (`slack_app.py`) — a thin client over the API. It posts
  questions to `/query` and replies in-thread; it does not load models or open
  DuckDB itself, so only the API process holds the DuckDB write lock.

---

## 7. Evaluation

`eval/run_eval.py` scores 82 questions across 14 documents on correctness,
faithfulness (claim-level, RAGAS-style judge), answer relevancy, refusal rate,
retrieval hit@k / MRR, and Excel accuracy. The eval drives the agent through
`stream_agent` directly; results land in `eval/results/`.
