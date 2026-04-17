[![CI](https://github.com/k-arvanitis/vault-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/k-arvanitis/vault-rag/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Qdrant](https://img.shields.io/badge/Qdrant-FF4136?style=for-the-badge&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-F55036?style=for-the-badge&logo=groq&logoColor=white)

# Vault RAG

Ask questions across all your business documents — PDFs, Word files, Excel sheets, and scanned images. Upload once, query instantly, answers with sources.

**Example questions:**
- "What are the payment terms in our supplier contract?"
- "What were total sales in Q3 according to the spreadsheet?"
- "Summarise the key risks identified in the audit report."

---

## Why Vault RAG is different

Most document RAG demos use PyMuPDF for parsing and fixed-size text chunking. Vault RAG makes different engineering choices at every layer:

- **Parsing**: LightOn OCR (a vision-language model running locally via vLLM) instead of rule-based PDF parsers. At only 1B parameters it is remarkably lightweight, yet outperforms much larger models on accuracy — including handwritten text, complex table layouts, and mixed-language content that PyMuPDF cannot handle at all.
- **Chunking**: 5-stage pipeline — header-aware splitting → token-limit enforcement → tiny chunk merging → contextual summary per chunk → table-aware batching. Chunks are never cut mid-table or mid-section.
- **Contextual retrieval**: Before embedding, a fast LLM writes one sentence per chunk describing its topic, entities, and purpose. That sentence is prepended to the chunk text. Every vector in the index captures both what the chunk *says* and what it is *about*.
- **Retrieval**: Hybrid search (dense nomic-embed-text + sparse BM25, RRF fusion in Qdrant) + HyDE query expansion + cross-encoder reranking. Short or vague queries still find the right chunks.
- **Privacy**: Parsing and embedding run entirely locally. Only retrieved chunks (not raw documents) are sent to the LLM API. Fully air-gappable by pointing generation and chunking endpoints at a local vLLM server.

---

## What it does

Upload any business document and ask questions in plain English. Vault RAG finds the most relevant passages across all your files and returns a precise, cited answer.

**Example questions:**
- "What are the payment terms in our supplier contract?"
- "What were total sales in Q3 according to the spreadsheet?"
- "Summarise the key risks identified in the audit report."

Under the hood:
- **Any file type** — PDFs (including scanned), Excel, CSV, Word, Markdown, and images are ingested into a unified search index.
- **LightOn OCR** converts scanned PDFs and complex layouts (tables, figures, multi-column) into clean Markdown locally before indexing.
- **5-stage chunking pipeline** preserves document structure, merges stubs, and prepends a contextual summary to every chunk before embedding.
- **Hybrid search + reranking** — dense + sparse vectors fused via RRF, then re-scored by a cross-encoder. Both semantic and exact-match queries work well.
- **ReAct agent** (LangGraph) issues multiple search calls, reasons step-by-step, and returns a cited answer. Fully traced in Langfuse.

The detailed technical breakdown of each component is in the Architecture and Chunking sections below.

---

## How chunking works

Chunking is the most important step in the ingestion pipeline — it directly determines retrieval quality. Vault RAG does not simply split text at fixed sizes. It goes through five stages:

### Stage 1 — Header-aware splitting
The parsed Markdown is split along heading boundaries (`#`, `##`, `###`). This keeps logically related content together: a section on "Payment Terms" stays in one chunk rather than being cut mid-sentence across two.

### Stage 2 — Token-limit enforcement
Any section that exceeds the `CHUNK_MAX_TOKENS` limit (default 1024) is further split using LangChain's `RecursiveCharacterTextSplitter`, which tries natural break points (paragraphs → sentences → words) before making a hard cut.

### Stage 3 — Tiny chunk merging
Chunks below `CHUNK_MIN_TOKENS` (default 256 tokens) or below 300 characters are merged into their neighbour. This prevents the index from filling up with stub chunks like a lone heading or a two-word section that would confuse the reranker.

### Stage 4 — Contextual summary (Contextual Retrieval)
For each chunk, `llama-3.1-8b-instant` (Groq) is called with the chunk text and asked to write **one sentence** describing the main topic, entities, and purpose of that specific chunk. For example:

> *"This chunk describes maximum permitted concentration limits for nitrates in drinking water under EU Directive 98/83/EC."*

This sentence is prepended to the chunk text before embedding:
```
CONTEXT: <one-sentence summary>

CONTENT:
<original chunk text>
```

The result is that each vector in the index captures both what the chunk is *about* and what it *says* — dramatically improving retrieval for short or indirect queries. This technique is known as **Contextual Retrieval** (Anthropic, 2024).

### Stage 5 — Table handling
PDF tables detected during OCR are stored as a separate `[TABLE_START]...[TABLE_END]` block. The chunker parses these ASCII grid tables into row-sentence batches (up to 20 rows each), with column headers and part numbers prepended so every table chunk is self-contained. Tables are never split mid-row.

---

## Supported file types

| Type | Parser | What it extracts |
|------|--------|-----------------|
| PDF | LightOn OCR | Text, tables, scanned content, complex layouts |
| Excel / CSV | openpyxl / pandas | All sheets, row batches with headers |
| Word (.docx) | python-docx (via LibreOffice → PDF) | Paragraphs with heading structure |
| Images (.png / .jpg) | Vision (Groq) | Text and visual content description |
| Markdown | Plain text | Heading-aware sections |

---

## Architecture

```
 INGESTION
 ─────────────────────────────────────────────────────────────────

 Upload
   │
   ▼
 ┌──────────────────┐     ┌───────────────────────────────────┐
 │  File type       │────▶│  Parser                           │
 │  detection       │     │  PDF/DOCX → LightOn OCR (local)   │
 └──────────────────┘     │  Excel/CSV → openpyxl             │
                          │  Markdown  → plain text           │
                          └────────────────┬──────────────────┘
                                           │ Markdown
                                           ▼
                          ┌───────────────────────────────────┐
                          │  Chunker                          │
                          │  1. Split on Markdown headers     │
                          │  2. Re-split chunks > 1024 tokens │
                          │  3. Merge tiny chunks             │
                          │  4. Contextual summary per chunk  │
                          │     (llama-3.1-8b-instant / Groq) │
                          └────────────────┬──────────────────┘
                                           │ chunks + context
                                           ▼
                          ┌───────────────────────────────────┐
                          │  Embedder (local Ollama)          │
                          │  nomic-embed-text → dense vector  │
                          │  BM25 (fastembed) → sparse vector │
                          └────────────────┬──────────────────┘
                                           │
                                           ▼
                          ┌───────────────────────────────────┐
                          │  Qdrant                           │
                          │  stores dense + sparse per chunk  │
                          └───────────────────────────────────┘

 QUERY
 ─────────────────────────────────────────────────────────────────

 User question
   │
   ▼
 ┌──────────────────┐     ┌───────────────────────────────────┐
 │  HyDE expansion  │────▶│  Hybrid search (Qdrant)           │
 │  generate a      │     │  dense + sparse → RRF fusion      │
 │  hypothetical    │     │  top-100 candidates               │
 │  answer, embed   │     └────────────────┬──────────────────┘
 └──────────────────┘                      │
                                           ▼
                          ┌───────────────────────────────────┐
                          │  Reranker                         │
                          │  cross-encoder/ms-marco-MiniLM    │
                          │  re-scores top-100 → top-10       │
                          └────────────────┬──────────────────┘
                                           │
                                           ▼
                          ┌───────────────────────────────────┐
                          │  ReAct agent (LangGraph)          │
                          │  llama-3.3-70b-versatile / Groq   │
                          │  can call search tool N times     │
                          └────────────────┬──────────────────┘
                                           │
                                           ▼
                          ┌───────────────────────────────────┐
                          │  Answer with cited sources        │
                          └───────────────────────────────────┘
```

---

## Key engineering decisions

These are the non-obvious choices made during implementation — the ones that are not visible in the architecture diagram but directly determine whether the system works well on real documents.

### Deterministic point IDs (idempotent re-ingestion)

Every chunk stored in Qdrant is assigned a SHA-1 hash of `(file_name, chunk_index)` rather than a random UUID. This means re-ingesting the same file overwrites the existing vectors in place instead of appending duplicates. The consequence: you can update a document, re-run ingestion, and the collection stays consistent without a manual delete step. With random IDs, re-running ingestion on an already-indexed file silently doubles every chunk in the index — a subtle bug that degrades retrieval precision without any visible error.

### Three-pass table repair (no LLM involvement)

Research papers and GHG inventory reports contain tables that OCR cannot handle correctly out of the box. Three specific failure modes are fixed deterministically before any chunk enters the index:

1. **Two-row HTML headers** — OCR outputs `<thead>` with group names (e.g. `KaLMv2`) and places sub-column names (`R@1 R@5 R@10`) in the first `<tbody>` row. The `_fix_html_multirow_header` function detects this pattern structurally (≥50% of the first body row cells match `R@K / NDCG@K` patterns) and derives flattened column names like `KaLMv2 R@1` programmatically.
2. **LaTeX `$\text{...}$` tables** — arXiv papers embed tables as raw LaTeX in the OCR output. The `_preprocess_latex_table` / `_derive_column_names` functions parse the `\textbf{}` header tokens, identify the repeating sub-column unit, and build a correctly aligned markdown table without calling any LLM.
3. **Missing rightmost column** — wide tables in scanned PDFs frequently have the last column dropped by the OCR (a known failure mode in vision-language models for wide layouts). `_fill_empty_last_column_from_text` falls back to the PDF's embedded text layer via `pypdf` and recovers the missing values by positional matching, including splitting concatenated decimals (`2.9524.141` → `2.952`, `4.141`) using fixed-decimal-place regex.

All three repairs are structural — no LLM is called, no hallucination risk, and alignment is guaranteed by counting data-row tokens rather than trusting model output.

### Context-overflow retry with dynamic `rerank_top_n` reduction

When the LLM returns a `400 Bad Request` due to input token overflow (a real failure mode when querying over very dense document collections), the agent automatically retries with `rerank_top_n` halved. The reduced limit persists only for that single request and is reset afterward. This means the system degrades gracefully under load — a shorter but valid answer is returned instead of a crash — without ever needing to set a permanently conservative chunk count.

### Hybrid-to-dense fallback

At query time, the retriever checks whether the Qdrant collection actually contains sparse vectors before attempting a hybrid query. If sparse vectors are absent (e.g. a collection indexed before the sparse embedder was added), it falls back to dense-only search silently. This means old index versions remain queryable without schema migration, and the system does not crash on collections that were ingested with an earlier pipeline version.

### Document-level summary chunk

At ingest time, a separate 3–5 sentence summary of the whole document is generated and stored as a dedicated chunk with `chunk_type: "document_summary"` in its metadata. General questions ("what is this paper about?", "give me an overview of the report") retrieve this chunk directly without scanning all content chunks. This avoids the common failure mode where a top-level question retrieves a random mid-document chunk because the summary information is distributed across too many pieces.

### HyDE improves recall on vague queries

Instead of embedding the raw user question, the agent first generates a *hypothetical answer* using a fast LLM call and embeds that. The embedded hypothetical answer is semantically much closer to the real document text than a short question, significantly improving recall for queries like "what were the sales figures" where the question itself shares few tokens with the answer. The trade-off is one extra LLM call per query (~200ms on Groq); the ablation study quantifies the metric gain.

---

## Tech stack

| Component | Technology | Why |
|-----------|-----------|-----|
| OCR / parsing | LightOn OCR (local vLLM) | State-of-the-art vision-language model for scanned PDFs; runs locally so raw document content never leaves the machine |
| Contextual summaries | llama-3.1-8b-instant (Groq) | Fast, cheap model writes one sentence per chunk at ingest time; improves retrieval without slowing queries |
| Embeddings | nomic-embed-text (Ollama) | Strong open embedding model; fast on CPU; runs locally so document text stays on-prem |
| Vector database | Qdrant | Native hybrid search (dense + sparse) with RRF fusion in a single query; straightforward Docker deployment |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 | Fast cross-encoder trained on MS MARCO; dramatically improves precision by comparing query and chunk together |
| Generation | llama-3.3-70b-versatile (Groq) | Best-in-class open LLM for complex reasoning; Groq free tier has low latency and no GPU required |
| UI | Streamlit | Rapid iteration on chat UI; built-in file upload handles the ingest trigger |
| Agent | LangGraph (ReAct) | Structured multi-step reasoning with explicit tool calls; agent can search multiple times before answering |
| Observability | Langfuse | Traces every agent run end-to-end; inspect tool calls, retrieved chunks, and token counts |

---

## Privacy & data

- **Parsing** runs entirely locally via the LightOn OCR server (vLLM on your GPU).
- **Embeddings** are generated locally by Ollama — document text never leaves the machine at indexing time.
- **Contextual summaries** (written per chunk at ingest time) are sent to the Groq API along with the chunk text. If this is unacceptable, point `CHUNK_LLM_API_BASE` at a local vLLM server and set `CHUNK_LLM_MODEL` accordingly.
- **LLM inference** at query time sends only the retrieved chunks and the user query to the Groq API. To go fully air-gapped, point `GENERATION_API_BASE` at a local vLLM server and set `GENERATION_MODEL` accordingly.

---

## Quickstart

```bash
git clone https://github.com/k-arvanitis/vault-rag.git && cd vault-rag
uv sync
cp .env.example .env          # set GROQ_API_KEY
cd docker/ingestion-stack && ./up.sh && cd ../..
ollama pull nomic-embed-text
uv run streamlit run app.py   # → http://localhost:8501
```

GPU required for LightOn OCR. See [Setup](#setup) for full prerequisites and configuration options.

---

## Setup

### Prerequisites

| Tool | Minimum version | Purpose |
|------|----------------|---------|
| Python | 3.11 | Application runtime |
| uv | 0.4+ | Fast dependency management |
| Docker + Compose | 24+ | Qdrant + optional LightOn OCR server |
| Ollama | 0.4+ | Local embedding model |
| GPU (optional) | CUDA 12 | Required for LightOn OCR; CPU fallback is slow |

### Installation

```bash
git clone https://github.com/k-arvanitis/vault-rag.git
cd vault-rag

# Install dependencies
uv sync

# Configure environment
cp .env.example .env
# Edit .env — at minimum set GROQ_API_KEY

# Start Qdrant + LightOn OCR
cd docker/ingestion-stack
cp .env.example .env        # set HUGGING_FACE_HUB_TOKEN
./up.sh                     # starts both services, waits for OCR to be ready
cd ../..

# Pull embedding model
ollama pull nomic-embed-text

# Start the app
uv run streamlit run app.py
```

---

## Configuration

All variables can be set in `.env`. See `.env.example` for a full annotated list.

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | _(required)_ | Groq API key — free tier at console.groq.com |
| `GENERATION_API_BASE` | `https://api.groq.com/openai/v1` | OpenAI-compatible generation endpoint |
| `GENERATION_MODEL` | `llama-3.3-70b-versatile` | LLM used for query answering |
| `CHUNK_LLM_API_BASE` | `https://api.groq.com/openai/v1` | Endpoint for contextual summary LLM (ingest only) |
| `CHUNK_LLM_MODEL` | `llama-3.1-8b-instant` | Model used to write one-sentence context per chunk |
| `OLLAMA_API_BASE` | `http://127.0.0.1:11434` | Ollama server URL |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Embedding model (Ollama) |
| `QDRANT_URL` | `http://localhost:7333` | Qdrant REST endpoint |
| `QDRANT_COLLECTION` | `documents_chunks` | Qdrant collection name |
| `OCR_API_BASE` | `http://127.0.0.1:8002` | LightOn OCR vLLM endpoint |
| `OCR_MODEL` | `lightonocr-2-1b-ocr-soup` | LightOn OCR model name |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | HuggingFace reranker model |
| `RERANKER_DEVICE` | `cpu` | Device for reranker (`cpu` or `cuda`) |
| `RETRIEVAL_TOP_K` | `100` | Candidates retrieved before reranking |
| `RERANK_TOP_N` | `10` | Top chunks passed to the LLM after reranking |
| `CHUNK_MAX_TOKENS` | `1024` | Maximum tokens per chunk |
| `CHUNK_MIN_TOKENS` | `256` | Minimum tokens per chunk (smaller chunks are merged) |

---

## Evaluation

An evaluation harness lives in `eval/evaluate_rag.py`. It runs a set of question–answer pairs through the full pipeline and scores each answer with a GPT-4o-mini judge across four metrics: faithfulness, context recall, answer relevance, and context precision.

Results are saved to `eval/results/` and auto-rotated (current + previous run kept).

| Metric | Score |
|--------|-------|
| Faithfulness | TBD |
| Context recall | TBD |
| Answer relevance | TBD |
| Context precision | TBD |

Run evaluation:

```bash
uv run python eval/evaluate_rag.py
```

---

## Failure modes

| Component | What fails | Symptom | Fix |
|---|---|---|---|
| LightOn OCR | vLLM server not running | Ingest crashes with `Connection refused` on port 8002 | `cd docker/ingestion-stack && ./up.sh` |
| Qdrant | Container not running | All queries return empty results | `docker compose up -d qdrant` |
| Ollama / bge-m3 | Model not pulled | Embedding step fails | `ollama pull bge-m3` |
| Groq API | Missing or invalid `GROQ_API_KEY` | Generation returns 401 | Set `GROQ_API_KEY` in `.env` |
| Groq API | Rate limit hit | Slow or failed responses | Retry or use a local vLLM via `GENERATION_API_BASE` |
| Reranker | Model not downloaded | First query is slow (~30s download) | Pre-download: `uv run python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"` |
| File ingestion | Unsupported format uploaded | Silent skip with error toast | Only PDF, Excel, CSV, MD, DOCX, images are supported |
