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

## What it does

Upload any business document and ask questions in plain English. Vault RAG finds the most relevant passages across all your files and returns a precise, cited answer.

Under the hood:

- **Any file type** — PDFs (including scanned), Excel, CSV, Word (.docx), Markdown, and images are all ingested into the same unified search index.
- **LightOn OCR** converts scanned PDFs and complex layouts into clean Markdown before indexing, preserving tables and figures.
- **Multi-level table repair** — research papers and reports often contain tables with two-row headers (group names + sub-column metrics like R@1/R@5/R@10). At ingest time, Vault RAG programmatically detects and flattens these into a single correct header row (e.g. `KaLMv2 + Qwen3-235B R@1`), handling both HTML tables (OCR output) and LaTeX-encoded `$\text{...}$` table blocks. Column names are derived structurally — no LLM guessing — so alignment is always correct.
- **PDF text-layer recovery** — when OCR misses values in the rightmost column of wide tables (a common failure mode), the pipeline falls back to the PDF's embedded text layer to recover those values, splitting concatenated decimals (`2.9524.141` → `2.952`, `4.141`) using fixed-decimal-place matching.
- **Figure/image analysis** — pages containing figures, charts, or diagrams are sent to a Groq vision model (`llama-4-scout`) which writes a description of each visual element inline in the Markdown, so figure content is searchable.
- **Document-level summary chunk** — at ingest time, a 3–5 sentence summary of the whole document is generated and stored as a dedicated chunk. General questions like "what is this paper about?" retrieve the summary directly without scanning all content. The summary is also shown in the Document Inspector UI.
- **Contextual retrieval** — after splitting a document into chunks, a fast LLM (`llama-3.1-8b-instant` via Groq) writes a one-sentence summary for each chunk describing its topic, entities, and purpose. This summary is prepended to the chunk text before embedding. The result is that each vector in the index carries both the context ("this chunk is about Q3 sales in the EMEA region") and the content — significantly improving retrieval accuracy on short or ambiguous queries.
- **Hybrid search** combines dense semantic vectors (bge-m3) with sparse BM25-style keyword vectors, fused via Reciprocal Rank Fusion (RRF) in Qdrant, so both conceptual and exact-match queries work well.
- **HyDE query expansion** generates a hypothetical answer to the query and embeds that instead of the raw question, improving recall on short or vague questions.
- **Cross-encoder reranking** re-scores the top-100 candidates with a dedicated reranker to surface the most relevant chunks before passing them to the LLM.
- **ReAct agent** (LangGraph) can issue multiple search calls, compare results, and reason step-by-step before writing the final answer — useful for multi-part or comparative questions.
- **Document inspector** shows the original PDF page side-by-side with the parsed Markdown, including a document summary panel and correctly rendered tables.

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
                          │  bge-m3 → dense vector  │
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

## Tech stack

| Component | Technology | Why |
|-----------|-----------|-----|
| OCR / parsing | LightOn OCR (local vLLM) | State-of-the-art vision-language model for scanned PDFs; runs locally so raw document content never leaves the machine |
| Contextual summaries | llama-3.1-8b-instant (Groq) | Fast, cheap model writes one sentence per chunk at ingest time; improves retrieval without slowing queries |
| Embeddings | bge-m3 (Ollama) | High-quality English embedding model; fast on CPU; runs locally so document text stays on-prem |
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
ollama pull bge-m3

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
| `OLLAMA_EMBED_MODEL` | `bge-m3` | Embedding model (Ollama) |
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
