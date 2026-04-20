[![CI](https://github.com/k-arvanitis/vault-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/k-arvanitis/vault-rag/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Qdrant](https://img.shields.io/badge/Qdrant-FF4136?style=for-the-badge&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-F55036?style=for-the-badge&logo=groq&logoColor=white)
![Langfuse](https://img.shields.io/badge/Langfuse-111827?style=for-the-badge&logoColor=white)

# Vault RAG

A retrieval backend for business document intelligence. Upload your company's contracts, reports, invoices, and spreadsheets — then query across all of them at once in plain English, with cited answers.

The demo runs as a Streamlit app. The pipeline is designed to be embedded: as a Slack bot, an internal API, or a customer-facing assistant. Swap the UI layer; the retrieval backend stays unchanged.

---

## Why Vault RAG is different

Most document RAG demos query a single clean PDF with fixed-size chunking. Vault RAG is built for real business document collections:

- **Cross-document retrieval** — queries run across all uploaded documents simultaneously. Answers can draw from a contract, a spreadsheet, and a scanned invoice in the same response.
- **Two-path PDF parsing** — each page is routed independently: born-digital pages use pymupdf4llm (CPU, no API call, zero hallucination risk) while scanned pages use LightOn OCR (local vision-language model via vLLM). Mixed documents — a contract where most pages are digital but one page is a faxed addendum — are handled correctly without any manual configuration.
- **5-stage chunking pipeline** — header-aware splitting, token-limit enforcement, tiny chunk merging, contextual summary per chunk (Anthropic Contextual Retrieval), and table-aware batching. Structure is preserved; chunks are never cut mid-table or mid-section.
- **Hybrid search + reranking** — dense (nomic-embed-text) + sparse (BM25) vectors fused via RRF in Qdrant, HyDE query expansion, cross-encoder reranking. Both semantic and exact-match queries work on short or ambiguous input.
- **Privacy-first** — parsing and embedding run entirely locally. Only retrieved chunks (not raw documents) leave the machine. Fully air-gappable by pointing generation endpoints at a local vLLM server.

---

## What it does

Upload any business document and ask questions in plain English. Vault RAG searches across all your files simultaneously and returns a precise, cited answer — pulling from whichever documents contain the relevant information.

**Example questions across a real document collection:**
- "What are the payment terms in our supplier contract?"
- "What were total sales in Q3 according to the spreadsheet?"
- "Summarise the key risks identified in the audit report."
- "Which invoices from last quarter exceeded the budget cap in the procurement policy?"

Under the hood:
- **Any file type** — PDFs (including scanned), Excel, CSV, Word, Markdown, and images ingested into a single unified search index.
- **Two-path PDF parser** routes each page independently — pymupdf4llm for born-digital pages (fast, no API call), LightOn OCR for scanned pages (local vLLM). Embedded figures are described by a vision model (Groq) and injected inline as `[Figure: ...]` before chunking.
- **5-stage chunking pipeline** preserves document structure and prepends a contextual summary to every chunk before embedding.
- **Hybrid search + reranking** — dense + sparse vectors fused via RRF, re-scored by a cross-encoder. Semantic and exact-match queries both work.
- **ReAct agent** (LangGraph) issues multiple search calls, reasons across results, and returns a cited answer. Every run traced in Langfuse.

Full technical breakdown in the Architecture and Chunking sections below.

---

## Interfaces

Vault RAG separates the operator experience from the end-user experience.

### Streamlit UI — for testing and tuning

Before going live, use the Streamlit app to validate your document collection:
inspect how each file chunks, verify OCR output on scanned PDFs, run test
queries and see exactly which chunks were retrieved and why. This is your
tuning environment, not your production UI.

```bash
uv run streamlit run app.py   # → http://localhost:8501
```

### Slack — for your team

The Streamlit UI is for the admin who owns the document collection. Slack is
for everyone else. Once documents are indexed, the whole team can query them
from where they already work — no new tool, no access to manage.

The Slack bot is a **query interface only**. It does not accept file uploads.
Documents are sensitive; they belong in your infrastructure, not in Slack.
The admin indexes them once via Streamlit; the team queries from that point on.

```
@vault what are the payment terms in the supplier contract?
→ Payment is due within 30 days of invoice date. [1]
  [1] contract.pdf, Section 4.2
```

Works in channels (@mention) and DMs (message directly).

**Setup:** create a Slack app, enable Socket Mode, add `SLACK_BOT_TOKEN`
and `SLACK_APP_TOKEN` to `.env`, then:

```bash
make slack            # run locally
make docker-slack-up  # run in Docker
```

See [Slack app setup](#slack-setup) for the full walkthrough.

---

## Deployment model

Each customer runs their own instance — their own Docker containers,
their own Qdrant collection, their own Slack workspace connection.
No shared infrastructure. Document content never leaves the customer's
environment (parsing and embeddings run locally; only retrieved chunks
are sent to the LLM API).

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
| PDF (born-digital) | pymupdf4llm | Text layer extraction, tables, embedded figures (VLM description) |
| PDF (scanned) | LightOn OCR (local vLLM) | Vision-language OCR, complex layouts, multi-column content |
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
 │  detection       │     │  Excel/CSV → openpyxl / pandas    │
 └──────────────────┘     │  Markdown  → plain text           │
                          │  PDF/DOCX  → per-page router ─────┼──┐
                          └───────────────────────────────────┘  │
                                                                  │
                          ┌──────────── PDF page ────────────┐   │
                          │                                  │◀──┘
                          │  text layer ≥ 50 chars?          │
                          │     YES → pymupdf4llm            │
                          │           (CPU, no API call)     │
                          │           figures → VLM (Groq)   │
                          │     NO  → LightOn OCR            │
                          │           (local vLLM, GPU)      │
                          └────────────────┬─────────────────┘
                                           │ Markdown (per page)
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

### Per-page routing: pymupdf4llm vs LightOn OCR

Every PDF page is routed independently to one of two parsers based on whether it has an extractable text layer.

**pymupdf4llm — born-digital pages**

| Property | Detail |
|---|---|
| Runs on | CPU only — no GPU, no model, no API call |
| Speed | Near-instant — a 50-page research paper parses in under a second |
| Accuracy | Lossless — reads characters directly from the PDF byte stream |
| Failure modes | None for text; figures require a separate VLM call (see below) |

Born-digital PDFs (exported from Word, LaTeX, or a browser) embed their text as selectable characters. Running OCR on these is counterproductive: the vision model transcribes what it *sees* in a rendered image, introducing errors on numbers, equations, and code that are already represented perfectly as text. pymupdf4llm skips the model entirely and reads the text layer directly — structure like tables and bold text is preserved in Markdown.

**LightOn OCR — scanned pages**

| Property | Detail |
|---|---|
| Runs on | GPU via local vLLM server |
| Speed | ~2–5 seconds per page depending on GPU |
| Accuracy | State-of-the-art for document OCR; handles multi-column, tables, mixed-language |
| Failure modes | Requires the vLLM server running; born-digital pages are unaffected if it goes down |

Scanned pages have no text layer — pymupdf4llm returns an empty string. A vision-language model is the only option.

The routing threshold is 50 characters of extractable text per page. Mixed documents — a contract where pages 1–10 are digital and page 11 is a faxed addendum — are handled correctly per page with no configuration required.

**Figure descriptions (pymupdf4llm path only)**

Figures in born-digital PDFs come in two forms: raster images written to disk by pymupdf4llm (appear as `![](path.png)` in the Markdown) and vector graphics whose underlying text pymupdf4llm extracts as raw labels (wrapped in `--- Start of picture text ---` blocks — typical for matplotlib or D3 charts). Both types are intercepted, the page region is cropped via fitz, and the image is sent to a vision model (Groq) for a natural-language description. The description is injected inline as `[Figure: ...]` so it enters the chunk index and is searchable.

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
| PDF parsing — text layer | pymupdf4llm | CPU-only, no model, near-instant. Born-digital PDFs embed selectable text — reading it directly is lossless and avoids the transcription errors a vision model introduces on numbers, equations, and code |
| PDF parsing — scanned | LightOn OCR (local vLLM) | Scanned pages have no text layer; a vision-language model running locally on GPU is the only option. Keeping it local means raw document bytes never leave the machine |
| Figure descriptions | llama-4-scout-17b (Groq vision) | Converts embedded charts and diagrams to searchable text; both raster images and vector graphics (extracted as raw label text by pymupdf4llm) are rasterised and described before chunking |
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

- **Parsing** runs locally. Born-digital PDF pages are processed by pymupdf4llm on CPU with no network calls. Scanned pages go to the LightOn OCR server (vLLM on your GPU, also local). Raw document bytes never leave the machine.
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
| `IMAGE_SIZE_LIMIT` | `0.05` | Fraction of page area; images smaller than this are skipped by pymupdf4llm |
| `VLM_ENABLED` | `true` | Set `false` to skip VLM figure descriptions entirely (faster ingestion, no Groq calls) |
| `VLM_PROVIDER` | `groq` | Provider for figure description calls |
| `VLM_MODEL` | `meta-llama/llama-4-scout-17b-16e-instruct` | Vision model used for figure descriptions |

---

## Slack setup

### 1. Create a Slack app

Go to [api.slack.com/apps](https://api.slack.com/apps) → **Create New App** → **From scratch**.

### 2. Enable Socket Mode

**Settings → Socket Mode** → Enable. This creates the App-Level Token (`xapp-...`) — copy it to `SLACK_APP_TOKEN`.

### 3. Add Bot Token Scopes

**OAuth & Permissions → Scopes → Bot Token Scopes**, add:

| Scope | Purpose |
|-------|---------|
| `app_mentions:read` | Receive @mention events |
| `chat:write` | Post messages and replies |
| `im:history` | Receive DM messages |
| `im:write` | Reply in DMs |

### 4. Subscribe to events

**Event Subscriptions → Subscribe to bot events**, add:
- `app_mention`
- `message.im`

### 5. Install and configure

**OAuth & Permissions → Install to Workspace** → copy the Bot User OAuth Token to `SLACK_BOT_TOKEN`.

Add both tokens to `.env`:

```
SLACK_BOT_TOKEN=xoxb-...
SLACK_APP_TOKEN=xapp-...
```

### 6. Start the bot

```bash
make slack            # local
make docker-slack-up  # Docker
```

Invite the bot to a channel: `/invite @vault-rag`

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

## Tests

71 tests, all mocked — no live services (Ollama, Qdrant, Groq) required to run CI.

| File | Tests | What's covered |
|------|-------|----------------|
| `test_chunker.py` | 3 | Token limits, minimum chunk size, required output fields |
| `test_config.py` | 5 | Config invariants: types, model family constraints |
| `test_retriever.py` | 14 | Cosine similarity math, text filter shape, retrieve from local JSON, dense and hybrid Qdrant paths |
| `test_reranker.py` | 10 | BGEReranker and QwenReranker: output format, sort order, top-n, score ranges |
| `test_rag_agent.py` | 17 | `ask_agent` history injection and answer extraction, `stream_agent` token streaming and `<think>` suppression, chunk collection, system prompt invariants |
| `test_table_repair.py` | 16 | Three-pass table repair: two-row HTML headers, LaTeX column derivation, missing last-column recovery |
| `test_pdf_parser.py` | 6 | Per-page routing (text-layer vs scanned), VLM enable/disable, VLM exception handling, mixed documents |

```bash
uv run pytest tests/ -v
```

---

## Known limitations

- **Very large Excel files** — files with 100k+ rows or 20+ sheets are ingested fully into memory. Expect slow ingestion and possible OOM errors; consider pre-filtering sheets before upload.
- **Password-protected PDFs** — LightOn OCR receives the raw bytes and will fail or return empty output. There is no detection or user-facing warning; the file silently ingests as an empty document.
- **Low-quality scans** — severely skewed, low-DPI, or handwritten content degrades LightOn OCR accuracy. Ingestion does not fail, but retrieved text may contain OCR artefacts that hurt retrieval precision. Born-digital pages are unaffected (they bypass OCR entirely).
- **VLM figure descriptions** — vector graphics in PDFs (e.g. matplotlib charts exported as PDF vectors) are rasterised and sent to Groq for description. If `GROQ_API_KEY` is unset or the Groq API is unavailable, figures are replaced with `[Figure: description unavailable]` and ingestion continues. Set `VLM_ENABLED=false` to skip all figure description calls.
- **Complex PDF table layouts** — merged cells, rotated headers, and tables spanning multiple pages are parsed heuristically. The three-pass table repair handles common academic/GHG-report patterns; novel layouts may produce misaligned row-sentence chunks.
- **Very long documents (500+ pages)** — the contextual enrichment step makes one LLM call per chunk. A 500-page technical report can generate 600+ chunks; at Groq free-tier rate limits this takes several minutes and may hit the requests-per-minute cap.
- **Multi-language documents** — cross-lingual retrieval (Greek ↔ English) was removed from the pipeline. Documents in non-English languages can still be ingested and searched in their native language, but English queries will not retrieve non-English content and vice versa.
- **Streaming and context overflow** — the adaptive `rerank_top_n` retry on context overflow is implemented only in `ask_agent`. The streaming path (`stream_agent`) does not retry; it will raise a `BadRequestError` on overflow. Use `ask_agent` for large document collections.

---

## Failure modes

| Component | What fails | Symptom | Fix |
|---|---|---|---|
| LightOn OCR | vLLM server not running | Scanned pages fail with `Connection refused` on port 8002; born-digital pages are unaffected | `cd docker/ingestion-stack && ./up.sh` |
| VLM (figure descriptions) | Groq API unavailable or `VLM_ENABLED=false` | Figures replaced with `[Figure: description unavailable]`; ingestion continues | Set `GROQ_API_KEY` or set `VLM_ENABLED=false` to opt out |
| Qdrant | Container not running | All queries return empty results | `docker compose up -d qdrant` |
| Ollama / nomic-embed-text | Model not pulled | Embedding step fails | `ollama pull nomic-embed-text` |
| Groq API | Missing or invalid `GROQ_API_KEY` | Generation returns 401 | Set `GROQ_API_KEY` in `.env` |
| Groq API | Rate limit hit | Slow or failed responses | Retry or use a local vLLM via `GENERATION_API_BASE` |
| Reranker | Model not downloaded | First query is slow (~30s download) | Pre-download: `uv run python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"` |
| File ingestion | Unsupported format uploaded | Silent skip with error toast | Only PDF, Excel, CSV, MD, DOCX, images are supported |
