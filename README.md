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
- **Hybrid search** combines dense semantic vectors with sparse BM25-style keyword vectors, fused via Reciprocal Rank Fusion (RRF) in Qdrant, so both conceptual and exact-match queries work well.
- **HyDE query expansion** generates a hypothetical answer to the query and embeds that instead of the raw question, significantly improving recall on short or ambiguous questions.
- **Cross-encoder reranking** re-scores the top-100 candidates with a dedicated reranker model to surface the most relevant chunks.
- **ReAct agent** (LangGraph) can issue multiple search calls, compare results, and reason step-by-step before writing the final answer — useful for multi-part or comparative questions.
- **Document inspector** shows the original PDF page side-by-side with the parsed Markdown so you can verify extraction quality.

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
 Upload
   │
   ▼
 ┌─────────────┐     ┌──────────────────┐     ┌────────────────┐
 │  File type  │────▶│  Parser          │────▶│  Chunker       │
 │  detection  │     │  LightOn OCR /   │     │  Markdown      │
 └─────────────┘     │  openpyxl /      │     │  header split  │
                     │  LibreOffice     │     │  + LLM context │
                     └──────────────────┘     └───────┬────────┘
                                                       │
                                                       ▼
                                              ┌────────────────┐
                                              │  Embedder      │
                                              │  nomic-embed   │
                                              │  + sparse BM25 │
                                              └───────┬────────┘
                                                       │
                                                       ▼
                                              ┌────────────────┐
                                              │  Qdrant        │
                                              │  dense+sparse  │
                                              └────────────────┘

 Query
   │
   ▼
 ┌─────────────┐     ┌──────────────────┐     ┌────────────────┐
 │  HyDE       │────▶│  Hybrid search   │────▶│  Reranker      │
 │  expansion  │     │  RRF fusion      │     │  cross-encoder │
 └─────────────┘     └──────────────────┘     └───────┬────────┘
                                                       │
                                                       ▼
                                              ┌────────────────┐
                                              │  ReAct agent   │
                                              │  (LangGraph)   │
                                              └───────┬────────┘
                                                       │
                                                       ▼
                                              ┌────────────────┐
                                              │  Answer with   │
                                              │  cited sources │
                                              └────────────────┘
```

---

## Tech stack

| Component | Technology | Why |
|-----------|-----------|-----|
| OCR / parsing | LightOn OCR | State-of-the-art vision-language model for complex PDFs and scanned documents; runs locally via vLLM |
| Embeddings | nomic-embed-text | High-quality, English-optimised open embedding model; fast on CPU; runs locally via Ollama |
| Vector database | Qdrant | Native hybrid search (dense + sparse) with RRF fusion; easy Docker deployment |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 | Fast, accurate cross-encoder trained on MS MARCO; dramatically improves precision over retrieval-only |
| Generation | llama-3.3-70b-versatile (Groq) | Best-in-class open LLM; Groq provides free-tier inference with low latency |
| UI | Streamlit | Rapid iteration on chat UI with minimal boilerplate |
| Agent | LangGraph (ReAct) | Structured multi-step reasoning with tool calls; transparent decision trace |
| Observability | Langfuse | Trace every agent run, inspect tool calls and retrieval results |

---

## Privacy & data

- **Parsing** runs entirely locally via the LightOn OCR server (vLLM on your GPU or CPU).
- **Embeddings** are generated locally by Ollama — your document text never leaves the machine at indexing time.
- **LLM inference** sends only the retrieved chunks and the user query to the Groq API. If this is unacceptable, point `GENERATION_API_BASE` at a local vLLM server running any compatible model (e.g. `llama-3.3-70b`) and set `GENERATION_MODEL` accordingly — the pipeline becomes fully air-gapped.

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

# Start Qdrant
docker compose -f docker/ingestion-stack/docker-compose.yaml up -d qdrant

# Pull embedding model
ollama pull nomic-embed-text

# Start the app
uv run streamlit run app.py
```

### LightOn OCR (optional — for scanned PDFs)

```bash
docker compose -f docker/ingestion-stack/docker-compose.yaml --profile ingest up -d lightonocr-vllm
```

---

## Configuration

All variables can be set in `.env`. See `.env.example` for a full annotated list.

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | _(required)_ | Groq API key — free tier at console.groq.com |
| `GENERATION_API_BASE` | `https://api.groq.com/openai/v1` | OpenAI-compatible generation endpoint |
| `GENERATION_MODEL` | `llama-3.3-70b-versatile` | LLM model name |
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
| LightOn OCR | vLLM server not running | Ingest crashes with `Connection refused` on port 8002 | `docker compose --profile ingest up -d` |
| Qdrant | Container not running | All queries return empty results | `docker compose up -d qdrant` |
| Ollama / nomic-embed-text | Model not pulled | Embedding step fails | `ollama pull nomic-embed-text` |
| Groq API | Missing or invalid `GROQ_API_KEY` | Generation returns 401 | Set `GROQ_API_KEY` in `.env` |
| Groq API | Rate limit hit | Slow or failed responses | Retry or use a local vLLM via `GENERATION_API_BASE` |
| Reranker | Model not downloaded | First query is slow (~30s download) | Pre-download: `uv run python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"` |
| File ingestion | Unsupported format uploaded | Silent skip with error toast | Only PDF, Excel, CSV, MD, DOCX, images are supported |
