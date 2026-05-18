"""Central configuration for the multi-modal RAG pipeline.

All defaults can be overridden via environment variables.
"""
from __future__ import annotations

import os

from dotenv import load_dotenv
load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Service endpoints
# ---------------------------------------------------------------------------
OLLAMA_API_BASE: str = os.getenv("OLLAMA_API_BASE", "http://127.0.0.1:11434")

# Embedding endpoint — defaults to Ollama, but can point to scripts/embedding_server.py
# for models not available in Ollama.
EMBED_API_BASE: str = os.getenv("EMBED_API_BASE", os.getenv("OLLAMA_API_BASE", "http://127.0.0.1:11435"))
OLLAMA_EMBED_MODEL: str = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:7333")
QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "documents_chunks")

GENERATION_API_BASE: str = os.getenv("GENERATION_API_BASE", "http://localhost:4000/v1")
GENERATION_MODEL: str = os.getenv("GENERATION_MODEL", "qwen/qwen3-32b")

# Required — get a free key at console.groq.com
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

# Optional — NVIDIA NIM fallback (build.nvidia.com)
NVIDIA_API_KEY: str = os.getenv("NVIDIA_API_KEY", "")

# LiteLLM proxy master key — used as the API key when routing through the proxy.
# Leave blank to run the proxy without authentication.
LITELLM_MASTER_KEY: str = os.getenv("LITELLM_MASTER_KEY", "")

# Contextual summary LLM used during chunking (smaller/faster model than generation)
CHUNK_LLM_API_BASE: str = os.getenv("CHUNK_LLM_API_BASE", "https://openrouter.ai/api/v1")
CHUNK_LLM_MODEL: str = os.getenv("CHUNK_LLM_MODEL", "google/gemma-4-31b-it:free")
# Key for the chunk LLM provider — defaults to OPENROUTER_API_KEY, falls back to GROQ_API_KEY
CHUNK_LLM_API_KEY: str = os.getenv("CHUNK_LLM_API_KEY", os.getenv("OPENROUTER_API_KEY", os.getenv("GROQ_API_KEY", "")))

OCR_API_BASE: str = os.getenv("OCR_API_BASE", "http://127.0.0.1:8002")
OCR_MODEL: str = os.getenv("OCR_MODEL", "lightonocr-2-1b-ocr-soup")

# Scanned-page parser selector. "auto" (default) routes scanned pages through
# the LightOn OCR vLLM server (requires CUDA). "cpu" routes them through
# unstructured + tesseract — slower (~10x) but no GPU required.
PDF_PARSER: str = os.getenv("PDF_PARSER", "auto").lower()

RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
RERANKER_DEVICE: str = os.getenv("RERANKER_DEVICE", "cpu")
EMBED_DEVICE: str = os.getenv("EMBED_DEVICE", "cpu")

# ---------------------------------------------------------------------------
# Retrieval & ranking
# ---------------------------------------------------------------------------
RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "100"))
RERANK_TOP_N: int = int(os.getenv("RERANK_TOP_N", "10"))
DOC_MIN_SCORE: float = float(os.getenv("DOC_MIN_SCORE", "0.01"))

# Max chars of a chunk/table shown to the LLM (prevents context overflow)
MAX_CHUNK_CHARS: int = int(os.getenv("MAX_CHUNK_CHARS", "1500"))
MAX_TABLE_CHARS: int = int(os.getenv("MAX_TABLE_CHARS", "3000"))

# Max chunks a single search_knowledge_base call may return to the agent
MAX_TOOL_RESULTS: int = int(os.getenv("MAX_TOOL_RESULTS", "8"))

# Max chars per ingested table chunk — wide sheets are split further at ingest time
MAX_CHARS_PER_TABLE_CHUNK: int = int(os.getenv("MAX_CHARS_PER_TABLE_CHUNK", "6000"))

# ---------------------------------------------------------------------------
# Table ingestion
# ---------------------------------------------------------------------------
MAX_ROWS_PER_CHUNK: int = int(os.getenv("MAX_ROWS_PER_CHUNK", "50"))

SKIP_SHEET_KEYWORDS: tuple[str, ...] = (
    "index", "contents", "cover", "notes", "summary", "overview", "total",
)
SKIP_ROW_VALUES: frozenset[str] = frozenset({"documentation box", "doc box"})
# UNFCCC and similar "no data" notations — rows where ALL data cells are these add noise
NO_DATA_TOKENS: frozenset[str] = frozenset({"no", "na", "ne", "ie", "n/a", "n.a.", "-", "–", ""})

# ---------------------------------------------------------------------------
# PDF chunking
# ---------------------------------------------------------------------------
CHUNK_MAX_TOKENS: int = int(os.getenv("CHUNK_MAX_TOKENS", "1024"))
CHUNK_MIN_TOKENS: int = int(os.getenv("CHUNK_MIN_TOKENS", "256"))

# ---------------------------------------------------------------------------
# Text-layer PDF parsing (pymupdf4llm path)
# ---------------------------------------------------------------------------
# Fraction of page area — images smaller than this are skipped by pymupdf4llm
IMAGE_SIZE_LIMIT: float = float(os.getenv("IMAGE_SIZE_LIMIT", "0.05"))
# Set False to skip VLM calls entirely (faster, no figure descriptions)
VLM_ENABLED: bool = os.getenv("VLM_ENABLED", "true").lower() not in ("false", "0", "no")
VLM_PROVIDER: str = os.getenv("VLM_PROVIDER", "groq")
VLM_MODEL: str = os.getenv("VLM_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")

# ---------------------------------------------------------------------------
# Excel → DuckDB structured query store
# ---------------------------------------------------------------------------
DUCKDB_PATH: str = os.getenv("DUCKDB_PATH", ".duckdb/excel_store.duckdb")

# ---------------------------------------------------------------------------
# Excel ReAct sub-agent (text-to-SQL with self-correction)
# ---------------------------------------------------------------------------
EXCEL_AGENT_MODEL: str = os.getenv("EXCEL_AGENT_MODEL", "gpt-4o-mini")
EXCEL_AGENT_API_BASE: str = os.getenv("EXCEL_AGENT_API_BASE", "https://api.openai.com/v1")
EXCEL_AGENT_API_KEY: str = os.getenv("EXCEL_AGENT_API_KEY", os.getenv("OPENAI_API_KEY", ""))

# ---------------------------------------------------------------------------
# FastAPI service
# ---------------------------------------------------------------------------
# Comma-separated list of allowed CORS origins; "*" disables the allow-list.
# The Next.js dev server runs on :3000 by default.
API_CORS_ORIGINS: str = os.getenv("API_CORS_ORIGINS", "http://localhost:3000")
# If set, every mutating endpoint (/ingest, DELETE /collection) requires the
# request header X-API-Key to match this value. Empty means no auth (dev only).
API_KEY: str = os.getenv("API_KEY", "")

# Base URL of the Vault RAG FastAPI server. The Slack bot queries through this
# so only the API process holds the DuckDB lock.
API_BASE: str = os.getenv("API_BASE", "http://localhost:8001")
