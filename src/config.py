"""Central configuration for the multi-modal RAG pipeline.

All defaults can be overridden via environment variables.
"""
from __future__ import annotations

import os

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

GENERATION_API_BASE: str = os.getenv("GENERATION_API_BASE", "https://api.groq.com/openai/v1")
GENERATION_MODEL: str = os.getenv("GENERATION_MODEL", "llama-3.3-70b-versatile")

# Required — get a free key at console.groq.com
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

OCR_API_BASE: str = os.getenv("OCR_API_BASE", "http://127.0.0.1:8002")
OCR_MODEL: str = os.getenv("OCR_MODEL", "lightonocr-2-1b-ocr-soup")

RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
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
