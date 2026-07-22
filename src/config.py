"""Central configuration for the multi-modal RAG pipeline.

All defaults can be overridden via environment variables.
Import individual names directly: ``from src.config import GROQ_API_KEY``
"""

from __future__ import annotations

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Service endpoints
    ollama_api_base: str = "http://127.0.0.1:11434"
    embed_api_base: str = ""
    ollama_embed_model: str = "nomic-embed-text"
    qdrant_url: str = "http://localhost:7333"
    qdrant_collection: str = "documents_chunks"
    generation_api_base: str = "http://localhost:4000/v1"
    generation_model: str = "qwen/qwen3-32b"
    # Per-request timeout for all query-time LLM calls (agent turns, HyDE, excel
    # sub-agent). No client-side timeout previously meant a stalled provider call
    # could block for the SDK default (~10 min); comparison questions chain
    # several such calls sequentially, so one slow turn could hang a whole request.
    llm_request_timeout_s: float = 60.0
    # Pin OpenRouter to one named backend provider (e.g. "DeepInfra"), empty =
    # no pin (OpenRouter picks per-request, can silently vary the actual engine/
    # quantization serving "the same" model call to call). Only applies when
    # generation_api_base is openrouter.ai. See TODO.md for why this exists.
    openrouter_provider_pin: str = ""

    # Secrets
    groq_api_key: SecretStr = SecretStr("")
    nvidia_api_key: SecretStr = SecretStr("")
    litellm_master_key: SecretStr = SecretStr("")
    openrouter_api_key: SecretStr = SecretStr("")
    free_llm_api_key: SecretStr = SecretStr("")
    chunk_llm_api_key: SecretStr = SecretStr("")
    openai_api_key: SecretStr = SecretStr("")
    excel_agent_api_key: SecretStr = SecretStr("")
    api_key: SecretStr = SecretStr("")

    # Optional admin/viewer access mode -- "open" (default) preserves today's
    # unauthenticated dev behavior everywhere. "admin_viewer" requires a
    # logged-in admin session (or the existing X-API-Key header) for
    # upload/reprocess/delete/clear/eval-run/feedback-resolve/drive-config,
    # while asking questions, browsing evidence, and reading conversations
    # stay open to anyone. See api.py's require_admin.
    access_mode: str = "open"
    admin_password: SecretStr = SecretStr("")
    session_secret: SecretStr = SecretStr("")
    # False for local HTTP dev (default); set True once served over HTTPS,
    # otherwise browsers silently refuse to send the cookie back at all.
    cookie_secure: bool = False

    # Chunking LLM
    chunk_llm_api_base: str = "https://openrouter.ai/api/v1"
    chunk_llm_model: str = "google/gemma-4-31b-it:free"

    # OCR
    ocr_api_base: str = "http://127.0.0.1:8002"
    ocr_model: str = "lightonocr-2-1b-ocr-soup"
    pdf_parser: str = "auto"

    # Reranker / embedder
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_device: str = "cpu"
    reranker_enabled: bool = True
    embed_device: str = "cpu"

    # Retrieval & ranking
    retrieval_top_k: int = 100
    rerank_top_n: int = 10
    doc_min_score: float = 0.01
    max_chunk_chars: int = 1500
    max_table_chars: int = 3000
    max_tool_results: int = 12
    max_chars_per_table_chunk: int = 6000

    # Post-generation grounding check — one extra LLM call per answered (non-
    # Unsupported) query, verifying the answer is actually supported by the
    # retrieved context before it's returned. Adds latency/cost; toggle off if
    # that's not an acceptable tradeoff for a given deployment.
    post_generation_verify_enabled: bool = True

    # Table ingestion
    max_rows_per_chunk: int = 50

    # PDF chunking
    chunk_max_tokens: int = 1024
    chunk_min_tokens: int = 256

    # VLM (figure descriptions)
    image_size_limit: float = 0.05
    vlm_enabled: bool = True
    vlm_provider: str = "groq"
    vlm_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"

    # DuckDB
    duckdb_path: str = ".duckdb/excel_store.duckdb"

    # Feedback queue
    feedback_path: str = "data/feedback.json"
    eval_regression_candidates_path: str = "eval/regression_candidates.jsonl"

    # Query cache -- avoids rerunning retrieval + generation for a question
    # already answered since the corpus was last changed.
    query_cache_enabled: bool = True
    query_cache_path: str = "data/query_cache.json"

    # Usage log -- per-question token/cost tracking for the admin usage panel.
    usage_log_path: str = "data/usage_log.json"

    # Conversation history
    conversation_path: str = "data/conversations.json"

    # Admin-set display titles -- overrides the extracted/fallback title
    # shown for a source without touching the underlying document or its
    # embeddings (see src/title_overrides.py).
    title_overrides_path: str = "data/title_overrides.json"

    # Google Drive folder sync connector
    input_dir: str = "data/input"
    google_drive_service_account_file: str = ""
    google_drive_sync_state_path: str = "data/google_drive_sync_state.json"

    # Excel sub-agent
    excel_agent_model: str = "gpt-4o-mini"
    excel_agent_api_base: str = "https://api.openai.com/v1"

    # FastAPI
    api_cors_origins: str = "http://localhost:3000"
    api_base: str = "http://localhost:8001"

    @model_validator(mode="after")
    def _apply_fallbacks(self) -> Settings:
        # embed_api_base defaults to ollama_api_base when not set
        if not self.embed_api_base:
            self.embed_api_base = self.ollama_api_base
        # chunk_llm_api_key falls back to openrouter then groq
        if not self.chunk_llm_api_key.get_secret_value():
            fallback = (
                self.openrouter_api_key.get_secret_value()
                or self.groq_api_key.get_secret_value()
            )
            self.chunk_llm_api_key = SecretStr(fallback)
        # excel_agent_api_key falls back to openai
        if not self.excel_agent_api_key.get_secret_value():
            self.excel_agent_api_key = SecretStr(self.openai_api_key.get_secret_value())
        # pdf_parser normalised to lowercase
        self.pdf_parser = self.pdf_parser.lower()
        return self


_s = Settings()

# ---------------------------------------------------------------------------
# Module-level exports — all existing import sites continue to work unchanged
# ---------------------------------------------------------------------------
OLLAMA_API_BASE: str = _s.ollama_api_base
EMBED_API_BASE: str = _s.embed_api_base
OLLAMA_EMBED_MODEL: str = _s.ollama_embed_model
QDRANT_URL: str = _s.qdrant_url
QDRANT_COLLECTION: str = _s.qdrant_collection
GENERATION_API_BASE: str = _s.generation_api_base
GENERATION_MODEL: str = _s.generation_model
LLM_REQUEST_TIMEOUT_S: float = _s.llm_request_timeout_s
OPENROUTER_PROVIDER_PIN: str = _s.openrouter_provider_pin

GROQ_API_KEY: str = _s.groq_api_key.get_secret_value()
NVIDIA_API_KEY: str = _s.nvidia_api_key.get_secret_value()
LITELLM_MASTER_KEY: str = _s.litellm_master_key.get_secret_value()
OPENROUTER_API_KEY: str = _s.openrouter_api_key.get_secret_value()
FREE_LLM_API_KEY: str = _s.free_llm_api_key.get_secret_value()
CHUNK_LLM_API_KEY: str = _s.chunk_llm_api_key.get_secret_value()
EXCEL_AGENT_API_KEY: str = _s.excel_agent_api_key.get_secret_value()
API_KEY: str = _s.api_key.get_secret_value()
ACCESS_MODE: str = _s.access_mode
ADMIN_PASSWORD: str = _s.admin_password.get_secret_value()
SESSION_SECRET: str = _s.session_secret.get_secret_value()
COOKIE_SECURE: bool = _s.cookie_secure

CHUNK_LLM_API_BASE: str = _s.chunk_llm_api_base
CHUNK_LLM_MODEL: str = _s.chunk_llm_model
OCR_API_BASE: str = _s.ocr_api_base
OCR_MODEL: str = _s.ocr_model
PDF_PARSER: str = _s.pdf_parser
RERANKER_MODEL: str = _s.reranker_model
RERANKER_DEVICE: str = _s.reranker_device
RERANKER_ENABLED: bool = _s.reranker_enabled
EMBED_DEVICE: str = _s.embed_device
RETRIEVAL_TOP_K: int = _s.retrieval_top_k
RERANK_TOP_N: int = _s.rerank_top_n
DOC_MIN_SCORE: float = _s.doc_min_score
MAX_CHUNK_CHARS: int = _s.max_chunk_chars
MAX_TABLE_CHARS: int = _s.max_table_chars
MAX_TOOL_RESULTS: int = _s.max_tool_results
POST_GENERATION_VERIFY_ENABLED: bool = _s.post_generation_verify_enabled
MAX_CHARS_PER_TABLE_CHUNK: int = _s.max_chars_per_table_chunk
MAX_ROWS_PER_CHUNK: int = _s.max_rows_per_chunk
CHUNK_MAX_TOKENS: int = _s.chunk_max_tokens
CHUNK_MIN_TOKENS: int = _s.chunk_min_tokens
IMAGE_SIZE_LIMIT: float = _s.image_size_limit
VLM_ENABLED: bool = _s.vlm_enabled
VLM_PROVIDER: str = _s.vlm_provider
VLM_MODEL: str = _s.vlm_model
DUCKDB_PATH: str = _s.duckdb_path
FEEDBACK_PATH: str = _s.feedback_path
EVAL_REGRESSION_CANDIDATES_PATH: str = _s.eval_regression_candidates_path
QUERY_CACHE_ENABLED: bool = _s.query_cache_enabled
QUERY_CACHE_PATH: str = _s.query_cache_path
USAGE_LOG_PATH: str = _s.usage_log_path

# Approximate list prices, USD per 1M tokens, (input, output) -- for the admin
# usage panel's cost estimate only, not a billing source of truth. Verify
# against the provider's current pricing page before trusting these for real
# budgeting. Unknown models (e.g. a local vLLM route) cost $0 here since
# there's no API bill.
USAGE_PRICE_PER_1M_TOKENS: dict[str, tuple[float, float]] = {
    "qwen/qwen3-32b": (0.29, 0.59),
    "openai/gpt-oss-120b": (0.15, 0.75),  # Groq pricing, see console.groq.com/pricing
}
INPUT_DIR: str = _s.input_dir
GOOGLE_DRIVE_SERVICE_ACCOUNT_FILE: str = _s.google_drive_service_account_file
GOOGLE_DRIVE_SYNC_STATE_PATH: str = _s.google_drive_sync_state_path
CONVERSATION_PATH: str = _s.conversation_path
TITLE_OVERRIDES_PATH: str = _s.title_overrides_path
EXCEL_AGENT_MODEL: str = _s.excel_agent_model
EXCEL_AGENT_API_BASE: str = _s.excel_agent_api_base
API_CORS_ORIGINS: str = _s.api_cors_origins
API_BASE: str = _s.api_base

# ---------------------------------------------------------------------------
# Non-env constants
# ---------------------------------------------------------------------------
SKIP_SHEET_KEYWORDS: tuple[str, ...] = (
    "index",
    "contents",
    "cover",
    "notes",
    "summary",
    "overview",
    "total",
)
SKIP_ROW_VALUES: frozenset[str] = frozenset({"documentation box", "doc box"})
# UNFCCC and similar "no data" notations — rows where ALL data cells are these add noise
NO_DATA_TOKENS: frozenset[str] = frozenset(
    {"no", "na", "ne", "ie", "n/a", "n.a.", "-", "–", ""}
)
