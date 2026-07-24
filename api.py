"""Vault RAG — FastAPI backend for the Next.js UI."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import io
import json
import logging
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from functools import lru_cache, partial
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv(override=True)

from fastapi import (  # noqa: E402
    Cookie,
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    Response,
    UploadFile,
)
from fastapi.concurrency import run_in_threadpool  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse, StreamingResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from src import query_cache  # noqa: E402
from src.config import (  # noqa: E402
    ACCESS_MODE,
    ADMIN_PASSWORD,
    API_CORS_ORIGINS,
    API_KEY,
    COOKIE_SECURE,
    EMBED_API_BASE,
    GENERATION_API_BASE,
    GENERATION_MODEL,
    GROQ_API_KEY,
    LITELLM_MASTER_KEY,
    OLLAMA_EMBED_MODEL,
    OPENROUTER_API_KEY,
    QDRANT_COLLECTION,
    QDRANT_URL,
    QUERY_CACHE_ENABLED,
    RERANK_TOP_N,
    RERANKER_MODEL,
    RETRIEVAL_TOP_K,
    SESSION_SECRET,
)
from src.retriever import _ollama_embed_query  # noqa: E402
from src.vector_store import _request as _qdrant  # noqa: E402
from src.vector_store import (  # noqa: E402
    delete_by_file,
    get_chunks_by_file,
    scroll_all_payloads,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent
INPUT_DIR = REPO_ROOT / "data" / "input"
INPUT_DIR.mkdir(parents=True, exist_ok=True)

MARKDOWN_DIRS = (
    REPO_ROOT / "data" / "output" / "processed",
    REPO_ROOT / "data" / "output" / "lightonocr",
    REPO_ROOT / "data" / "output" / "pymupdf",
    REPO_ROOT / "data" / "output" / "translated",
)

_jobs: dict[str, dict[str, Any]] = {}
_executor = ThreadPoolExecutor(max_workers=2)


# ── lifespan: warm the agent on startup ────────────────────────────────────────


def _validate_startup_env() -> None:
    """Log a clear, actionable error when GENERATION_API_BASE points at a
    provider that needs a key we don't have -- without this, the failure
    only ever surfaces as a cryptic 401 from the provider on the first real
    /query, long after startup looked clean."""
    base = GENERATION_API_BASE.lower()
    missing: str | None = None
    if "openrouter.ai" in base and not OPENROUTER_API_KEY:
        missing = "OPENROUTER_API_KEY"
    elif "groq.com" in base and not GROQ_API_KEY:
        missing = "GROQ_API_KEY"
    elif (
        "localhost:4000" in base or "127.0.0.1:4000" in base
    ) and not LITELLM_MASTER_KEY:
        # LiteLLM proxy accepts an unauthenticated request in some configs,
        # so this is a warning, not necessarily a hard failure.
        logger.warning(
            "GENERATION_API_BASE points at the LiteLLM proxy (%s) but "
            "LITELLM_MASTER_KEY is not set -- requests will fail unless the "
            "proxy itself has auth disabled.",
            GENERATION_API_BASE,
        )
        return
    if missing:
        logger.error(
            "GENERATION_API_BASE=%s requires %s, which is not set. Every "
            "/query will fail with an authentication error until this is "
            "fixed in .env. See .env.example.",
            GENERATION_API_BASE,
            missing,
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the agent once at startup so the first /query doesn't pay the cold-start penalty."""
    _validate_startup_env()
    try:
        _get_agent()
        logger.info("Agent warmed; ready to serve")
    except Exception:
        logger.exception("Agent warmup failed; /query will retry per-request")
    # Embedding warm-up: the reranker already warms up inside build_rag_agent(),
    # but the embedding call (every retrieval's first step) didn't -- so the
    # very first real query was still paying Ollama's cold-start latency.
    try:
        _ollama_embed_query(EMBED_API_BASE, OLLAMA_EMBED_MODEL, "warmup")
        logger.info("Embedding model warmed up")
    except Exception:
        logger.exception("Embedding warmup failed; first /query will retry per-request")
    yield


app = FastAPI(title="Vault RAG API", lifespan=lifespan)


# ── CORS — explicit origin list (use API_CORS_ORIGINS=* only intentionally) ───
#
# allow_credentials is required for the admin session cookie to survive a
# cross-origin request (frontend on :3000/:3001, API on :8001) -- browsers
# reject allow_credentials=True combined with a wildcard origin, so
# API_CORS_ORIGINS=* is incompatible with ACCESS_MODE=admin_viewer.

_cors_origins = [o.strip() for o in API_CORS_ORIGINS.split(",") if o.strip()] or ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials="*" not in _cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── auth dep — admin-only endpoints ────────────────────────────────────────────
#
# ACCESS_MODE=open (default): unchanged from before this existed -- gated only
# when API_KEY is set, via the X-API-Key header.
#
# ACCESS_MODE=admin_viewer: everyone can ask questions, browse evidence, and
# read conversations (those routes carry no dependency at all, see below);
# admin-only routes (upload/reprocess/delete/clear/eval-run/feedback-resolve/
# drive-config) require either the X-API-Key header or a session cookie set
# by POST /admin/login. Enforcement lives here in the backend, not just in
# what the frontend chooses to render.

ADMIN_COOKIE_NAME = "vault_admin_session"


def _admin_session_token() -> str:
    """Deterministic HMAC token for the admin session cookie.

    Deliberately simple, by design (this is a small optional feature, not
    full session management, per the brief): no per-login random token or
    session store, so this doesn't support server-side session revocation --
    POST /admin/logout only clears the browser's copy of the cookie. That's
    an accepted trade-off for a single-admin demo/portfolio deployment.
    """
    return hmac.new(
        SESSION_SECRET.encode(), b"vault-rag-admin", hashlib.sha256
    ).hexdigest()


def _is_valid_admin_session(token: str | None) -> bool:
    return bool(
        SESSION_SECRET and token and hmac.compare_digest(token, _admin_session_token())
    )


def _is_admin_caller(x_api_key: str | None, vault_admin_session: str | None) -> bool:
    """Same authorization check as require_admin, without raising -- for routes
    that stay open to everyone but need to know the caller's role (e.g. /query
    refusing corpus-enumeration questions from a non-admin)."""
    if ACCESS_MODE != "admin_viewer":
        return not API_KEY or x_api_key == API_KEY
    return (API_KEY and x_api_key == API_KEY) or _is_valid_admin_session(
        vault_admin_session
    )


async def require_admin(
    x_api_key: str | None = Header(default=None),
    vault_admin_session: str | None = Cookie(default=None),
) -> None:
    """Raise 401/403 unless the caller is authorized for an admin-only route."""
    if _is_admin_caller(x_api_key, vault_admin_session):
        return
    if ACCESS_MODE != "admin_viewer":
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")
    raise HTTPException(status_code=403, detail="Admin login required")


# Kept for any external caller/script still using the old name.
require_api_key = require_admin


class AdminLoginRequest(BaseModel):
    password: str


@app.post("/admin/login")
async def admin_login(req: AdminLoginRequest, response: Response) -> dict:
    """POST /admin/login — exchange ADMIN_PASSWORD for an admin session cookie."""
    if ACCESS_MODE != "admin_viewer":
        raise HTTPException(
            status_code=400,
            detail="Admin login is only available when ACCESS_MODE=admin_viewer",
        )
    if not ADMIN_PASSWORD or not hmac.compare_digest(req.password, ADMIN_PASSWORD):
        raise HTTPException(status_code=401, detail="Incorrect password")
    response.set_cookie(
        key=ADMIN_COOKIE_NAME,
        value=_admin_session_token(),
        httponly=True,
        samesite="lax",
        secure=COOKIE_SECURE,
        max_age=60 * 60 * 24 * 7,
    )
    return {"status": "ok"}


@app.post("/admin/logout")
async def admin_logout(response: Response) -> dict:
    """POST /admin/logout — clear the admin session cookie."""
    response.delete_cookie(ADMIN_COOKIE_NAME)
    return {"status": "ok"}


@app.get("/admin/session")
async def admin_session(vault_admin_session: str | None = Cookie(default=None)) -> dict:
    """GET /admin/session — lets the frontend know the access mode and whether
    the current browser session is logged in as admin, to show/hide admin UI."""
    return {
        "access_mode": ACCESS_MODE,
        "is_admin": ACCESS_MODE != "admin_viewer"
        or _is_valid_admin_session(vault_admin_session),
    }


# ── global exception handler — never leak stack traces ────────────────────────


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Log the exception and return a generic 500 instead of leaking str(exc)."""
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


# ── agent singleton ────────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def _get_agent() -> Any:
    """Build (and cache) the RAG agent — one instance reused across requests."""
    from src.llm_credentials import resolve_generation_override
    from src.rag_agent import build_rag_agent

    override = resolve_generation_override()
    api_base, model_name = override or (GENERATION_API_BASE, GENERATION_MODEL)

    return build_rag_agent(
        qdrant_url=QDRANT_URL,
        collection=QDRANT_COLLECTION,
        retrieval_top_k=RETRIEVAL_TOP_K,
        rerank_top_n=RERANK_TOP_N,
        reranker_model_name=RERANKER_MODEL or None,
        model_name=model_name,
        generation_api_base=api_base,
    )


# ── helpers ────────────────────────────────────────────────────────────────────


def _run_ingest_sync(
    job_id: str, dest: Path, force_pipeline: str | None = None
) -> None:
    """Ingest one uploaded file into Qdrant/DuckDB, updating the job record."""
    suffix = dest.suffix.lower()
    _jobs[job_id]["status"] = "processing"
    try:
        if suffix in {".xlsx", ".xls", ".csv"}:
            _jobs[job_id]["stage"] = "chunking"
            from src.ingest_table_rows import ingest_table_rows

            ingest_table_rows(str(dest), collection=QDRANT_COLLECTION)
            _jobs[job_id]["chunks_created"] = -1
        else:
            from src.ingest import run_ingest

            _jobs[job_id]["stage"] = "parsing"
            result = run_ingest(
                pdf_path=dest,
                collection=QDRANT_COLLECTION,
                force_pipeline=force_pipeline,
            )
            chunks_path = (
                result.get("chunks_path") if isinstance(result, dict) else None
            )
            if chunks_path and Path(chunks_path).exists():
                import json as _json

                _jobs[job_id]["chunks_created"] = len(
                    _json.loads(Path(chunks_path).read_text())
                )
        _jobs[job_id].update({"status": "done", "stage": "indexed"})
        # Refresh the cached agent so its doc_registry (filename/title -> doc_id,
        # built once at agent-construction time) picks up this document. Must
        # happen on actual completion, not when the background job is queued --
        # reindex_document() used to clear the cache immediately at request time,
        # before the async ingest even started, so a query arriving after the
        # job *finished* still got the pre-ingest registry. New uploads via
        # /ingest never cleared it at all, so a freshly uploaded document was
        # invisible to title-based routing until something else (a delete, a
        # reindex of a different file) happened to rebuild the agent.
        _get_agent.cache_clear()
        query_cache.clear()
    except Exception as exc:
        _jobs[job_id].update({"status": "failed", "stage": "failed", "error": str(exc)})


_SHEET_ROW_COUNT_RE = re.compile(r"Sheet summary: (\d+) rows\.")
_TITLE_LINE_RE = re.compile(r"^Title: (.+)$", re.MULTILINE)


def _payloads_to_docs(payloads: list[dict]) -> list[dict]:
    """Group Qdrant payloads into one document card per source file.

    page_count/sheet_count/row_count/display_title are derived from metadata
    already on these payloads (chunk page numbers, sheet_name, the
    sheet_summary chunk's own "N rows." text, the document_summary chunk's
    own "Title: ..." line -- same one src/answer_pipeline.py's title-shortcut
    answer already quotes verbatim) -- no extra file I/O per document.
    """
    from collections import defaultdict

    counts: dict[str, int] = defaultdict(int)
    last_indexed: dict[str, str] = {}
    max_page: dict[str, int] = {}
    sheets: dict[str, set[str]] = defaultdict(set)
    row_counts: dict[str, int] = defaultdict(int)
    titles: dict[str, str] = {}
    for p in payloads:
        meta = p.get("metadata", {}) or {}
        name = meta.get("source_file") or meta.get("file_name") or ""
        if not name:
            continue
        # Same doc ingested both as a bare filename and via the eval corpus's
        # "eval/data/raw/<name>" path shows up as two separate source_file
        # values for the same physical document -- normalize before grouping
        # so it collapses into one card instead of two (see answer_pipeline.py's
        # identical normalization for citations).
        if name.startswith("eval/data/raw/"):
            name = name[len("eval/data/raw/") :]
        counts[name] += 1
        ts = meta.get("ingested_at") or ""
        if ts > last_indexed.get(name, ""):
            last_indexed[name] = ts
        suffix = Path(name).suffix.lstrip(".").lower()
        page = meta.get("page")
        if suffix == "pdf" and isinstance(page, int):
            max_page[name] = max(max_page.get(name, 0), page)
        sheet_name = meta.get("sheet_name")
        # Guard by extension: a PDF's own embedded-table extraction can also
        # carry sheet_summary/table_N metadata (found live on a stray
        # duplicate doc_001 entry) -- without this it'd wrongly show
        # "N sheets" on what's actually a PDF, not a spreadsheet.
        if sheet_name and suffix in ("xlsx", "xls", "csv"):
            sheets[name].add(sheet_name)
        # Row counts only live in the document_summary chunk's concatenated
        # "Sheet summary: N rows." lines (one per sheet) -- the sheet_summary
        # points themselves carry a DuckDB-table discovery blurb, not a count.
        if (
            suffix in ("xlsx", "xls", "csv")
            and meta.get("chunk_type") == "document_summary"
        ):
            total = sum(
                int(n) for n in _SHEET_ROW_COUNT_RE.findall(p.get("content") or "")
            )
            if total:
                row_counts[name] = total
        if meta.get("chunk_type") == "document_summary" and name not in titles:
            title_match = _TITLE_LINE_RE.search(p.get("content") or "")
            if title_match:
                titles[name] = title_match.group(1).strip()
    type_map = {
        "pdf": "PDF",
        "xlsx": "Excel",
        "xls": "Excel",
        "csv": "CSV",
        "docx": "Word",
        "doc": "Word",
        "md": "MD",
        "png": "Image",
        "jpg": "Image",
        "jpeg": "Image",
    }
    return [
        {
            "filename": name,
            "display_title": titles.get(name) or None,
            "file_type": type_map.get(Path(name).suffix.lstrip(".").lower(), "File"),
            "chunk_count": count,
            "status": "indexed",
            "last_indexed_at": last_indexed.get(name) or None,
            "page_count": max_page.get(name),
            "sheet_count": len(sheets[name]) if name in sheets else None,
            "row_count": row_counts.get(name) or None,
        }
        for name, count in sorted(counts.items())
    ]


def _document_count_answer() -> str:
    """Real answer to "how many documents", from the same data /stats uses --
    not a RAG-agent guess. The agent has no counting tool, so routed to
    search_knowledge_base it was answering from whichever chunk got
    retrieved, which is neither reliable nor the right kind of question for
    a retrieval agent to attempt."""
    try:
        payloads = scroll_all_payloads(QDRANT_URL, QDRANT_COLLECTION)
    except Exception:
        return "The document count is temporarily unavailable."
    n = len(_payloads_to_docs(payloads))
    return f"There {'is' if n == 1 else 'are'} {n} document{'' if n == 1 else 's'} in the knowledge base."


# _TABLE_MARKER_RE is also used by _split_markdown_pages below.
_TABLE_MARKER_RE = re.compile(r"\[TABLE_START\]|\[TABLE_END\]")

# Header parsing, leaked-header/citation stripping, and source-card assembly
# now live in src/answer_pipeline.py (parse_sources / strip_leaked_headers) —
# shared with eval/run_eval.py so both measure identical behavior.


def _resolve_source_file_path(filename: str) -> Path | None:
    """Find the on-disk source file (PDF, xlsx, csv, ...) for a filename, or
    None if it cannot be located.

    Documents ingested via the eval corpus (make seed pulls a subset into
    data/input/, but most eval docs are only ever ingested straight from
    eval/data/raw/) have their markdown/chunks/embeddings under data/output/
    regardless of where the original source file lives — but the inspector's
    page-image, bbox-highlight, and raw-table-rows endpoints all need the
    actual source file, so eval/data/raw/ must be a real fallback location,
    not just data/input/.
    """
    p = Path(filename)
    candidates = [
        INPUT_DIR / p.name,
        REPO_ROOT / "eval" / "data" / "raw" / p.name,
        p if p.is_absolute() else None,
        REPO_ROOT / p,
    ]
    for c in candidates:
        if c and c.exists():
            return c
    return None


def _resolve_markdown_path(filename: str) -> Path | None:
    """Find the parsed-markdown file for a document, or None if absent."""
    stem = Path(filename).stem
    for d in MARKDOWN_DIRS:
        candidate = d / f"{stem}.md"
        if candidate.exists():
            return candidate
    return None


def _split_markdown_pages(md_text: str) -> tuple[dict[int, str], dict[int, str]]:
    """Split markdown by <!-- PAGE N | label --> markers."""
    parts = re.split(r"<!--\s*PAGE\s+(\d+)(?:\s*\|([^-]*))?\s*-->", md_text)
    pages: dict[int, str] = {}
    pipelines: dict[int, str] = {}
    i = 1
    while i < len(parts) - 2:
        page_num = int(parts[i])
        label = (parts[i + 1] or "").strip()
        content = _TABLE_MARKER_RE.sub("", parts[i + 2]).strip()
        pages[page_num] = content
        if label:
            pipelines[page_num] = label
        i += 3
    return pages, pipelines


# ── routes ─────────────────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    """Liveness probe — returns immediately, does not touch Qdrant or the agent."""
    return {"status": "ok"}


@app.post("/ingest", dependencies=[Depends(require_admin)])
async def ingest(file: UploadFile = File(...), pipeline: str = Form("auto")):
    """POST /ingest — save the uploaded file and start a background ingest job."""
    dest = INPUT_DIR / file.filename
    dest.write_bytes(await file.read())
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "pending", "stage": "parsing", "chunks_created": 0}
    force_pipeline = pipeline if pipeline in {"ocr", "text"} else None
    loop = asyncio.get_running_loop()
    loop.run_in_executor(_executor, _run_ingest_sync, job_id, dest, force_pipeline)
    return {"job_id": job_id, "status": "processing"}


@app.get("/ingest/status/{job_id}")
async def ingest_status(job_id: str):
    """GET /ingest/status — report a background ingest job's progress."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "status": job["status"],
        "stage": job["stage"],
        "chunks_created": job.get("chunks_created", 0),
        "error": job.get("error"),
    }


@app.get("/documents/exists")
async def documents_exist():
    """GET /documents/exists — whether the corpus is non-empty, nothing else.

    Non-admin callers need this to enable the chat input (see ChatPanel's
    hasSources) without learning the document count or any filename -- the
    full /documents listing and /stats counts are admin-only (see
    require_admin below); this is the one piece of corpus state a non-admin
    is allowed to see.
    """
    try:
        payloads = scroll_all_payloads(QDRANT_URL, QDRANT_COLLECTION)
    except Exception:
        return {"has_documents": False}
    return {"has_documents": len(payloads) > 0}


@app.get("/documents", dependencies=[Depends(require_admin)])
async def list_documents():
    """GET /documents — list ingested documents as UI cards. Admin-only: a
    non-admin viewer must not be able to enumerate the corpus (see
    /documents/exists for the one non-admin-safe signal)."""
    try:
        payloads = scroll_all_payloads(QDRANT_URL, QDRANT_COLLECTION)
    except Exception:
        return []
    from src.title_overrides import get_overrides

    docs = _payloads_to_docs(payloads)
    overrides = get_overrides()
    for doc in docs:
        if doc["filename"] in overrides:
            doc["display_title"] = overrides[doc["filename"]]
    return docs


class SetTitleRequest(BaseModel):
    title: str | None = None


@app.patch("/documents/{filename:path}/title", dependencies=[Depends(require_admin)])
async def set_document_title(filename: str, req: SetTitleRequest):
    """PATCH /documents/{filename}/title — set or clear this source's
    admin display title. A blank/missing title clears the override,
    reverting to the extracted title (or filename, if none)."""
    from src.title_overrides import clear_title, set_title

    if req.title and req.title.strip():
        set_title(filename, req.title.strip())
    else:
        clear_title(filename)
    return {"status": "ok"}


@app.get("/eval/summary")
async def eval_summary():
    """GET /eval/summary — serve the last computed benchmark results (from `make eval`)."""
    path = REPO_ROOT / "eval" / "results" / "summary.json"
    if not path.exists():
        raise HTTPException(
            status_code=404, detail="No eval results found. Run `make eval` first."
        )
    import json as _json

    return _json.loads(path.read_text())


_eval_jobs: dict[str, dict[str, Any]] = {}


def _run_eval_sync(job_id: str) -> None:
    """Run the full benchmark (real LLM calls, several minutes) and write summary.json."""
    from eval.run_eval import run

    _eval_jobs[job_id]["status"] = "running"
    try:
        summary = run()
        _eval_jobs[job_id].update({"status": "done", "summary": summary})
    except Exception as exc:
        _eval_jobs[job_id].update({"status": "failed", "error": str(exc)})


@app.post("/eval/run", dependencies=[Depends(require_admin)])
async def eval_run():
    """POST /eval/run — kick off the full benchmark in the background (real LLM calls)."""
    job_id = str(uuid.uuid4())
    _eval_jobs[job_id] = {"status": "pending"}
    loop = asyncio.get_running_loop()
    loop.run_in_executor(_executor, _run_eval_sync, job_id)
    return {"job_id": job_id, "status": "running"}


@app.get("/eval/status/{job_id}")
async def eval_status(job_id: str):
    """GET /eval/status — poll a background eval run started via POST /eval/run."""
    job = _eval_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/stats", dependencies=[Depends(require_admin)])
async def stats():
    """GET /stats — return total document and chunk counts. Admin-only, same
    reasoning as /documents above."""
    try:
        payloads = scroll_all_payloads(QDRANT_URL, QDRANT_COLLECTION)
    except Exception:
        return {"total_docs": 0, "total_chunks": 0}
    docs = _payloads_to_docs(payloads)
    return {"total_docs": len(docs), "total_chunks": len(payloads)}


@app.get("/usage", dependencies=[Depends(require_admin)])
async def usage():
    """GET /usage — per-question token/cost log and daily rollups. Admin-only."""
    from src.usage_log import stats as usage_stats

    return usage_stats()


class LLMCredentialsRequest(BaseModel):
    provider: str
    api_key: str | None = None
    model: str | None = None


@app.get("/admin/llm-credentials", dependencies=[Depends(require_admin)])
async def get_llm_credentials():
    """GET /admin/llm-credentials — the stored BYOK provider/model and whether
    a key is set, masked to its last 4 characters. Never returns the full key."""
    from src.llm_credentials import PROVIDERS, get_masked

    return {**get_masked(), "providers": list(PROVIDERS.keys())}


@app.post("/admin/llm-credentials", dependencies=[Depends(require_admin)])
async def set_llm_credentials(req: LLMCredentialsRequest):
    """POST /admin/llm-credentials — save the admin's chosen provider/key/model
    and rebuild the agent on the next request. A blank api_key keeps the
    existing stored key (see set_credentials's docstring)."""
    from src.llm_credentials import set_credentials

    try:
        set_credentials(req.provider, req.api_key, req.model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    _get_agent.cache_clear()
    return {"status": "ok"}


@app.delete("/admin/llm-credentials", dependencies=[Depends(require_admin)])
async def delete_llm_credentials():
    """DELETE /admin/llm-credentials — clear the stored override, reverting to
    env-configured generation, and rebuild the agent on the next request."""
    from src.llm_credentials import clear_credentials

    clear_credentials()
    _get_agent.cache_clear()
    return {"status": "ok"}


class FeedbackRequest(BaseModel):
    question: str
    answer: str
    rating: str
    reason: str | None = None
    sources: list[dict] = []


@app.post("/feedback")
async def submit_feedback(req: FeedbackRequest):
    """POST /feedback — record a thumbs up/down (with optional reason) on an answer."""
    from src.feedback_store import add_feedback

    return add_feedback(req.question, req.answer, req.rating, req.reason, req.sources)


@app.get("/feedback", dependencies=[Depends(require_admin)])
async def get_feedback():
    """GET /feedback — list all feedback records for the admin queue, newest first."""
    from src.feedback_store import list_feedback

    return list_feedback()


class FeedbackResolveRequest(BaseModel):
    action: str
    note: str | None = None


@app.patch("/feedback/{feedback_id}", dependencies=[Depends(require_admin)])
async def resolve_feedback_endpoint(feedback_id: str, req: FeedbackResolveRequest):
    """PATCH /feedback/{id} — mark a feedback record resolved with an admin action."""
    from src.feedback_store import resolve_feedback

    try:
        return resolve_feedback(feedback_id, req.action, req.note)
    except KeyError:
        raise HTTPException(status_code=404, detail="Feedback not found") from None


class ConversationSaveRequest(BaseModel):
    id: str | None = None
    messages: list[dict]


@app.post("/conversations")
async def save_conversation_endpoint(req: ConversationSaveRequest):
    """POST /conversations — create or update a saved conversation."""
    from src.conversation_store import save_conversation

    return save_conversation(req.id, req.messages)


@app.get("/conversations")
async def list_conversations_endpoint():
    """GET /conversations — list saved conversations (metadata only), newest first."""
    from src.conversation_store import list_conversations

    return list_conversations()


@app.get("/conversations/{conversation_id}")
async def get_conversation_endpoint(conversation_id: str):
    """GET /conversations/{id} — return one saved conversation with its full messages."""
    from src.conversation_store import get_conversation

    try:
        return get_conversation(conversation_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Conversation not found") from None


@app.delete("/conversations/{conversation_id}")
async def delete_conversation_endpoint(conversation_id: str):
    """DELETE /conversations/{id} — remove a saved conversation."""
    from src.conversation_store import delete_conversation

    try:
        delete_conversation(conversation_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Conversation not found") from None
    return {"status": "deleted"}


class QueryRequest(BaseModel):
    question: str
    doc_id: str | list[str] | None = None
    history: list[dict] | None = None


@app.post("/query")
async def query(
    req: QueryRequest,
    x_api_key: str | None = Header(default=None),
    vault_admin_session: str | None = Cookie(default=None),
):
    """POST /query — answer a question with the RAG agent.

    Routing, retries, and multi-part splitting all live in
    src/answer_pipeline.answer_query — shared with eval/run_eval.py so both
    the live app and the benchmark measure the exact same behavior.
    """
    from src.answer_pipeline import answer_query
    from src.guardrails import (
        REFUSAL_MESSAGE,
        check_corpus_enumeration,
        check_document_count_question,
        check_prompt_injection,
        check_system_prompt_leak,
    )
    from src.rag_agent import _get_langfuse

    is_admin = _is_admin_caller(x_api_key, vault_admin_session)

    if is_admin and check_document_count_question(req.question):
        return {
            "answer": _document_count_answer(),
            "sources": [],
            "rejected_sources": [],
            "sql": [],
            "tools_used": [],
        }

    if check_prompt_injection(req.question) or (
        not is_admin and check_corpus_enumeration(req.question)
    ):
        return {
            "answer": REFUSAL_MESSAGE,
            "sources": [],
            "rejected_sources": [],
            "sql": [],
            "tools_used": [],
        }

    # A history-carrying request's answer depends on prior turns, not just
    # this question's text -- caching it under the raw-question key would
    # serve a stale/wrong-context answer to a different conversation asking
    # the same follow-up text, so those requests skip the cache entirely.
    use_cache = QUERY_CACHE_ENABLED and not req.history
    key = query_cache.cache_key(req.question, req.doc_id)
    if use_cache and (cached := query_cache.get(key)) is not None:
        return cached

    agent = _get_agent()
    loop = asyncio.get_running_loop()
    lf = _get_langfuse()
    lf_trace = lf.trace(name="query", input=req.question) if lf else None

    _start = time.monotonic()
    result = await loop.run_in_executor(
        _executor,
        partial(
            answer_query,
            agent,
            req.question,
            trace=lf_trace,
            forced_doc_id=req.doc_id,
            history=req.history,
        ),
    )
    _latency_ms = (time.monotonic() - _start) * 1000

    from src.usage_log import log as log_usage

    log_usage(
        req.question,
        getattr(agent, "_generation_model", None),
        result.get("usage", {}),
        latency_ms=_latency_ms,
    )

    if lf_trace is not None:
        lf_trace.span(
            name="retrieval",
            input={"tools_used": result["tools"]},
            output={"sources": result["sources"], "sql": result["sql"]},
        )
        lf_trace.update(output=result["answer"])
        lf.flush()

    if check_system_prompt_leak(result["answer"]):
        return {
            "answer": REFUSAL_MESSAGE,
            "sources": [],
            "rejected_sources": [],
            "sql": [],
            "tools_used": [],
        }

    response = {
        "answer": result["answer"],
        "sources": result["sources"],
        "rejected_sources": result["rejected_sources"],
        "sql": result["sql"],
        "tools_used": _tools_used(result["tools"]),
    }
    if use_cache:
        query_cache.set(key, response)
    return response


@app.post("/query/stream")
async def query_stream(
    req: QueryRequest,
    x_api_key: str | None = Header(default=None),
    vault_admin_session: str | None = Cookie(default=None),
):
    """POST /query/stream — SSE version of /query, for perceived-latency UX.

    src.answer_pipeline.stream_answer is a plain sync generator (it drives
    LangGraph's own sync stream_agent) — bridged to an async SSE response by
    running it in the executor thread and relaying its items through an
    asyncio.Queue, since a sync generator can't be iterated directly inside
    an async endpoint. Each `data: ` line is one JSON event: {"token": str}
    while generating, then one final {"done": true, "answer", "sources",
    "sql", "tools_used"} — see stream_answer's docstring for which question
    types actually stream live vs. arrive as a single lump event.

    Known limitation: if the client disconnects (stop generation, or the tab
    closes) mid-stream, the executor thread keeps running stream_answer to
    completion — a sync generator running in a worker thread can't be
    cancelled from the async side without extra plumbing, so its remaining
    output just piles up in the now-unread queue until the request context
    is garbage collected. Wasted compute on an aborted answer, not a
    correctness bug (nothing is read from the queue after disconnect).
    """
    from src.answer_pipeline import stream_answer
    from src.guardrails import (
        REFUSAL_MESSAGE,
        check_corpus_enumeration,
        check_document_count_question,
        check_prompt_injection,
        check_system_prompt_leak,
    )

    is_admin = _is_admin_caller(x_api_key, vault_admin_session)

    if is_admin and check_document_count_question(req.question):

        async def count_stream():
            event = {
                "done": True,
                "answer": _document_count_answer(),
                "sources": [],
                "rejected_sources": [],
                "sql": [],
                "tools_used": [],
            }
            yield f"data: {json.dumps(event)}\n\n"

        return StreamingResponse(count_stream(), media_type="text/event-stream")

    if check_prompt_injection(req.question) or (
        not is_admin and check_corpus_enumeration(req.question)
    ):

        async def refused_stream():
            event = {
                "done": True,
                "answer": REFUSAL_MESSAGE,
                "sources": [],
                "rejected_sources": [],
                "sql": [],
                "tools_used": [],
            }
            yield f"data: {json.dumps(event)}\n\n"

        return StreamingResponse(refused_stream(), media_type="text/event-stream")

    use_cache = QUERY_CACHE_ENABLED and not req.history
    key = query_cache.cache_key(req.question, req.doc_id)
    cached = query_cache.get(key) if use_cache else None
    if cached is not None:

        async def cached_stream():
            yield f"data: {json.dumps({'done': True, **cached})}\n\n"

        return StreamingResponse(cached_stream(), media_type="text/event-stream")

    agent = _get_agent()
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    _SENTINEL = object()

    def produce() -> None:
        # Cache the write here, not in event_stream() -- a client disconnect
        # (tab close, abort, or the frontend remounting mid-request) cancels
        # event_stream()'s queue.get() before it ever sees the done event,
        # while this executor thread keeps running to completion regardless
        # (see the disconnect limitation in this endpoint's docstring). Caching
        # only from event_stream() meant an abandoned or aborted question
        # computed its full expensive answer and then threw it away, so the
        # next identical ask always missed the cache too.
        _start = time.monotonic()
        try:
            for event in stream_answer(agent, req.question, req.doc_id, req.history):
                if event.get("done") and "tools" in event:
                    event["tools_used"] = _tools_used(event.pop("tools"))
                if event.get("done"):
                    from src.usage_log import log as log_usage

                    log_usage(
                        req.question,
                        getattr(agent, "_generation_model", None),
                        event.get("usage") or {},
                        latency_ms=(time.monotonic() - _start) * 1000,
                    )
                if event.get("done") and check_system_prompt_leak(
                    event.get("answer") or ""
                ):
                    event = {
                        "done": True,
                        "answer": REFUSAL_MESSAGE,
                        "sources": [],
                        "rejected_sources": [],
                        "sql": [],
                        "tools_used": [],
                    }
                if event.get("done") and use_cache and not event.get("error"):
                    query_cache.set(
                        key,
                        {
                            "answer": event.get("answer"),
                            "sources": event.get("sources", []),
                            "rejected_sources": event.get("rejected_sources", []),
                            "sql": event.get("sql", []),
                            "tools_used": event.get("tools_used", []),
                        },
                    )
                loop.call_soon_threadsafe(queue.put_nowait, event)
        except Exception as exc:  # noqa: BLE001 - report to the client, don't crash the thread
            loop.call_soon_threadsafe(
                queue.put_nowait, {"done": True, "error": str(exc)}
            )
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, _SENTINEL)

    async def event_stream():
        loop.run_in_executor(_executor, produce)
        while True:
            item = await queue.get()
            if item is _SENTINEL:
                break
            yield f"data: {json.dumps(item)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# Real tool names → frontend pill keys (see TOOL_META in TraceSidebar.tsx).
_TOOL_DISPLAY = {
    "search_knowledge_base": "search_documents",
    "query_excel": "query_excel",
}


def _tools_used(tool_calls: list[str]) -> list[str]:
    """Map the agent's actual tool invocations to display keys, deduped in call order."""
    seen: list[str] = []
    for name in tool_calls:
        key = _TOOL_DISPLAY.get(name, name)
        if key not in seen:
            seen.append(key)
    return seen


@app.delete("/collection", dependencies=[Depends(require_admin)])
async def clear_collection():
    """DELETE /collection — drop the Qdrant collection and reset the agent."""
    base = QDRANT_URL.rstrip("/")
    _qdrant("DELETE", f"{base}/collections/{QDRANT_COLLECTION}")
    _get_agent.cache_clear()
    query_cache.clear()
    return {"status": "cleared"}


@app.delete("/documents/{filename:path}", dependencies=[Depends(require_admin)])
async def delete_document(filename: str):
    """Remove all Qdrant points for a single file."""
    try:
        deleted = delete_by_file(QDRANT_URL, QDRANT_COLLECTION, filename)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    _get_agent.cache_clear()
    query_cache.clear()
    return {"status": "deleted", "filename": filename, "points_deleted": deleted}


@app.post("/documents/{filename:path}/reindex", dependencies=[Depends(require_admin)])
async def reindex_document(filename: str, pipeline: str = Form("auto")):
    """Re-run ingestion on a file already stored in data/input — idempotent point
    IDs overwrite its existing Qdrant points in place rather than duplicating them."""
    dest = INPUT_DIR / filename
    if not dest.exists():
        raise HTTPException(
            status_code=404, detail=f"Original file not found: {filename}"
        )
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "pending", "stage": "parsing", "chunks_created": 0}
    force_pipeline = pipeline if pipeline in {"ocr", "text"} else None
    loop = asyncio.get_running_loop()
    loop.run_in_executor(_executor, _run_ingest_sync, job_id, dest, force_pipeline)
    return {"job_id": job_id, "status": "processing"}


# ── Google Drive connector endpoints ────────────────────────────────────────────


class DriveConfigureRequest(BaseModel):
    folder_id: str
    service_account_file: str | None = None


class DriveSyncRequest(BaseModel):
    remove_deleted: bool = False


@app.post("/connectors/google-drive/configure", dependencies=[Depends(require_admin)])
async def configure_google_drive(req: DriveConfigureRequest):
    """POST /connectors/google-drive/configure — set which Drive folder to sync from.

    Authenticates via a service-account key file (see GOOGLE_DRIVE_SERVICE_ACCOUNT_FILE
    in .env) -- share the target folder with that service account's email, no
    interactive login required.
    """
    from src.connectors.google_drive import configure

    return configure(req.folder_id, req.service_account_file)


@app.post("/connectors/google-drive/sync", dependencies=[Depends(require_admin)])
async def sync_google_drive(req: DriveSyncRequest):
    """POST /connectors/google-drive/sync — pull new/changed files and ingest them.

    Runs synchronously (not backgrounded like /ingest) since a folder sync is
    typically small and bounded; each file's own ingestion failure is captured
    per-file in the response rather than aborting the whole sync.
    """
    from src.connectors.google_drive import sync

    try:
        result = sync(remove_deleted=req.remove_deleted)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    _get_agent.cache_clear()
    query_cache.clear()
    return result


@app.get("/connectors/google-drive/status")
async def google_drive_status():
    """GET /connectors/google-drive/status — configured folder + last sync summary."""
    from src.connectors.google_drive import status

    return status()


@app.get("/connectors/google-drive/files")
async def google_drive_files():
    """GET /connectors/google-drive/files — every Drive file currently tracked."""
    from src.connectors.google_drive import list_files

    return list_files()


# ── inspector endpoints ────────────────────────────────────────────────────────


@app.get("/documents/{filename:path}/chunks")
async def document_chunks(filename: str):
    """All Qdrant chunks for a file, grouped and sorted for the inspector."""
    try:
        # get_chunks_by_file is a blocking network call -- run it off the
        # event loop so a slow Qdrant scroll doesn't stall every other
        # request (including the SSE /query/stream) for its duration.
        payloads = await run_in_threadpool(
            get_chunks_by_file, QDRANT_URL, QDRANT_COLLECTION, filename
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    if not payloads:
        return {"summary": None, "chunks": []}

    summary_payload = next(
        (
            p
            for p in payloads
            if (p.get("metadata") or {}).get("chunk_type") == "document_summary"
        ),
        None,
    )
    data_chunks = [
        {
            "content": p.get("content", ""),
            "metadata": p.get("metadata", {}),
        }
        for p in payloads
        if (p.get("metadata") or {}).get("chunk_type") != "document_summary"
    ]
    data_chunks.sort(
        key=lambda c: c["metadata"].get("chunk_index", c["metadata"].get("part", 0))
    )

    return {
        "summary": summary_payload.get("content") if summary_payload else None,
        "chunks": data_chunks,
    }


@app.get("/documents/{filename:path}/markdown")
async def document_markdown(filename: str):
    """Return parsed markdown split into pages (for PDF inspector)."""
    md_path = _resolve_markdown_path(filename)
    if not md_path:
        raise HTTPException(status_code=404, detail="Markdown not found")
    raw = md_path.read_text(encoding="utf-8")
    pages, pipelines = _split_markdown_pages(raw)
    has_markers = bool(pages)
    return {
        "has_page_markers": has_markers,
        "full_text": raw if not has_markers else None,
        "pages": [
            {"page": n, "content": pages[n], "pipeline": pipelines.get(n, "")}
            for n in sorted(pages)
        ],
    }


@app.get("/documents/{filename:path}/pdf/info")
async def document_pdf_info(filename: str):
    """Return total page count for a PDF.

    Registered ahead of /pdf/{page} deliberately -- Starlette matches routes
    by trying them in registration order against the untyped path template
    (page's "int" typing lives in the function signature, not the URL
    pattern), so a /pdf/{page} declared first would greedily match
    "/pdf/info" too (page="info") and 422 on int coercion instead of ever
    reaching this route. Reproduced live 2026-07-25: this endpoint was
    completely unreachable with the routes in the other order.
    """
    pdf_path = _resolve_source_file_path(filename)
    if not pdf_path:
        raise HTTPException(status_code=404, detail="PDF not found")
    try:
        import pypdfium2 as pdfium

        doc = pdfium.PdfDocument(str(pdf_path))
        return {"total_pages": len(doc)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/documents/{filename:path}/pdf/{page}")
async def document_pdf_page(filename: str, page: int):
    """Render a single PDF page (1-indexed) and return it as a base64 PNG."""
    pdf_path = _resolve_source_file_path(filename)
    if not pdf_path:
        raise HTTPException(status_code=404, detail="PDF not found")
    try:
        import pypdfium2 as pdfium

        doc = pdfium.PdfDocument(str(pdf_path))
        n = len(doc)
        if page < 1 or page > n:
            raise HTTPException(
                status_code=404, detail=f"Page {page} out of range (1–{n})"
            )
        bitmap = doc[page - 1].render(scale=1.5)
        img = bitmap.to_pil()
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return {"image_b64": b64, "page": page, "total_pages": n}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/documents/{filename:path}/pdf/{page}/highlight")
async def document_pdf_highlight(filename: str, page: int, quote: str):
    """Locate a cited passage on a born-digital PDF page and return its bbox.

    Uses fitz's exact text search against the PDF's real text layer — no
    ingestion-time storage, no external service. A long excerpt often fails
    fitz's exact match as one string (line-wrap/whitespace differences), so
    falls back to matching sentence-by-sentence and unioning every sentence
    found — covering the actual passage instead of an arbitrary word-count
    cutoff. Returns bbox=None (never an invented region, and never a partial
    fragment that stops short of the cited content) when nothing matches.
    """
    pdf_path = _resolve_source_file_path(filename)
    if not pdf_path:
        raise HTTPException(status_code=404, detail="PDF not found")
    try:
        import fitz

        doc = fitz.open(str(pdf_path))
        if page < 1 or page > len(doc):
            raise HTTPException(
                status_code=404, detail=f"Page {page} out of range (1–{len(doc)})"
            )
        fitz_page = doc[page - 1]
        # A mid-chunk excerpt is prefixed/suffixed with a literal "…" to mark
        # truncation (see retrieval_tool.py's _best_snippet) -- that marker
        # isn't in the PDF's real text, so strip it before an exact search.
        search_text = quote.strip().strip("…").strip()
        rects = fitz_page.search_for(search_text)
        if not rects:
            sentences = [
                s.strip()
                for s in re.split(r"(?<=[.:!?])\s+", search_text)
                if len(s.strip()) > 15
            ]
            rects = [r for s in sentences for r in fitz_page.search_for(s)]
        # A multi-line match returns one rect per line it spans — union them into
        # one box covering the whole passage, not just its first line.
        bbox = (
            [
                min(r.x0 for r in rects),
                min(r.y0 for r in rects),
                max(r.x1 for r in rects),
                max(r.y1 for r in rects),
            ]
            if rects
            else None
        )
        return {"bbox": bbox, "coordinate_system": "pdf_points" if bbox else None}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/documents/{filename:path}/pdf/{page}/crop")
async def document_pdf_crop(filename: str, page: int, bbox: str):
    """Crop a region of a born-digital PDF page and return it as a base64 PNG.

    Used by the evidence panel to show the actual source figure/chart instead
    of only its VLM-generated text description. `bbox` is "x0,y0,x1,y1" in PDF
    points, as stored on the chunk at ingestion time.
    """
    pdf_path = _resolve_source_file_path(filename)
    if not pdf_path:
        raise HTTPException(status_code=404, detail="PDF not found")
    try:
        coords = [float(v) for v in bbox.split(",")]
        if len(coords) != 4:
            raise ValueError("bbox must have 4 comma-separated values")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid bbox: {exc}")
    try:
        import fitz

        doc = fitz.open(str(pdf_path))
        if page < 1 or page > len(doc):
            raise HTTPException(
                status_code=404, detail=f"Page {page} out of range (1–{len(doc)})"
            )
        fitz_page = doc[page - 1]
        pix = fitz_page.get_pixmap(dpi=150, clip=fitz.Rect(*coords))
        b64 = base64.b64encode(pix.tobytes("png")).decode()
        return {"image_b64": b64}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


_TABLE_MD_MAX_ROWS = 60


def _truncate_markdown_table(md: str, max_rows: int = _TABLE_MD_MAX_ROWS) -> str:
    """Cap a sheet's full markdown table to its first `max_rows` data rows.

    The stored .md file holds the whole sheet (thousands of rows for a real
    spreadsheet) -- rendering that through ReactMarkdown+remarkGfm client-side
    freezes the tab (found live: a 6701-row sheet produced a 920KB cleaned_md
    that never finished rendering, no network error possible since nothing
    ever hangs on the wire). Bounded to match raw_rows' own nrows=60 cap.
    """
    lines = md.split("\n")
    table_start = next(
        (i for i, ln in enumerate(lines) if ln.lstrip().startswith("|")), None
    )
    if table_start is None:
        return md
    header_lines = lines[:table_start]
    # Only the contiguous run of pipe-prefixed lines is "the table" -- content
    # appended after it (e.g. extracted notes) must survive truncation intact,
    # not get silently cut off mid-line once the table's own row count grows.
    table_end = table_start
    while table_end < len(lines) and lines[table_end].lstrip().startswith("|"):
        table_end += 1
    table_lines = lines[table_start:table_end]
    trailing_lines = lines[table_end:]
    kept = table_lines[: 2 + max_rows]  # header row + separator row + data rows
    omitted = len(table_lines) - len(kept)
    if omitted > 0:
        kept.append(f"\n_{omitted} more rows omitted — showing first {max_rows}._")
    return "\n".join(header_lines + kept + trailing_lines)


def _normalize_cell(value: object) -> str:
    """Mirror frontend/lib/tableMatch.ts's normalizeCell exactly, so the
    server-side row search agrees with the client-side one it replaces the
    input for."""
    return str(value).strip().lower()


def _find_header_row_index(rows: list[list[str]]) -> int:
    """Mirror frontend/lib/tableMatch.ts's findHeaderRowIndex: the index of
    the first row with more than one non-empty cell -- some sheets carry a
    report-title/notice preamble (single populated cell) above the real
    header row, so "row 0 is the header" isn't always true. See that
    function's docstring for the live repro (doc_006's DataAnalysis sheet)."""
    for i, row in enumerate(rows):
        if sum(1 for cell in row if str(cell).strip()) > 1:
            return i
    return 0


def _find_matched_row_index(rows: list[list[str]], quote: str | None) -> int:
    """Mirror frontend/lib/tableMatch.ts's findMatchedRowIndex: the index of
    the row with the MOST cells (len > 2) that appear verbatim in `quote`,
    or -1 if none match / quote is empty.

    Ties broken by earliest row index. Scoring by match COUNT rather than
    stopping at the first single-cell match matters once the search spans a
    whole sheet instead of only 60 rows (see _window_rows_around_match):
    reproduced live -- a category value like "PLAY EQUIPMENT TEAM" repeats
    across many unrelated rows, so "first row with any one matching cell"
    picked a real but WRONG transaction. The quote also contains the row's
    other distinguishing values (e.g. the exact total "317.50"), which only
    the true row matches on top of the category -- counting cells picks it.
    """
    quote_norm = _normalize_cell(quote or "")
    if not quote_norm:
        return -1
    best_idx = -1
    best_count = 0
    for i, row in enumerate(rows):
        count = sum(
            1
            for cell in row
            if len(cell_norm := _normalize_cell(cell)) > 2 and cell_norm in quote_norm
        )
        if count > best_count:
            best_count = count
            best_idx = i
    return best_idx


def _window_rows_around_match(
    header: list[list[str]],
    body: list[list[str]],
    quote: str | None,
    max_rows: int,
) -> list[list[str]] | None:
    """Return header + a max_rows-sized window of body centered on the row
    matching `quote`, or None if no match was found anywhere in `body` (the
    caller falls back to the existing "first max_rows" behavior in that case).
    """
    matched_idx = _find_matched_row_index(body, quote)
    if matched_idx < 0:
        return None
    half = max_rows // 2
    start = max(0, matched_idx - half)
    end = min(len(body), start + max_rows)
    return header + body[start:end]


@app.get("/documents/{filename:path}/table-sheet/{sheet}")
async def document_table_sheet(filename: str, sheet: str, quote: str | None = None):
    """Return raw rows and cleaned markdown for one sheet of an Excel/CSV file.

    quote: a citation's quoted evidence text. When given, the row it was
    drawn from is searched for across the WHOLE sheet (not just the first 60
    rows the response caps at) and a window of rows around it is returned
    instead of always "the first 60" -- reproduced live: a citation whose
    real row fell past row 60 in a larger CSV (doc_007, hundreds of rows)
    could never be found or highlighted, no matter how good the client-side
    matching was, since it was never in the data sent to the browser at all.
    Without a quote (or no match found), behavior is unchanged: first 60 rows.
    """
    TABLE_MD_DIR = REPO_ROOT / "data" / "output" / "table_markdowns"
    file_stem = Path(filename).stem
    safe_sheet = sheet.replace("/", "_").replace("\\", "_")
    md_path = TABLE_MD_DIR / f"{file_stem}__{safe_sheet}.md"
    raw_path = _resolve_source_file_path(filename) or INPUT_DIR / filename
    suffix = Path(filename).suffix.lower()

    cleaned_md: str | None = (
        _truncate_markdown_table(md_path.read_text(encoding="utf-8"))
        if md_path.exists()
        else None
    )

    def _read_raw_rows() -> list[list[str]] | None:
        try:
            import pandas as pd

            if quote:
                # Search the full sheet for the citation's real row -- reading
                # the whole file server-side is fine (this is what earlier
                # froze: rendering thousands of rows through ReactMarkdown
                # client-side, not pandas parsing them here); only the
                # windowed slice actually sent to the browser stays bounded.
                if suffix == ".csv":
                    # encoding_errors="replace": some source CSVs have a stray
                    # non-UTF8 byte (e.g. a Windows-1252 non-breaking space)
                    # that the old nrows=60 cap never reached -- reading the
                    # whole file to search it now does, and a hard decode
                    # error there must not take down row search entirely.
                    full_df = pd.read_csv(
                        raw_path, header=None, dtype=str, encoding_errors="replace"
                    ).fillna("")
                else:
                    full_df = pd.read_excel(
                        raw_path, sheet_name=sheet, header=None, dtype=str
                    ).fillna("")
                full_rows = full_df.values.tolist()
                header_idx = _find_header_row_index(full_rows)
                header, body = full_rows[: header_idx + 1], full_rows[header_idx + 1 :]
                windowed = _window_rows_around_match(
                    header, body, quote, _TABLE_MD_MAX_ROWS
                )
                if windowed is not None:
                    return windowed
                # No match anywhere in the sheet -- fall through to the
                # existing "first 60" behavior below, same as no quote given.

            if suffix == ".csv":
                df = pd.read_csv(
                    raw_path,
                    header=None,
                    dtype=str,
                    nrows=_TABLE_MD_MAX_ROWS,
                    encoding_errors="replace",
                ).fillna("")
            else:
                df = pd.read_excel(
                    raw_path,
                    sheet_name=sheet,
                    header=None,
                    dtype=str,
                    nrows=_TABLE_MD_MAX_ROWS,
                ).fillna("")
            return df.values.tolist()
        except Exception:
            return None

    raw_rows: list[list[str]] | None = None
    if raw_path.exists():
        # pandas/openpyxl parses the whole workbook before `nrows` truncates
        # it -- a large .xlsx can block for real time, so run it off the
        # event loop instead of stalling every other request.
        raw_rows = await run_in_threadpool(_read_raw_rows)

    if cleaned_md is None and raw_rows is None:
        raise HTTPException(status_code=404, detail="No data found for this sheet")

    return {"sheet": sheet, "raw_rows": raw_rows, "cleaned_md": cleaned_md}
