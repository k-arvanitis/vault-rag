"""Vault RAG — FastAPI backend for the Next.js UI."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv(override=True)

from fastapi import (  # noqa: E402
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.concurrency import run_in_threadpool  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse, StreamingResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from src.config import (  # noqa: E402
    API_CORS_ORIGINS,
    API_KEY,
    GENERATION_API_BASE,
    GENERATION_MODEL,
    QDRANT_COLLECTION,
    QDRANT_URL,
    RERANK_TOP_N,
    RERANKER_MODEL,
    RETRIEVAL_TOP_K,
)
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the agent once at startup so the first /query doesn't pay the cold-start penalty."""
    try:
        _get_agent()
        logger.info("Agent warmed; ready to serve")
    except Exception:
        logger.exception("Agent warmup failed; /query will retry per-request")
    yield


app = FastAPI(title="Vault RAG API", lifespan=lifespan)


# ── CORS — explicit origin list (use API_CORS_ORIGINS=* only intentionally) ───

_cors_origins = [o.strip() for o in API_CORS_ORIGINS.split(",") if o.strip()] or ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── auth dep — required on mutating endpoints when API_KEY is set ─────────────


async def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Raise 401 unless the X-API-Key header matches API_KEY (when configured)."""
    if not API_KEY:
        return
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


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
    from src.rag_agent import build_rag_agent

    return build_rag_agent(
        qdrant_url=QDRANT_URL,
        collection=QDRANT_COLLECTION,
        retrieval_top_k=RETRIEVAL_TOP_K,
        rerank_top_n=RERANK_TOP_N,
        reranker_model_name=RERANKER_MODEL or None,
        model_name=GENERATION_MODEL,
        generation_api_base=GENERATION_API_BASE,
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
    except Exception as exc:
        _jobs[job_id].update({"status": "failed", "stage": "failed", "error": str(exc)})


def _payloads_to_docs(payloads: list[dict]) -> list[dict]:
    """Group Qdrant payloads into one document card per source file."""
    from collections import defaultdict

    counts: dict[str, int] = defaultdict(int)
    last_indexed: dict[str, str] = {}
    for p in payloads:
        meta = p.get("metadata", {}) or {}
        name = meta.get("source_file") or meta.get("file_name") or ""
        if name:
            counts[name] += 1
            ts = meta.get("ingested_at") or ""
            if ts > last_indexed.get(name, ""):
                last_indexed[name] = ts
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
            "file_type": type_map.get(Path(name).suffix.lstrip(".").lower(), "File"),
            "chunk_count": count,
            "status": "indexed",
            "last_indexed_at": last_indexed.get(name) or None,
        }
        for name, count in sorted(counts.items())
    ]


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


@app.post("/ingest", dependencies=[Depends(require_api_key)])
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
    }


@app.get("/documents")
async def list_documents():
    """GET /documents — list ingested documents as UI cards."""
    try:
        payloads = scroll_all_payloads(QDRANT_URL, QDRANT_COLLECTION)
    except Exception:
        return []
    return _payloads_to_docs(payloads)


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


@app.post("/eval/run", dependencies=[Depends(require_api_key)])
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


@app.get("/stats")
async def stats():
    """GET /stats — return total document and chunk counts."""
    try:
        payloads = scroll_all_payloads(QDRANT_URL, QDRANT_COLLECTION)
    except Exception:
        return {"total_docs": 0, "total_chunks": 0}
    docs = _payloads_to_docs(payloads)
    return {"total_docs": len(docs), "total_chunks": len(payloads)}


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


@app.get("/feedback", dependencies=[Depends(require_api_key)])
async def get_feedback():
    """GET /feedback — list all feedback records for the admin queue, newest first."""
    from src.feedback_store import list_feedback

    return list_feedback()


class FeedbackResolveRequest(BaseModel):
    action: str
    note: str | None = None


@app.patch("/feedback/{feedback_id}", dependencies=[Depends(require_api_key)])
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


@app.post("/query")
async def query(req: QueryRequest):
    """POST /query — answer a question with the RAG agent.

    Routing, retries, and multi-part splitting all live in
    src/answer_pipeline.answer_query — shared with eval/run_eval.py so both
    the live app and the benchmark measure the exact same behavior.
    """
    from src.answer_pipeline import answer_query
    from src.rag_agent import _get_langfuse

    agent = _get_agent()
    loop = asyncio.get_running_loop()
    lf = _get_langfuse()
    lf_trace = lf.trace(name="query", input=req.question) if lf else None

    result = await loop.run_in_executor(
        _executor, answer_query, agent, req.question, lf_trace, req.doc_id
    )

    if lf_trace is not None:
        lf_trace.span(
            name="retrieval",
            input={"tools_used": result["tools"]},
            output={"sources": result["sources"], "sql": result["sql"]},
        )
        lf_trace.update(output=result["answer"])
        lf.flush()

    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "rejected_sources": result["rejected_sources"],
        "sql": result["sql"],
        "tools_used": _tools_used(result["tools"]),
    }


@app.post("/query/stream")
async def query_stream(req: QueryRequest):
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

    agent = _get_agent()
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    _SENTINEL = object()

    def produce() -> None:
        try:
            for event in stream_answer(agent, req.question, req.doc_id):
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
            if item.get("done") and "tools" in item:
                item["tools_used"] = _tools_used(item.pop("tools"))
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


@app.delete("/collection", dependencies=[Depends(require_api_key)])
async def clear_collection():
    """DELETE /collection — drop the Qdrant collection and reset the agent."""
    base = QDRANT_URL.rstrip("/")
    _qdrant("DELETE", f"{base}/collections/{QDRANT_COLLECTION}")
    _get_agent.cache_clear()
    return {"status": "cleared"}


@app.delete("/documents/{filename:path}", dependencies=[Depends(require_api_key)])
async def delete_document(filename: str):
    """Remove all Qdrant points for a single file."""
    try:
        deleted = delete_by_file(QDRANT_URL, QDRANT_COLLECTION, filename)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    _get_agent.cache_clear()
    return {"status": "deleted", "filename": filename, "points_deleted": deleted}


@app.post("/documents/{filename:path}/reindex", dependencies=[Depends(require_api_key)])
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


@app.post("/connectors/google-drive/configure", dependencies=[Depends(require_api_key)])
async def configure_google_drive(req: DriveConfigureRequest):
    """POST /connectors/google-drive/configure — set which Drive folder to sync from.

    Authenticates via a service-account key file (see GOOGLE_DRIVE_SERVICE_ACCOUNT_FILE
    in .env) -- share the target folder with that service account's email, no
    interactive login required.
    """
    from src.connectors.google_drive import configure

    return configure(req.folder_id, req.service_account_file)


@app.post("/connectors/google-drive/sync", dependencies=[Depends(require_api_key)])
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
    ingestion-time storage, no external service. Falls back to a shorter
    prefix of the quote since markdown reformatting (tables, headings) can
    break an exact match on the full excerpt. Returns bbox=None (never an
    invented region) when nothing matches.
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
        rects = fitz_page.search_for(quote)
        if not rects:
            prefix = " ".join(quote.split()[:12])
            rects = fitz_page.search_for(prefix) if prefix else []
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
    table_start = next((i for i, ln in enumerate(lines) if ln.lstrip().startswith("|")), None)
    if table_start is None:
        return md
    header_lines = lines[:table_start]
    table_lines = lines[table_start:]
    kept = table_lines[: 2 + max_rows]  # header row + separator row + data rows
    omitted = len(table_lines) - len(kept)
    if omitted > 0:
        kept.append(f"\n_{omitted} more rows omitted — showing first {max_rows}._")
    return "\n".join(header_lines + kept)


@app.get("/documents/{filename:path}/table-sheet/{sheet}")
async def document_table_sheet(filename: str, sheet: str):
    """Return raw rows and cleaned markdown for one sheet of an Excel/CSV file."""
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

            if suffix == ".csv":
                df = pd.read_csv(raw_path, header=None, dtype=str, nrows=60).fillna("")
            else:
                df = pd.read_excel(
                    raw_path, sheet_name=sheet, header=None, dtype=str, nrows=60
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


@app.get("/documents/{filename:path}/pdf/info")
async def document_pdf_info(filename: str):
    """Return total page count for a PDF."""
    pdf_path = _resolve_source_file_path(filename)
    if not pdf_path:
        raise HTTPException(status_code=404, detail="PDF not found")
    try:
        import pypdfium2 as pdfium

        doc = pdfium.PdfDocument(str(pdf_path))
        return {"total_pages": len(doc)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
