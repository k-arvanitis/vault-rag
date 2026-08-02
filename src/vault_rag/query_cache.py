"""JSON-file-backed cache for /query and /query/stream answers.

Keyed on (question, doc_id) so re-asking the exact same question against the
same source scope skips retrieval + generation entirely. Cleared whenever the
corpus changes (ingest, reindex, delete, collection wipe, Google Drive sync)
so a cached answer can never outlive the documents it was computed from.
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path

from vault_rag.config import QUERY_CACHE_PATH

# Same lost-update race as src/conversation_store.py / src/feedback_store.py --
# guards the load-modify-save sequence against FastAPI's threadpool executor
# running handlers concurrently.
_lock = threading.Lock()


def _path() -> Path:
    return Path(QUERY_CACHE_PATH)


def _load() -> dict[str, dict]:
    path = _path()
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _save(items: dict[str, dict]) -> None:
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(items, indent=2))


def cache_key(question: str, doc_id: str | list[str] | None) -> str:
    """Build a stable key from the question text and source scope."""
    doc_key = doc_id if isinstance(doc_id, list) else [doc_id] if doc_id else []
    normalized = json.dumps([question.strip().lower(), sorted(doc_key)])
    return hashlib.sha256(normalized.encode()).hexdigest()


def get(key: str) -> dict | None:
    """Return the cached result dict for `key`, or None on a miss."""
    with _lock:
        return _load().get(key)


def set(key: str, result: dict) -> None:  # noqa: A001 - matches get/clear's dict.get-style naming
    """Store `result` under `key`, overwriting any prior entry."""
    with _lock:
        items = _load()
        items[key] = result
        _save(items)


def clear() -> None:
    """Drop every cached answer -- call after any corpus-changing operation."""
    with _lock:
        _save({})
