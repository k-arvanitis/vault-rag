"""JSON-file-backed store for admin-set display titles on a source.

Overrides what the Sources UI shows for a document (extracted display_title
or the filename fallback) without touching the document's own content,
chunks, or embeddings -- purely a presentation-layer rename.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from src.config import TITLE_OVERRIDES_PATH

# Guards the load-modify-save sequence -- same rationale as feedback_store.py's
# lock: FastAPI's threadpool executor can run handlers concurrently, and
# without this a second write can silently discard the first's change.
_lock = threading.Lock()


def _path() -> Path:
    """Return the title-overrides store path, resolved relative to the repo root."""
    return Path(TITLE_OVERRIDES_PATH)


def _load() -> dict[str, str]:
    """Return {filename: custom_title} for every document with a manual override."""
    path = _path()
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _save(overrides: dict[str, str]) -> None:
    """Write overrides back to disk, creating the parent directory if needed."""
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(overrides, indent=2, ensure_ascii=False))


def get_overrides() -> dict[str, str]:
    """Return all current filename -> custom_title overrides."""
    return _load()


def set_title(filename: str, title: str) -> None:
    """Set (or replace) a document's custom display title."""
    with _lock:
        overrides = _load()
        overrides[filename] = title
        _save(overrides)


def clear_title(filename: str) -> None:
    """Remove a document's custom title, reverting to the extracted/fallback name.

    A no-op (not an error) if the document had no override -- resetting an
    already-default title should be idempotent, not a 404 for the caller.
    """
    with _lock:
        overrides = _load()
        if filename in overrides:
            del overrides[filename]
            _save(overrides)
