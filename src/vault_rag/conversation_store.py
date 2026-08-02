"""JSON-file-backed store for saved chat conversations (history sidebar)."""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from vault_rag.config import CONVERSATION_PATH

# Same lost-update race as src/feedback_store.py -- guards the load-modify-save
# sequence against FastAPI's threadpool executor running handlers concurrently.
_lock = threading.Lock()


def _path() -> Path:
    """Return the conversation store path, resolved relative to the repo root."""
    return Path(CONVERSATION_PATH)


def _load() -> list[dict]:
    """Return all stored conversations, or an empty list if none exist yet."""
    path = _path()
    if not path.exists():
        return []
    return json.loads(path.read_text())


def _save(items: list[dict]) -> None:
    """Write conversations back to disk, creating the parent directory if needed."""
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(items, indent=2))


def _title_from_messages(messages: list[dict]) -> str:
    """Derive a short title from the first user message, for the history list."""
    for m in messages:
        if m.get("role") == "user" and m.get("content"):
            text = m["content"].strip()
            return text[:60] + ("…" if len(text) > 60 else "")
    return "Untitled conversation"


def save_conversation(conversation_id: str | None, messages: list[dict]) -> dict:
    """Create or update a conversation record; returns the stored record."""
    with _lock:
        items = _load()
        now = datetime.now(timezone.utc).isoformat()
        if conversation_id:
            for item in items:
                if item["id"] == conversation_id:
                    item["messages"] = messages
                    item["title"] = _title_from_messages(messages)
                    item["updated_at"] = now
                    _save(items)
                    return item
        item = {
            "id": conversation_id or str(uuid.uuid4()),
            "title": _title_from_messages(messages),
            "messages": messages,
            "created_at": now,
            "updated_at": now,
        }
        items.append(item)
        _save(items)
        return item


def list_conversations() -> list[dict]:
    """Return conversation metadata (no message bodies), newest first."""
    items = sorted(_load(), key=lambda i: i["updated_at"], reverse=True)
    return [
        {
            "id": i["id"],
            "title": i["title"],
            "message_count": len(i["messages"]),
            "created_at": i["created_at"],
            "updated_at": i["updated_at"],
        }
        for i in items
    ]


def get_conversation(conversation_id: str) -> dict:
    """Return one full conversation record, including messages; raises KeyError if not found."""
    for item in _load():
        if item["id"] == conversation_id:
            return item
    raise KeyError(conversation_id)


def delete_conversation(conversation_id: str) -> None:
    """Remove a conversation record; raises KeyError if not found."""
    with _lock:
        items = _load()
        remaining = [i for i in items if i["id"] != conversation_id]
        if len(remaining) == len(items):
            raise KeyError(conversation_id)
        _save(remaining)
