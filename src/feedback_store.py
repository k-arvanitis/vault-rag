"""JSON-file-backed store for user feedback on answers (thumbs up/down, admin queue)."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from src.config import EVAL_REGRESSION_CANDIDATES_PATH, FEEDBACK_PATH


def _path() -> Path:
    """Return the feedback store path, resolved relative to the repo root."""
    return Path(FEEDBACK_PATH)


def _regression_candidates_path() -> Path:
    """Return the eval regression-candidates path, resolved relative to the repo root."""
    return Path(EVAL_REGRESSION_CANDIDATES_PATH)


def _append_regression_candidate(item: dict) -> None:
    """Append a flagged answer to the eval regression-candidates file.

    Not auto-graded: the flagged predicted_answer isn't a verified gold
    answer, so gold_answer/question_type are left for a human to fill in
    before the question counts in a real eval run.
    """
    path = _regression_candidates_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    candidate = {
        "qa_id": f"feedback_{item['id']}",
        "question": item["question"],
        "predicted_answer": item["answer"],
        "sources": item["sources"],
        "reason": item["reason"],
        "gold_answer": None,  # TODO: fill in before including in an eval run
        "question_type": None,  # TODO: classify
        "flagged_at": item["created_at"],
    }
    with path.open("a") as f:
        f.write(json.dumps(candidate) + "\n")


def _load() -> list[dict]:
    """Return all stored feedback records, or an empty list if none exist yet."""
    path = _path()
    if not path.exists():
        return []
    return json.loads(path.read_text())


def _save(items: list[dict]) -> None:
    """Write feedback records back to disk, creating the parent directory if needed."""
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(items, indent=2))


def add_feedback(
    question: str,
    answer: str,
    rating: str,
    reason: str | None,
    sources: list[dict],
) -> dict:
    """Record a thumbs up/down on an answer and return the stored record."""
    item = {
        "id": str(uuid.uuid4()),
        "question": question,
        "answer": answer,
        "rating": rating,
        "reason": reason,
        "sources": sources,
        "status": "open",
        "action": None,
        "note": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    items = _load()
    items.append(item)
    _save(items)
    return item


def list_feedback() -> list[dict]:
    """Return all feedback records, newest first, for the admin queue view."""
    return list(reversed(_load()))


def resolve_feedback(feedback_id: str, action: str, note: str | None) -> dict:
    """Mark a feedback record resolved with the admin's chosen action; raises KeyError if not found."""
    items = _load()
    for item in items:
        if item["id"] == feedback_id:
            item["status"] = "resolved"
            item["action"] = action
            item["note"] = note
            _save(items)
            if action == "add_to_eval_set":
                _append_regression_candidate(item)
            return item
    raise KeyError(feedback_id)
