"""Tests for src/feedback_store.py."""
import pytest

import src.feedback_store as feedback_store


@pytest.fixture(autouse=True)
def _isolated_feedback_path(tmp_path, monkeypatch):
    """Point the feedback store at a scratch file so tests never touch real data."""
    monkeypatch.setattr(feedback_store, "_path", lambda: tmp_path / "feedback.json")


def test_add_and_list_feedback():
    feedback_store.add_feedback("q1", "a1", "down", "hallucinated", [])
    feedback_store.add_feedback("q2", "a2", "up", None, [])
    items = feedback_store.list_feedback()
    assert len(items) == 2
    assert items[0]["question"] == "q2"  # newest first
    assert items[0]["status"] == "open"


def test_resolve_feedback_updates_status():
    item = feedback_store.add_feedback("q1", "a1", "down", "wrong_source", [])
    resolved = feedback_store.resolve_feedback(item["id"], "add_to_eval_set", "note")
    assert resolved["status"] == "resolved"
    assert resolved["action"] == "add_to_eval_set"
    assert resolved["note"] == "note"


def test_resolve_unknown_feedback_raises():
    with pytest.raises(KeyError):
        feedback_store.resolve_feedback("nonexistent", "dismissed", None)
