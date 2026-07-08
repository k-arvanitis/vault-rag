"""Tests for src/conversation_store.py."""
import pytest

import src.conversation_store as conversation_store


@pytest.fixture(autouse=True)
def _isolated_conversation_path(tmp_path, monkeypatch):
    """Point the conversation store at a scratch file so tests never touch real data."""
    monkeypatch.setattr(conversation_store, "_path", lambda: tmp_path / "conversations.json")


def test_save_creates_new_conversation():
    messages = [{"id": "m1", "role": "user", "content": "what is the deadline?"}]
    item = conversation_store.save_conversation(None, messages)
    assert item["title"] == "what is the deadline?"
    assert item["messages"] == messages


def test_save_updates_existing_conversation():
    first = conversation_store.save_conversation(None, [{"id": "m1", "role": "user", "content": "q1"}])
    updated = conversation_store.save_conversation(
        first["id"],
        [
            {"id": "m1", "role": "user", "content": "q1"},
            {"id": "m2", "role": "assistant", "content": "a1"},
        ],
    )
    assert updated["id"] == first["id"]
    assert len(updated["messages"]) == 2
    assert len(conversation_store.list_conversations()) == 1


def test_list_conversations_newest_first():
    conversation_store.save_conversation(None, [{"id": "m1", "role": "user", "content": "first"}])
    conversation_store.save_conversation(None, [{"id": "m1", "role": "user", "content": "second"}])
    items = conversation_store.list_conversations()
    assert items[0]["title"] == "second"


def test_get_conversation_not_found_raises():
    with pytest.raises(KeyError):
        conversation_store.get_conversation("nonexistent")


def test_delete_conversation():
    item = conversation_store.save_conversation(None, [{"id": "m1", "role": "user", "content": "q"}])
    conversation_store.delete_conversation(item["id"])
    assert conversation_store.list_conversations() == []
    with pytest.raises(KeyError):
        conversation_store.delete_conversation(item["id"])
