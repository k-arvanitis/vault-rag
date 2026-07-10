"""Tests for slack_app.py — Slack calls and the RAG API call mocked.

Handler functions are imported directly so no live Slack connection is needed.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from slack_app import _strip_mention, handle_dm, handle_mention

# ---------------------------------------------------------------------------
# _strip_mention
# ---------------------------------------------------------------------------


class TestStripMention:
    def test_removes_leading_mention(self):
        assert _strip_mention("<@U123ABC> what are the payment terms?") == "what are the payment terms?"

    def test_strips_surrounding_whitespace(self):
        assert _strip_mention("<@UBOT>   hello") == "hello"

    def test_no_mention_unchanged(self):
        assert _strip_mention("just a plain question") == "just a plain question"

    def test_only_mention_returns_empty_string(self):
        assert _strip_mention("<@UBOT123>") == ""


# ---------------------------------------------------------------------------
# handle_mention
# ---------------------------------------------------------------------------


class TestHandleMention:
    """@mention events — verify query extraction, threading, and help fallback."""

    def _run(self, text: str, answer: str = "The answer."):
        event = {"text": text, "ts": "123.456"}
        say = MagicMock()
        with patch("slack_app._query_api", return_value=answer):
            handle_mention(event, say)
        return say

    def test_posts_agent_answer(self):
        say = self._run("<@UBOT> what are the payment terms?", "Net 30 days.")
        assert "Net 30 days." in say.call_args.kwargs["text"]

    def test_replies_in_thread(self):
        say = self._run("<@UBOT> any question")
        assert say.call_args.kwargs.get("thread_ts") is not None

    def test_uses_existing_thread_ts(self):
        event = {"text": "<@UBOT> q", "ts": "111.0", "thread_ts": "999.0"}
        say = MagicMock()
        with patch("slack_app._query_api", return_value="a"):
            handle_mention(event, say)
        assert say.call_args.kwargs["thread_ts"] == "999.0"

    def test_bare_mention_posts_help(self):
        say = self._run("<@UBOT>")
        text = say.call_args.kwargs["text"]
        assert text
        assert "example" in text.lower() or "question" in text.lower()

    def test_empty_text_posts_help(self):
        say = self._run("")
        assert say.call_args.kwargs["text"]

    def test_query_api_called_with_stripped_query(self):
        event = {"text": "<@UBOT> payment terms", "ts": "1.0"}
        say = MagicMock()
        with patch("slack_app._query_api", return_value="a") as mock_query:
            handle_mention(event, say)
        assert mock_query.call_args[0][0] == "payment terms"


# ---------------------------------------------------------------------------
# handle_dm
# ---------------------------------------------------------------------------


class TestHandleDm:
    """Direct-message events — verify agent call and filtering."""

    def _run(self, event: dict, answer: str = "DM answer."):
        say = MagicMock()
        with patch("slack_app._query_api", return_value=answer):
            handle_dm(event, say)
        return say

    def test_dm_triggers_agent_and_posts_answer(self):
        say = self._run({"text": "what is this about?", "ts": "1.0"})
        assert "DM answer." in say.call_args.kwargs["text"]

    def test_replies_in_thread(self):
        say = self._run({"text": "question", "ts": "1.0"})
        assert say.call_args.kwargs.get("thread_ts") is not None

    def test_bot_message_ignored(self):
        say = self._run({"text": "hi", "ts": "1.0", "bot_id": "B123"})
        say.assert_not_called()

    def test_subtype_message_ignored(self):
        say = self._run({"text": "edited", "ts": "1.0", "subtype": "message_changed"})
        say.assert_not_called()

    def test_empty_text_ignored(self):
        say = self._run({"text": "", "ts": "1.0"})
        say.assert_not_called()

    def test_whitespace_only_text_ignored(self):
        say = self._run({"text": "   ", "ts": "1.0"})
        say.assert_not_called()
