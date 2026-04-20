"""Slack Bolt app for Vault RAG — query interface only (Socket Mode).

Documents are indexed by admins via the Streamlit UI. Slack is read-only:
  app_mention  → run RAG agent, reply in thread
  message (DM) → run RAG agent, reply in thread
"""

from __future__ import annotations

import os
import re

from dotenv import load_dotenv

load_dotenv(override=True)

from src.rag_agent import ask_agent, build_rag_agent  # noqa: E402

_agent = None


def _get_agent():
    """Build and cache the RAG agent — loaded once on first query."""
    global _agent
    if _agent is None:
        _agent = build_rag_agent()
    return _agent


def _strip_mention(text: str) -> str:
    """Remove the leading <@USERID> mention token from a Slack message."""
    return re.sub(r"^<@[A-Z0-9]+>\s*", "", text).strip()


# ---------------------------------------------------------------------------
# Event handlers — pure functions, importable and testable without a live App
# ---------------------------------------------------------------------------


def handle_mention(event: dict, say) -> None:
    """Answer an @mention with a RAG-grounded response posted in thread."""
    text = event.get("text", "")
    query = _strip_mention(text)
    thread_ts = event.get("thread_ts") or event.get("ts")

    if not query:
        say(
            text=(
                "Hi! Mention me with a question to search your documents.\n"
                "Example: `@vault what are the payment terms in the supplier contract?`"
            ),
            thread_ts=thread_ts,
        )
        return

    answer = ask_agent(_get_agent(), query)
    say(text=answer, thread_ts=thread_ts)


def handle_dm(event: dict, say) -> None:
    """Answer a direct message with a RAG-grounded response."""
    if event.get("subtype") or event.get("bot_id"):
        return
    query = (event.get("text") or "").strip()
    if not query:
        return
    thread_ts = event.get("thread_ts") or event.get("ts")
    answer = ask_agent(_get_agent(), query)
    say(text=answer, thread_ts=thread_ts)


# ---------------------------------------------------------------------------
# App wiring — only called at runtime, not at import time
# ---------------------------------------------------------------------------


def create_app():
    """Create and register all Slack Bolt event handlers."""
    from slack_bolt import App

    app = App(token=os.environ["SLACK_BOT_TOKEN"])

    @app.event("app_mention")
    def _on_mention(event, say):
        handle_mention(event, say)

    @app.event("message")
    def _on_message(event, say):
        if event.get("channel_type") not in ("im", "mpim"):
            return
        handle_dm(event, say)

    return app


def main() -> None:
    """Start the Vault RAG Slack bot in Socket Mode."""
    from slack_bolt.adapter.socket_mode import SocketModeHandler

    app = create_app()
    print("[slack] Vault RAG bot starting (Socket Mode)...")
    print("[slack] Warming up RAG agent...")
    _get_agent()
    print("[slack] Ready.")
    SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()


if __name__ == "__main__":
    main()
