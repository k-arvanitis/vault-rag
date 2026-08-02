"""Slack Bolt app for Vault RAG — query interface only (Socket Mode).

Documents are indexed by admins via the web UI. Slack is a thin client:
  app_mention  → POST /query to the Vault RAG API, reply in thread
  message (DM) → POST /query to the Vault RAG API, reply in thread

The bot does not build the RAG agent or open DuckDB itself — the FastAPI server
is the single backend, so the bot and the API never contend for the DuckDB lock.
"""

from __future__ import annotations

import os
import re

import httpx
from dotenv import load_dotenv

load_dotenv(override=True)

from vault_rag.config import API_BASE  # noqa: E402

_QUERY_TIMEOUT = 120.0


def _strip_mention(text: str) -> str:
    """Remove the leading <@USERID> mention token from a Slack message."""
    return re.sub(r"^<@[A-Z0-9]+>\s*", "", text).strip()


def _query_api(question: str) -> str:
    """Send a question to the Vault RAG API and return the answer text."""
    try:
        resp = httpx.post(
            f"{API_BASE}/query",
            json={"question": question},
            timeout=_QUERY_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json().get("answer") or "Unsupported"
    except httpx.HTTPError as exc:
        return f":warning: Could not reach the Vault RAG API — {exc}"


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

    say(text=_query_api(query), thread_ts=thread_ts)


def handle_dm(event: dict, say) -> None:
    """Answer a direct message with a RAG-grounded response."""
    if event.get("subtype") or event.get("bot_id"):
        return
    query = (event.get("text") or "").strip()
    if not query:
        return
    thread_ts = event.get("thread_ts") or event.get("ts")
    say(text=_query_api(query), thread_ts=thread_ts)


# ---------------------------------------------------------------------------
# App wiring — only called at runtime, not at import time
# ---------------------------------------------------------------------------


def create_app():
    """Create and register all Slack Bolt event handlers."""
    from slack_bolt import App

    app = App(token=os.environ["SLACK_BOT_TOKEN"])

    @app.event("app_mention")
    def _on_mention(event, say):
        """Slack app_mention handler — delegate to the testable handle_mention."""
        handle_mention(event, say)

    @app.event("message")
    def _on_message(event, say):
        """Slack message handler — handle DMs only, ignore channel messages."""
        if event.get("channel_type") not in ("im", "mpim"):
            return
        handle_dm(event, say)

    return app


def main() -> None:
    """Start the Vault RAG Slack bot in Socket Mode."""
    from slack_bolt.adapter.socket_mode import SocketModeHandler

    app = create_app()
    print(f"[slack] Vault RAG bot starting (Socket Mode) — API at {API_BASE}")
    SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()
    print("[slack] Ready.")


if __name__ == "__main__":
    main()
