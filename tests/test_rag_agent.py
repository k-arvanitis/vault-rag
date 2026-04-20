"""Tests for src/rag_agent.py.

No real LLM, Qdrant, or reranker is used. The LangGraph agent is replaced
with a mock whose invoke/stream output is controlled per test.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage, ToolMessage

from src.rag_agent import (
    SYSTEM_PROMPT,
    _extract_refs,
    ask_agent,
    stream_agent,
)


# ---------------------------------------------------------------------------
# _extract_refs
# ---------------------------------------------------------------------------

class TestExtractRefs:
    def test_document_chunk_headers_extracted(self):
        content = "[1] file=report.pdf chunk=3 score=0.92\nsome content here"
        refs = _extract_refs(content)
        assert "[1] file=report.pdf chunk=3 score=0.92" in refs

    def test_multiple_chunks_all_extracted(self):
        content = (
            "[1] file=a.pdf chunk=0 score=0.9\ncontent\n\n"
            "[2] file=b.pdf chunk=1 score=0.8\ncontent"
        )
        refs = _extract_refs(content)
        assert "[1]" in refs
        assert "[2]" in refs

    def test_no_matches_returns_fallback(self):
        assert _extract_refs("no chunk headers here") == "(no results)"

    def test_empty_string_returns_fallback(self):
        assert _extract_refs("") == "(no results)"


# ---------------------------------------------------------------------------
# SYSTEM_PROMPT sanity checks
# ---------------------------------------------------------------------------

class TestSystemPrompt:
    def test_contains_tool_name(self):
        assert "search_knowledge_base" in SYSTEM_PROMPT

    def test_contains_citation_instruction(self):
        assert "[1]" in SYSTEM_PROMPT or "cite" in SYSTEM_PROMPT.lower()

    def test_no_bilingual_instruction(self):
        assert "Greek" not in SYSTEM_PROMPT
        assert "bilingual" not in SYSTEM_PROMPT.lower()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_invoke_result(answer: str, tool_content: str = "") -> dict:
    """Build a minimal agent.invoke() result dict."""
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    if tool_content:
        tool_msg = ToolMessage(content=tool_content, tool_call_id="tc1", name="search_knowledge_base")
        messages.append(tool_msg)
    messages.append(AIMessage(content=answer))
    return {"messages": messages}


def _make_stream_chunks(tokens: list[str], tool_content: str = "") -> list[tuple]:
    """Build the (chunk, metadata) tuples that agent.stream() would yield."""
    pairs = []
    if tool_content:
        pairs.append((ToolMessage(content=tool_content, tool_call_id="tc1", name="search"), None))
    for token in tokens:
        pairs.append((AIMessageChunk(content=token), None))
    return pairs


# ---------------------------------------------------------------------------
# ask_agent
# ---------------------------------------------------------------------------

class TestAskAgent:
    def test_returns_final_answer(self):
        agent = MagicMock()
        agent.invoke.return_value = _make_invoke_result("The answer is 42.")
        agent._rag_limits = {}
        result = ask_agent(agent, "What is the answer?")
        assert result == "The answer is 42."

    def test_strips_think_blocks(self):
        agent = MagicMock()
        agent.invoke.return_value = _make_invoke_result("<think>hidden reasoning</think>Clean answer.")
        agent._rag_limits = {}
        result = ask_agent(agent, "q")
        assert "hidden reasoning" not in result
        assert "Clean answer." in result

    def test_history_prepended_to_messages(self):
        agent = MagicMock()
        agent.invoke.return_value = _make_invoke_result("ok")
        agent._rag_limits = {}

        history = [
            {"role": "user",      "content": "first question"},
            {"role": "assistant", "content": "first answer"},
        ]
        ask_agent(agent, "follow-up", history=history)

        call_messages = agent.invoke.call_args[0][0]["messages"]
        contents = [m.content for m in call_messages]
        assert "first question" in contents
        assert "first answer" in contents

    def test_current_query_is_last_human_message(self):
        agent = MagicMock()
        agent.invoke.return_value = _make_invoke_result("ok")
        agent._rag_limits = {}

        ask_agent(agent, "my query", history=[{"role": "user", "content": "prev"}])

        call_messages = agent.invoke.call_args[0][0]["messages"]
        last = call_messages[-1]
        assert isinstance(last, HumanMessage)
        assert last.content == "my query"

    def test_first_message_is_system_prompt(self):
        agent = MagicMock()
        agent.invoke.return_value = _make_invoke_result("ok")
        agent._rag_limits = {}

        ask_agent(agent, "q")

        first = agent.invoke.call_args[0][0]["messages"][0]
        assert isinstance(first, SystemMessage)
        assert first.content == SYSTEM_PROMPT

    def test_no_answer_returns_fallback(self):
        agent = MagicMock()
        # Only a ToolMessage, no final AIMessage
        agent.invoke.return_value = {"messages": [ToolMessage(content="chunks", tool_call_id="x", name="s")]}
        agent._rag_limits = {}
        result = ask_agent(agent, "q")
        assert result == "No answer generated."


# ---------------------------------------------------------------------------
# stream_agent
# ---------------------------------------------------------------------------

class TestStreamAgent:
    def test_yields_tokens(self):
        agent = MagicMock()
        agent.stream.return_value = _make_stream_chunks(["Hello", " world"])
        tokens = list(stream_agent(agent, "q"))
        assert "".join(tokens) == "Hello world"

    def test_strips_think_blocks(self):
        agent = MagicMock()
        agent.stream.return_value = _make_stream_chunks(["<think>hidden</think>visible"])
        tokens = list(stream_agent(agent, "q"))
        full = "".join(tokens)
        assert "hidden" not in full
        assert "visible" in full

    def test_think_block_content_across_tokens(self):
        """<think> open/close tags arrive as whole tokens; content spans multiple tokens."""
        agent = MagicMock()
        agent.stream.return_value = _make_stream_chunks(
            ["<think>", "hidden ", "reasoning", "</think>", "real answer"]
        )
        full = "".join(stream_agent(agent, "q"))
        assert "hidden" not in full
        assert "real answer" in full

    def test_tool_messages_not_yielded(self):
        agent = MagicMock()
        agent.stream.return_value = _make_stream_chunks(["answer"], tool_content="chunk data")
        tokens = list(stream_agent(agent, "q"))
        assert "chunk data" not in "".join(tokens)
        assert "answer" in "".join(tokens)

    def test_collected_chunks_populated(self):
        tool_content = "[1] file=a.pdf chunk=0 score=0.9\ncontent A\n\n[2] file=b.pdf chunk=1 score=0.8\ncontent B"
        agent = MagicMock()
        agent.stream.return_value = _make_stream_chunks(["ok"], tool_content=tool_content)
        chunks: list[str] = []
        list(stream_agent(agent, "q", collected_chunks=chunks))
        assert len(chunks) == 2

    def test_history_included_in_messages(self):
        agent = MagicMock()
        agent.stream.return_value = []
        history = [
            {"role": "user",      "content": "prev q"},
            {"role": "assistant", "content": "prev a"},
        ]
        list(stream_agent(agent, "new q", history=history))
        call_messages = agent.stream.call_args[0][0]["messages"]
        contents = [m.content for m in call_messages]
        assert "prev q" in contents
        assert "prev a" in contents
        assert call_messages[-1].content == "new q"

    def test_no_history_still_works(self):
        agent = MagicMock()
        agent.stream.return_value = _make_stream_chunks(["fine"])
        result = "".join(stream_agent(agent, "q"))
        assert result == "fine"
