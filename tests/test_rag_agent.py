"""Tests for src/rag_agent.py.

No real LLM, Qdrant, or reranker is used. The LangGraph agent is replaced
with a mock whose invoke/stream output is controlled per test.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage, ToolMessage

from src.rag_agent import (
    SYSTEM_PROMPT,
    _best_snippet,
    _context_source_count,
    _enrich_search_query,
    _extract_key_value_answer,
    _repair_incomplete_answer,
    _extract_refs,
    _is_thinking_model,
    _build_system_prompt,
    _hyde,
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

    def test_no_matches_returns_preview(self):
        # Non-chunk content returns a short preview (for query_excel output etc.)
        assert _extract_refs("no chunk headers here") == "no chunk headers here"

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
# _is_thinking_model
# ---------------------------------------------------------------------------

class TestIsThinkingModel:
    def test_qwen_detected(self):
        assert _is_thinking_model("qwen3-30b-a3b") is True

    def test_qwq_detected(self):
        assert _is_thinking_model("qwq-32b") is True

    def test_deepseek_r_detected(self):
        assert _is_thinking_model("deepseek-r1-distill-qwen-32b") is True

    def test_r1_detected(self):
        assert _is_thinking_model("r1-32b") is True

    def test_case_insensitive(self):
        assert _is_thinking_model("QWEN3-30B") is True
        assert _is_thinking_model("DeepSeek-R1") is True

    def test_gpt_not_thinking(self):
        assert _is_thinking_model("gpt-4o") is False

    def test_claude_not_thinking(self):
        assert _is_thinking_model("claude-3-5-sonnet") is False

    def test_llama_not_thinking(self):
        assert _is_thinking_model("llama-3.3-70b") is False

    def test_mistral_not_thinking(self):
        assert _is_thinking_model("mistral-large") is False


# ---------------------------------------------------------------------------
# _build_system_prompt
# ---------------------------------------------------------------------------

class TestBuildSystemPrompt:
    def test_thinking_model_gets_no_think_prefix(self):
        prompt = _build_system_prompt("qwen3-30b")
        assert prompt.startswith("/no_think ")

    def test_regular_model_no_prefix(self):
        prompt = _build_system_prompt("gpt-4o")
        assert not prompt.startswith("/no_think ")

    def test_body_preserved_for_thinking_model(self):
        prompt = _build_system_prompt("deepseek-r1")
        assert "search_knowledge_base" in prompt

    def test_body_preserved_for_regular_model(self):
        prompt = _build_system_prompt("claude-3-5-sonnet")
        assert "search_knowledge_base" in prompt


# ---------------------------------------------------------------------------
# _hyde
# ---------------------------------------------------------------------------

class TestHyde:
    def test_thinking_model_prompt_has_no_think_prefix(self):
        with patch("src.rag_agent._llm_call") as mock_llm:
            mock_llm.return_value = "A hypothetical passage."
            result = _hyde("What is RAG?", "http://localhost:4000", "qwen3-30b")
            call_args = mock_llm.call_args
            prompt = call_args[0][0]
            assert prompt.startswith("/no_think ")
            assert "What is RAG?" in prompt
            assert result == "A hypothetical passage."

    def test_regular_model_prompt_no_prefix(self):
        with patch("src.rag_agent._llm_call") as mock_llm:
            mock_llm.return_value = "Another passage."
            result = _hyde("Explain vector DBs", "http://localhost:4000", "gpt-4o")
            call_args = mock_llm.call_args
            prompt = call_args[0][0]
            assert not prompt.startswith("/no_think ")
            assert "Explain vector DBs" in prompt
            assert result == "Another passage."


# ---------------------------------------------------------------------------
# Snippet selection
# ---------------------------------------------------------------------------

class TestBestSnippet:
    def test_keeps_query_relevant_window(self):
        content = "intro " * 300 + "Original Issue Date: December 15, 2005\n" + "tail " * 300
        snippet = _best_snippet(
            content,
            "What is the original issue date?",
            max_chars=300,
        )
        assert "Original Issue Date: December 15, 2005" in snippet
        assert len(snippet) <= 304

    def test_prefers_answer_dense_later_window_over_first_generic_match(self):
        content = (
            "Contract documents may include a Renewal or extension of an existing Contract. "
            + "general policy text " * 120
            + "\n## Contract Term\n"
            + "Contracts shall be limited to a maximum of five consecutive years "
            + "with an optional extension of up to two years.\n"
            + "tail " * 80
        )
        snippet = _best_snippet(
            content,
            "contract extension or renewal period",
            max_chars=420,
        )
        assert "optional extension of up to two years" in snippet


class TestSearchQueryEnrichment:
    def test_pass_through_returns_query_unchanged(self):
        # _enrich_search_query is intentionally a pass-through seam — query
        # rewriting was removed to avoid eval-set-specific shortcuts.
        assert _enrich_search_query("new topic areas in the 2024 report") == (
            "new topic areas in the 2024 report"
        )


class TestExtractKeyValueAnswer:
    def test_extracts_markdown_heading_label_value(self):
        answer = _extract_key_value_answer(
            "What approval level is listed?",
            ["## **Approval Level: Board of Retirement**"],
        )
        assert answer == "Board of Retirement"


class TestIncompleteAnswerRepair:
    def test_counts_distinct_context_sources(self):
        contexts = [
            "[1] file=doc_001_procurement_policy.pdf chunk=0 score=0.9\ncontent",
            "[2] file=doc_002_services_contract.pdf chunk=0 score=0.8\ncontent",
        ]
        assert _context_source_count(contexts) == 2

    def test_one_source_multi_part_answer_forces_decomposed_retrieval(self):
        contexts = [
            "[1] file=doc_002_services_contract.pdf chunk=0 score=0.9\n"
            "Payment will be made within thirty (30) days after receipt of a valid invoice."
        ]
        query = (
            "Which specifies a 30-day payment deadline, and what does the other say "
            "about invoices?"
        )

        with (
            patch("src.rag_agent._llm_split_subqueries") as split,
            patch("src.rag_agent._coverage_check") as coverage,
            patch("src.rag_agent.retrieve") as retrieve_mock,
            patch("src.rag_agent._direct_answer_from_context") as answer_from_context,
        ):
            split.return_value = [
                "services contract 30-day payment deadline valid invoice",
                "procurement policy invoices original PO delivery documents before payment",
            ]
            retrieve_mock.side_effect = [
                [
                    {
                        "metadata": {
                            "file_name": "doc_002_services_contract.pdf",
                            "chunk_index": 0,
                        },
                        "content": "Payment will be made within thirty (30) days after receipt of a valid invoice.",
                    }
                ],
                [
                    {
                        "metadata": {
                            "file_name": "doc_001_procurement_policy.pdf",
                            "chunk_index": 1,
                        },
                        "content": (
                            "All Invoices must be validated against the original PO and "
                            "delivery documents before payment is made."
                        ),
                    }
                ],
            ]
            answer_from_context.return_value = (
                "Services contract: 30 days after receipt of a valid invoice. "
                "Procurement policy: invoices must be validated against the original PO "
                "and delivery documents before payment is made."
            )

            repaired = _repair_incomplete_answer(
                query,
                "Services contract: 30 days after receipt of a valid invoice.",
                contexts,
                "http://llm/v1",
                "gpt-test",
            )

        assert "Procurement policy" in repaired
        assert "original PO" in repaired
        assert retrieve_mock.call_count == 2
        coverage.assert_not_called()


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

    def test_no_manual_system_prompt_in_messages(self):
        """System prompt is now injected via state_modifier, not manually."""
        agent = MagicMock()
        agent.invoke.return_value = _make_invoke_result("ok")
        agent._rag_limits = {}

        ask_agent(agent, "q")

        messages = agent.invoke.call_args[0][0]["messages"]
        assert not any(isinstance(m, SystemMessage) for m in messages)

    def test_no_answer_returns_fallback(self):
        agent = MagicMock()
        # Only a ToolMessage, no final AIMessage
        agent.invoke.return_value = {"messages": [ToolMessage(content="chunks", tool_call_id="x", name="s")]}
        agent._rag_limits = {}
        result = ask_agent(agent, "q")
        assert result == "No answer generated."

    def test_bad_answer_uses_direct_context_fallback(self):
        agent = MagicMock()
        agent.invoke.return_value = _make_invoke_result(
            "Unsupported",
            tool_content="[1] file=a.md chunk=0 score=0.9\nOriginal Issue Date: December 15, 2005",
        )
        agent._rag_limits = {}
        agent._generation_api_base = "http://llm/v1"
        agent._generation_model = "gpt-test"

        with patch("src.rag_agent._direct_answer_from_context", return_value="December 15, 2005") as fallback:
            result = ask_agent(agent, "What is the original issue date?")

        assert result == "December 15, 2005"
        fallback.assert_called_once()

    def test_raw_tool_call_without_tool_result_becomes_unsupported(self):
        agent = MagicMock()
        agent.invoke.return_value = _make_invoke_result(
            '<function=search_knowledge_base {"query": "x"}</function>'
        )
        agent._rag_limits = {}
        result = ask_agent(agent, "q")
        assert result == "Unsupported"


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
        # Tool-call boundary marker + 2 content chunks.
        assert chunks[0] == "---CALL_BOUNDARY---"
        assert len([c for c in chunks if c != "---CALL_BOUNDARY---"]) == 2

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

    def test_bad_stream_answer_uses_direct_context_fallback(self):
        agent = MagicMock()
        agent.stream.return_value = _make_stream_chunks(
            ["Unsupported"],
            tool_content="[1] file=a.md chunk=0 score=0.9\nOriginal Issue Date: December 15, 2005",
        )
        agent._generation_api_base = "http://llm/v1"
        agent._generation_model = "gpt-test"

        with patch("src.rag_agent._direct_answer_from_context", return_value="December 15, 2005") as fallback:
            result = "".join(stream_agent(agent, "What is the original issue date?"))

        assert result == "December 15, 2005"
        fallback.assert_called_once()

    def test_raw_stream_tool_call_without_tool_result_becomes_unsupported(self):
        agent = MagicMock()
        agent.stream.return_value = _make_stream_chunks(
            ['<function=search_knowledge_base {"query": "x"}</function>']
        )
        result = "".join(stream_agent(agent, "q"))
        assert result == "Unsupported"
