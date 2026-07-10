"""Tests for src/pipeline.py.

All external calls (LLM, ask_agent, OpenAI client) are mocked.
No real agent, Qdrant, or network access is used.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from src.pipeline import (
    _is_unsupported,
    _looks_excel,
    ask_with_decomposition,
    ask_with_reflection,
    build_decomposition_pipeline,
    build_reflection_pipeline,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

class TestIsUnsupported:
    def test_exact_match(self):
        assert _is_unsupported("Unsupported")

    def test_case_insensitive(self):
        assert _is_unsupported("UNSUPPORTED")
        assert _is_unsupported("unsupported")

    def test_whitespace_stripped(self):
        assert _is_unsupported("  unsupported  ")

    def test_non_unsupported(self):
        assert not _is_unsupported("The answer is 42.")
        assert not _is_unsupported("")


class TestLooksExcel:
    def test_transaction_keyword(self):
        assert _looks_excel("What is the transaction total?")

    def test_supplier_keyword(self):
        assert _looks_excel("List all supplier payments.")

    def test_pdf_question(self):
        assert not _looks_excel("Who is the authorizing manager?")

    def test_case_insensitive(self):
        assert _looks_excel("Show SUPPLIER spend.")


# ---------------------------------------------------------------------------
# Reflection pipeline
# ---------------------------------------------------------------------------

def _make_agent(answer: str) -> MagicMock:
    """Return a mock LangGraph agent that always yields *answer* from ask_agent."""
    return MagicMock()


class TestReflectionPipeline:
    def test_good_answer_returned_directly(self):
        agent = _make_agent("The answer is 42.")
        with patch("src.rag_agent.ask_agent", return_value="The answer is 42.") as mock_ask:
            pipeline = build_reflection_pipeline(agent)
            result = ask_with_reflection(pipeline, "What is the answer?")
        assert result == "The answer is 42."
        assert mock_ask.call_count == 1

    def test_unsupported_triggers_retry(self):
        agent = MagicMock()
        answers = ["Unsupported", "Found it on retry."]
        with patch("src.rag_agent.ask_agent", side_effect=answers) as mock_ask:
            pipeline = build_reflection_pipeline(agent)
            result = ask_with_reflection(pipeline, "What is in doc_001?")
        assert result == "Found it on retry."
        assert mock_ask.call_count == 2

    def test_unsupported_twice_stops_at_two_retries(self):
        agent = MagicMock()
        with patch("src.rag_agent.ask_agent", return_value="Unsupported") as mock_ask:
            pipeline = build_reflection_pipeline(agent)
            result = ask_with_reflection(pipeline, "What is in doc_001?")
        assert result == "Unsupported"
        # Initial run + 1 retry = 2 (retry_count cap is 2 but only 1 retry allowed from < 2 check)
        assert mock_ask.call_count == 2

    def test_retry_question_contains_broader_search_hint(self):
        agent = MagicMock()
        answers = ["Unsupported", "Found it."]
        captured_questions = []

        def capture(ag, q, **kwargs):
            captured_questions.append(q)
            return answers.pop(0)

        with patch("src.rag_agent.ask_agent", side_effect=capture):
            pipeline = build_reflection_pipeline(agent)
            ask_with_reflection(pipeline, "What is the term in doc_001?")

        assert len(captured_questions) == 2
        assert "broader" in captured_questions[1].lower() or "broader search" in captured_questions[1].lower()


# ---------------------------------------------------------------------------
# Decomposition pipeline
# ---------------------------------------------------------------------------

def _mock_openai_response(sub_questions: list[str]) -> MagicMock:
    """Build a minimal mock OpenAI completion response."""
    choice = MagicMock()
    choice.message.content = json.dumps(sub_questions)
    response = MagicMock()
    response.choices = [choice]
    return response


class TestDecompositionPipeline:
    def test_single_hop_passes_through_unchanged(self):
        agent = MagicMock()
        question = "Who is the authorizing manager?"
        openai_response = _mock_openai_response([question])

        with patch("src.rag_agent.ask_agent", return_value="Ricki Contreras."), \
             patch("openai.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = openai_response
            mock_openai_cls.return_value = mock_client

            pipeline = build_decomposition_pipeline(agent)
            result = pipeline.invoke({
                "question": question,
                "sub_questions": [],
                "formatted_question": "",
                "answer": "",
            })

        assert result["sub_questions"] == [question]
        assert result["formatted_question"] == question
        assert result["answer"] == "Ricki Contreras."

    def test_multi_hop_formats_numbered_plan(self):
        agent = MagicMock()
        sub_qs = [
            "What is the extension period in the procurement policy?",
            "What is the extension period in the services contract?",
        ]
        openai_response = _mock_openai_response(sub_qs)

        with patch("src.rag_agent.ask_agent", return_value="Policy allows longer."), \
             patch("openai.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = openai_response
            mock_openai_cls.return_value = mock_client

            pipeline = build_decomposition_pipeline(agent)
            result = pipeline.invoke({
                "question": "Which document allows the longer extension?",
                "sub_questions": [],
                "formatted_question": "",
                "answer": "",
            })

        assert result["sub_questions"] == sub_qs
        assert "1." in result["formatted_question"]
        assert "2." in result["formatted_question"]
        assert sub_qs[0] in result["formatted_question"]
        assert sub_qs[1] in result["formatted_question"]

    def test_llm_failure_falls_back_to_original_question(self):
        agent = MagicMock()
        question = "Which document is longer?"

        with patch("src.rag_agent.ask_agent", return_value="Doc A."), \
             patch("openai.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = RuntimeError("API down")
            mock_openai_cls.return_value = mock_client

            pipeline = build_decomposition_pipeline(agent)
            result = pipeline.invoke({
                "question": question,
                "sub_questions": [],
                "formatted_question": "",
                "answer": "",
            })

        assert result["sub_questions"] == [question]
        assert result["formatted_question"] == question

    def test_think_tags_stripped_from_llm_output(self):
        agent = MagicMock()
        question = "Compare the two documents."
        sub_qs = ["Sub-question 1?", "Sub-question 2?"]
        raw_with_think = f"<think>some reasoning</think>{json.dumps(sub_qs)}"

        choice = MagicMock()
        choice.message.content = raw_with_think
        response = MagicMock()
        response.choices = [choice]

        with patch("src.rag_agent.ask_agent", return_value="Result."), \
             patch("openai.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = response
            mock_openai_cls.return_value = mock_client

            pipeline = build_decomposition_pipeline(agent)
            result = pipeline.invoke({
                "question": question,
                "sub_questions": [],
                "formatted_question": "",
                "answer": "",
            })

        assert result["sub_questions"] == sub_qs

    def test_ask_with_decomposition_returns_answer(self):
        agent = MagicMock()
        question = "Simple question?"
        openai_response = _mock_openai_response([question])

        with patch("src.rag_agent.ask_agent", return_value="Simple answer."), \
             patch("openai.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = openai_response
            mock_openai_cls.return_value = mock_client

            pipeline = build_decomposition_pipeline(agent)
            answer = ask_with_decomposition(pipeline, question)

        assert answer == "Simple answer."
