"""Unit tests for the post-generation grounding check (_verify_grounded /
_apply_grounding_check) — no live LLM calls, everything mocked."""
from __future__ import annotations

from unittest.mock import patch

from src.answer_quality import _verify_grounded
from src.rag_agent import _apply_grounding_check


class TestVerifyGrounded:
    def test_true_when_no_context_to_check(self):
        assert _verify_grounded("Q", "A", [], "http://x", "model") is True

    def test_true_when_judge_says_yes(self):
        with patch("src.answer_quality._llm_call", return_value="YES"):
            assert _verify_grounded("Q", "A", ["ctx"], "http://x", "model") is True

    def test_false_when_judge_says_no(self):
        with patch("src.answer_quality._llm_call", return_value="NO"):
            assert _verify_grounded("Q", "A", ["ctx"], "http://x", "model") is False

    def test_fails_open_on_llm_error(self):
        with patch("src.answer_quality._llm_call", side_effect=RuntimeError("boom")):
            assert _verify_grounded("Q", "A", ["ctx"], "http://x", "model") is True


class TestApplyGroundingCheck:
    def test_noop_when_flag_disabled(self):
        with patch("src.rag_agent.POST_GENERATION_VERIFY_ENABLED", False), patch(
            "src.rag_agent._verify_grounded"
        ) as mock_verify:
            out = _apply_grounding_check("Q", "The answer", ["ctx"], "http://x", "model")
        assert out == "The answer"
        mock_verify.assert_not_called()

    def test_noop_when_no_tool_contexts(self):
        with patch("src.rag_agent._verify_grounded") as mock_verify:
            out = _apply_grounding_check("Q", "The answer", [], "http://x", "model")
        assert out == "The answer"
        mock_verify.assert_not_called()

    def test_noop_when_already_unsupported(self):
        with patch("src.rag_agent._verify_grounded") as mock_verify:
            out = _apply_grounding_check("Q", "Unsupported", ["ctx"], "http://x", "model")
        assert out == "Unsupported"
        mock_verify.assert_not_called()

    def test_downgrades_to_unsupported_when_ungrounded(self):
        with patch("src.rag_agent._verify_grounded", return_value=False):
            out = _apply_grounding_check("Q", "The answer", ["ctx"], "http://x", "model")
        assert out == "Unsupported"

    def test_leaves_answer_unchanged_when_grounded(self):
        with patch("src.rag_agent._verify_grounded", return_value=True):
            out = _apply_grounding_check("Q", "The answer", ["ctx"], "http://x", "model")
        assert out == "The answer"
