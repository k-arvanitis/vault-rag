"""Tests for src/reranker.py.

Model weights are never downloaded — AutoTokenizer and AutoModel* are fully
mocked. Real torch tensors are used for logit/score arithmetic so the
mathematical path through each reranker is actually exercised.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch

# ---------------------------------------------------------------------------
# BGEReranker
# ---------------------------------------------------------------------------

class TestBGEReranker:
    """Cross-encoder reranker backed by a sequence-classification model."""

    def _make_ranker(self, scores: list[float]):
        """Build a BGEReranker with mocked model that returns `scores` as logits."""
        with patch("vault_rag.reranker.AutoTokenizer.from_pretrained") as mock_tok_cls, \
             patch("vault_rag.reranker.AutoModelForSequenceClassification.from_pretrained") as mock_model_cls:

            mock_tokenizer = MagicMock()
            mock_tok_cls.return_value = mock_tokenizer

            mock_model = MagicMock()
            mock_model_cls.return_value.to.return_value.eval.return_value = mock_model

            # Tokenizer returns inputs that support .to(device)
            fake_inputs = MagicMock()
            fake_inputs.to.return_value = fake_inputs
            mock_tokenizer.return_value = fake_inputs

            # Model returns real tensor so squeeze/float/cpu/tolist work correctly
            logits = torch.tensor([[s] for s in scores])
            mock_output = MagicMock()
            mock_output.logits = logits
            mock_model.return_value = mock_output

            from vault_rag.reranker import BGEReranker
            ranker = BGEReranker(model_name="fake-model", device="cpu")

        return ranker

    def test_output_has_required_keys(self):
        ranker = self._make_ranker([0.8, 0.2, 0.5])
        results = ranker.rerank("query", ["a", "b", "c"], top_n=3)
        for r in results:
            assert "index" in r
            assert "text" in r
            assert "score" in r

    def test_results_sorted_descending(self):
        ranker = self._make_ranker([0.1, 0.9, 0.5])
        results = ranker.rerank("q", ["low", "high", "mid"], top_n=3)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_n_limits_output(self):
        ranker = self._make_ranker([0.3, 0.7, 0.5, 0.9])
        results = ranker.rerank("q", ["a", "b", "c", "d"], top_n=2)
        assert len(results) == 2

    def test_index_maps_to_original_position(self):
        ranker = self._make_ranker([0.2, 0.8])
        results = ranker.rerank("q", ["first", "second"], top_n=2)
        # Highest score is second doc → index 1
        assert results[0]["index"] == 1
        assert results[0]["text"] == "second"

    def test_single_document(self):
        ranker = self._make_ranker([0.6])
        results = ranker.rerank("q", ["only doc"], top_n=1)
        assert len(results) == 1
        assert results[0]["index"] == 0


# ---------------------------------------------------------------------------
# QwenReranker
# ---------------------------------------------------------------------------

class TestQwenReranker:
    """Causal-LM reranker that scores relevance via yes/no token probabilities."""

    # Use token id 1 for "yes" and 0 for "no" in all mocks
    _YES_ID = 1
    _NO_ID = 0

    def _make_ranker(self, yes_logits: list[float]):
        """Build a QwenReranker whose model returns controlled yes-logits per doc."""
        with patch("vault_rag.reranker.AutoTokenizer.from_pretrained") as mock_tok_cls, \
             patch("vault_rag.reranker.AutoModelForCausalLM.from_pretrained") as mock_model_cls:

            mock_tokenizer = MagicMock()
            mock_tok_cls.return_value = mock_tokenizer
            mock_tokenizer.convert_tokens_to_ids.side_effect = (
                lambda t: self._YES_ID if t == "yes" else self._NO_ID
            )

            mock_model = MagicMock()
            mock_model.device = "cpu"
            mock_model_cls.return_value.eval.return_value = mock_model

            from vault_rag.reranker import QwenReranker
            ranker = QwenReranker(model_name="fake-qwen", device="cpu")

        # Wire tokenizer to return fake inputs per call
        fake_inputs = MagicMock()
        fake_inputs.to.return_value = fake_inputs
        ranker.tokenizer.return_value = fake_inputs

        # Each call to model(**inputs) returns the next yes_logit
        call_iter = iter(yes_logits)

        def _model_side_effect(**kwargs):
            y = next(call_iter)
            # logits shape [1, seq_len, vocab_size]; code slices [:, -1, :]
            logits = torch.zeros(1, 1, 10)
            logits[0, 0, self._YES_ID] = y
            out = MagicMock()
            out.logits = logits
            return out

        ranker.model.side_effect = _model_side_effect
        return ranker

    def test_output_has_required_keys(self):
        ranker = self._make_ranker([2.0, 0.0, 1.0])
        results = ranker.rerank("q", ["a", "b", "c"], top_n=3)
        for r in results:
            assert "index" in r
            assert "text" in r
            assert "score" in r

    def test_scores_are_probabilities(self):
        ranker = self._make_ranker([2.0, 0.0])
        results = ranker.rerank("q", ["relevant", "irrelevant"], top_n=2)
        for r in results:
            assert 0.0 <= r["score"] <= 1.0

    def test_high_yes_logit_yields_high_score(self):
        ranker = self._make_ranker([5.0, -5.0])
        results = ranker.rerank("q", ["relevant", "irrelevant"], top_n=2)
        assert results[0]["text"] == "relevant"
        assert results[0]["score"] > 0.5

    def test_top_n_limits_output(self):
        ranker = self._make_ranker([1.0, 2.0, 3.0, 0.5])
        results = ranker.rerank("q", ["a", "b", "c", "d"], top_n=2)
        assert len(results) == 2

    def test_results_sorted_descending(self):
        ranker = self._make_ranker([0.5, 3.0, 1.0])
        results = ranker.rerank("q", ["low", "high", "mid"], top_n=3)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)
