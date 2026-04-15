"""Tests for src/config.py."""
import src.config as config


def test_groq_api_key_var_exists():
    assert hasattr(config, "GROQ_API_KEY")


def test_retrieval_top_k_is_int():
    assert isinstance(config.RETRIEVAL_TOP_K, int)


def test_rerank_top_n_less_than_retrieval_top_k():
    assert config.RERANK_TOP_N < config.RETRIEVAL_TOP_K


def test_no_multilingual_models():
    assert "multilingual" not in config.OLLAMA_EMBED_MODEL.lower()
    assert "m3" not in config.RERANKER_MODEL.lower()
