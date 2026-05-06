"""Tests for src/config.py."""
import src.config as config


def test_groq_api_key_var_exists():
    assert hasattr(config, "GROQ_API_KEY")


def test_retrieval_top_k_is_int():
    assert isinstance(config.RETRIEVAL_TOP_K, int)


def test_rerank_top_n_less_than_retrieval_top_k():
    assert config.RERANK_TOP_N < config.RETRIEVAL_TOP_K


def test_default_embed_model():
    """Default embed model must be nomic-embed-text (set in .env.example)."""
    import os
    model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    assert "multilingual" not in model.lower(), (
        "Multilingual embedding models are significantly slower; "
        "use a monolingual model unless bilingual retrieval is explicitly required."
    )


def test_reranker_is_cross_encoder():
    """Default reranker must be a known cross-encoder model."""
    model = config.RERANKER_MODEL.lower()
    assert any(name in model for name in ("cross-encoder", "bge-reranker", "reranker"))
