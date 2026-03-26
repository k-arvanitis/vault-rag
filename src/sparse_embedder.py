"""Sparse (BM42) embeddings via fastembed for hybrid Qdrant search."""
from __future__ import annotations

_SPARSE_MODEL_NAME = "Qdrant/bm42-all-minilm-l6-v2-attentions"
_instance: "SparseEmbedder | None" = None


class SparseEmbedder:
    def __init__(self, model_name: str = _SPARSE_MODEL_NAME) -> None:
        from fastembed import SparseTextEmbedding
        self._model = SparseTextEmbedding(model_name=model_name)

    def embed(self, text: str) -> tuple[list[int], list[float]]:
        result = list(self._model.embed([text]))[0]
        return result.indices.tolist(), result.values.tolist()


def get_sparse_embedder() -> SparseEmbedder:
    """Return a module-level singleton — model loads only once."""
    global _instance
    if _instance is None:
        _instance = SparseEmbedder()
    return _instance
