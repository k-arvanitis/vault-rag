"""Tests for src/embedder.py."""

from unittest.mock import patch

from src.embedder import embed_chunks


def _chunk(content: str) -> dict:
    return {"content": content, "metadata": {}}


class TestEmbedChunksNaNFallback:
    def test_nan_on_single_chunk_falls_back_to_placeholder_instead_of_crashing(self):
        """Reproduced live 2026-07-25: Ollama/bge-m3 returned a NaN embedding
        for one chunk ("failed to encode response: json: unsupported value:
        NaN"), which used to crash the entire file's ingest -- all other
        chunks already embedded successfully got thrown away too."""
        chunks = [_chunk("normal chunk one"), _chunk("chunk that produces NaN")]
        calls: list[list[str]] = []

        def fake_embed(api_base, model_name, texts):
            calls.append(texts)
            if "chunk that produces NaN" in texts:
                raise RuntimeError(
                    'Ollama embed request failed (500): {"error":"failed to '
                    'encode response: json: unsupported value: NaN"}'
                )
            return [[0.1, 0.2] for _ in texts]

        with patch("src.embedder._ollama_embed_batch", side_effect=fake_embed):
            result = embed_chunks(chunks, model_name="bge-m3", api_base="http://x", batch_size=2)

        assert len(result) == 2
        assert all(len(r["embedding"]) == 2 for r in result)
        # The placeholder text was actually re-embedded, not a raw zero-vector.
        assert ["[content omitted: embedding failed]"] in calls

    def test_non_nan_single_chunk_failure_still_raises(self):
        """A real failure (backend down, wrong model, etc.) must not be
        silently swallowed the same way -- only the specific NaN-encoding
        error gets the placeholder fallback."""
        chunks = [_chunk("some chunk")]

        def fake_embed(api_base, model_name, texts):
            raise RuntimeError("Could not connect to Ollama at http://x.")

        with patch("src.embedder._ollama_embed_batch", side_effect=fake_embed):
            try:
                embed_chunks(chunks, model_name="bge-m3", api_base="http://x", batch_size=1)
                assert False, "expected RuntimeError to propagate"
            except RuntimeError as e:
                assert "Could not connect" in str(e)

    def test_all_chunks_embed_normally_when_no_failure(self):
        chunks = [_chunk("a"), _chunk("b")]
        with patch(
            "src.embedder._ollama_embed_batch",
            return_value=[[1.0], [2.0]],
        ):
            result = embed_chunks(chunks, model_name="bge-m3", api_base="http://x", batch_size=2)
        assert [r["embedding"] for r in result] == [[1.0], [2.0]]
