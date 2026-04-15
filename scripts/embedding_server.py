"""Local embedding server for models not available in Ollama.

Exposes two OpenAI/Ollama-compatible endpoints so existing code needs no changes:
  POST /api/embed        — Ollama format  (used by embedder.py, retriever.py)
  POST /v1/embeddings    — OpenAI format  (used by ingest_table_rows.py)

Usage:
  python scripts/embedding_server.py --model intfloat/multilingual-e5-large --port 11435
"""

import argparse
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

_model: SentenceTransformer | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # model is loaded before uvicorn starts, nothing to do here
    yield


app = FastAPI(lifespan=lifespan)


class OllamaEmbedRequest(BaseModel):
    model: str
    input: str | list[str]


class OpenAIEmbedRequest(BaseModel):
    model: str
    input: str | list[str]


def _encode(texts: list[str]) -> list[list[float]]:
    return _model.encode(texts, normalize_embeddings=True).tolist()


@app.post("/api/embed")
def ollama_embed(req: OllamaEmbedRequest):
    """Ollama-compatible endpoint used by embedder.py and retriever.py."""
    texts = [req.input] if isinstance(req.input, str) else req.input
    return {"embeddings": _encode(texts), "model": req.model}


@app.post("/v1/embeddings")
def openai_embed(req: OpenAIEmbedRequest):
    """OpenAI-compatible endpoint used by ingest_table_rows.py."""
    texts = [req.input] if isinstance(req.input, str) else req.input
    vecs = _encode(texts)
    return {
        "object": "list",
        "model": req.model,
        "data": [{"object": "embedding", "embedding": v, "index": i} for i, v in enumerate(vecs)],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="intfloat/multilingual-e5-large")
    parser.add_argument("--port", type=int, default=11435)
    args = parser.parse_args()

    print(f"Loading {args.model} ...")
    _model = SentenceTransformer(args.model)
    dim = _model.get_sentence_embedding_dimension()
    print(f"Model ready — dim={dim}, serving on port {args.port}")

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")
