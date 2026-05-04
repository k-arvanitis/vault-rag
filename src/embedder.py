import argparse
import json
import os
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import tiktoken
from dotenv import load_dotenv

from src.config import EMBED_API_BASE, OLLAMA_EMBED_MODEL

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_chunks(input_path: Path) -> list[dict[str, Any]]:
    raw = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Expected a list in {input_path}, got {type(raw).__name__}")
    return raw


_MAX_EMBED_CHARS = int(os.getenv("MAX_EMBED_CHARS", "8000"))
_MODEL_MAX_EMBED_CHARS = {
    "mxbai-embed-large:latest": 4000,
    "mxbai-embed-large": 4000,
    "nomic-embed-text:latest": 8000,
    "nomic-embed-text": 8000,
}
_MODEL_MAX_EMBED_TOKENS = {
    "mxbai-embed-large:latest": 512,
    "mxbai-embed-large": 512,
    "nomic-embed-text:latest": 2048,
    "nomic-embed-text": 2048,
}
_TOKENIZER = tiktoken.get_encoding("cl100k_base")


def _text_for_embedding(chunk: dict[str, Any], model_name: str) -> str:
    vector_text = chunk.get("vector_text")
    content = chunk.get("content", "")
    text = ""
    if isinstance(vector_text, str) and vector_text.strip():
        text = vector_text
    elif isinstance(content, str) and content.strip():
        text = content
    max_chars = _MODEL_MAX_EMBED_CHARS.get(model_name, _MAX_EMBED_CHARS)
    truncated = text[:max_chars]
    max_tokens = _MODEL_MAX_EMBED_TOKENS.get(model_name)
    if max_tokens is None:
        return truncated
    token_ids = _TOKENIZER.encode(truncated)
    if len(token_ids) <= max_tokens:
        return truncated
    return _TOKENIZER.decode(token_ids[:max_tokens])


def _batched(items: list[Any], batch_size: int) -> list[list[Any]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def _ollama_embed_batch(api_base: str, model_name: str, texts: list[str]) -> list[list[float]]:
    url = f"{api_base.rstrip('/')}/api/embed"
    payload = json.dumps({"model": model_name, "input": texts}).encode("utf-8")
    req = Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(req, timeout=300) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama embed request failed ({exc.code}): {detail}") from exc
    except URLError as exc:
        raise RuntimeError(
            f"Could not connect to Ollama at {api_base}. Ensure `ollama serve` is running."
        ) from exc

    embeddings = body.get("embeddings")
    if not isinstance(embeddings, list):
        raise RuntimeError(f"Unexpected Ollama response: {body}")
    return embeddings


def embed_chunks(
    chunks: list[dict[str, Any]],
    model_name: str,
    api_base: str,
    batch_size: int = 32,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    texts = [_text_for_embedding(chunk, model_name) for chunk in chunks]
    payload: list[dict[str, Any]] = []

    def _embed_with_fallback(batch: list[str]) -> list[list[float]]:
        try:
            return _ollama_embed_batch(api_base=api_base, model_name=model_name, texts=batch)
        except RuntimeError:
            if len(batch) == 1:
                raise
            mid = max(1, len(batch) // 2)
            return _embed_with_fallback(batch[:mid]) + _embed_with_fallback(batch[mid:])

    for batch_index, batch in enumerate(_batched(texts, batch_size), start=1):
        embeddings = _embed_with_fallback(batch)
        if verbose:
            print(f"Embedded batch {batch_index} ({len(batch)} chunks)")
        base_idx = (batch_index - 1) * batch_size
        for idx, embedding in enumerate(embeddings):
            source_chunk = chunks[base_idx + idx]
            payload.append(
                {
                    "id": str(source_chunk.get("id") or source_chunk.get("metadata", {}).get("chunk_index", len(payload))),
                    "embedding": embedding,
                    "content": source_chunk.get("content", ""),
                    "vector_text": batch[idx],
                    "metadata": source_chunk.get("metadata", {}),
                }
            )

    return payload


def embed_chunks_file(
    input_path: Path,
    output_path: Path | None = None,
    api_base: str = EMBED_API_BASE,
    model_name: str = OLLAMA_EMBED_MODEL,
    batch_size: int = 32,
    verbose: bool = True,
) -> Path:
    chunks = _load_chunks(input_path)
    embeddings = embed_chunks(
        chunks,
        model_name=model_name,
        api_base=api_base,
        batch_size=batch_size,
        verbose=verbose,
    )

    if output_path is None:
        output_path = REPO_ROOT / "data/output/embeddings" / f"{input_path.stem}_embeddings.json"
    elif not output_path.is_absolute():
        output_path = REPO_ROOT / output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(embeddings, ensure_ascii=False), encoding="utf-8")
    if verbose:
        print(f"Saved embeddings: {output_path} ({len(embeddings)} vectors)")
    return output_path


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Embed chunk JSON with an Ollama embedding model.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/output/chunks/chunks.json"),
        help="Path to chunk JSON file produced by src/chunker.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file path. Default: <repo_root>/data/output/embeddings/<input_stem>_embeddings.json",
    )
    parser.add_argument(
        "--api-base",
        default=EMBED_API_BASE,
        help="Embedding server base URL",
    )
    parser.add_argument(
        "--model-name",
        default=OLLAMA_EMBED_MODEL,
        help="Ollama embedding model name",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of chunks per embeddings request",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input chunk file not found: {args.input}")

    output_path = args.output

    embed_chunks_file(
        input_path=args.input,
        output_path=output_path,
        api_base=args.api_base,
        model_name=args.model_name,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
