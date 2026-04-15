import argparse
import json
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv

from src.config import OLLAMA_API_BASE, OLLAMA_EMBED_MODEL, GENERATION_MODEL
from src.retriever import retrieve


def _build_context(chunks: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for i, ch in enumerate(chunks, start=1):
        meta = ch.get("metadata", {}) or {}
        chunk_idx = meta.get("chunk_index", "unknown")
        file_name = meta.get("file_name", "unknown")
        content = (ch.get("content", "") or "").strip()
        lines.append(f"[{i}] file={file_name} chunk_index={chunk_idx}\n{content}")
    return "\n\n".join(lines)


def _ollama_chat(api_base: str, model_name: str, system_prompt: str, user_prompt: str) -> str:
    url = f"{api_base.rstrip('/')}/api/chat"
    payload = {
        "model": model_name,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    req = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=180) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama chat request failed ({exc.code}): {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not connect to Ollama at {api_base}. Ensure `ollama serve` is running.") from exc

    message = body.get("message", {})
    content = message.get("content")
    if not isinstance(content, str):
        raise RuntimeError(f"Unexpected Ollama response: {body}")
    return content.strip()


def generate_answer(
    query: str,
    retrieved_chunks: list[dict[str, Any]],
    api_base: str = OLLAMA_API_BASE,
    model_name: str = GENERATION_MODEL,
) -> dict[str, Any]:
    context = _build_context(retrieved_chunks)
    system_prompt = (
        "You are a precise RAG assistant. Answer only from the provided context. "
        "If the context is insufficient, say you do not know."
    )
    user_prompt = (
        f"Question:\n{query}\n\n"
        f"Context chunks:\n{context}\n\n"
        "Write a concise answer and cite supporting chunks like [1], [2]."
    )

    answer = _ollama_chat(
        api_base=api_base,
        model_name=model_name,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )
    return {"query": query, "answer": answer, "chunks": retrieved_chunks}


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Generate an answer from retrieved chunks using Ollama.")
    parser.add_argument("--query", required=True, help="User query")
    parser.add_argument(
        "--retrieved",
        type=Path,
        default=None,
        help="Optional JSON file containing retrieved chunks",
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=None,
        help="Embeddings JSON path (if set, retrieval runs first)",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Top-k chunks for retrieval")
    parser.add_argument(
        "--ollama-api-base",
        default=OLLAMA_API_BASE,
        help="Ollama base URL",
    )
    parser.add_argument(
        "--embed-model",
        default=OLLAMA_EMBED_MODEL,
        help="Embedding model used for retrieval",
    )
    parser.add_argument(
        "--gen-model",
        default=GENERATION_MODEL,
        help="Generator model",
    )
    args = parser.parse_args()

    if args.retrieved is None and args.embeddings is None:
        raise ValueError("Provide either --retrieved or --embeddings")

    if args.retrieved is not None:
        retrieved_chunks = json.loads(args.retrieved.read_text(encoding="utf-8"))
    else:
        retrieved_chunks = retrieve(
            query=args.query,
            embeddings_path=args.embeddings,
            top_k=args.top_k,
            use_qdrant=False,
            api_base=args.ollama_api_base,
            model_name=args.embed_model,
        )

    result = generate_answer(
        query=args.query,
        retrieved_chunks=retrieved_chunks,
        api_base=args.ollama_api_base,
        model_name=args.gen_model,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
