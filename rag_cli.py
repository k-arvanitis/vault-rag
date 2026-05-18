"""Command-line runner for the Vault RAG agent — for local debugging.

Builds the agent and answers a single --query, optionally streaming. The API,
Slack bot and eval do not use this; they import build_rag_agent directly.

Usage: uv run python rag_cli.py --query "..." [--stream] [--show-tool-uses]
"""
from __future__ import annotations

import argparse
import json

from src.config import (
    GENERATION_API_BASE,
    GENERATION_MODEL,
    QDRANT_COLLECTION,
    QDRANT_URL,
    RERANK_TOP_N,
    RERANKER_MODEL,
    RETRIEVAL_TOP_K,
)
from src.rag_agent import ask_agent, build_rag_agent, stream_agent


def main() -> None:
    """Parse CLI args, build the agent, and print the answer to one question."""
    parser = argparse.ArgumentParser(description="Agentic RAG: unified document + table search.")
    parser.add_argument("--query", required=True, help="User question")
    parser.add_argument("--qdrant-url", default=QDRANT_URL)
    parser.add_argument("--collection", default=QDRANT_COLLECTION)
    parser.add_argument("--top-k", type=int, default=RETRIEVAL_TOP_K)
    parser.add_argument("--rerank-top-n", type=int, default=RERANK_TOP_N)
    parser.add_argument(
        "--reranker-model-name",
        default=RERANKER_MODEL,
        help="Optional local reranker model name. Empty disables reranker.",
    )
    parser.add_argument("--model-name", default=GENERATION_MODEL)
    parser.add_argument("--generation-api-base", default=GENERATION_API_BASE)
    parser.add_argument("--show-tool-uses", action="store_true")
    parser.add_argument("--stream", action="store_true", help="Stream the answer token-by-token.")
    parser.add_argument("--no-hyde", action="store_true", help="Disable HyDE query expansion.")
    args = parser.parse_args()

    agent = build_rag_agent(
        qdrant_url=args.qdrant_url,
        collection=args.collection,
        retrieval_top_k=args.top_k,
        rerank_top_n=args.rerank_top_n,
        reranker_model_name=args.reranker_model_name or None,
        model_name=args.model_name,
        generation_api_base=args.generation_api_base,
        use_hyde=not args.no_hyde,
    )

    if args.stream:
        print(f"[query] {args.query}\n[answer] ", end="", flush=True)
        for token in stream_agent(agent, args.query, show_tool_uses=args.show_tool_uses):
            print(token, end="", flush=True)
        print()
    else:
        answer = ask_agent(agent, args.query, show_tool_uses=args.show_tool_uses)
        print(json.dumps({"query": args.query, "answer": answer}, ensure_ascii=False, indent=2))

    try:
        from langsmith import Client
        Client().flush()
    except Exception:
        pass


if __name__ == "__main__":
    main()
