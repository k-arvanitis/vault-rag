"""Cross-encoder / generative rerankers for second-stage retrieval ranking.

Provides two reranker classes that score (query, document) pairs more
accurately than the first-stage dense/sparse Qdrant search:
  - BGEReranker: a classification cross-encoder (BAAI/bge-reranker-*)
  - QwenReranker: a generative model scored on yes/no token probabilities

Called by: src/tools/retrieval_tool.py (_fetch_docs uses a QwenReranker
instance to rerank candidate chunks before formatting them for the agent).
Calls into: transformers (model + tokenizer loading), torch, and the
RERANKER_MODEL setting from src/config.py.
"""

import torch
from langsmith import traceable
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from src.config import RERANKER_MODEL

# ---------------------------------------------------------------------------
# BGE cross-encoder reranker — single forward pass scores all pairs at once
# ---------------------------------------------------------------------------


class BGEReranker:
    """Cross-encoder reranker for BAAI/bge-reranker-* models."""

    def __init__(self, model_name: str = RERANKER_MODEL, device: str = "cuda"):
        # Pick GPU when available, otherwise fall back to CPU.
        self.device = device if torch.cuda.is_available() else "cpu"
        # Load the tokenizer and the sequence-classification model (fp16 on GPU,
        # fp32 on CPU), move it to the device and put it in eval mode.
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            )
            .to(self.device)
            .eval()
        )

    @traceable(name="bge-reranker", metadata={"model": RERANKER_MODEL})
    @torch.no_grad()
    def rerank(
        self, query: str, documents: list[str], top_n: int = 10, **_
    ) -> list[dict]:
        # Pair the query with every document and tokenize the whole batch.
        pairs = [[query, doc] for doc in documents]
        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)
        # Run the model — logits are the raw relevance scores (one per pair).
        scores = self.model(**inputs).logits.squeeze(-1).float().cpu().tolist()
        if isinstance(scores, float):
            scores = [scores]
        # Attach each score to its original index and sort by relevance.
        results = sorted(
            [
                {"index": i, "text": doc, "score": s}
                for i, (doc, s) in enumerate(zip(documents, scores))
            ],
            key=lambda x: x["score"],
            reverse=True,
        )
        return results[:top_n]


# ---------------------------------------------------------------------------
# Qwen generative reranker — scores each doc via yes/no token probability
# ---------------------------------------------------------------------------


class QwenReranker:
    """Generative reranker using Qwen3 token-probability scoring (yes/no)."""

    def __init__(
        self, model_name: str = "Qwen/Qwen3-Reranker-0.6B", device: str = "cuda"
    ) -> None:
        """0.6B is fast for CPU/low-end GPU; 4B or 7B for complex reasoning."""
        # Pick GPU when available, otherwise fall back to CPU.
        self.device = device if torch.cuda.is_available() else "cpu"
        # Load tokenizer (left padding, required for causal-LM batching) and the
        # causal-LM model placed on the chosen device, in eval mode.
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", device_map=self.device
        ).eval()

        # Cache the vocab IDs of the "yes" / "no" tokens — relevance scoring
        # later compares only these two logits.
        self.yes_token_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.no_token_id = self.tokenizer.convert_tokens_to_ids("no")

        # Prebuild the Qwen3 chat-template fragments wrapped around each pair.
        self.system_prompt = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
        self.user_prefix = "<|im_start|>user\n"
        self.assistant_suffix = (
            "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )

    def _format_input(self, query: str, doc: str, instruction: str) -> str:
        """Wrap one (query, document) pair in the Qwen3 judge prompt template."""
        # Assemble the full prompt: system judge instruction + the instruct/
        # query/document triple + the assistant turn the model completes.
        return (
            f"{self.system_prompt}{self.user_prefix}"
            f"<Instruct>: {instruction}\n"
            f"<Query>: {query}\n"
            f"<Document>: {doc}{self.assistant_suffix}"
        )

    @traceable(name="qwen-reranker")
    @torch.no_grad()
    def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int = 5,
        instruction: str = "Given a search query, retrieve relevant passages that answer the query.",
    ) -> list[dict]:
        """
        Scores a list of documents against a query.
        Returns: List of dictionaries containing the original index, text, and relevance score.
        """
        scored_results = []

        # Score each document independently with one forward pass.
        for idx, doc in enumerate(documents):
            # Build the judge prompt for this pair and tokenize it.
            text_to_score = self._format_input(query, doc, instruction)
            inputs = self.tokenizer(
                text_to_score, return_tensors="pt", truncation=True, max_length=512
            ).to(self.model.device)

            # Run the model and take the logits at the last position — the slot
            # where the model would emit "yes" or "no".
            outputs = self.model(**inputs)
            # We look at the logits of the last generated token
            logits = outputs.logits[:, -1, :]

            # Extract only the 'yes' and 'no' logits
            # Softmax over just those two logits gives a normalized relevance prob.
            relevant_logits = torch.stack(
                (logits[0, self.no_token_id], logits[0, self.yes_token_id])
            )
            probs = torch.nn.functional.softmax(relevant_logits, dim=0)

            # The 'yes' probability is our relevance score (0.0 to 1.0)
            score = probs[1].item()
            scored_results.append({"index": idx, "text": doc, "score": score})

        # Sort by score descending and return the top_n most relevant.
        scored_results.sort(key=lambda x: x["score"], reverse=True)
        return scored_results[:top_n]


# ---------------------------------------------------------------------------
# Usage example — manual smoke test when run as a script
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Initialize once
    ranker = QwenReranker()

    # Sample data (usually coming from Qdrant search results)
    query = "How do I scale Qdrant?"
    docs = [
        "Qdrant is a vector database written in Rust.",
        "To scale Qdrant, you can use distributed mode with multiple nodes and sharding.",
        "You can install Qdrant using Docker with a simple run command.",
    ]

    results = ranker.rerank(query, docs, top_n=2)

    for i, res in enumerate(results):
        print(f"Rank {i + 1} (Score: {res['score']:.4f}): {res['text']}")
