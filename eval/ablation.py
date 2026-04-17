"""Ablation study: measure the contribution of each retrieval technique.

An ablation study isolates how much each individual component contributes to
overall pipeline quality by systematically removing one component at a time
and measuring the drop in evaluation metrics. The name comes from neuroscience,
where researchers lesion specific brain regions to understand their function.

Here, we run the full evaluation pipeline three times with increasing complexity:

  1. baseline — raw query embedding only; no HyDE, no reranker
  2. +hyde     — adds HyDE query expansion (embed a hypothetical answer
                 instead of the raw question, improving recall on vague queries)
  3. full      — HyDE + cross-encoder reranking (re-scores top-100 candidates
                 with a model that sees both query and chunk together,
                 improving precision); production config

The delta rows in the output table show exactly how many metric points each
component adds — useful for deciding whether the latency cost is worth it.

Usage:
    uv run python -m eval.ablation [--dataset eval/dataset.json]
                                   [--output-dir eval/ablation/]
                                   [--n 50]

Results are saved per-run to <output_dir>/<config_name>_results.json
and a comparison table is printed + saved to <output_dir>/ablation_summary.json.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = REPO_ROOT / "eval" / "dataset.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "eval" / "ablation"

METRIC_THRESHOLD = 0.7

CONFIGS: list[dict] = [
    {"name": "baseline", "use_hyde": False, "use_reranker": False},
    {"name": "+hyde",    "use_hyde": True,  "use_reranker": False},
    {"name": "full",     "use_hyde": True,  "use_reranker": True},
]


def _parse_retrieval_context(tool_content: str) -> list[str]:
    if not tool_content or tool_content.strip() == "No relevant information found.":
        return []
    parts = re.split(r"\n\n(?=\[\d+\])", tool_content.strip())
    return [p.strip() for p in parts if p.strip()]


def run_agent(agent, query: str, system_prompt: str) -> tuple[str, list[str]]:
    from openai import BadRequestError
    from src.config import RERANK_TOP_N

    _input = {"messages": [SystemMessage(content=system_prompt), HumanMessage(content=query)]}
    _config = {"recursion_limit": 20}
    _limits: dict = getattr(agent, "_rag_limits", {})

    try:
        result = agent.invoke(_input, config=_config)
    except BadRequestError as exc:
        err_str = str(exc).lower()
        if "input tokens" in err_str or "context" in err_str or "400" in err_str:
            print(f"  [WARN] Context overflow — retrying with fewer chunks.")
            _limits["rerank_top_n"] = max(3, _limits.get("rerank_top_n", RERANK_TOP_N) // 2)
            try:
                result = agent.invoke(_input, config=_config)
            finally:
                _limits["rerank_top_n"] = RERANK_TOP_N
        else:
            raise
    messages: list = result.get("messages", [])

    retrieval_context: list[str] = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            retrieval_context.extend(_parse_retrieval_context(content))

    actual_output = "No answer generated."
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            text = re.sub(r"(?is)<think>.*?</think>\s*", "", str(msg.content)).strip()
            if text:
                actual_output = text
                break

    return actual_output, retrieval_context


def run_config(
    cfg: dict,
    pairs: list[dict],
    output_path: Path,
) -> dict[str, float]:
    """Run one ablation configuration. Returns {metric_name: avg_score}."""
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        FaithfulnessMetric,
    )
    from deepeval.test_case import LLMTestCase

    from eval.judge_llm import OpenAICompatibleJudge
    from src.config import (
        GENERATION_API_BASE,
        GENERATION_MODEL,
        QDRANT_COLLECTION,
        QDRANT_URL,
        RERANK_TOP_N,
        RERANKER_MODEL,
        RETRIEVAL_TOP_K,
    )
    from src.rag_agent import SYSTEM_PROMPT, build_rag_agent

    reranker = RERANKER_MODEL if cfg["use_reranker"] else None

    print(f"\n{'=' * 60}")
    print(f"  Config: {cfg['name']}")
    print(f"    HyDE={cfg['use_hyde']}  Reranker={bool(reranker)}")
    print(f"{'=' * 60}")

    agent = build_rag_agent(
        qdrant_url=QDRANT_URL,
        collection=QDRANT_COLLECTION,
        retrieval_top_k=RETRIEVAL_TOP_K,
        rerank_top_n=RERANK_TOP_N,
        reranker_model_name=reranker,
        model_name=GENERATION_MODEL,
        generation_api_base=GENERATION_API_BASE,
        use_hyde=cfg["use_hyde"],
    )

    judge = OpenAICompatibleJudge()
    metrics = [
        AnswerRelevancyMetric(threshold=METRIC_THRESHOLD, model=judge, include_reason=False),
        FaithfulnessMetric(threshold=METRIC_THRESHOLD, model=judge, include_reason=False),
        ContextualRecallMetric(threshold=METRIC_THRESHOLD, model=judge, include_reason=False),
        ContextualPrecisionMetric(threshold=METRIC_THRESHOLD, model=judge, include_reason=False),
    ]
    metric_names = [m.__class__.__name__ for m in metrics]
    score_accum: dict[str, list[float]] = {n: [] for n in metric_names}

    results: list[dict] = []
    for idx, pair in enumerate(pairs, start=1):
        question = pair["input"]
        expected = pair["expected_output"]
        context_gt = pair.get("context", [])
        print(f"  [{idx}/{len(pairs)}] {question[:70]}…")

        try:
            actual_output, retrieval_context = run_agent(agent, question, SYSTEM_PROMPT)
        except Exception as exc:
            print(f"    [WARN] Agent failed: {exc}")
            actual_output = ""
            retrieval_context = []

        test_case = LLMTestCase(
            input=question,
            actual_output=actual_output,
            expected_output=expected,
            context=context_gt,
            retrieval_context=retrieval_context,
        )

        case_result: dict = {
            "input": question,
            "actual_output": actual_output,
            "metrics": {},
        }
        for metric in metrics:
            name = metric.__class__.__name__
            try:
                metric.measure(test_case)
                score = metric.score
            except Exception as exc:
                score = 0.0
                print(f"    [WARN] {name} failed: {exc}")
            case_result["metrics"][name] = round(score, 4)
            score_accum[name].append(score)

        results.append(case_result)
        time.sleep(0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Saved → {output_path}")

    return {name: sum(v) / len(v) if v else 0.0 for name, v in score_accum.items()}


def ablation(dataset_path: Path, output_dir: Path, n: int | None) -> None:
    if not dataset_path.exists():
        sys.exit(f"[ERROR] Dataset not found: {dataset_path}\n       Run: uv run python -m eval.generate_dataset")

    pairs: list[dict] = json.loads(dataset_path.read_text(encoding="utf-8"))
    if n:
        pairs = pairs[:n]
    print(f"Loaded {len(pairs)} QA pairs (n={n or 'all'})")

    all_scores: dict[str, dict[str, float]] = {}
    for cfg in CONFIGS:
        out = output_dir / f"{cfg['name']}_results.json"
        scores = run_config(cfg, pairs, out)
        all_scores[cfg["name"]] = scores

    # Print comparison table
    metric_names = list(next(iter(all_scores.values())).keys())
    col_w = 14

    print(f"\n{'=' * (20 + col_w * len(metric_names))}")
    header = f"{'Config':<20}" + "".join(f"{m[-col_w+2:]:>{col_w}}" for m in metric_names)
    print(header)
    print("-" * (20 + col_w * len(metric_names)))
    for cfg_name, scores in all_scores.items():
        row = f"{cfg_name:<20}" + "".join(f"{scores[m]:>{col_w}.3f}" for m in metric_names)
        print(row)
    print("=" * (20 + col_w * len(metric_names)))

    # Delta rows (improvement over baseline)
    baseline = all_scores.get("baseline", {})
    if baseline:
        print("\nDelta vs baseline:")
        for cfg_name, scores in all_scores.items():
            if cfg_name == "baseline":
                continue
            row = f"  {cfg_name:<18}" + "".join(
                f"{scores[m] - baseline[m]:>+{col_w}.3f}" for m in metric_names
            )
            print(row)

    # Save summary
    summary = {
        "n_pairs": len(pairs),
        "threshold": METRIC_THRESHOLD,
        "metric_names": metric_names,
        "configs": all_scores,
    }
    summary_path = output_dir / "ablation_summary.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSummary saved → {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ablation study for the RAG pipeline.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n", type=int, default=None, help="Limit to first N pairs (default: all)")
    args = parser.parse_args()
    ablation(dataset_path=args.dataset, output_dir=args.output_dir, n=args.n)
