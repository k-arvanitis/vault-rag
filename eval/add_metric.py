"""Add a single metric to existing results.json without re-running the agent.

Usage:
    uv run python -m eval.add_metric --metric AnswerRelevancy
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = REPO_ROOT / "eval" / "results.json"
METRIC_THRESHOLD = 0.7


def main(metric_name: str) -> None:
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        FaithfulnessMetric,
    )
    from deepeval.test_case import LLMTestCase
    from eval.judge_llm import OpenAICompatibleJudge

    metric_map = {
        "AnswerRelevancy": AnswerRelevancyMetric,
        "Faithfulness": FaithfulnessMetric,
        "ContextualRecall": ContextualRecallMetric,
        "ContextualPrecision": ContextualPrecisionMetric,
    }
    if metric_name not in metric_map:
        raise ValueError(f"Unknown metric '{metric_name}'. Choose from: {list(metric_map)}")

    results: list[dict] = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    judge = OpenAICompatibleJudge()
    print(f"Judge    : {judge.get_model_name()}")
    print(f"Metric   : {metric_name}")
    print(f"Cases    : {len(results)}\n")

    metric_cls = metric_map[metric_name]
    metric = metric_cls(threshold=METRIC_THRESHOLD, model=judge, include_reason=True)
    full_name = metric.__class__.__name__

    scores: list[float] = []

    for idx, case in enumerate(results, start=1):
        question = case["input"]
        expected = case.get("expected_output", "")
        actual = case.get("actual_output", "")
        context = case.get("retrieval_context", [])

        print(f"[{idx}/{len(results)}] {question[:80]}…")

        test_case = LLMTestCase(
            input=question,
            actual_output=actual,
            expected_output=expected,
            retrieval_context=context,
        )

        try:
            metric.measure(test_case)
            score = metric.score
            passed = metric.is_successful()
            reason = getattr(metric, "reason", "")
        except Exception as exc:
            score = 0.0
            passed = False
            reason = str(exc)

        case["metrics"][full_name] = {"score": score, "passed": passed, "reason": reason}
        scores.append(score)
        print(f"  {'✓' if passed else '✗'} {full_name}: {score:.3f}")
        time.sleep(0.1)

    # Save updated results
    RESULTS_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    avg = sum(scores) / len(scores)
    pass_rate = sum(1 for s in scores if s >= METRIC_THRESHOLD) / len(scores)
    print(f"\n{full_name}: avg={avg:.3f}  pass_rate={pass_rate:.1%}")
    print(f"Results updated → {RESULTS_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", required=True, help="Metric name: AnswerRelevancy, Faithfulness, ContextualRecall, ContextualPrecision")
    args = parser.parse_args()
    main(args.metric)
