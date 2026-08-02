"""Tests for src/usage_log.py."""

from datetime import datetime, timezone
from unittest.mock import patch

import pytest

import vault_rag.usage_log as usage_log


@pytest.fixture(autouse=True)
def _isolated_usage_path(tmp_path, monkeypatch):
    """Point the usage log at a scratch file so tests never touch real data."""
    monkeypatch.setattr(usage_log, "_path", lambda: tmp_path / "usage_log.json")


def test_log_skips_zero_usage():
    usage_log.log(
        "q1", "model-a", {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    )
    assert usage_log.stats()["total_questions"] == 0


def test_log_and_stats_aggregate():
    usage = {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}
    usage_log.log("q1", "qwen/qwen3-32b", usage, latency_ms=1000)
    usage_log.log("q2", "qwen/qwen3-32b", usage, latency_ms=2000)
    result = usage_log.stats()
    assert result["total_questions"] == 2
    assert result["total_tokens"] == 300
    assert len(result["daily"]) == 1
    assert result["daily"][0]["questions"] == 2
    assert result["daily"][0]["avg_latency_ms"] == 1500
    assert result["recent"][0]["question"] == "q2"  # newest first


def test_stats_handles_missing_latency():
    usage = {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20}
    usage_log.log("q1", "model-a", usage)
    result = usage_log.stats()
    assert result["daily"][0]["avg_latency_ms"] is None
    assert result["recent"][0]["latency_ms"] is None


def test_daily_includes_today_zeroed_when_last_activity_was_a_prior_day():
    """Reproduced live: the admin panel's "Today" card reads daily[0], which
    used to be whichever date last had a logged question -- with zero
    questions asked today, that silently showed yesterday's numbers
    mislabeled as "Today." daily[0] must be today's real date even with
    nothing logged for it yet."""
    usage = {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20}
    with patch("vault_rag.usage_log.datetime") as mock_dt:
        mock_dt.now.return_value = datetime(2026, 7, 23, 18, 0, tzinfo=timezone.utc)
        mock_dt.now.side_effect = None
        usage_log.log("q1", "model-a", usage)

    with patch("vault_rag.usage_log.datetime") as mock_dt:
        mock_dt.now.return_value = datetime(2026, 7, 24, 9, 0, tzinfo=timezone.utc)
        result = usage_log.stats()

    assert result["daily"][0]["date"] == "2026-07-24"
    assert result["daily"][0]["questions"] == 0
    assert result["daily"][1]["date"] == "2026-07-23"
    assert result["daily"][1]["questions"] == 1


def test_estimate_cost_unknown_model_is_free():
    assert usage_log.estimate_cost("some-local-model", 1_000_000, 1_000_000) == 0.0


def test_estimate_cost_known_model():
    cost = usage_log.estimate_cost("qwen/qwen3-32b", 1_000_000, 1_000_000)
    assert cost == pytest.approx(0.29 + 0.59)
