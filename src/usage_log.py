"""JSON-file-backed log of per-question LLM token usage, for the admin usage panel.

Same load/modify/save-under-lock pattern as query_cache.py/feedback_store.py.
Cost is a rough estimate from a static price table (see config.py) -- not a
substitute for the provider's actual bill.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

from src.config import USAGE_LOG_PATH, USAGE_PRICE_PER_1M_TOKENS

_lock = threading.Lock()


def _path() -> Path:
    return Path(USAGE_LOG_PATH)


def _load() -> list[dict]:
    path = _path()
    if not path.exists():
        return []
    return json.loads(path.read_text())


def _save(entries: list[dict]) -> None:
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2))


def estimate_cost(model: str | None, input_tokens: int, output_tokens: int) -> float:
    """Rough USD cost from the static price table; $0 for an unlisted model."""
    price_in, price_out = USAGE_PRICE_PER_1M_TOKENS.get(model or "", (0.0, 0.0))
    return (input_tokens / 1_000_000) * price_in + (output_tokens / 1_000_000) * price_out


def log(
    question: str, model: str | None, usage: dict, latency_ms: float | None = None
) -> None:
    """Append one usage entry. No-ops if usage has no tokens (nothing to log)."""
    total = usage.get("total_tokens", 0)
    if not total:
        return
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question[:200],
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total,
        "cost_usd": round(estimate_cost(model, input_tokens, output_tokens), 6),
        "latency_ms": round(latency_ms) if latency_ms is not None else None,
    }
    with _lock:
        entries = _load()
        entries.append(entry)
        _save(entries)


def stats() -> dict:
    """Aggregate the log into recent per-question entries plus daily totals."""
    with _lock:
        entries = _load()

    daily: dict[str, dict] = {}
    for e in entries:
        day = e["timestamp"][:10]
        d = daily.setdefault(
            day,
            {
                "date": day,
                "questions": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
                "_latency_sum": 0.0,
                "_latency_count": 0,
            },
        )
        d["questions"] += 1
        d["total_tokens"] += e["total_tokens"]
        d["cost_usd"] += e["cost_usd"]
        if e.get("latency_ms") is not None:
            d["_latency_sum"] += e["latency_ms"]
            d["_latency_count"] += 1

    for d in daily.values():
        d["cost_usd"] = round(d["cost_usd"], 6)
        d["avg_latency_ms"] = (
            round(d["_latency_sum"] / d["_latency_count"]) if d["_latency_count"] else None
        )
        del d["_latency_sum"]
        del d["_latency_count"]

    # The admin panel's "Today" card reads daily[0], relying on it being
    # the current date -- but `daily` only contains dates that actually
    # logged a question, so daily[0] silently became "yesterday" (or
    # whenever the log was last written) on any day with zero questions
    # so far, mislabeled as "Today." Force today's real date to exist,
    # zeroed if nothing has been logged for it yet.
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if today not in daily:
        daily[today] = {
            "date": today,
            "questions": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
            "avg_latency_ms": None,
        }

    return {
        "recent": list(reversed(entries[-50:])),
        "daily": sorted(daily.values(), key=lambda d: d["date"], reverse=True),
        "total_questions": len(entries),
        "total_tokens": sum(e["total_tokens"] for e in entries),
        "total_cost_usd": round(sum(e["cost_usd"] for e in entries), 6),
    }
