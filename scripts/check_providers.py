#!/usr/bin/env python3
"""Quick sanity check: ping all providers listed in litellm_config.yaml.

Usage:
    uv run python scripts/check_providers.py

Requires .env with the API keys referenced by the config.
"""

import os
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv
from openai import OpenAI

ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
CONFIG_PATH = Path(__file__).resolve().parents[1] / "litellm_config.yaml"
load_dotenv(ENV_PATH)

PROMPT = "Say 'pong' and nothing else."

# Map LiteLLM provider prefix → OpenAI-compatible base URL
BASE_URLS = {
    "groq": "https://api.groq.com/openai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}


def parse_config(path: Path) -> list[dict]:
    """Read litellm_config.yaml and return a list of provider dicts."""
    raw = yaml.safe_load(path.read_text())
    providers = []
    for entry in raw.get("model_list", []):
        params = entry.get("litellm_params", {})
        model = params.get("model", "")
        if "/" not in model:
            continue
        prefix, actual_model = model.split("/", 1)
        api_key_env = (
            params.get("api_key", "").removeprefix("os.environ/")
            if params.get("api_key", "").startswith("os.environ/")
            else None
        )
        base_url = params.get("api_base") or BASE_URLS.get(prefix)
        providers.append(
            {
                "name": prefix.capitalize(),
                "base_url": base_url,
                "model": actual_model,
                "api_key_env": api_key_env,
            }
        )
    return providers


def check_provider(cfg: dict) -> dict:
    """Ping a single provider and return timing + response metadata."""
    key = os.getenv(cfg["api_key_env"]) if cfg["api_key_env"] else None
    if not key:
        return {"ok": False, "error": f"missing {cfg['api_key_env']}"}

    client = OpenAI(base_url=cfg["base_url"], api_key=key)
    t0 = time.perf_counter()
    try:
        resp = client.chat.completions.create(
            model=cfg["model"],
            messages=[{"role": "user", "content": PROMPT}],
            max_tokens=10,
            temperature=0,
        )
        latency = time.perf_counter() - t0
        content = resp.choices[0].message.content or ""
        return {
            "ok": True,
            "latency_ms": round(latency * 1000, 1),
            "content": content.strip(),
            "model_used": resp.model,
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


def main() -> None:
    providers = parse_config(CONFIG_PATH)
    if not providers:
        print("No providers found in litellm_config.yaml")
        return

    print("Provider sanity check (from litellm_config.yaml)")
    print("=" * 50)
    all_ok = True
    for cfg in providers:
        result = check_provider(cfg)
        status = "✅" if result["ok"] else "❌"
        print(f"\n{status} {cfg['name']}  ({cfg['model']})")
        if result["ok"]:
            print(f"   latency : {result['latency_ms']} ms")
            print(f"   response: {result['content']!r}")
            print(f"   model   : {result['model_used']}")
        else:
            print(f"   error   : {result['error']}")
            all_ok = False
    print("\n" + "=" * 50)
    print("All green ✅" if all_ok else "Some providers failed ❌")


if __name__ == "__main__":
    main()
