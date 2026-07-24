"""JSON-file-backed store for admin-provided LLM credentials (bring-your-own-key).

Lets an admin override the env-configured generation provider/key/model at
runtime (no .env edit, no restart -- POST /admin/llm-credentials clears the
agent cache) instead of always using the operator's own keys. One of three
known providers, each with its own (base_url, default_model) -- picking a
provider without knowing this pairing is the easiest way to silently ship a
"model not in catalog" error (reproduced live 2026-07-24 with the unrelated
CHUNK_LLM_MODEL misconfiguration).

ponytail: plaintext on disk, same trust model as .env (which already holds
these same keys in the clear) -- gitignored, not encrypted at rest. No
per-user keys, no rotation.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from src.config import (
    FREE_LLM_API_KEY,
    GROQ_API_KEY,
    LITELLM_MASTER_KEY,
    LLM_CREDENTIALS_PATH,
    OPENROUTER_API_KEY,
)

# Guards the load-modify-save sequence -- same lost-update race as
# feedback_store.py/query_cache.py (FastAPI's threadpool executor can run
# handlers concurrently).
_lock = threading.Lock()

PROVIDERS: dict[str, dict[str, str]] = {
    "groq": {"base_url": "https://api.groq.com/openai/v1", "default_model": "llama-3.3-70b-versatile"},
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "default_model": "meta-llama/llama-3.3-70b-instruct",
    },
    "openai": {"base_url": "https://api.openai.com/v1", "default_model": "gpt-4o-mini"},
}


def _path() -> Path:
    return Path(LLM_CREDENTIALS_PATH)


def _load() -> dict | None:
    path = _path()
    if not path.exists():
        return None
    return json.loads(path.read_text()) or None


def _save(record: dict | None) -> None:
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record or {}, indent=2))


def get_masked() -> dict:
    """Return {provider, model, key_set, key_last4} for the admin UI -- never the full key."""
    with _lock:
        record = _load()
    if not record or not record.get("api_key"):
        return {"provider": None, "model": None, "key_set": False, "key_last4": None}
    key = record["api_key"]
    return {
        "provider": record["provider"],
        "model": record.get("model") or "",
        "key_set": True,
        "key_last4": key[-4:] if len(key) >= 4 else "***",
    }


def set_credentials(provider: str, api_key: str | None, model: str | None) -> None:
    """Save the admin's chosen provider/key/model.

    A blank/omitted api_key keeps the existing stored key -- the GET endpoint
    only ever returns a mask, so the admin form round-trips "unchanged" as
    blank, not the mask itself; treating blank as "clear the key" would wipe
    a working key every time the form is resaved without retyping it.
    """
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider: {provider}")
    with _lock:
        existing = _load() or {}
        key = api_key if api_key else existing.get("api_key", "")
        _save({"provider": provider, "api_key": key, "model": model or ""})


def clear_credentials() -> None:
    """Drop the stored override, reverting to env-configured generation."""
    with _lock:
        _save(None)


def resolve_generation_override() -> tuple[str, str] | None:
    """Return (api_base, model_name) if a BYOK override is set, else None.

    None means "no override" -- the caller (api.py's _get_agent) falls back
    to today's env-configured GENERATION_API_BASE/GENERATION_MODEL exactly
    as before this feature existed.
    """
    with _lock:
        record = _load()
    if not record or not record.get("api_key"):
        return None
    cfg = PROVIDERS[record["provider"]]
    model = record.get("model") or cfg["default_model"]
    return cfg["base_url"], model


def key_for_base(api_base: str) -> str:
    """Resolve the API key for a generation base_url.

    Single resolver for both build_rag_agent's build-time key and
    llm_utils._llm_call's per-call key -- they used to each duplicate this
    base_url-to-key matching independently, which is exactly the kind of
    split that let a BYOK override reach the agent's own answers but silently
    miss the auxiliary calls (multi-turn condense, repair, grounding check)
    that also route through _llm_call. Checked first against the BYOK store
    (matched by base_url, so an active override wins even though its
    base_url isn't one of the hardcoded hosts below), then the existing
    env-key-by-host fallback -- unchanged from before this feature existed.
    """
    with _lock:
        record = _load()
    base = api_base.lower()
    if record and record.get("api_key"):
        cfg = PROVIDERS.get(record["provider"])
        if cfg and cfg["base_url"].lower() in base:
            return record["api_key"]
    if "localhost:4000" in base or "127.0.0.1:4000" in base:
        return LITELLM_MASTER_KEY or "EMPTY"
    if "localhost:3011" in base or "127.0.0.1:3011" in base:
        return FREE_LLM_API_KEY or "EMPTY"
    if "openrouter.ai" in base:
        return OPENROUTER_API_KEY or "EMPTY"
    if "groq.com" in base:
        return GROQ_API_KEY or "EMPTY"
    return GROQ_API_KEY or LITELLM_MASTER_KEY or "EMPTY"
