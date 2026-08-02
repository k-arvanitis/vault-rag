"""Tests for src/llm_credentials.py."""

import pytest

import vault_rag.llm_credentials as llm_credentials


@pytest.fixture(autouse=True)
def _isolated_path(tmp_path, monkeypatch):
    """Point the credentials store at a scratch file so tests never touch real data."""
    monkeypatch.setattr(llm_credentials, "_path", lambda: tmp_path / "llm_credentials.json")


class TestStoreRoundTrip:
    def test_set_then_get_masked_round_trips_provider_and_model(self):
        llm_credentials.set_credentials("groq", "sk-real-secret-key", "llama-3.3-70b-versatile")
        masked = llm_credentials.get_masked()
        assert masked["provider"] == "groq"
        assert masked["model"] == "llama-3.3-70b-versatile"
        assert masked["key_set"] is True

    def test_unknown_provider_rejected(self):
        with pytest.raises(ValueError):
            llm_credentials.set_credentials("anthropic", "sk-key", None)

    def test_clear_credentials_reverts_to_unset(self):
        llm_credentials.set_credentials("openai", "sk-key", None)
        llm_credentials.clear_credentials()
        masked = llm_credentials.get_masked()
        assert masked["key_set"] is False
        assert masked["provider"] is None


class TestMaskedGetNeverReturnsFullKey:
    def test_key_last4_only(self):
        llm_credentials.set_credentials("openrouter", "sk-abcdefgh1234", None)
        masked = llm_credentials.get_masked()
        assert masked["key_last4"] == "1234"
        assert "sk-abcdefgh1234" not in str(masked)

    def test_no_credentials_set_reports_unset(self):
        masked = llm_credentials.get_masked()
        assert masked == {"provider": None, "model": None, "key_set": False, "key_last4": None}


class TestBlankKeyKeepsExisting:
    def test_blank_api_key_on_resave_keeps_the_real_key(self):
        """The admin GET only ever returns a mask, so the edit form round-trips
        blank for "unchanged" -- saving with a blank key must not wipe the
        working key."""
        llm_credentials.set_credentials("groq", "sk-original-key", "llama-3.3-70b-versatile")
        llm_credentials.set_credentials("groq", None, "llama-3.3-70b-versatile")
        masked = llm_credentials.get_masked()
        assert masked["key_last4"] == "-key"

    def test_blank_api_key_can_still_change_provider_and_model(self):
        llm_credentials.set_credentials("groq", "sk-original-key", None)
        llm_credentials.set_credentials("openai", "", "gpt-4o-mini")
        masked = llm_credentials.get_masked()
        assert masked["provider"] == "openai"
        assert masked["model"] == "gpt-4o-mini"
        assert masked["key_last4"] == "-key"


class TestResolveGenerationOverride:
    def test_no_credentials_returns_none(self):
        assert llm_credentials.resolve_generation_override() is None

    def test_credentials_set_returns_provider_base_and_model(self):
        llm_credentials.set_credentials("openai", "sk-key", "gpt-4o-mini")
        result = llm_credentials.resolve_generation_override()
        assert result == ("https://api.openai.com/v1", "gpt-4o-mini")

    def test_credentials_set_without_model_uses_provider_default(self):
        llm_credentials.set_credentials("groq", "sk-key", None)
        base, model = llm_credentials.resolve_generation_override()
        assert base == llm_credentials.PROVIDERS["groq"]["base_url"]
        assert model == llm_credentials.PROVIDERS["groq"]["default_model"]


class TestKeyForBase:
    def test_store_override_wins_when_base_matches(self):
        llm_credentials.set_credentials("openai", "sk-byok-key", None)
        assert llm_credentials.key_for_base("https://api.openai.com/v1") == "sk-byok-key"

    def test_falls_back_to_env_key_when_no_override(self):
        # No store set -- should fall through to the groq.com env-key branch
        # without raising, regardless of what GROQ_API_KEY actually is.
        result = llm_credentials.key_for_base("https://api.groq.com/openai/v1")
        assert isinstance(result, str) and result

    def test_store_override_does_not_leak_to_unrelated_base(self):
        """An OpenAI override must not be handed to a call against a
        different host (e.g. the local proxy) -- only used when the
        provider's own base_url is actually being called."""
        llm_credentials.set_credentials("openai", "sk-byok-key", None)
        result = llm_credentials.key_for_base("http://127.0.0.1:3011/v1")
        assert result != "sk-byok-key"
