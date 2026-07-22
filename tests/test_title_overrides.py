"""Tests for src/title_overrides.py."""
import pytest

import src.title_overrides as title_overrides


@pytest.fixture(autouse=True)
def _isolated_path(tmp_path, monkeypatch):
    """Point the store at a scratch file so tests never touch real data."""
    monkeypatch.setattr(title_overrides, "_path", lambda: tmp_path / "title_overrides.json")


def test_set_and_get_override():
    title_overrides.set_title("doc_002_services_contract_terms.pdf", "Services Contract Terms")
    assert title_overrides.get_overrides() == {
        "doc_002_services_contract_terms.pdf": "Services Contract Terms"
    }


def test_set_replaces_existing_override():
    title_overrides.set_title("doc_002.pdf", "First Name")
    title_overrides.set_title("doc_002.pdf", "Second Name")
    assert title_overrides.get_overrides() == {"doc_002.pdf": "Second Name"}


def test_clear_removes_override():
    title_overrides.set_title("doc_002.pdf", "Custom Name")
    title_overrides.clear_title("doc_002.pdf")
    assert title_overrides.get_overrides() == {}


def test_clear_unknown_filename_is_a_noop():
    title_overrides.clear_title("nonexistent.pdf")
    assert title_overrides.get_overrides() == {}


def test_get_overrides_empty_when_no_file():
    assert title_overrides.get_overrides() == {}
