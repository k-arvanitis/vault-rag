"""Tests for src/connectors/google_drive.py -- the Drive API client is always mocked."""

from unittest.mock import MagicMock, patch

import pytest

import src.connectors.google_drive as gdrive


@pytest.fixture(autouse=True)
def _isolated_state_path(tmp_path, monkeypatch):
    """Point the sync-state store at a scratch file so tests never touch real data."""
    monkeypatch.setattr(gdrive, "_state_path", lambda: tmp_path / "state.json")


def test_configure_persists_folder_id():
    result = gdrive.configure("folder123")
    assert result["folder_id"] == "folder123"
    assert gdrive.status()["configured"] is True
    assert gdrive.status()["folder_id"] == "folder123"


def test_status_unconfigured_by_default():
    result = gdrive.status()
    assert result["configured"] is False
    assert result["file_count"] == 0


def test_sync_without_configure_raises():
    with pytest.raises(RuntimeError, match="No Drive folder configured"):
        gdrive.sync()


def test_sync_downloads_and_ingests_new_file(tmp_path):
    gdrive.configure("folder123", service_account_file="/fake/key.json")

    drive_file = {
        "id": "file1",
        "name": "policy.pdf",
        "mimeType": "application/pdf",
        "modifiedTime": "2026-01-01T00:00:00Z",
    }
    mock_service = MagicMock()

    with (
        patch.object(gdrive, "_get_service", return_value=mock_service),
        patch.object(gdrive, "_list_drive_files", return_value=[drive_file]),
        patch.object(gdrive, "_download_file", return_value=tmp_path / "policy.pdf"),
        patch.object(gdrive, "_ingest_file") as mock_ingest,
        patch("src.connectors.google_drive.INPUT_DIR", str(tmp_path)),
    ):
        result = gdrive.sync()

    mock_ingest.assert_called_once()
    assert result["synced"] == [{"name": "policy.pdf", "status": "synced"}]
    files = gdrive.list_files()
    assert len(files) == 1
    assert files[0]["name"] == "policy.pdf"


def test_sync_skips_unchanged_file():
    gdrive.configure("folder123", service_account_file="/fake/key.json")
    drive_file = {
        "id": "file1",
        "name": "policy.pdf",
        "mimeType": "application/pdf",
        "modifiedTime": "2026-01-01T00:00:00Z",
    }

    with (
        patch.object(gdrive, "_get_service", return_value=MagicMock()),
        patch.object(gdrive, "_list_drive_files", return_value=[drive_file]),
        patch.object(gdrive, "_download_file", return_value="ignored") as mock_download,
        patch.object(gdrive, "_ingest_file"),
    ):
        gdrive.sync()  # first sync: downloads and records the file
        result = gdrive.sync()  # second sync: modifiedTime unchanged

    assert result["synced"] == [{"name": "policy.pdf", "status": "unchanged"}]
    mock_download.assert_called_once()  # only the first sync call downloaded


def test_sync_records_per_file_error_without_aborting():
    gdrive.configure("folder123", service_account_file="/fake/key.json")
    drive_file = {
        "id": "file1",
        "name": "bad.pdf",
        "mimeType": "application/pdf",
        "modifiedTime": "2026-01-01T00:00:00Z",
    }

    with (
        patch.object(gdrive, "_get_service", return_value=MagicMock()),
        patch.object(gdrive, "_list_drive_files", return_value=[drive_file]),
        patch.object(gdrive, "_download_file", side_effect=RuntimeError("boom")),
    ):
        result = gdrive.sync()

    assert result["synced"][0]["status"] == "error"
    assert "boom" in result["synced"][0]["error"]


def test_sync_remove_deleted_clears_removed_file_from_index():
    gdrive.configure("folder123", service_account_file="/fake/key.json")
    drive_file = {
        "id": "file1",
        "name": "policy.pdf",
        "mimeType": "application/pdf",
        "modifiedTime": "2026-01-01T00:00:00Z",
    }

    with (
        patch.object(gdrive, "_get_service", return_value=MagicMock()),
        patch.object(gdrive, "_list_drive_files", return_value=[drive_file]),
        patch.object(gdrive, "_download_file", return_value="ignored"),
        patch.object(gdrive, "_ingest_file"),
    ):
        gdrive.sync()

    with (
        patch.object(gdrive, "_get_service", return_value=MagicMock()),
        patch.object(gdrive, "_list_drive_files", return_value=[]),  # file gone from Drive
        patch.object(gdrive, "_remove_from_index") as mock_remove,
    ):
        result = gdrive.sync(remove_deleted=True)

    mock_remove.assert_called_once_with("policy.pdf")
    assert result["removed"] == ["policy.pdf"]
    assert gdrive.list_files() == []
