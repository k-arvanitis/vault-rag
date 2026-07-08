"""Tests for src/integrations/drive_sync.py -- a scaffold, not a real implementation."""
import pytest

from src.integrations.drive_sync import (
    detect_changed_files,
    list_drive_files,
    sync_drive_folder,
)


def test_list_drive_files_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        list_drive_files("some-folder-id")


def test_detect_changed_files_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        detect_changed_files("some-folder-id", [])


def test_sync_drive_folder_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        sync_drive_folder("some-folder-id")
