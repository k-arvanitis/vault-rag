"""Google Drive folder sync — SCAFFOLD ONLY, not wired to a real Drive connection.

This defines the data model and call shape a real implementation would fill in.
Deliberately not implemented tonight (2026-07-09): live OAuth needs a Google Cloud
project, a configured OAuth consent screen, client_id/client_secret, a redirect URI,
and a token storage/refresh strategy -- all of which need the user present to set up
and review, not something to build unsupervised overnight.

What a real implementation needs, concretely:
1. OAuth: `google-auth-oauthlib` InstalledAppFlow (desktop) or a web OAuth flow
   (server-side redirect), scope `drive.readonly` at minimum.
2. Token storage: refresh token persisted per connected folder (encrypted at rest,
   not just dropped in a JSON file like feedback/conversation stores are).
3. `list_drive_files(folder_id)`: Drive API v3 `files.list` with
   `q="'{folder_id}' in parents"`, fields `id, name, modifiedTime, md5Checksum`.
4. `detect_changed_files(folder_id)`: diff the current listing's
   `(file_id, modifiedTime or md5Checksum)` against what's stored in the doc
   registry (see `src/file_resolver.py`) to find new/changed/deleted files.
5. `sync_drive_folder(folder_id)`: download changed files to `data/input/`, call the
   existing `run_ingest()` (src/ingest.py) per file, then re-run for changed files and
   call `delete_by_file()` (src/vector_store.py) for files no longer in the folder.
6. Sync status: a small store (same JSON-file pattern as feedback_store.py /
   conversation_store.py) tracking last_synced_at, per-file status, and errors, for
   the "Show sync status in UI" requirement.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DriveFile:
    """One file as reported by the Drive API file listing."""

    file_id: str
    name: str
    modified_time: str
    checksum: str | None


def list_drive_files(folder_id: str) -> list[DriveFile]:
    """Return the current file listing for a connected Drive folder.

    Not implemented -- needs a real OAuth-authenticated Drive API client (see module
    docstring, point 1-3). Raises so nothing silently pretends to sync.
    """
    raise NotImplementedError(
        "Google Drive sync is a scaffold, not implemented. "
        "See src/integrations/drive_sync.py module docstring for what's needed."
    )


def detect_changed_files(folder_id: str, known_files: list[DriveFile]) -> dict[str, list[DriveFile]]:
    """Diff a fresh Drive listing against previously known files.

    Returns {"added": [...], "changed": [...], "removed": [...]}. Not implemented --
    needs list_drive_files() to work first.
    """
    raise NotImplementedError(
        "Google Drive sync is a scaffold, not implemented. "
        "See src/integrations/drive_sync.py module docstring for what's needed."
    )


def sync_drive_folder(folder_id: str) -> dict[str, int]:
    """Download changed files and re-run ingestion; mark removed files as deleted.

    Returns a summary count, e.g. {"added": 0, "changed": 0, "removed": 0}. Not
    implemented -- needs OAuth, file download, and wiring into run_ingest()/
    delete_by_file() (see module docstring, points 4-6).
    """
    raise NotImplementedError(
        "Google Drive sync is a scaffold, not implemented. "
        "See src/integrations/drive_sync.py module docstring for what's needed."
    )
