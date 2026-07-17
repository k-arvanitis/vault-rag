"""Google Drive folder sync connector.

Non-interactive by design: authenticates with a service-account JSON key file,
not an interactive OAuth login flow. Share the target Drive folder with the
service account's email address and it can read files without any user ever
signing in -- there is no login UI to build or secure here. Configure once via
configure(), then call sync() to pull new/changed files into the same ingestion
pipeline manual uploads use.

TODO if a real interactive OAuth flow is ever wanted instead (e.g. syncing a
user's own personal Drive rather than a shared service folder): this would
need a consent-screen redirect, a token exchange endpoint, and secure refresh-
token storage -- a materially different, credential-sensitive feature, out of
scope here.
"""

from __future__ import annotations

import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from src.config import (
    GOOGLE_DRIVE_SERVICE_ACCOUNT_FILE,
    GOOGLE_DRIVE_SYNC_STATE_PATH,
    INPUT_DIR,
    QDRANT_COLLECTION,
    QDRANT_URL,
)

_SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

# Native Google Docs/Sheets have no binary file to download -- export them to
# a format the existing ingestion pipeline already handles.
_MIME_EXPORTS = {
    "application/vnd.google-apps.document": ("application/pdf", ".pdf"),
    "application/vnd.google-apps.spreadsheet": ("text/csv", ".csv"),
}


def _state_path() -> Path:
    """Return the sync-state file path, resolved relative to the repo root."""
    return Path(GOOGLE_DRIVE_SYNC_STATE_PATH)


def _load_state() -> dict[str, Any]:
    """Return the persisted sync state, or fresh defaults if never configured."""
    path = _state_path()
    if not path.exists():
        return {
            "folder_id": None,
            "service_account_file": None,
            "last_synced_at": None,
            "files": {},
        }
    return json.loads(path.read_text())


def _save_state(state: dict[str, Any]) -> None:
    """Persist sync state, creating the parent directory if needed."""
    path = _state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2))


def configure(
    folder_id: str, service_account_file: str | None = None
) -> dict[str, Any]:
    """Persist which Drive folder (and optionally which key file) to sync from."""
    state = _load_state()
    state["folder_id"] = folder_id
    if service_account_file:
        state["service_account_file"] = service_account_file
    _save_state(state)
    return state


def _get_service() -> Any:
    """Build an authenticated Drive API client from the configured service account key."""
    state = _load_state()
    key_path = state.get("service_account_file") or GOOGLE_DRIVE_SERVICE_ACCOUNT_FILE
    if not key_path or not Path(key_path).exists():
        raise RuntimeError(
            "No Google service-account key configured. Set "
            "GOOGLE_DRIVE_SERVICE_ACCOUNT_FILE in .env, or pass "
            "service_account_file to POST /connectors/google-drive/configure."
        )
    creds = service_account.Credentials.from_service_account_file(
        key_path, scopes=_SCOPES
    )
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def _list_drive_files(service: Any, folder_id: str) -> list[dict[str, Any]]:
    """List non-trashed files directly inside the configured folder."""
    files: list[dict[str, Any]] = []
    page_token = None
    query = f"'{folder_id}' in parents and trashed = false"
    while True:
        resp = (
            service.files()
            .list(
                q=query,
                fields="nextPageToken, files(id, name, mimeType, modifiedTime)",
                pageToken=page_token,
            )
            .execute()
        )
        files.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return files


def _download_file(service: Any, drive_file: dict[str, Any], dest_dir: Path) -> Path:
    """Download (or export, for native Google Docs/Sheets) one Drive file to dest_dir."""
    mime = drive_file["mimeType"]
    name = drive_file["name"]
    if mime in _MIME_EXPORTS:
        export_mime, suffix = _MIME_EXPORTS[mime]
        request = service.files().export_media(
            fileId=drive_file["id"], mimeType=export_mime
        )
        if not name.endswith(suffix):
            name = f"{name}{suffix}"
    else:
        request = service.files().get_media(fileId=drive_file["id"])

    dest = dest_dir / name
    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    dest.write_bytes(buf.getvalue())
    return dest


def _ingest_file(path: Path) -> None:
    """Ingest one downloaded file through the same pipeline a manual upload uses."""
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls", ".csv"}:
        from src.ingest_table_rows import ingest_table_rows

        ingest_table_rows(str(path), collection=QDRANT_COLLECTION)
    else:
        from src.ingest import run_ingest

        run_ingest(pdf_path=path, collection=QDRANT_COLLECTION, force_pipeline=None)


def _remove_from_index(filename: str) -> None:
    """Delete a removed file's chunks from the vector store."""
    from src.vector_store import delete_by_file

    delete_by_file(QDRANT_URL, QDRANT_COLLECTION, filename)


def sync(remove_deleted: bool = False) -> dict[str, Any]:
    """Pull new/changed files from the configured Drive folder and ingest them.

    Each file is ingested independently -- one file's failure is recorded in
    its result entry and does not stop the rest of the sync.
    """
    state = _load_state()
    folder_id = state.get("folder_id")
    if not folder_id:
        raise RuntimeError("No Drive folder configured. Call configure() first.")

    service = _get_service()
    Path(INPUT_DIR).mkdir(parents=True, exist_ok=True)
    drive_files = _list_drive_files(service, folder_id)
    seen_ids = set()
    results: list[dict[str, Any]] = []

    for f in drive_files:
        seen_ids.add(f["id"])
        prior = state["files"].get(f["id"])
        if prior and prior.get("modified_time") == f["modifiedTime"]:
            results.append({"name": f["name"], "status": "unchanged"})
            continue
        try:
            dest = _download_file(service, f, Path(INPUT_DIR))
            _ingest_file(dest)
            state["files"][f["id"]] = {
                "name": f["name"],
                "modified_time": f["modifiedTime"],
                "local_path": str(dest),
                "indexed_at": datetime.now(timezone.utc).isoformat(),
            }
            results.append({"name": f["name"], "status": "synced"})
        except Exception as exc:
            results.append({"name": f["name"], "status": "error", "error": str(exc)})

    removed = []
    if remove_deleted:
        for file_id in list(state["files"].keys()):
            if file_id not in seen_ids:
                removed_entry = state["files"].pop(file_id)
                _remove_from_index(removed_entry["name"])
                removed.append(removed_entry["name"])

    state["last_synced_at"] = datetime.now(timezone.utc).isoformat()
    _save_state(state)
    return {
        "synced": results,
        "removed": removed,
        "last_synced_at": state["last_synced_at"],
    }


def status() -> dict[str, Any]:
    """Return the current sync configuration and last-known state."""
    state = _load_state()
    return {
        "configured": bool(state.get("folder_id")),
        "folder_id": state.get("folder_id"),
        "last_synced_at": state.get("last_synced_at"),
        "file_count": len(state.get("files", {})),
    }


def list_files() -> list[dict[str, Any]]:
    """Return the locally tracked sync state for every known Drive file."""
    state = _load_state()
    return [
        {"file_id": file_id, **info} for file_id, info in state.get("files", {}).items()
    ]
