"use client";

import { useCallback, useEffect, useState } from "react";
import { X, RefreshCw, Loader2 } from "lucide-react";
import {
  configureDrive,
  getDriveFiles,
  getDriveStatus,
  syncDrive,
  type DriveFile,
  type DriveStatus,
} from "@/lib/api";

interface Props {
  onClose: () => void;
}

export default function GoogleDrivePanel({ onClose }: Props) {
  const [status, setStatus] = useState<DriveStatus | null>(null);
  const [files, setFiles] = useState<DriveFile[]>([]);
  const [folderId, setFolderId] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [removeDeleted, setRemoveDeleted] = useState(false);

  const load = useCallback(() => {
    getDriveStatus()
      .then((s) => {
        setStatus(s);
        if (s.folder_id) setFolderId(s.folder_id);
        setError(null);
      })
      .catch((e) => setError(e instanceof Error ? e.message : "Failed to load status"));
    getDriveFiles()
      .then(setFiles)
      .catch(() => setFiles([]));
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const handleConfigure = async () => {
    if (!folderId.trim()) return;
    setBusy(true);
    setError(null);
    try {
      await configureDrive(folderId.trim());
      load();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to configure folder");
    } finally {
      setBusy(false);
    }
  };

  const handleSync = async () => {
    setBusy(true);
    setError(null);
    try {
      await syncDrive(removeDeleted);
      load();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Sync failed");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="fixed inset-0 z-30 flex">
      <div className="flex h-full w-full flex-col bg-ink-50">
        <div className="flex shrink-0 items-center gap-3 border-b border-ink-200 bg-surface px-5 py-3">
          <div className="min-w-0 flex-1">
            <p className="text-xs font-semibold text-ink-800">Google Drive sync</p>
            <p className="text-[10px] text-ink-400">
              {status?.configured
                ? `Folder ${status.folder_id} · ${status.file_count} file(s) tracked`
                : "Not configured"}
            </p>
          </div>
          <button
            onClick={onClose}
            className="rounded-md p-1.5 text-ink-500 transition-colors hover:bg-ink-100 hover:text-ink-800"
            aria-label="Close Google Drive panel"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-5">
          <div className="mx-auto max-w-xl">
            {error && <p className="mb-3 text-sm text-red-600">{error}</p>}

            <div className="rounded-lg border border-ink-200 bg-surface p-3">
              <label className="text-[11px] font-medium text-ink-600">Drive folder ID</label>
              <div className="mt-1.5 flex gap-2">
                <input
                  value={folderId}
                  onChange={(e) => setFolderId(e.target.value)}
                  placeholder="e.g. 1AbCdEfGhIjKlMnOpQrStUvWxYz"
                  className="flex-1 rounded-md border border-ink-200 bg-surface px-2 py-1.5 text-xs text-ink-700 placeholder:text-ink-400 focus:outline-none focus:ring-1 focus:ring-ink-300"
                />
                <button
                  onClick={handleConfigure}
                  disabled={busy || !folderId.trim()}
                  className="rounded-md border border-ink-200 px-3 py-1.5 text-xs font-medium text-ink-600 transition-colors hover:bg-ink-100 disabled:opacity-50"
                >
                  Save
                </button>
              </div>
              <p className="mt-2 text-[10px] text-ink-400">
                Share the folder with the configured service-account email (read access) —
                see .env.example for GOOGLE_DRIVE_SERVICE_ACCOUNT_FILE. No login required.
              </p>
            </div>

            <div className="mt-3 flex items-center justify-between rounded-lg border border-ink-200 bg-surface p-3">
              <label className="flex items-center gap-2 text-[11px] text-ink-600">
                <input
                  type="checkbox"
                  checked={removeDeleted}
                  onChange={(e) => setRemoveDeleted(e.target.checked)}
                />
                Remove documents deleted from Drive
              </label>
              <button
                onClick={handleSync}
                disabled={busy || !status?.configured}
                className="flex items-center gap-1.5 rounded-md border border-ink-200 px-3 py-1.5 text-xs font-medium text-ink-600 transition-colors hover:bg-ink-100 disabled:opacity-50"
              >
                {busy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <RefreshCw className="h-3.5 w-3.5" />}
                Sync now
              </button>
            </div>

            {status?.last_synced_at && (
              <p className="mt-2 text-[10px] text-ink-400">
                Last synced: {new Date(status.last_synced_at).toLocaleString()}
              </p>
            )}

            <div className="mt-4">
              <p className="mb-1.5 text-[11px] font-medium text-ink-600">Tracked files</p>
              {files.length === 0 ? (
                <p className="text-xs text-ink-400">No files synced yet.</p>
              ) : (
                <div className="grid gap-1.5">
                  {files.map((f) => (
                    <div
                      key={f.file_id}
                      className="rounded-md border border-ink-200 bg-surface px-2.5 py-1.5 text-[11px] text-ink-700"
                    >
                      <span className="font-medium">{f.name}</span>{" "}
                      <span className="text-ink-400">
                        — indexed {new Date(f.indexed_at).toLocaleString()}
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
