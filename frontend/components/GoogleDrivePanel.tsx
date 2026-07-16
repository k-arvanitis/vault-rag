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
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Skeleton } from "@/components/ui/skeleton";

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
    <div className="flex h-full w-full">
      <div className="flex h-full w-full flex-col bg-background">
        <div className="flex shrink-0 items-center gap-3 border-b border-border bg-card px-5 py-3">
          <div className="min-w-0 flex-1">
            <p className="text-xs font-semibold text-foreground">Google Drive sync</p>
            <div className="mt-0.5 flex items-center gap-1.5">
              {status ? (
                status.configured ? (
                  <>
                    <Badge variant="default">Connected</Badge>
                    <span className="text-[10px] text-muted-foreground">
                      Folder {status.folder_id} · {status.file_count} file(s) tracked
                    </span>
                  </>
                ) : (
                  <Badge variant="outline">Not configured</Badge>
                )
              ) : (
                <Skeleton className="h-4 w-32" />
              )}
            </div>
          </div>
          <Button variant="ghost" size="icon-sm" onClick={onClose} aria-label="Close Google Drive panel">
            <X />
          </Button>
        </div>

        <div className="flex-1 overflow-y-auto p-5">
          <div className="mx-auto max-w-xl space-y-3">
            {error && (
              <Alert variant="destructive">
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}

            <Card size="sm">
              <CardContent className="space-y-1.5">
                <label className="text-[11px] font-medium text-muted-foreground">Drive folder ID</label>
                <div className="flex gap-2">
                  <Input
                    value={folderId}
                    onChange={(e) => setFolderId(e.target.value)}
                    placeholder="e.g. 1AbCdEfGhIjKlMnOpQrStUvWxYz"
                    className="flex-1 text-xs"
                  />
                  <Button variant="outline" size="sm" onClick={handleConfigure} disabled={busy || !folderId.trim()}>
                    Save
                  </Button>
                </div>
                <p className="text-[10px] text-muted-foreground">
                  Share the folder with the configured service-account email (read access) —
                  see .env.example for GOOGLE_DRIVE_SERVICE_ACCOUNT_FILE. No login required.
                </p>
              </CardContent>
            </Card>

            <Card size="sm">
              <CardContent className="flex items-center justify-between">
                <label className="flex items-center gap-2 text-[11px] text-muted-foreground">
                  <input
                    type="checkbox"
                    checked={removeDeleted}
                    onChange={(e) => setRemoveDeleted(e.target.checked)}
                  />
                  Remove documents deleted from Drive
                </label>
                <Button variant="outline" size="sm" onClick={handleSync} disabled={busy || !status?.configured}>
                  {busy ? <Loader2 className="animate-spin" /> : <RefreshCw data-icon="inline-start" />}
                  Sync now
                </Button>
              </CardContent>
            </Card>

            {status?.last_synced_at && (
              <p className="text-[10px] text-muted-foreground">
                Last synced: {new Date(status.last_synced_at).toLocaleString()}
              </p>
            )}

            <div>
              <p className="mb-1.5 text-[11px] font-medium text-muted-foreground">Tracked files</p>
              {files.length === 0 ? (
                <p className="text-xs text-muted-foreground">No files synced yet.</p>
              ) : (
                <div className="grid gap-1.5">
                  {files.map((f) => (
                    <div
                      key={f.file_id}
                      className="rounded-md border border-border bg-card px-2.5 py-1.5 text-[11px] text-foreground"
                    >
                      <span className="font-medium">{f.name}</span>{" "}
                      <span className="text-muted-foreground">
                        — added {new Date(f.indexed_at).toLocaleString()}
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
