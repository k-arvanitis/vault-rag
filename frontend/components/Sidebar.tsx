"use client";

import { useEffect, useState, useCallback } from "react";
import { FileText, FileSpreadsheet, Image, FileCode, File, ScanSearch, Trash2, RefreshCw } from "lucide-react";
import {
  getDocuments,
  getStats,
  deleteDocument,
  reindexDocument,
  clearCollection,
  type Document,
  type Stats,
} from "@/lib/api";
import { cn } from "@/lib/utils";
import UploadZone from "./UploadZone";

const STATUS_CHIP: Record<string, string> = {
  indexed: "bg-emerald-50 text-emerald-700",
  processing: "bg-ink-100 text-ink-500",
  failed: "bg-red-50 text-red-700",
};

function fileExt(name: string) {
  return name.split("/").pop()?.split(".").pop()?.toLowerCase() ?? "";
}

function FileIcon({ ext }: { ext: string }) {
  const cls = "h-3.5 w-3.5 shrink-0 text-ink-400";
  if (ext === "pdf") return <FileText className={cls} />;
  if (["xlsx", "xls", "csv"].includes(ext)) return <FileSpreadsheet className={cls} />;
  if (["png", "jpg", "jpeg"].includes(ext)) return <Image className={cls} />;
  if (ext === "md") return <FileCode className={cls} />;
  return <File className={cls} />;
}

const INSPECTABLE = new Set(["pdf", "xlsx", "xls", "csv"]);

interface Props {
  onToast: (msg: string, variant?: "error") => void;
  onInspect: (filename: string) => void;
  onCollectionCleared: () => void;
}

export default function Sidebar({ onToast, onInspect, onCollectionCleared }: Props) {
  const [docs, setDocs] = useState<Document[]>([]);
  const [stats, setStats] = useState<Stats | null>(null);
  const [confirmClear, setConfirmClear] = useState(false);

  const refresh = useCallback(async () => {
    try {
      const [d, s] = await Promise.all([getDocuments(), getStats()]);
      const byName = (f: string) => (f.split("/").pop() ?? f).toLowerCase();
      setDocs([...d].sort((a, b) => byName(a.filename).localeCompare(byName(b.filename))));
      setStats(s);
    } catch {
      // backend unreachable — handled by the offline banner upstream
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  return (
    <aside className="flex w-[300px] flex-shrink-0 flex-col gap-3 border-r border-ink-200 bg-ink-50 p-3">
      {/* Documents header + clear control */}
      <div className="flex items-center justify-between px-0.5">
        <span className="text-xs font-semibold uppercase tracking-wide text-ink-500">Documents</span>
        {!confirmClear ? (
          <button
            onClick={() => setConfirmClear(true)}
            title="Clear collection"
            className="flex items-center gap-1 text-[10px] text-ink-400 hover:text-red-600"
          >
            <Trash2 className="h-3 w-3" />
            Clear all
          </button>
        ) : (
          <div className="flex items-center gap-1.5 text-[10px]">
            <button
              onClick={async () => {
                try {
                  await clearCollection();
                  setConfirmClear(false);
                  onCollectionCleared();
                  refresh();
                } catch (e) {
                  onToast(e instanceof Error ? e.message : "Clear failed", "error");
                  setConfirmClear(false);
                }
              }}
              className="font-medium text-red-600 hover:text-red-700"
            >
              Confirm
            </button>
            <button onClick={() => setConfirmClear(false)} className="text-ink-500 hover:text-ink-700">
              Cancel
            </button>
          </div>
        )}
      </div>

      <div className="min-h-0 flex-1 space-y-1.5 overflow-y-auto">
        {docs.length === 0 ? (
          <p className="px-0.5 pt-1 text-xs text-ink-400">No documents indexed yet.</p>
        ) : (
          docs.map((doc) => {
            const ext = fileExt(doc.filename);
            const basename = doc.filename.split("/").pop() ?? doc.filename;
            const typeLabel = doc.file_type || ext.toUpperCase();
            const canInspect = INSPECTABLE.has(ext);
            return (
              <div
                key={doc.filename}
                className="group flex items-start gap-2 rounded-md border border-ink-200 bg-surface px-2.5 py-2 transition-colors hover:border-ink-300"
              >
                <span className="pt-0.5">
                  <FileIcon ext={ext} />
                </span>
                <div className="min-w-0 flex-1 space-y-1">
                  <p className="truncate text-xs font-medium leading-tight text-ink-800">{basename}</p>
                  <div className="flex items-center gap-1.5">
                    <span className="rounded bg-ink-100 px-1.5 py-0.5 text-[10px] uppercase tracking-wide text-ink-600">
                      {typeLabel}
                    </span>
                    <span
                      className={cn(
                        "rounded px-1.5 py-0.5 text-[10px]",
                        STATUS_CHIP[doc.status] ?? "bg-ink-100 text-ink-500"
                      )}
                    >
                      {doc.status}
                    </span>
                    {doc.last_indexed_at && (
                      <span className="text-[10px] text-ink-400">
                        {new Date(doc.last_indexed_at).toLocaleDateString()}
                      </span>
                    )}
                  </div>
                </div>
                <div className="flex shrink-0 items-center gap-0.5 opacity-0 transition-opacity group-hover:opacity-100">
                  <button
                    onClick={async () => {
                      try {
                        await reindexDocument(doc.filename);
                        onToast(`Re-indexing ${basename}…`);
                        refresh();
                      } catch (e) {
                        onToast(e instanceof Error ? e.message : "Re-index failed", "error");
                      }
                    }}
                    title="Re-index document"
                    className="rounded p-1 text-ink-400 hover:bg-ink-100 hover:text-ink-700"
                  >
                    <RefreshCw className="h-3.5 w-3.5" />
                  </button>
                  {canInspect && (
                    <button
                      onClick={() => onInspect(doc.filename)}
                      title="Inspect document"
                      className="rounded p-1 text-ink-400 hover:bg-ink-100 hover:text-ink-700"
                    >
                      <ScanSearch className="h-3.5 w-3.5" />
                    </button>
                  )}
                  <button
                    onClick={async () => {
                      if (!confirm(`Delete "${basename}" from the collection?`)) return;
                      try {
                        await deleteDocument(doc.filename);
                        onToast(`Deleted ${basename}`);
                        refresh();
                      } catch (e) {
                        onToast(e instanceof Error ? e.message : "Delete failed", "error");
                      }
                    }}
                    title="Delete document"
                    className="rounded p-1 text-ink-400 hover:bg-ink-100 hover:text-red-600"
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                  </button>
                </div>
              </div>
            );
          })
        )}
      </div>

      {stats && (
        <div className="flex gap-5 border-t border-ink-200 px-0.5 pt-3">
          <div>
            <p className="text-[10px] text-ink-400">Docs</p>
            <p className="text-sm font-semibold text-ink-800">{stats.total_docs}</p>
          </div>
          <div>
            <p className="text-[10px] text-ink-400">Chunks</p>
            <p className="text-sm font-semibold text-ink-800">{stats.total_chunks.toLocaleString()}</p>
          </div>
        </div>
      )}

      <UploadZone onUploaded={refresh} onToast={onToast} />
    </aside>
  );
}
