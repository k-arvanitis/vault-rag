"use client";

import { useEffect, useState, useCallback } from "react";
import { FileText, FileSpreadsheet, Image, FileCode, File, ScanSearch } from "lucide-react";
import { getDocuments, getStats, type Document, type Stats } from "@/lib/api";
import { cn } from "@/lib/utils";
import UploadZone from "./UploadZone";

const TYPE_COLORS: Record<string, string> = {
  pdf: "bg-red-900 text-red-300",
  xlsx: "bg-green-900 text-green-300",
  xls: "bg-green-900 text-green-300",
  csv: "bg-yellow-900 text-yellow-300",
  docx: "bg-blue-900 text-blue-300",
  doc: "bg-blue-900 text-blue-300",
  md: "bg-zinc-700 text-zinc-300",
  png: "bg-purple-900 text-purple-300",
  jpg: "bg-purple-900 text-purple-300",
  jpeg: "bg-purple-900 text-purple-300",
};

const STATUS_COLORS: Record<string, string> = {
  indexed: "bg-emerald-900/60 text-emerald-400",
  processing: "bg-zinc-700 text-zinc-400",
  failed: "bg-red-900/60 text-red-400",
};

function fileExt(name: string) {
  return name.split("/").pop()?.split(".").pop()?.toLowerCase() ?? "";
}

function FileIcon({ ext }: { ext: string }) {
  const cls = "h-3.5 w-3.5 shrink-0 text-zinc-500";
  if (["pdf"].includes(ext)) return <FileText className={cls} />;
  if (["xlsx", "xls", "csv"].includes(ext)) return <FileSpreadsheet className={cls} />;
  if (["png", "jpg", "jpeg"].includes(ext)) return <Image className={cls} />;
  if (["md"].includes(ext)) return <FileCode className={cls} />;
  return <File className={cls} />;
}

const INSPECTABLE = new Set(["pdf", "xlsx", "xls", "csv"]);

interface Props {
  onToast: (msg: string, variant?: "error") => void;
  onInspect: (filename: string) => void;
}

export default function Sidebar({ onToast, onInspect }: Props) {
  const [docs, setDocs] = useState<Document[]>([]);
  const [stats, setStats] = useState<Stats | null>(null);

  const refresh = useCallback(async () => {
    try {
      const [d, s] = await Promise.all([getDocuments(), getStats()]);
      setDocs(d);
      setStats(s);
    } catch {
      // backend unreachable — handled by banner upstream
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  return (
    <aside className="w-[280px] shrink-0 flex flex-col h-full bg-zinc-950 border-r border-zinc-800 px-3 py-4">
      {/* Wordmark */}
      <div className="mb-5 px-1">
        <span className="text-base font-semibold tracking-tight text-zinc-100">Vault RAG</span>
      </div>

      {/* Documents section */}
      <p className="text-[10px] uppercase tracking-widest text-zinc-600 mb-2 px-1">Documents</p>

      <div className="flex-1 overflow-y-auto min-h-0 space-y-1">
        {docs.length === 0 ? (
          <p className="text-xs text-zinc-600 px-1 mt-2">No documents indexed yet.</p>
        ) : (
          docs.map((doc) => {
            const ext = fileExt(doc.filename);
            const basename = doc.filename.split("/").pop() ?? doc.filename;
            const typeLabel = doc.file_type || ext.toUpperCase();
            const canInspect = INSPECTABLE.has(ext);
            return (
              <div
                key={doc.filename}
                className="group flex items-start gap-2 rounded-md px-2 py-2 hover:bg-zinc-900 transition-colors"
              >
                <FileIcon ext={ext} />
                <div className="flex-1 min-w-0">
                  <p className="text-xs text-zinc-300 truncate leading-tight">{basename}</p>
                  <div className="flex items-center gap-1.5 mt-1">
                    <span
                      className={cn(
                        "text-[9px] px-1.5 py-0.5 rounded font-medium uppercase tracking-wide",
                        TYPE_COLORS[ext] ?? "bg-zinc-800 text-zinc-400"
                      )}
                    >
                      {typeLabel}
                    </span>
                    <span
                      className={cn(
                        "text-[9px] px-1.5 py-0.5 rounded font-medium",
                        STATUS_COLORS[doc.status] ?? "bg-zinc-800 text-zinc-400"
                      )}
                    >
                      {doc.status}
                    </span>
                  </div>
                </div>
                {canInspect && (
                  <button
                    onClick={() => onInspect(doc.filename)}
                    title="Inspect document"
                    className="shrink-0 p-1 rounded text-zinc-600 hover:text-zinc-300 hover:bg-zinc-800 opacity-0 group-hover:opacity-100 transition-all"
                  >
                    <ScanSearch className="h-3.5 w-3.5" />
                  </button>
                )}
              </div>
            );
          })
        )}
      </div>

      {/* Stats row */}
      {stats && (
        <div className="flex gap-4 px-1 pt-3 pb-2 border-t border-zinc-800 mt-2">
          <div>
            <p className="text-[10px] text-zinc-600">Docs</p>
            <p className="text-sm font-semibold text-zinc-300">{stats.total_docs}</p>
          </div>
          <div>
            <p className="text-[10px] text-zinc-600">Chunks</p>
            <p className="text-sm font-semibold text-zinc-300">{stats.total_chunks.toLocaleString()}</p>
          </div>
        </div>
      )}

      <UploadZone onUploaded={refresh} onToast={onToast} />
    </aside>
  );
}
