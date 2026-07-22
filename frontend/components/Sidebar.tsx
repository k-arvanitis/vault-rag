"use client";

import { useEffect, useState, useCallback } from "react";
import { FileText, FileSpreadsheet, Image, FileCode, File, ScanSearch, Trash2, RefreshCw, Search, Pencil } from "lucide-react";
import { getDocuments, getStats, deleteDocument, type Document, type Stats } from "@/lib/api";
import { toSourceLibraryItem, resolveDisplayTitle, SOURCE_STATUS_LABEL } from "@/lib/product";
import { useJobTracker } from "@/lib/jobTracker";
import { useAdminSession } from "@/lib/useAdminSession";
import { cn } from "@/lib/utils";
import {
  Sidebar as SidebarRoot,
  SidebarContent,
  SidebarFooter,
  SidebarHeader,
} from "@/components/ui/sidebar";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import ReprocessDialog from "./ReprocessDialog";
import RenameDialog from "./RenameDialog";
import UploadZone from "./UploadZone";

function fileExt(name: string) {
  return name.split("/").pop()?.split(".").pop()?.toLowerCase() ?? "";
}

function FileIcon({ ext }: { ext: string }) {
  const cls = "size-3.5 shrink-0 text-muted-foreground";
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
  offline?: boolean;
}

export default function Sidebar({ onToast, onInspect, offline }: Props) {
  const [docs, setDocs] = useState<Document[]>([]);
  const [search, setSearch] = useState("");
  const [stats, setStats] = useState<Stats | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<string | null>(null);
  const [reprocessTarget, setReprocessTarget] = useState<string | null>(null);
  const [renameTarget, setRenameTarget] = useState<string | null>(null);
  const jobs = useJobTracker();
  const { is_admin: isAdmin } = useAdminSession();

  // getDocuments()/getStats() are admin-only (see api.py's require_admin on
  // those routes) -- a non-admin viewer must not be able to enumerate the
  // corpus, so this never even calls them and the list/count stay empty.
  const refresh = useCallback(async () => {
    if (!isAdmin) return;
    try {
      const [d, s] = await Promise.all([getDocuments(), getStats()]);
      const byName = (f: string) => (f.split("/").pop() ?? f).toLowerCase();
      setDocs([...d].sort((a, b) => byName(a.filename).localeCompare(byName(b.filename))));
      setStats(s);
    } catch {
      // backend unreachable — handled by the offline banner upstream
    }
  }, [isAdmin]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  // Job completion settles the shared tracker a couple seconds after success
  // (see lib/jobTracker.ts) — re-fetch then so the row picks up the new
  // last_indexed_at / chunk state instead of staying on stale data.
  useEffect(() => {
    refresh();
  }, [jobs, refresh]);

  const filteredDocs = search.trim()
    ? docs.filter((d) => {
        const q = search.trim().toLowerCase();
        const basename = (d.filename.split("/").pop() ?? d.filename).toLowerCase();
        return basename.includes(q) || resolveDisplayTitle(d).toLowerCase().includes(q);
      })
    : docs;

  return (
    <SidebarRoot collapsible="offcanvas">
      <SidebarHeader className="gap-2 px-3 py-3">
        <span className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
          Knowledge base
        </span>
        {isAdmin && docs.length > 0 && (
          <div className="relative">
            <Search className="pointer-events-none absolute left-2 top-1/2 size-3 -translate-y-1/2 text-muted-foreground" />
            <Input
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search knowledge base…"
              className="h-7 pl-6 text-xs"
            />
          </div>
        )}
      </SidebarHeader>

      {/* SidebarContent already scrolls itself (overflow-auto, see
          components/ui/sidebar.tsx) -- nesting a second ScrollArea inside it
          created two competing scroll containers, and the inner one's h-full
          didn't reliably bound itself against the outer's auto-sized height,
          letting content render taller than the visible area and get sliced
          by SidebarFooter instead of scrolling clear of it. One scroll
          container, not two. */}
      <SidebarContent className="px-3">
        {/* pb-28 reserves space for the fixed SidebarFooter (stats line +
            Add sources button, ~90px) plus a margin so the last card's
            metadata never renders clipped behind it. */}
        <div className="flex flex-col gap-1 pb-28">
            {!isAdmin ? (
              <p className="px-0.5 pt-1 text-xs text-muted-foreground">
                Ask questions in the chat — individual document details aren&apos;t shown here.
              </p>
            ) : docs.length === 0 ? (
              <p className="px-0.5 pt-1 text-xs text-muted-foreground">No sources added yet.</p>
            ) : filteredDocs.length === 0 ? (
              <p className="px-0.5 pt-1 text-xs text-muted-foreground">No sources match &ldquo;{search}&rdquo;.</p>
            ) : (
              filteredDocs.map((doc) => {
                const ext = fileExt(doc.filename);
                const basename = doc.filename.split("/").pop() ?? doc.filename;
                const displayName = resolveDisplayTitle(doc);
                const typeLabel = doc.file_type || ext.toUpperCase();
                const canInspect = INSPECTABLE.has(ext);
                const status = toSourceLibraryItem(doc, jobs.get(doc.filename)).status;
                return (
                  <div
                    key={doc.filename}
                    className="group flex items-start gap-2 rounded-md border border-border/60 bg-card px-2 py-1.5 transition-colors hover:border-foreground/20"
                  >
                    <span className="pt-0.5">
                      <FileIcon ext={ext} />
                    </span>
                    <div className="min-w-0 flex-1 space-y-0.5">
                      <p
                        className="line-clamp-2 text-xs font-medium leading-tight text-foreground"
                        title={displayName}
                      >
                        {displayName}
                      </p>
                      {displayName !== basename && (
                        <p className="truncate text-[10px] text-muted-foreground" title={basename}>
                          {basename}
                        </p>
                      )}
                      <div className="flex items-center gap-1.5">
                        <Badge variant="outline" className="uppercase">
                          {typeLabel}
                        </Badge>
                        <span className="flex items-center gap-1 text-[10px] text-muted-foreground">
                          <span
                            className={cn(
                              "size-1.5 rounded-full",
                              status === "ready" && "bg-emerald-600",
                              (status === "processing" || status === "updating") &&
                                "animate-pulse bg-blue-500",
                              (status === "attention" || status === "failed") && "bg-destructive"
                            )}
                          />
                          {SOURCE_STATUS_LABEL[status]}
                        </span>
                        {doc.last_indexed_at && (
                          <span className="text-[10px] text-muted-foreground">
                            {new Date(doc.last_indexed_at).toLocaleDateString()}
                          </span>
                        )}
                      </div>
                      {(doc.page_count || doc.sheet_count) && (
                        <p className="text-[10px] text-muted-foreground">
                          {doc.page_count && `${doc.page_count} page${doc.page_count === 1 ? "" : "s"}`}
                          {doc.sheet_count &&
                            `${doc.sheet_count} sheet${doc.sheet_count === 1 ? "" : "s"}${
                              doc.row_count ? ` · ${doc.row_count.toLocaleString()} rows` : ""
                            }`}
                        </p>
                      )}
                    </div>
                    <div className="flex shrink-0 items-center gap-0.5 opacity-0 transition-opacity group-hover:opacity-100">
                      {isAdmin && (
                        <Tooltip>
                          <TooltipTrigger
                            render={
                              <Button
                                variant="ghost"
                                size="icon-xs"
                                aria-label="Rename source"
                                onClick={() => setRenameTarget(doc.filename)}
                              />
                            }
                          >
                            <Pencil />
                          </TooltipTrigger>
                          <TooltipContent>Rename source</TooltipContent>
                        </Tooltip>
                      )}
                      {isAdmin && (
                        <Tooltip>
                          <TooltipTrigger
                            render={
                              <Button
                                variant="ghost"
                                size="icon-xs"
                                aria-label="Reprocess document"
                                onClick={() => setReprocessTarget(doc.filename)}
                              />
                            }
                          >
                            <RefreshCw />
                          </TooltipTrigger>
                          <TooltipContent>Reprocess document</TooltipContent>
                        </Tooltip>
                      )}
                      {canInspect && (
                        <Tooltip>
                          <TooltipTrigger
                            render={
                              <Button
                                variant="ghost"
                                size="icon-xs"
                                aria-label="Inspect document"
                                onClick={() => onInspect(doc.filename)}
                              />
                            }
                          >
                            <ScanSearch />
                          </TooltipTrigger>
                          <TooltipContent>Inspect document</TooltipContent>
                        </Tooltip>
                      )}
                      {isAdmin && (
                        <Tooltip>
                          <TooltipTrigger
                            render={
                              <Button
                                variant="ghost"
                                size="icon-xs"
                                className="hover:text-destructive"
                                aria-label="Delete document"
                                onClick={() => setDeleteTarget(doc.filename)}
                              />
                            }
                          >
                            <Trash2 />
                          </TooltipTrigger>
                          <TooltipContent>Delete document</TooltipContent>
                        </Tooltip>
                      )}
                    </div>
                  </div>
                );
              })
            )}
        </div>
      </SidebarContent>

      <SidebarFooter className="gap-3 border-t border-border px-3 py-3">
        {stats && (
          <p className="px-0.5 text-xs text-muted-foreground">
            <span className="font-semibold text-foreground">{stats.total_docs}</span> approved source
            {stats.total_docs === 1 ? "" : "s"}
          </p>
        )}
        {isAdmin && <UploadZone onUploaded={refresh} onToast={onToast} offline={offline} />}
      </SidebarFooter>

      {reprocessTarget && (
        <ReprocessDialog
          filename={reprocessTarget}
          open={!!reprocessTarget}
          onOpenChange={(open) => !open && setReprocessTarget(null)}
          onReprocessed={refresh}
          onToast={onToast}
        />
      )}

      {renameTarget && (
        <RenameDialog
          filename={renameTarget}
          currentName={resolveDisplayTitle(docs.find((d) => d.filename === renameTarget) ?? { filename: renameTarget, display_title: null })}
          open={!!renameTarget}
          onOpenChange={(open) => !open && setRenameTarget(null)}
          onRenamed={refresh}
          onToast={onToast}
        />
      )}

      <AlertDialog open={deleteTarget !== null} onOpenChange={(open) => !open && setDeleteTarget(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete this source?</AlertDialogTitle>
            <AlertDialogDescription>
              &ldquo;{deleteTarget?.split("/").pop()}&rdquo; will be removed from the collection. This cannot be
              undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={async () => {
                if (!deleteTarget) return;
                const basename = deleteTarget.split("/").pop() ?? deleteTarget;
                try {
                  await deleteDocument(deleteTarget);
                  onToast(`Deleted ${basename}`);
                  refresh();
                } catch (e) {
                  onToast(e instanceof Error ? e.message : "Delete failed", "error");
                } finally {
                  setDeleteTarget(null);
                }
              }}
            >
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </SidebarRoot>
  );
}
