"use client";

import { useEffect, useState, useCallback } from "react";
import { FileText, FileSpreadsheet, Image, FileCode, File, ScanSearch, Trash2, RefreshCw } from "lucide-react";
import { getDocuments, getStats, deleteDocument, type Document, type Stats } from "@/lib/api";
import { toSourceLibraryItem, SOURCE_STATUS_LABEL, SOURCE_STATUS_BADGE_VARIANT } from "@/lib/product";
import { useJobTracker } from "@/lib/jobTracker";
import { useAdminSession } from "@/lib/useAdminSession";
import {
  Sidebar as SidebarRoot,
  SidebarContent,
  SidebarFooter,
  SidebarHeader,
} from "@/components/ui/sidebar";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
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
  const [stats, setStats] = useState<Stats | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<string | null>(null);
  const [reprocessTarget, setReprocessTarget] = useState<string | null>(null);
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

  return (
    <SidebarRoot collapsible="offcanvas">
      <SidebarHeader className="flex-row items-center justify-between px-3 py-3">
        <span className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
          {isAdmin ? "Sources" : "Knowledge base"}
        </span>
      </SidebarHeader>

      <SidebarContent className="px-3">
        <ScrollArea className="h-full">
          <div className="flex flex-col gap-1.5 pb-2">
            {!isAdmin ? (
              <p className="px-0.5 pt-1 text-xs text-muted-foreground">
                Ask questions in the chat — individual document details aren&apos;t shown here.
              </p>
            ) : docs.length === 0 ? (
              <p className="px-0.5 pt-1 text-xs text-muted-foreground">No sources added yet.</p>
            ) : (
              docs.map((doc) => {
                const ext = fileExt(doc.filename);
                const basename = doc.filename.split("/").pop() ?? doc.filename;
                const typeLabel = doc.file_type || ext.toUpperCase();
                const canInspect = INSPECTABLE.has(ext);
                const status = toSourceLibraryItem(doc, jobs.get(doc.filename)).status;
                return (
                  <div
                    key={doc.filename}
                    className="group flex items-start gap-2 rounded-md border border-border bg-card px-2.5 py-2 transition-colors hover:border-foreground/20"
                  >
                    <span className="pt-0.5">
                      <FileIcon ext={ext} />
                    </span>
                    <div className="min-w-0 flex-1 space-y-1">
                      <p
                        className="truncate text-xs font-medium leading-tight text-foreground"
                        title={basename}
                      >
                        {basename}
                      </p>
                      <div className="flex items-center gap-1.5">
                        <Badge variant="outline" className="uppercase">
                          {typeLabel}
                        </Badge>
                        <Badge variant={SOURCE_STATUS_BADGE_VARIANT[status]}>{SOURCE_STATUS_LABEL[status]}</Badge>
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
        </ScrollArea>
      </SidebarContent>

      <SidebarFooter className="gap-3 border-t border-border px-3 py-3">
        {stats && (
          <div className="flex gap-5 px-0.5">
            <div>
              <p className="text-[10px] text-muted-foreground">Sources</p>
              <p className="text-sm font-semibold text-foreground">{stats.total_docs}</p>
            </div>
          </div>
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
