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
import UploadZone from "./UploadZone";

const STATUS_BADGE: Record<string, "default" | "secondary" | "destructive"> = {
  indexed: "default",
  processing: "secondary",
  failed: "destructive",
};

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
  onCollectionCleared: () => void;
  offline?: boolean;
}

export default function Sidebar({ onToast, onInspect, onCollectionCleared, offline }: Props) {
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
    <SidebarRoot collapsible="offcanvas">
      <SidebarHeader className="flex-row items-center justify-between px-3 py-3">
        <span className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Documents</span>
        <AlertDialog open={confirmClear} onOpenChange={setConfirmClear}>
          <Button variant="ghost" size="xs" onClick={() => setConfirmClear(true)}>
            <Trash2 data-icon="inline-start" />
            Clear all
          </Button>
          <AlertDialogContent>
            <AlertDialogHeader>
              <AlertDialogTitle>Clear the collection?</AlertDialogTitle>
              <AlertDialogDescription>
                This deletes every indexed document and chunk. This cannot be undone.
              </AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter>
              <AlertDialogCancel>Cancel</AlertDialogCancel>
              <AlertDialogAction
                onClick={async () => {
                  try {
                    await clearCollection();
                    onCollectionCleared();
                    refresh();
                  } catch (e) {
                    onToast(e instanceof Error ? e.message : "Clear failed", "error");
                  }
                }}
              >
                Confirm
              </AlertDialogAction>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>
      </SidebarHeader>

      <SidebarContent className="px-3">
        <ScrollArea className="h-full">
          <div className="flex flex-col gap-1.5 pb-2">
            {docs.length === 0 ? (
              <p className="px-0.5 pt-1 text-xs text-muted-foreground">No documents indexed yet.</p>
            ) : (
              docs.map((doc) => {
                const ext = fileExt(doc.filename);
                const basename = doc.filename.split("/").pop() ?? doc.filename;
                const typeLabel = doc.file_type || ext.toUpperCase();
                const canInspect = INSPECTABLE.has(ext);
                return (
                  <div
                    key={doc.filename}
                    className="group flex items-start gap-2 rounded-md border border-border bg-card px-2.5 py-2 transition-colors hover:border-foreground/20"
                  >
                    <span className="pt-0.5">
                      <FileIcon ext={ext} />
                    </span>
                    <div className="min-w-0 flex-1 space-y-1">
                      <p className="truncate text-xs font-medium leading-tight text-foreground">{basename}</p>
                      <div className="flex items-center gap-1.5">
                        <Badge variant="outline" className="uppercase">
                          {typeLabel}
                        </Badge>
                        <Badge variant={STATUS_BADGE[doc.status] ?? "secondary"}>{doc.status}</Badge>
                        {doc.last_indexed_at && (
                          <span className="text-[10px] text-muted-foreground">
                            {new Date(doc.last_indexed_at).toLocaleDateString()}
                          </span>
                        )}
                      </div>
                    </div>
                    <div className="flex shrink-0 items-center gap-0.5 opacity-0 transition-opacity group-hover:opacity-100">
                      <Tooltip>
                        <TooltipTrigger
                          render={
                            <Button
                              variant="ghost"
                              size="icon-xs"
                              aria-label="Re-index document"
                              onClick={async () => {
                                try {
                                  await reindexDocument(doc.filename);
                                  onToast(`Re-indexing ${basename}…`);
                                  refresh();
                                } catch (e) {
                                  onToast(e instanceof Error ? e.message : "Re-index failed", "error");
                                }
                              }}
                            />
                          }
                        >
                          <RefreshCw />
                        </TooltipTrigger>
                        <TooltipContent>Re-index document</TooltipContent>
                      </Tooltip>
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
                      <Tooltip>
                        <TooltipTrigger
                          render={
                            <Button
                              variant="ghost"
                              size="icon-xs"
                              className="hover:text-destructive"
                              aria-label="Delete document"
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
                            />
                          }
                        >
                          <Trash2 />
                        </TooltipTrigger>
                        <TooltipContent>Delete document</TooltipContent>
                      </Tooltip>
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
              <p className="text-[10px] text-muted-foreground">Docs</p>
              <p className="text-sm font-semibold text-foreground">{stats.total_docs}</p>
            </div>
            <div>
              <p className="text-[10px] text-muted-foreground">Chunks</p>
              <p className="text-sm font-semibold text-foreground">{stats.total_chunks.toLocaleString()}</p>
            </div>
          </div>
        )}
        <UploadZone onUploaded={refresh} onToast={onToast} offline={offline} />
      </SidebarFooter>
    </SidebarRoot>
  );
}
