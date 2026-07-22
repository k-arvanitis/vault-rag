"use client";

import { useCallback, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { MoreVertical, ScanSearch, MessageCircle, RefreshCw, Pencil, Trash2, ArrowLeft } from "lucide-react";
import { getDocuments, deleteDocument, clearCollection, type Document } from "@/lib/api";
import { toSourceLibraryItem, resolveDisplayTitle, SOURCE_STATUS_LABEL, SOURCE_STATUS_BADGE_VARIANT } from "@/lib/product";
import { useJobTracker } from "@/lib/jobTracker";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Table, TableHeader, TableRow, TableHead, TableBody, TableCell } from "@/components/ui/table";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu";
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
import ReprocessDialog from "@/components/ReprocessDialog";
import RenameDialog from "@/components/RenameDialog";
import { toast } from "sonner";

const INSPECTABLE = new Set(["pdf", "xlsx", "xls", "csv"]);

function fileExt(name: string) {
  return name.split("/").pop()?.split(".").pop()?.toLowerCase() ?? "";
}

/** "Source Library" — the full Sources screen (product spec §7), a table view
 * complementing the persistent sidebar list rather than replacing it. Uses
 * lib/product.ts's status mapping so this table and the sidebar never drift
 * into showing different terminology for the same backend status. Also the
 * home of "Clear all" — a whole-collection destructive action belongs on the
 * management screen, not one click away in the everyday sidebar. */
export default function SourcesPage() {
  const router = useRouter();
  const [docs, setDocs] = useState<Document[]>([]);
  const [loading, setLoading] = useState(true);
  const [pendingDelete, setPendingDelete] = useState<Document | null>(null);
  const [reprocessTarget, setReprocessTarget] = useState<Document | null>(null);
  const [renameTarget, setRenameTarget] = useState<Document | null>(null);
  const [confirmClearAll, setConfirmClearAll] = useState(false);
  const jobs = useJobTracker();

  const refresh = useCallback(async () => {
    try {
      const d = await getDocuments();
      setDocs(d);
    } catch {
      toast.error("Failed to load sources");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  useEffect(() => {
    refresh();
  }, [jobs, refresh]);

  const askAbout = (filename: string) => {
    router.push(`/?doc=${encodeURIComponent(filename)}`);
  };

  return (
    <div className="flex h-full flex-col overflow-hidden bg-background">
      <div className="flex shrink-0 items-center gap-3 border-b border-border bg-card px-5 py-3">
        <Button variant="ghost" size="icon-sm" onClick={() => router.push("/")} aria-label="Back to Ask">
          <ArrowLeft />
        </Button>
        <div className="min-w-0 flex-1">
          <p className="text-sm font-semibold text-foreground">Sources</p>
          <p className="text-[11px] text-muted-foreground">{docs.length} source{docs.length === 1 ? "" : "s"}</p>
        </div>
        {docs.length > 0 && (
          <Button variant="outline" size="sm" onClick={() => setConfirmClearAll(true)}>
            <Trash2 data-icon="inline-start" />
            Clear all
          </Button>
        )}
      </div>

      <div className="flex-1 overflow-y-auto p-5">
        <div className="mx-auto max-w-4xl">
          {loading ? (
            <p className="text-sm text-muted-foreground">Loading…</p>
          ) : docs.length === 0 ? (
            <p className="text-sm text-muted-foreground">No sources added yet.</p>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Name</TableHead>
                  <TableHead>Type</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead>Last updated</TableHead>
                  <TableHead className="text-right">Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {docs.map((doc) => {
                  const item = toSourceLibraryItem(doc, jobs.get(doc.filename));
                  const ext = fileExt(doc.filename);
                  const canInspect = INSPECTABLE.has(ext);
                  return (
                    <TableRow key={doc.filename}>
                      <TableCell className="max-w-[280px]">
                        <p className="truncate font-medium" title={item.name}>
                          {item.name}
                        </p>
                        {item.name !== item.filename && (
                          <p className="truncate text-xs text-muted-foreground" title={item.filename}>
                            {item.filename}
                          </p>
                        )}
                      </TableCell>
                      <TableCell>
                        <Badge variant="outline" className="uppercase">
                          {doc.file_type || ext.toUpperCase()}
                        </Badge>
                      </TableCell>
                      <TableCell>
                        <Badge variant={SOURCE_STATUS_BADGE_VARIANT[item.status]}>
                          {SOURCE_STATUS_LABEL[item.status]}
                        </Badge>
                      </TableCell>
                      <TableCell className="text-muted-foreground">
                        {item.updatedAt ? new Date(item.updatedAt).toLocaleDateString() : "—"}
                      </TableCell>
                      <TableCell className="text-right">
                        <DropdownMenu>
                          <DropdownMenuTrigger
                            render={<Button variant="ghost" size="icon-xs" aria-label="Source actions" />}
                          >
                            <MoreVertical />
                          </DropdownMenuTrigger>
                          <DropdownMenuContent align="end">
                            {canInspect && (
                              <DropdownMenuItem onClick={() => router.push(`/?inspect=${encodeURIComponent(doc.filename)}`)}>
                                <ScanSearch data-icon="inline-start" />
                                Open
                              </DropdownMenuItem>
                            )}
                            <DropdownMenuItem onClick={() => askAbout(doc.filename)}>
                              <MessageCircle data-icon="inline-start" />
                              Ask about this source
                            </DropdownMenuItem>
                            <DropdownMenuItem onClick={() => setRenameTarget(doc)}>
                              <Pencil data-icon="inline-start" />
                              Rename
                            </DropdownMenuItem>
                            <DropdownMenuItem onClick={() => setReprocessTarget(doc)}>
                              <RefreshCw data-icon="inline-start" />
                              Reprocess
                            </DropdownMenuItem>
                            <DropdownMenuItem variant="destructive" onClick={() => setPendingDelete(doc)}>
                              <Trash2 data-icon="inline-start" />
                              Delete
                            </DropdownMenuItem>
                          </DropdownMenuContent>
                        </DropdownMenu>
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          )}
        </div>
      </div>

      {reprocessTarget && (
        <ReprocessDialog
          filename={reprocessTarget.filename}
          open={!!reprocessTarget}
          onOpenChange={(open) => !open && setReprocessTarget(null)}
          onReprocessed={refresh}
          onToast={(msg, variant) => (variant === "error" ? toast.error(msg) : toast(msg))}
        />
      )}

      {renameTarget && (
        <RenameDialog
          filename={renameTarget.filename}
          currentName={resolveDisplayTitle(renameTarget)}
          open={!!renameTarget}
          onOpenChange={(open) => !open && setRenameTarget(null)}
          onRenamed={refresh}
          onToast={(msg, variant) => (variant === "error" ? toast.error(msg) : toast(msg))}
        />
      )}

      <AlertDialog open={!!pendingDelete} onOpenChange={(open) => !open && setPendingDelete(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete this source?</AlertDialogTitle>
            <AlertDialogDescription>
              &ldquo;{pendingDelete?.filename.split("/").pop()}&rdquo; will be permanently removed. This cannot be
              undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={async () => {
                if (!pendingDelete) return;
                try {
                  await deleteDocument(pendingDelete.filename);
                  toast(`Deleted ${pendingDelete.filename.split("/").pop()}`);
                  setPendingDelete(null);
                  refresh();
                } catch (e) {
                  toast.error(e instanceof Error ? e.message : "Delete failed");
                }
              }}
            >
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      <AlertDialog open={confirmClearAll} onOpenChange={setConfirmClearAll}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Clear the entire collection?</AlertDialogTitle>
            <AlertDialogDescription>
              All {docs.length} source{docs.length === 1 ? "" : "s"} will be removed. This cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={async () => {
                try {
                  await clearCollection();
                  toast("Collection cleared");
                  setConfirmClearAll(false);
                  refresh();
                } catch (e) {
                  toast.error(e instanceof Error ? e.message : "Clear failed");
                }
              }}
            >
              Clear all
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
