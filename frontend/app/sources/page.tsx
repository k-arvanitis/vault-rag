"use client";

import { useCallback, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { MoreVertical, ScanSearch, MessageCircle, RefreshCw, Trash2, ArrowLeft } from "lucide-react";
import { getDocuments, deleteDocument, reindexDocument, type Document } from "@/lib/api";
import { toSourceLibraryItem, SOURCE_STATUS_LABEL, type SourceStatus } from "@/lib/product";
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
import { toast } from "sonner";

const STATUS_BADGE: Record<SourceStatus, "default" | "secondary" | "destructive"> = {
  ready: "default",
  processing: "secondary",
  attention: "destructive",
  failed: "destructive",
};

const INSPECTABLE = new Set(["pdf", "xlsx", "xls", "csv"]);

function fileExt(name: string) {
  return name.split("/").pop()?.split(".").pop()?.toLowerCase() ?? "";
}

/** "Source Library" — the full Sources screen (product spec §7), a table view
 * complementing the persistent sidebar list rather than replacing it. Uses
 * lib/product.ts's status mapping so this table and the sidebar never drift
 * into showing different terminology for the same backend status. */
export default function SourcesPage() {
  const router = useRouter();
  const [docs, setDocs] = useState<Document[]>([]);
  const [loading, setLoading] = useState(true);
  const [pendingDelete, setPendingDelete] = useState<Document | null>(null);

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

  const askAbout = (filename: string) => {
    router.push(`/?doc=${encodeURIComponent(filename)}`);
  };

  return (
    <div className="flex h-screen flex-col overflow-hidden bg-background">
      <div className="flex shrink-0 items-center gap-3 border-b border-border bg-card px-5 py-3">
        <Button variant="ghost" size="icon-sm" onClick={() => router.push("/")} aria-label="Back to Ask">
          <ArrowLeft />
        </Button>
        <div className="min-w-0 flex-1">
          <p className="text-sm font-semibold text-foreground">Sources</p>
          <p className="text-[11px] text-muted-foreground">{docs.length} source{docs.length === 1 ? "" : "s"}</p>
        </div>
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
                  const item = toSourceLibraryItem(doc);
                  const ext = fileExt(doc.filename);
                  const canInspect = INSPECTABLE.has(ext);
                  return (
                    <TableRow key={doc.filename}>
                      <TableCell className="max-w-[280px] truncate font-medium">{item.name}</TableCell>
                      <TableCell>
                        <Badge variant="outline" className="uppercase">
                          {doc.file_type || ext.toUpperCase()}
                        </Badge>
                      </TableCell>
                      <TableCell>
                        <Badge variant={STATUS_BADGE[item.status]}>{SOURCE_STATUS_LABEL[item.status]}</Badge>
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
                            <DropdownMenuItem
                              onClick={async () => {
                                try {
                                  await reindexDocument(doc.filename);
                                  toast(`Reprocessing ${item.name}…`);
                                  refresh();
                                } catch (e) {
                                  toast.error(e instanceof Error ? e.message : "Reprocess failed");
                                }
                              }}
                            >
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

      <AlertDialog open={!!pendingDelete} onOpenChange={(open) => !open && setPendingDelete(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete this source?</AlertDialogTitle>
            <AlertDialogDescription>
              "{pendingDelete?.filename.split("/").pop()}" will be permanently removed. This cannot be undone.
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
    </div>
  );
}
