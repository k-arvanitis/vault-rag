"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { X, MoreVertical, Trash2 } from "lucide-react";
import { listConversations, getConversation, deleteConversation, type Conversation, type ConversationSummary } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Skeleton } from "@/components/ui/skeleton";
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

interface Props {
  onClose: () => void;
  onSelect: (conversation: Conversation) => void;
}

export default function HistoryPanel({ onClose, onSelect }: Props) {
  const [items, setItems] = useState<ConversationSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [pendingDelete, setPendingDelete] = useState<ConversationSummary | null>(null);

  const load = useCallback(() => {
    listConversations()
      .then((c) => {
        setItems(c);
        setError(null);
      })
      .catch((e) => setError(e instanceof Error ? e.message : "Failed to load conversations"));
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const handleSelect = async (id: string) => {
    try {
      const conv = await getConversation(id);
      onSelect(conv);
      onClose();
    } catch {
      setError("Failed to load that conversation");
    }
  };

  const handleDelete = async (id: string) => {
    await deleteConversation(id);
    setItems((prev) => prev?.filter((c) => c.id !== id) ?? null);
    setPendingDelete(null);
  };

  const filtered = useMemo(() => {
    if (!items) return items;
    const q = query.trim().toLowerCase();
    if (!q) return items;
    return items.filter((c) => c.title.toLowerCase().includes(q));
  }, [items, query]);

  return (
    <div className="fixed inset-0 z-30 flex">
      <div className="flex h-full w-full flex-col bg-background">
        <div className="flex shrink-0 items-center gap-3 border-b border-border bg-card px-5 py-3">
          <div className="min-w-0 flex-1">
            <p className="text-xs font-semibold text-foreground">Conversation history</p>
            <p className="text-[10px] text-muted-foreground">{items ? `${items.length} saved` : "Loading…"}</p>
          </div>
          <Button variant="ghost" size="icon-sm" onClick={onClose} aria-label="Close history">
            <X />
          </Button>
        </div>

        <div className="flex-1 overflow-y-auto p-5">
          <div className="mx-auto max-w-2xl space-y-3">
            {items && items.length > 0 && (
              <Input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search conversations…"
                className="text-sm"
              />
            )}

            {error && (
              <Alert variant="destructive">
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}
            {!error && !items && (
              <div className="space-y-2">
                <Skeleton className="h-12 w-full" />
                <Skeleton className="h-12 w-full" />
                <Skeleton className="h-12 w-full" />
              </div>
            )}
            {items && items.length === 0 && (
              <p className="text-center text-sm text-muted-foreground">No saved conversations yet.</p>
            )}
            {filtered && filtered.length === 0 && items && items.length > 0 && (
              <p className="text-center text-sm text-muted-foreground">No conversations match "{query}".</p>
            )}
            {filtered && filtered.length > 0 && (
              <div className="grid gap-2">
                {filtered.map((c) => (
                  <div
                    key={c.id}
                    role="button"
                    tabIndex={0}
                    onClick={() => handleSelect(c.id)}
                    onKeyDown={(e) => e.key === "Enter" && handleSelect(c.id)}
                    className="flex items-center justify-between gap-2 rounded-lg border border-border bg-card p-3 text-left transition-colors hover:bg-muted"
                  >
                    <div className="min-w-0 flex-1">
                      <p className="truncate text-xs font-medium text-foreground">{c.title}</p>
                      <p className="text-[10px] text-muted-foreground">
                        {c.message_count} messages · {new Date(c.updated_at).toLocaleString()}
                      </p>
                    </div>
                    <DropdownMenu>
                      <DropdownMenuTrigger
                        render={
                          <Button
                            variant="ghost"
                            size="icon-xs"
                            aria-label="Conversation actions"
                            onClick={(e) => e.stopPropagation()}
                          />
                        }
                      >
                        <MoreVertical />
                      </DropdownMenuTrigger>
                      <DropdownMenuContent>
                        <DropdownMenuItem
                          variant="destructive"
                          onClick={(e) => {
                            e.stopPropagation();
                            setPendingDelete(c);
                          }}
                        >
                          <Trash2 data-icon="inline-start" />
                          Delete
                        </DropdownMenuItem>
                      </DropdownMenuContent>
                    </DropdownMenu>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      <AlertDialog open={!!pendingDelete} onOpenChange={(open) => !open && setPendingDelete(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete this conversation?</AlertDialogTitle>
            <AlertDialogDescription>
              "{pendingDelete?.title}" will be permanently removed. This cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction onClick={() => pendingDelete && handleDelete(pendingDelete.id)}>
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
