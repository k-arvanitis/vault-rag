"use client";

import { useCallback, useEffect, useState } from "react";
import { X, Loader2, Trash2 } from "lucide-react";
import { listConversations, getConversation, deleteConversation, type Conversation, type ConversationSummary } from "@/lib/api";

interface Props {
  onClose: () => void;
  onSelect: (conversation: Conversation) => void;
}

export default function HistoryPanel({ onClose, onSelect }: Props) {
  const [items, setItems] = useState<ConversationSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);

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

  const handleDelete = async (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    await deleteConversation(id);
    setItems((prev) => prev?.filter((c) => c.id !== id) ?? null);
  };

  return (
    <div className="fixed inset-0 z-30 flex">
      <div className="flex h-full w-full flex-col bg-ink-50">
        <div className="flex shrink-0 items-center gap-3 border-b border-ink-200 bg-surface px-5 py-3">
          <div className="min-w-0 flex-1">
            <p className="text-xs font-semibold text-ink-800">Conversation history</p>
            <p className="text-[10px] text-ink-400">{items ? `${items.length} saved` : "Loading…"}</p>
          </div>
          <button
            onClick={onClose}
            className="rounded-md p-1.5 text-ink-500 transition-colors hover:bg-ink-100 hover:text-ink-800"
            aria-label="Close history"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-5">
          {error && <p className="text-sm text-red-600">{error}</p>}
          {!error && !items && (
            <div className="flex flex-1 items-center justify-center py-20">
              <Loader2 className="h-5 w-5 animate-spin text-ink-400" />
            </div>
          )}
          {items && items.length === 0 && (
            <p className="text-center text-sm text-ink-400">No saved conversations yet.</p>
          )}
          {items && items.length > 0 && (
            <div className="mx-auto grid max-w-2xl gap-2">
              {items.map((c) => (
                <button
                  key={c.id}
                  onClick={() => handleSelect(c.id)}
                  className="flex items-center justify-between gap-2 rounded-lg border border-ink-200 bg-surface p-3 text-left transition-colors hover:bg-ink-100"
                >
                  <div className="min-w-0 flex-1">
                    <p className="truncate text-xs font-medium text-ink-800">{c.title}</p>
                    <p className="text-[10px] text-ink-400">
                      {c.message_count} messages · {new Date(c.updated_at).toLocaleString()}
                    </p>
                  </div>
                  <span
                    onClick={(e) => handleDelete(e, c.id)}
                    className="shrink-0 rounded p-1 text-ink-400 transition-colors hover:bg-ink-200 hover:text-red-600"
                    aria-label="Delete conversation"
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                  </span>
                </button>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
