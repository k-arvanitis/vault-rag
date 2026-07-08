"use client";

import { useCallback, useEffect, useState } from "react";
import { X, ThumbsUp, ThumbsDown, Loader2 } from "lucide-react";
import { getFeedback, resolveFeedback, type Feedback } from "@/lib/api";
import { cn } from "@/lib/utils";

interface Props {
  onClose: () => void;
}

const ACTIONS = [
  { value: "mark_correct_source", label: "Mark correct source" },
  { value: "add_to_eval_set", label: "Add to eval set" },
  { value: "dismissed", label: "Dismiss" },
];

function FeedbackRow({ item, onResolved }: { item: Feedback; onResolved: (f: Feedback) => void }) {
  const [busy, setBusy] = useState(false);

  const resolve = async (action: string) => {
    setBusy(true);
    try {
      const updated = await resolveFeedback(item.id, action);
      onResolved(updated);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="rounded-lg border border-ink-200 bg-surface p-3">
      <div className="flex items-start justify-between gap-2">
        <div className="flex items-center gap-1.5">
          {item.rating === "up" ? (
            <ThumbsUp className="h-3.5 w-3.5 text-emerald-600" />
          ) : (
            <ThumbsDown className="h-3.5 w-3.5 text-red-600" />
          )}
          {item.reason && (
            <span className="rounded bg-ink-100 px-1.5 py-0.5 text-[10px] text-ink-600">
              {item.reason.replace(/_/g, " ")}
            </span>
          )}
          <span
            className={cn(
              "rounded px-1.5 py-0.5 text-[10px]",
              item.status === "resolved" ? "bg-emerald-50 text-emerald-700" : "bg-amber-50 text-amber-700"
            )}
          >
            {item.status}
          </span>
        </div>
        <span className="shrink-0 text-[10px] text-ink-400">
          {new Date(item.created_at).toLocaleString()}
        </span>
      </div>

      <p className="mt-2 text-xs font-medium text-ink-800">{item.question}</p>
      <p className="mt-1 line-clamp-3 text-[11px] text-ink-500">{item.answer}</p>

      {item.status === "open" ? (
        <div className="mt-2 flex flex-wrap gap-1.5">
          {ACTIONS.map((a) => (
            <button
              key={a.value}
              onClick={() => resolve(a.value)}
              disabled={busy}
              className="rounded-md border border-ink-200 px-2 py-1 text-[10px] font-medium text-ink-600 transition-colors hover:bg-ink-100 disabled:opacity-50"
            >
              {busy ? <Loader2 className="h-3 w-3 animate-spin" /> : a.label}
            </button>
          ))}
        </div>
      ) : (
        <p className="mt-2 text-[10px] text-ink-400">
          Resolved: {item.action?.replace(/_/g, " ")}
        </p>
      )}
    </div>
  );
}

export default function FeedbackPanel({ onClose }: Props) {
  const [items, setItems] = useState<Feedback[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(() => {
    getFeedback()
      .then((f) => {
        setItems(f);
        setError(null);
      })
      .catch((e) => setError(e instanceof Error ? e.message : "Failed to load feedback"));
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const handleResolved = (updated: Feedback) => {
    setItems((prev) => prev?.map((f) => (f.id === updated.id ? updated : f)) ?? null);
  };

  const openCount = items?.filter((f) => f.status === "open").length ?? 0;

  return (
    <div className="fixed inset-0 z-30 flex">
      <div className="flex h-full w-full flex-col bg-ink-50">
        <div className="flex shrink-0 items-center gap-3 border-b border-ink-200 bg-surface px-5 py-3">
          <div className="min-w-0 flex-1">
            <p className="text-xs font-semibold text-ink-800">Feedback queue</p>
            <p className="text-[10px] text-ink-400">
              {items ? `${openCount} open, ${items.length} total` : "Loading…"}
            </p>
          </div>
          <button
            onClick={onClose}
            className="rounded-md p-1.5 text-ink-500 transition-colors hover:bg-ink-100 hover:text-ink-800"
            aria-label="Close feedback"
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
            <p className="text-center text-sm text-ink-400">No feedback yet.</p>
          )}
          {items && items.length > 0 && (
            <div className="mx-auto grid max-w-2xl gap-2.5">
              {items.map((f) => (
                <FeedbackRow key={f.id} item={f} onResolved={handleResolved} />
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
