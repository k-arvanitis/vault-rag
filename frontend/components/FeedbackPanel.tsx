"use client";

import { useCallback, useEffect, useState } from "react";
import { X, ThumbsUp, ThumbsDown, Loader2 } from "lucide-react";
import { getFeedback, resolveFeedback, type Feedback } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Textarea } from "@/components/ui/textarea";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Skeleton } from "@/components/ui/skeleton";
import { Table, TableBody, TableCell, TableRow } from "@/components/ui/table";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

interface Props {
  onClose: () => void;
}

const ACTIONS = [
  { value: "mark_correct_source", label: "Mark correct source" },
  { value: "add_to_eval_set", label: "Add to eval set" },
  { value: "dismissed", label: "Dismiss" },
];

function FeedbackDetail({ item, onResolved }: { item: Feedback; onResolved: (f: Feedback) => void }) {
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState(item.note ?? "");
  const [action, setAction] = useState(ACTIONS[0].value);

  const resolve = async () => {
    setBusy(true);
    try {
      const updated = await resolveFeedback(item.id, action, note || undefined);
      onResolved(updated);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-1.5">
        {item.rating === "up" ? (
          <ThumbsUp className="size-3.5 text-emerald-600" />
        ) : (
          <ThumbsDown className="size-3.5 text-destructive" />
        )}
        {item.reason && <Badge variant="secondary">{item.reason.replace(/_/g, " ")}</Badge>}
        <Badge variant={item.status === "resolved" ? "default" : "outline"}>{item.status}</Badge>
        <span className="ml-auto text-[10px] text-muted-foreground">
          {new Date(item.created_at).toLocaleString()}
        </span>
      </div>

      <div>
        <p className="text-xs font-medium text-foreground">{item.question}</p>
        <p className="mt-1 text-[11px] text-muted-foreground">{item.answer}</p>
      </div>

      {item.sources.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {item.sources.map((s, i) => (
            <Badge key={`${s.filename}-${i}`} variant="outline" title={s.excerpt || s.quote}>
              {s.filename}
              {s.page != null ? ` p.${s.page}` : ""}
            </Badge>
          ))}
        </div>
      )}

      {item.status === "open" ? (
        <div className="space-y-2">
          <Textarea
            value={note}
            onChange={(e) => setNote(e.target.value)}
            placeholder="Admin note (optional)"
            className="min-h-16 text-xs"
          />
          <div className="flex items-center gap-2">
            <Select value={action} onValueChange={(v) => v && setAction(v)}>
              <SelectTrigger className="flex-1">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {ACTIONS.map((a) => (
                  <SelectItem key={a.value} value={a.value}>
                    {a.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Button size="sm" onClick={resolve} disabled={busy}>
              {busy ? <Loader2 className="animate-spin" /> : "Resolve"}
            </Button>
          </div>
        </div>
      ) : (
        <p className="text-[10px] text-muted-foreground">
          Resolved: {item.action?.replace(/_/g, " ")}
          {item.note ? ` — ${item.note}` : ""}
        </p>
      )}
    </div>
  );
}

export default function FeedbackPanel({ onClose }: Props) {
  const [items, setItems] = useState<Feedback[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<Feedback | null>(null);

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
    setSelected(updated);
  };

  const openCount = items?.filter((f) => f.status === "open").length ?? 0;

  return (
    <div className="fixed inset-0 z-30 flex">
      <div className="flex h-full w-full flex-col bg-background">
        <div className="flex shrink-0 items-center gap-3 border-b border-border bg-card px-5 py-3">
          <div className="min-w-0 flex-1">
            <p className="text-xs font-semibold text-foreground">Feedback queue</p>
            <p className="text-[10px] text-muted-foreground">
              {items ? `${openCount} open, ${items.length} total` : "Loading…"}
            </p>
          </div>
          <Button variant="ghost" size="icon-sm" onClick={onClose} aria-label="Close feedback">
            <X />
          </Button>
        </div>

        <div className="flex-1 overflow-y-auto p-5">
          {error && (
            <Alert variant="destructive" className="mx-auto mb-3 max-w-2xl">
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}
          {!error && !items && (
            <div className="mx-auto max-w-2xl space-y-2">
              <Skeleton className="h-10 w-full" />
              <Skeleton className="h-10 w-full" />
              <Skeleton className="h-10 w-full" />
            </div>
          )}
          {items && items.length === 0 && (
            <p className="text-center text-sm text-muted-foreground">No feedback yet.</p>
          )}
          {items && items.length > 0 && (
            <div className="mx-auto max-w-2xl overflow-hidden rounded-lg border border-border">
              <Table>
                <TableBody>
                  {items.map((f) => (
                    <TableRow
                      key={f.id}
                      className="cursor-pointer"
                      onClick={() => setSelected(f)}
                    >
                      <TableCell className="w-6">
                        {f.rating === "up" ? (
                          <ThumbsUp className="size-3.5 text-emerald-600" />
                        ) : (
                          <ThumbsDown className="size-3.5 text-destructive" />
                        )}
                      </TableCell>
                      <TableCell className="max-w-0 truncate whitespace-nowrap text-foreground">
                        {f.question}
                      </TableCell>
                      <TableCell className="w-24">
                        <Badge variant={f.status === "resolved" ? "default" : "outline"}>{f.status}</Badge>
                      </TableCell>
                      <TableCell className="w-36 text-right text-[10px] text-muted-foreground">
                        {new Date(f.created_at).toLocaleDateString()}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          )}
        </div>
      </div>

      <Dialog open={!!selected} onOpenChange={(open) => !open && setSelected(null)}>
        <DialogContent className="sm:max-w-lg">
          <DialogHeader>
            <DialogTitle>Feedback detail</DialogTitle>
            <DialogDescription className="sr-only">Full question, answer, and resolution controls</DialogDescription>
          </DialogHeader>
          {selected && <FeedbackDetail item={selected} onResolved={handleResolved} />}
        </DialogContent>
      </Dialog>
    </div>
  );
}
