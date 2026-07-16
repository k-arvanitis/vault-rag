"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { X, Loader2, Play } from "lucide-react";
import { getEvalSummary, runEval, getEvalStatus, type EvalSummary } from "@/lib/api";
import { useAdminSession } from "@/lib/useAdminSession";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableRow } from "@/components/ui/table";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Skeleton } from "@/components/ui/skeleton";
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
}

function pct(v: unknown): string {
  return typeof v === "number" ? `${(v * 100).toFixed(1)}%` : String(v);
}

/** Looks up a known metric name at the top level or one level of nesting —
 * doesn't assume a fixed schema, just surfaces it as a headline if present. */
function findMetric(summary: EvalSummary, key: string): number | null {
  const top = summary[key];
  if (typeof top === "number") return top;
  for (const v of Object.values(summary)) {
    if (v && typeof v === "object" && !Array.isArray(v) && key in (v as Record<string, unknown>)) {
      const nested = (v as Record<string, unknown>)[key];
      if (typeof nested === "number") return nested;
    }
  }
  return null;
}

const HEADLINE_METRICS: { key: string; label: string }[] = [
  { key: "correctness", label: "Correctness" },
  { key: "faithfulness", label: "Faithfulness" },
  { key: "answer_relevancy", label: "Answer relevancy" },
  { key: "hit_at_5", label: "Hit@5" },
  { key: "answer_accuracy", label: "Structured accuracy" },
  { key: "correct_refusal_rate", label: "Refusal rate" },
];

function MetricTable({ title, metrics }: { title: string; metrics: Record<string, unknown> }) {
  const rows = Object.entries(metrics).filter(([, v]) => typeof v === "number" || typeof v === "string");
  if (rows.length === 0) return null;
  return (
    <Card size="sm">
      <CardHeader>
        <CardTitle className="text-xs uppercase tracking-wide text-muted-foreground">{title}</CardTitle>
      </CardHeader>
      <CardContent className="px-0">
        <Table>
          <TableBody>
            {rows.map(([k, v]) => (
              <TableRow key={k}>
                <TableCell className="text-muted-foreground">{k.replace(/_/g, " ")}</TableCell>
                <TableCell className="text-right font-mono font-medium text-foreground">
                  {typeof v === "number" && v <= 1 && v >= 0 ? pct(v) : String(v)}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  );
}

export default function EvalPanel({ onClose }: Props) {
  const [summary, setSummary] = useState<EvalSummary | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [confirmRun, setConfirmRun] = useState(false);
  const { is_admin: isAdmin } = useAdminSession();
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const load = useCallback(() => {
    getEvalSummary()
      .then((s) => {
        setSummary(s);
        setError(null);
      })
      .catch((e) => setError(e instanceof Error ? e.message : "Failed to load eval results"));
  }, []);

  useEffect(() => {
    load();
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [load]);

  const handleRun = useCallback(async () => {
    setRunning(true);
    try {
      const { job_id } = await runEval();
      pollRef.current = setInterval(async () => {
        const status = await getEvalStatus(job_id);
        if (status.status === "done" || status.status === "failed") {
          if (pollRef.current) clearInterval(pollRef.current);
          setRunning(false);
          if (status.status === "done") load();
          else setError(status.error ?? "Eval run failed");
        }
      }, 5000);
    } catch (e) {
      setRunning(false);
      setError(e instanceof Error ? e.message : "Failed to start eval run");
    }
  }, [load]);

  const headline = summary
    ? HEADLINE_METRICS.map((m) => ({ ...m, value: findMetric(summary, m.key) })).filter((m) => m.value != null)
    : [];

  return (
    <div className="fixed inset-0 z-30 flex">
      <div className="flex h-full w-full flex-col bg-background">
        <div className="flex shrink-0 items-center gap-3 border-b border-border bg-card px-5 py-3">
          <div className="min-w-0 flex-1">
            <p className="text-xs font-semibold text-foreground">Evaluation</p>
            <p className="text-[10px] text-muted-foreground">
              Last computed benchmark results (run <code className="font-mono">make eval</code> to refresh)
            </p>
          </div>
          {running && (
            <Badge variant="secondary" className="gap-1">
              <Loader2 className="size-3 animate-spin" />
              Running
            </Badge>
          )}
          <AlertDialog open={confirmRun} onOpenChange={setConfirmRun}>
            {isAdmin && (
              <Button variant="outline" size="sm" onClick={() => setConfirmRun(true)} disabled={running}>
                <Play data-icon="inline-start" />
                Run eval
              </Button>
            )}
            <AlertDialogContent>
              <AlertDialogHeader>
                <AlertDialogTitle>Run the evaluation suite?</AlertDialogTitle>
                <AlertDialogDescription>
                  This invokes real models against the full benchmark set and can take several minutes.
                </AlertDialogDescription>
              </AlertDialogHeader>
              <AlertDialogFooter>
                <AlertDialogCancel>Cancel</AlertDialogCancel>
                <AlertDialogAction
                  onClick={() => {
                    setConfirmRun(false);
                    handleRun();
                  }}
                >
                  Run
                </AlertDialogAction>
              </AlertDialogFooter>
            </AlertDialogContent>
          </AlertDialog>
          <Button variant="ghost" size="icon-sm" onClick={onClose} aria-label="Close evaluation">
            <X />
          </Button>
        </div>

        <div className="flex-1 overflow-y-auto p-5">
          {error && (
            <Alert variant="destructive" className="mx-auto mb-3 max-w-2xl">
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}
          {!error && !summary && (
            <div className="mx-auto max-w-2xl space-y-3">
              <Skeleton className="h-20 w-full" />
              <Skeleton className="h-32 w-full" />
              <Skeleton className="h-32 w-full" />
            </div>
          )}
          {summary && (
            <div className="mx-auto max-w-2xl space-y-4">
              <p className="text-xs text-muted-foreground">{summary.question_count} benchmark questions</p>

              {headline.length > 0 && (
                <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
                  {headline.map((m) => (
                    <Card key={m.key} size="sm">
                      <CardContent className="px-3">
                        <p className="text-[10px] uppercase tracking-wide text-muted-foreground">{m.label}</p>
                        <p className="mt-1 text-lg font-semibold text-foreground">
                          {m.value! <= 1 && m.value! >= 0 ? pct(m.value) : m.value}
                        </p>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              )}

              <div className="grid gap-3">
                {Object.entries(summary)
                  .filter(([k, v]) => k !== "question_count" && v && typeof v === "object" && !Array.isArray(v))
                  .map(([key, value]) => (
                    <MetricTable key={key} title={key.replace(/_/g, " ")} metrics={value as Record<string, unknown>} />
                  ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
