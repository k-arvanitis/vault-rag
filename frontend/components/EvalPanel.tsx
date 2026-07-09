"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { X, Loader2, Play } from "lucide-react";
import { getEvalSummary, runEval, getEvalStatus, type EvalSummary } from "@/lib/api";

interface Props {
  onClose: () => void;
}

function pct(v: unknown): string {
  return typeof v === "number" ? `${(v * 100).toFixed(1)}%` : String(v);
}

function MetricTable({ title, metrics }: { title: string; metrics: Record<string, unknown> }) {
  const rows = Object.entries(metrics).filter(([, v]) => typeof v === "number" || typeof v === "string");
  if (rows.length === 0) return null;
  return (
    <div className="overflow-hidden rounded-lg border border-ink-200 bg-surface">
      <p className="border-b border-ink-100 px-3 py-2 text-xs font-semibold uppercase tracking-wide text-ink-500">
        {title}
      </p>
      <table className="w-full text-xs">
        <tbody>
          {rows.map(([k, v]) => (
            <tr key={k} className="border-b border-ink-100 last:border-0">
              <td className="px-3 py-1.5 text-ink-500">{k.replace(/_/g, " ")}</td>
              <td className="px-3 py-1.5 text-right font-mono font-medium text-ink-800">
                {typeof v === "number" && v <= 1 && v >= 0 ? pct(v) : String(v)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function EvalPanel({ onClose }: Props) {
  const [summary, setSummary] = useState<EvalSummary | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
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

  return (
    <div className="fixed inset-0 z-30 flex">
      <div className="flex h-full w-full flex-col bg-ink-50">
        <div className="flex shrink-0 items-center gap-3 border-b border-ink-200 bg-surface px-5 py-3">
          <div className="min-w-0 flex-1">
            <p className="text-xs font-semibold text-ink-800">Evaluation</p>
            <p className="text-[10px] text-ink-400">
              Last computed benchmark results (run <code className="font-mono">make eval</code> to refresh)
            </p>
          </div>
          <button
            onClick={handleRun}
            disabled={running}
            className="flex items-center gap-1.5 rounded-md border border-ink-200 px-2.5 py-1.5 text-xs font-medium text-ink-600 transition-colors hover:bg-ink-100 disabled:opacity-50"
          >
            {running ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Play className="h-3.5 w-3.5" />}
            {running ? "Running…" : "Run eval"}
          </button>
          <button
            onClick={onClose}
            className="rounded-md p-1.5 text-ink-500 transition-colors hover:bg-ink-100 hover:text-ink-800"
            aria-label="Close evaluation"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-5">
          {error && <p className="text-sm text-red-600">{error}</p>}
          {!error && !summary && (
            <div className="flex flex-1 items-center justify-center py-20">
              <Loader2 className="h-5 w-5 animate-spin text-ink-400" />
            </div>
          )}
          {summary && (
            <div className="mx-auto grid max-w-2xl gap-3">
              <p className="text-xs text-ink-500">{summary.question_count} benchmark questions</p>
              {Object.entries(summary)
                .filter(([k, v]) => k !== "question_count" && v && typeof v === "object" && !Array.isArray(v))
                .map(([key, value]) => (
                  <MetricTable key={key} title={key.replace(/_/g, " ")} metrics={value as Record<string, unknown>} />
                ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
