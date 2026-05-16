"use client";

import { useState } from "react";
import { ChevronDown, ChevronRight, Database, FileText, Wrench } from "lucide-react";
import { type Source } from "@/lib/api";
import SourceCard from "./SourceCard";

export interface Trace {
  sources: Source[];
  sql: string[];
  tools_used: string[];
}

interface Props {
  trace: Trace | null;
}

const TOOL_META: Record<string, { label: string; icon: typeof Wrench }> = {
  query_excel: { label: "SQL · DuckDB", icon: Database },
  search_documents: { label: "RAG · Qdrant", icon: FileText },
};

function Panel({
  title,
  count,
  defaultOpen,
  children,
}: {
  title: string;
  count?: number;
  defaultOpen?: boolean;
  children: React.ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen ?? true);
  return (
    <div className="overflow-hidden rounded-lg border border-ink-200 bg-surface">
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center justify-between border-b border-ink-100 px-3 py-2 text-left hover:bg-ink-50"
      >
        <span className="text-xs font-semibold uppercase tracking-wide text-ink-500">
          {title}
          {typeof count === "number" && (
            <span className="ml-1.5 font-mono text-[10px] text-ink-400">({count})</span>
          )}
        </span>
        {open ? (
          <ChevronDown className="h-3.5 w-3.5 text-ink-400" />
        ) : (
          <ChevronRight className="h-3.5 w-3.5 text-ink-400" />
        )}
      </button>
      {open && <div className="p-3">{children}</div>}
    </div>
  );
}

export default function TraceSidebar({ trace }: Props) {
  if (!trace) {
    return (
      <aside className="hidden w-[320px] shrink-0 overflow-y-auto border-l border-ink-200 bg-ink-50 p-3 lg:block">
        <div className="rounded-lg border border-dashed border-ink-200 bg-surface px-3 py-6 text-center">
          <p className="text-[11px] uppercase tracking-wide text-ink-400">Trace</p>
          <p className="mt-1.5 text-xs text-ink-500">
            Ask a question — tools, SQL, and retrieved chunks for the latest turn appear here.
          </p>
        </div>
      </aside>
    );
  }

  const { sources, sql, tools_used } = trace;

  return (
    <aside className="hidden w-[320px] shrink-0 overflow-y-auto border-l border-ink-200 bg-ink-50 p-3 lg:block">
      <div className="space-y-3">
        {tools_used.length > 0 && (
          <Panel title="Tools" count={tools_used.length}>
            <div className="flex flex-wrap gap-1.5">
              {tools_used.map((t) => {
                const meta = TOOL_META[t] ?? { label: t, icon: Wrench };
                const Icon = meta.icon;
                return (
                  <span
                    key={t}
                    className="inline-flex items-center gap-1 rounded-full border border-ink-200 bg-ink-50 px-2 py-0.5 text-[10px] font-medium text-ink-700"
                  >
                    <Icon className="h-3 w-3 text-brand-dark" />
                    {meta.label}
                  </span>
                );
              })}
            </div>
          </Panel>
        )}

        {sql.length > 0 && (
          <Panel title="Generated SQL" count={sql.length}>
            <div className="space-y-2">
              {sql.map((q, i) => (
                <pre
                  key={i}
                  className="overflow-x-auto rounded bg-slate-900 p-2 font-mono text-[10px] leading-relaxed text-slate-50"
                >
                  {q}
                </pre>
              ))}
            </div>
          </Panel>
        )}

        {sources.length > 0 && (
          <Panel title="Retrieved chunks" count={sources.length}>
            <div className="space-y-2">
              {sources.map((s, i) => (
                <SourceCard key={i} source={s} index={i + 1} />
              ))}
            </div>
          </Panel>
        )}

        {sources.length === 0 && sql.length === 0 && (
          <div className="rounded-lg border border-dashed border-ink-200 bg-surface px-3 py-4 text-center">
            <p className="text-xs text-ink-500">No tool calls for this turn.</p>
          </div>
        )}
      </div>
    </aside>
  );
}
