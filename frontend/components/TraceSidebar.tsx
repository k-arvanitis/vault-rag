"use client";

import { ChevronDown, ChevronRight, Database, FileText, Wrench } from "lucide-react";
import { type Source, type RejectedSource, type InspectTarget, sourceInspectTarget } from "@/lib/api";
import { Badge } from "@/components/ui/badge";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import SourceCard from "./SourceCard";

export interface Trace {
  sources: Source[];
  rejected_sources?: RejectedSource[];
  sql: string[];
  tools_used: string[];
  /** Final answer text — used to pick which citation the Evidence panel
   * defaults to (the first one actually cited, not just index 0). */
  answer?: string;
}

interface Props {
  trace: Trace | null;
  onInspect?: (target: InspectTarget) => void;
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
  return (
    <Collapsible defaultOpen={defaultOpen ?? true} className="overflow-hidden rounded-lg border border-border bg-card">
      <CollapsibleTrigger className="group flex w-full items-center justify-between border-b border-border px-3 py-2 text-left hover:bg-muted">
        <span className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
          {title}
          {typeof count === "number" && (
            <span className="ml-1.5 font-mono text-[10px] text-muted-foreground/70">({count})</span>
          )}
        </span>
        <ChevronDown className="size-3.5 text-muted-foreground group-data-[panel-open]:hidden" />
        <ChevronRight className="hidden size-3.5 text-muted-foreground group-data-[panel-open]:block" />
      </CollapsibleTrigger>
      <CollapsibleContent className="p-3">{children}</CollapsibleContent>
    </Collapsible>
  );
}

export default function TraceSidebar({ trace, onInspect }: Props) {
  if (!trace) {
    return (
      <aside className="h-full w-full overflow-y-auto bg-background p-3 lg:w-[320px] lg:shrink-0 lg:border-l lg:border-border">
        <div className="rounded-lg border border-dashed border-border bg-card px-3 py-6 text-center">
          <p className="text-[11px] uppercase tracking-wide text-muted-foreground">Trace</p>
          <p className="mt-1.5 text-xs text-muted-foreground">
            Ask a question — tools, SQL, and retrieved chunks for the latest turn appear here.
          </p>
        </div>
      </aside>
    );
  }

  const { sources, sql, tools_used, rejected_sources = [] } = trace;

  return (
    <aside className="h-full w-full overflow-y-auto bg-background p-3 lg:w-[320px] lg:shrink-0 lg:border-l lg:border-border">
      <div className="space-y-3">
        {tools_used.length > 0 && (
          <Panel title="Tools" count={tools_used.length}>
            <div className="flex flex-wrap gap-1.5">
              {tools_used.map((t) => {
                const meta = TOOL_META[t] ?? { label: t, icon: Wrench };
                const Icon = meta.icon;
                return (
                  <Badge key={t} variant="secondary" className="gap-1">
                    <Icon data-icon="inline-start" />
                    {meta.label}
                  </Badge>
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
                <SourceCard
                  key={i}
                  source={s}
                  index={i + 1}
                  onInspect={onInspect ? () => onInspect(sourceInspectTarget(s)) : undefined}
                />
              ))}
            </div>
          </Panel>
        )}

        {rejected_sources.length > 0 && (
          <Panel title="Retrieved but rejected" count={rejected_sources.length} defaultOpen={false}>
            <ul className="space-y-1.5">
              {rejected_sources.map((r, i) => (
                <li key={i} className="flex items-center justify-between gap-2 text-[11px] text-muted-foreground">
                  <span className="truncate">{r.filename}</span>
                  {r.score !== null && (
                    <span className="shrink-0 font-mono text-[10px] text-muted-foreground/70">
                      {r.score.toFixed(2)}
                    </span>
                  )}
                </li>
              ))}
            </ul>
            <p className="mt-2 text-[10px] text-muted-foreground/70">
              Lower reranker relevance than the sources used above.
            </p>
          </Panel>
        )}

        {sources.length === 0 && sql.length === 0 && (
          <div className="rounded-lg border border-dashed border-border bg-card px-3 py-4 text-center">
            <p className="text-xs text-muted-foreground">No tool calls for this turn.</p>
          </div>
        )}
      </div>
    </aside>
  );
}
