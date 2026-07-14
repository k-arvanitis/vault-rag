"use client";

import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import { ThumbsUp, ThumbsDown, Check } from "lucide-react";
import { submitFeedback, type Source, type FeedbackReason, type InspectTarget } from "@/lib/api";
import { toCitation, type Citation } from "@/lib/product";
import { cn } from "@/lib/utils";
import { Skeleton } from "@/components/ui/skeleton";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
}

interface Props {
  messages: Message[];
  streaming: boolean;
  onInspect?: (target: InspectTarget) => void;
}

function LoadingPill() {
  return <Skeleton className="h-2 w-12 rounded-full" />;
}

/** A compact numbered citation button. Hover or focus opens a Popover with the
 * exact supporting quote — click selects the citation and opens the Evidence
 * panel on it (see product-level Citation type, lib/product.ts). Citations
 * currently only render as a summary after the answer, not inline after each
 * claim, because the backend doesn't map individual claims to sources (see
 * TODO.md) — a fragile client-side text match was rejected in favor of this
 * honest fallback, which the product spec explicitly allows. */
function CitationChip({
  citation,
  onSelect,
}: {
  citation: Citation;
  onSelect?: (target: InspectTarget) => void;
}) {
  const [open, setOpen] = useState(false);
  const target: InspectTarget = { filename: citation.sourceId, page: citation.page, sheet: citation.sheet };

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger
        render={
          <button
            type="button"
            onMouseEnter={() => setOpen(true)}
            onMouseLeave={() => setOpen(false)}
            onFocus={() => setOpen(true)}
            onBlur={() => setOpen(false)}
            onClick={() => onSelect?.(target)}
            className="inline-flex size-4 items-center justify-center rounded-full bg-muted font-mono text-[10px] font-medium text-muted-foreground hover:bg-primary hover:text-primary-foreground"
          />
        }
      >
        {citation.id}
      </PopoverTrigger>
      <PopoverContent className="w-80 text-xs" onMouseEnter={() => setOpen(true)} onMouseLeave={() => setOpen(false)}>
        <p className="font-medium text-foreground">{citation.sourceName}</p>
        <p className="text-[11px] text-muted-foreground">
          {citation.page != null && `Page ${citation.page}`}
          {citation.sheet && `Sheet: ${citation.sheet}`}
          {citation.section && ` · ${citation.section}`}
        </p>
        <p className="mt-1.5 whitespace-pre-wrap border-l-2 border-border pl-2 italic leading-relaxed text-muted-foreground">
          {citation.quote || "—"}
        </p>
        {onSelect && (
          <Button variant="link" size="xs" className="mt-1 h-auto p-0" onClick={() => onSelect(target)}>
            Open source
          </Button>
        )}
      </PopoverContent>
    </Popover>
  );
}

/** "Sources used · N" compact summary — one row per source, not the full
 * retrieved-chunk cards (those live under Technical details). */
function SourcesUsed({ sources, onInspect }: { sources: Source[]; onInspect?: (target: InspectTarget) => void }) {
  if (sources.length === 0) return null;
  const citations = sources.map((s, i) => toCitation(s, i + 1));

  return (
    <div className="mt-2">
      <p className="text-[11px] font-medium text-muted-foreground">Sources used · {sources.length}</p>
      <div className="mt-1 space-y-1">
        {citations.map((c) => (
          <div key={c.id} className="flex items-center gap-1.5 text-[11px] text-muted-foreground">
            <CitationChip citation={c} onSelect={onInspect} />
            <button
              type="button"
              className="min-w-0 flex-1 truncate text-left hover:text-foreground hover:underline"
              onClick={() => onInspect?.({ filename: c.sourceId, page: c.page, sheet: c.sheet })}
            >
              {c.sourceName}
              {c.page != null && ` · Page ${c.page}`}
              {c.sheet && ` · ${c.sheet}`}
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}

const REASON_OPTIONS: { value: FeedbackReason; label: string }[] = [
  { value: "wrong_source", label: "Wrong source" },
  { value: "hallucinated", label: "Hallucinated" },
  { value: "should_have_refused", label: "Should have refused" },
  { value: "missing_document", label: "Missing document" },
  { value: "other", label: "Other" },
];

/** Thumbs up/down on an answer, with a reason dropdown on thumbs-down — feeds the
 * admin feedback queue so bad answers surface for review instead of vanishing. */
function FeedbackWidget({ question, answer, sources }: { question: string; answer: string; sources: Source[] }) {
  const [rating, setRating] = useState<"up" | "down" | null>(null);
  const [reason, setReason] = useState<FeedbackReason | "">("");
  const [submitted, setSubmitted] = useState(false);

  const submit = async (r: "up" | "down", reasonValue: FeedbackReason | null) => {
    setRating(r);
    try {
      await submitFeedback(question, answer, r, reasonValue, sources);
      setSubmitted(true);
    } catch {
      // Feedback is best-effort UI polish — a failed submit shouldn't block chat use.
    }
  };

  if (submitted) {
    return (
      <p className="mt-1.5 flex items-center gap-1 text-[10px] text-muted-foreground">
        <Check className="size-3" /> Thanks for the feedback
      </p>
    );
  }

  return (
    <div className="mt-1.5 flex items-center gap-1.5">
      <Tooltip>
        <TooltipTrigger
          render={
            <Button
              variant="ghost"
              size="icon-xs"
              onClick={() => submit("up", null)}
              aria-label="Good answer"
              className={cn(rating === "up" && "text-emerald-600")}
            />
          }
        >
          <ThumbsUp />
        </TooltipTrigger>
        <TooltipContent>Good answer</TooltipContent>
      </Tooltip>
      <Tooltip>
        <TooltipTrigger
          render={
            <Button
              variant="ghost"
              size="icon-xs"
              onClick={() => setRating(rating === "down" ? null : "down")}
              aria-label="Bad answer"
              className={cn(rating === "down" && "text-destructive")}
            />
          }
        >
          <ThumbsDown />
        </TooltipTrigger>
        <TooltipContent>Bad answer</TooltipContent>
      </Tooltip>
      {rating === "down" && (
        <select
          value={reason}
          onChange={(e) => {
            const value = e.target.value as FeedbackReason;
            setReason(value);
            submit("down", value);
          }}
          className="rounded border border-input bg-background px-1.5 py-0.5 text-[10px] text-foreground"
        >
          <option value="" disabled>
            Why?
          </option>
          {REASON_OPTIONS.map((o) => (
            <option key={o.value} value={o.value}>
              {o.label}
            </option>
          ))}
        </select>
      )}
    </div>
  );
}

export default function MessageList({ messages, streaming, onInspect }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, streaming]);

  if (messages.length === 0 && !streaming) {
    return (
      <div className="flex flex-1 select-none flex-col items-center justify-center px-8 text-center">
        <svg
          className="mb-3 h-10 w-10 text-muted-foreground/50"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
          strokeWidth={1}
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5M16.5 12L12 16.5m0 0L7.5 12m4.5 4.5V3"
          />
        </svg>
        <p className="text-sm text-muted-foreground">Upload a document and ask anything.</p>
      </div>
    );
  }

  return (
    <div className="flex-1 space-y-4 overflow-y-auto px-6 py-4">
      {messages.map((msg, i) =>
        msg.role === "user" ? (
          <div key={msg.id} className="flex items-end justify-end">
            <div className="max-w-[70%] whitespace-pre-wrap rounded-xl rounded-br-sm bg-muted px-4 py-2 text-sm text-foreground">
              {msg.content}
            </div>
          </div>
        ) : (
          <div key={msg.id} className="flex items-start justify-start">
            <div className="max-w-[85%]">
              <div className="prose-ui rounded-xl rounded-bl-sm border border-border bg-card px-4 py-2 text-sm text-card-foreground">
                <ReactMarkdown>{msg.content}</ReactMarkdown>
              </div>
              {msg.sources && msg.sources.length > 0 && (
                <SourcesUsed sources={msg.sources} onInspect={onInspect} />
              )}
              <FeedbackWidget
                question={messages[i - 1]?.content ?? ""}
                answer={msg.content}
                sources={msg.sources ?? []}
              />
            </div>
          </div>
        )
      )}
      {streaming && (
        <div className="flex items-start justify-start">
          <div className="rounded-xl rounded-bl-sm border border-border bg-card px-4 py-3">
            <LoadingPill />
          </div>
        </div>
      )}
      <div ref={bottomRef} />
    </div>
  );
}
