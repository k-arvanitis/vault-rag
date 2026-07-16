"use client";

import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import { ThumbsUp, ThumbsDown, Check, RotateCcw } from "lucide-react";
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
  elapsedSeconds?: number;
  /** Primary citation action: selects this specific citation's evidence (and
   * the message's full source list, so Evidence shows the right set even for
   * an older turn) and switches the right panel to Evidence — does NOT leave
   * the chat. */
  onSelectEvidence?: (target: InspectTarget, messageSources: Source[]) => void;
  /** Secondary action: opens the full-screen document inspector. Only reached
   * via an explicit "Open source" link, never the citation click itself. */
  onOpenFullSource?: (target: InspectTarget) => void;
  /** Re-asks the question preceding this assistant message and replaces its
   * content/sources in place. No history is sent today, so this is just
   * firing the same question again — see ChatPanel.tsx's ask()/retry(). */
  onRetry?: (assistantMessageId: string) => void;
  /** Whether at least one source is indexed — drives the empty state's copy
   * and whether the example prompts are clickable. */
  hasSources?: boolean;
  onExamplePick?: (text: string) => void;
}

const EXAMPLE_PROMPTS = [
  "Summarize the key points of this document.",
  "What are the payment terms, and where do they appear?",
  "Compare the two most recent versions — what changed?",
];

/** Honest waiting state: one neutral message plus how long the request has
 * actually been running — no fake sequential "searching... verifying..."
 * steps, since the backend doesn't emit real progress events (see review
 * item H). The elapsed counter is the only truthful signal available. */
function WaitingIndicator({ elapsedSeconds }: { elapsedSeconds: number }) {
  return (
    <div className="flex items-center gap-2">
      <Skeleton className="h-2 w-2 rounded-full" />
      <p className="text-xs text-muted-foreground">
        Searching and verifying sources…{elapsedSeconds > 0 && ` (${elapsedSeconds}s)`}
      </p>
    </div>
  );
}

/** A compact numbered citation button. Hover or focus opens a Popover with the
 * title, page/sheet, section, and exact supporting quote. Clicking the chip
 * itself (or the row label) is the PRIMARY action — it selects this citation
 * in the Evidence panel and switches to it, staying in the chat. "Open
 * source" inside the popover is a deliberately separate, secondary action
 * that leaves the chat for the full document inspector. Used both inline
 * (AnswerContent, when the backend resolves a real [N] marker) and in the
 * SourcesUsed summary below the answer. */
function CitationChip({
  citation,
  messageSources,
  onSelectEvidence,
  onOpenFullSource,
}: {
  citation: Citation;
  messageSources: Source[];
  onSelectEvidence?: (target: InspectTarget, messageSources: Source[]) => void;
  onOpenFullSource?: (target: InspectTarget) => void;
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
            onClick={() => onSelectEvidence?.(target, messageSources)}
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
        {onOpenFullSource && (
          <Button variant="link" size="xs" className="mt-1 h-auto p-0" onClick={() => onOpenFullSource(target)}>
            Open full source
          </Button>
        )}
      </PopoverContent>
    </Popover>
  );
}

const INLINE_CITATION_RE = /\[(\d+)\]/g;

/** Renders the answer with inline [N] markers as real CitationChips instead of
 * plain text — the backend only ever emits an [N] that resolves to a real
 * position in `sources` (see build_citation_map in answer_pipeline.py); any
 * marker it couldn't resolve was already stripped server-side. Splits on the
 * marker and renders each text segment through its own ReactMarkdown (p
 * unwrapped to a fragment so it flows inline) — a reasonable v1 for the
 * common single-paragraph case; a multi-paragraph answer with citations
 * would have paragraph breaks flattened, since markdown block structure
 * isn't reparsed across the split. Falls back to plain ReactMarkdown when
 * there are no sources to resolve against or no markers in the text. */
function AnswerContent({
  content,
  sources,
  onSelectEvidence,
  onOpenFullSource,
}: {
  content: string;
  sources: Source[];
  onSelectEvidence?: (target: InspectTarget, messageSources: Source[]) => void;
  onOpenFullSource?: (target: InspectTarget) => void;
}) {
  INLINE_CITATION_RE.lastIndex = 0;
  if (sources.length === 0 || !INLINE_CITATION_RE.test(content)) {
    return <ReactMarkdown>{content}</ReactMarkdown>;
  }

  const segments: (string | number)[] = [];
  let lastIndex = 0;
  INLINE_CITATION_RE.lastIndex = 0;
  let match: RegExpExecArray | null;
  while ((match = INLINE_CITATION_RE.exec(content))) {
    segments.push(content.slice(lastIndex, match.index));
    segments.push(Number(match[1]));
    lastIndex = match.index + match[0].length;
  }
  segments.push(content.slice(lastIndex));

  const unwrapP = { p: ({ children }: { children?: React.ReactNode }) => <>{children}</> };

  return (
    <p>
      {segments.map((segment, i) => {
        if (typeof segment === "string") {
          return segment ? (
            <ReactMarkdown key={i} components={unwrapP}>
              {segment}
            </ReactMarkdown>
          ) : null;
        }
        const source = sources[segment - 1];
        if (!source) return `[${segment}]`;
        return (
          <CitationChip
            key={i}
            citation={toCitation(source, segment)}
            messageSources={sources}
            onSelectEvidence={onSelectEvidence}
            onOpenFullSource={onOpenFullSource}
          />
        );
      })}
    </p>
  );
}

/** "Sources used · N" compact summary — one row per source, not the full
 * retrieved-chunk cards (those live under Technical details). Every row's
 * primary click selects Evidence; only the popover's "Open full source" link
 * leaves the chat. */
function SourcesUsed({
  sources,
  onSelectEvidence,
  onOpenFullSource,
}: {
  sources: Source[];
  onSelectEvidence?: (target: InspectTarget, messageSources: Source[]) => void;
  onOpenFullSource?: (target: InspectTarget) => void;
}) {
  if (sources.length === 0) return null;
  const citations = sources.map((s, i) => toCitation(s, i + 1));

  return (
    <div className="mt-2">
      <p className="text-[11px] font-medium text-muted-foreground">Sources used · {sources.length}</p>
      <div className="mt-1 space-y-1">
        {citations.map((c) => (
          <div key={c.id} className="flex items-center gap-1.5 text-[11px] text-muted-foreground">
            <CitationChip
              citation={c}
              messageSources={sources}
              onSelectEvidence={onSelectEvidence}
              onOpenFullSource={onOpenFullSource}
            />
            <button
              type="button"
              className="min-w-0 flex-1 truncate text-left hover:text-foreground hover:underline"
              onClick={() => onSelectEvidence?.({ filename: c.sourceId, page: c.page, sheet: c.sheet }, sources)}
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

export default function MessageList({
  messages,
  streaming,
  elapsedSeconds = 0,
  onSelectEvidence,
  onOpenFullSource,
  onRetry,
  hasSources,
  onExamplePick,
}: Props) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, streaming]);

  if (messages.length === 0 && !streaming) {
    return (
      <div className="flex flex-1 select-none flex-col items-center justify-center px-8 text-center">
        <h2 className="text-lg font-semibold text-foreground">Ask questions across your documents.</h2>
        <p className="mt-1.5 max-w-sm text-sm text-muted-foreground">
          Every answer can be checked against the original page, sheet, or row it came from.
        </p>
        {hasSources ? (
          <div className="mt-5 flex max-w-md flex-col gap-1.5 select-text">
            {EXAMPLE_PROMPTS.map((p) => (
              <button
                key={p}
                type="button"
                onClick={() => onExamplePick?.(p)}
                className="rounded-lg border border-border bg-card px-3 py-2 text-left text-xs text-foreground transition-colors hover:border-foreground/20 hover:bg-muted"
              >
                {p}
              </button>
            ))}
          </div>
        ) : (
          <p className="mt-5 text-xs text-muted-foreground">
            Add a source in the sidebar to get started.
          </p>
        )}
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
                <AnswerContent
                  content={msg.content}
                  sources={msg.sources ?? []}
                  onSelectEvidence={onSelectEvidence}
                  onOpenFullSource={onOpenFullSource}
                />
              </div>
              {msg.sources && msg.sources.length > 0 ? (
                <SourcesUsed
                  sources={msg.sources}
                  onSelectEvidence={onSelectEvidence}
                  onOpenFullSource={onOpenFullSource}
                />
              ) : (
                msg.content.trim().toLowerCase() !== "unsupported" && (
                  <p className="mt-1.5 text-[10px] text-muted-foreground">
                    No source citation available for this answer yet.
                  </p>
                )
              )}
              <div className="flex items-center">
                <FeedbackWidget
                  question={messages[i - 1]?.content ?? ""}
                  answer={msg.content}
                  sources={msg.sources ?? []}
                />
                {onRetry && (
                  <Tooltip>
                    <TooltipTrigger
                      render={
                        <Button
                          variant="ghost"
                          size="icon-xs"
                          onClick={() => onRetry(msg.id)}
                          disabled={streaming}
                          aria-label="Retry this answer"
                        />
                      }
                    >
                      <RotateCcw />
                    </TooltipTrigger>
                    <TooltipContent>Retry</TooltipContent>
                  </Tooltip>
                )}
              </div>
            </div>
          </div>
        )
      )}
      {streaming && (
        <div className="flex items-start justify-start">
          <div className="rounded-xl rounded-bl-sm border border-border bg-card px-4 py-3">
            <WaitingIndicator elapsedSeconds={elapsedSeconds} />
          </div>
        </div>
      )}
      <div ref={bottomRef} />
    </div>
  );
}
