"use client";

import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeRaw from "rehype-raw";
import { ThumbsUp, ThumbsDown, Check, RotateCcw } from "lucide-react";
import { submitFeedback, type Source, type FeedbackReason, type InspectTarget } from "@/lib/api";
import { toCitation, citedIndices, citedOnlySources, type Citation } from "@/lib/product";
import { useAdminSession } from "@/lib/useAdminSession";
import { cn } from "@/lib/utils";
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
  /** Primary citation action: selects this specific citation's evidence (and
   * the message's full source list, so Evidence shows the right set even for
   * an older turn) and switches the right panel to Evidence — does NOT leave
   * the chat. */
  onSelectEvidence?: (target: InspectTarget, messageSources: Source[]) => void;
  /** Secondary action: opens the full-screen document inspector. Only reached
   * via an explicit "Open source" link, never the citation click itself. */
  onOpenFullSource?: (target: InspectTarget) => void;
  /** The citation currently shown in the Evidence panel — matching chips/rows
   * render a selected state so the answer, source list, and panel agree. */
  selectedCitation?: InspectTarget | null;
  /** Re-asks the question preceding this assistant message and replaces its
   * content/sources in place. No history is sent today, so this is just
   * firing the same question again — see ChatPanel.tsx's ask()/retry(). */
  onRetry?: (assistantMessageId: string) => void;
  /** Whether at least one source is indexed — drives the empty state's copy
   * and whether the example prompts are clickable. */
  hasSources?: boolean;
  onExamplePick?: (text: string) => void;
  /** Current "Ask across" source scope — [] means every indexed source, one
   * filename means a single source, several means a multi-source compare.
   * Drives the empty state's headline/subtext (see review: "make the empty
   * state respond to scope"). */
  scopedDocIds?: string[];
}

/** Real, verified-passing questions against this project's fixed demo
 * corpus (see eval/data/qa_pairs/) — not generic placeholders that might
 * miss the indexed collection. One single-doc factoid, one cross-document
 * compare, one structured (DuckDB) lookup. */
const EXAMPLE_PROMPTS = [
  "Who must approve a Sole Source Procurement under the procurement policy?",
  "Compare the procurement policy and the services contract terms — which allows a longer contract extension?",
  "What is the NET Amount for supplier Citycoseals?",
];

/** Honest waiting state: one neutral message, no fake sequential
 * "searching... verifying..." steps, since the backend doesn't emit real
 * progress events (see review item H). A shimmering gradient sweep across
 * the label itself instead of an elapsed-time counter or a spinner/dots --
 * the number invited "why is this taking N seconds" more than it reassured
 * anyone the request was still alive. */
function WaitingIndicator() {
  return (
    <p className="animate-text-shimmer text-xs font-medium">
      Searching and verifying sources…
    </p>
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
/** Whether an InspectTarget refers to the same evidence location — same file,
 * and same page/sheet when either target specifies one. */
function isSameTarget(a: InspectTarget, b: InspectTarget | null | undefined): boolean {
  if (!b || a.filename !== b.filename) return false;
  if (a.page != null || b.page != null) return a.page === b.page;
  if (a.sheet || b.sheet) return a.sheet === b.sheet;
  return true;
}

function CitationChip({
  citation,
  messageSources,
  selectedCitation,
  onSelectEvidence,
  onOpenFullSource,
}: {
  citation: Citation;
  messageSources: Source[];
  selectedCitation?: InspectTarget | null;
  onSelectEvidence?: (target: InspectTarget, messageSources: Source[]) => void;
  onOpenFullSource?: (target: InspectTarget) => void;
}) {
  const [open, setOpen] = useState(false);
  const target: InspectTarget = { filename: citation.sourceId, page: citation.page, sheet: citation.sheet };
  const selected = isSameTarget(target, selectedCitation);

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger
        render={
          <button
            type="button"
            onClick={() => {
              onSelectEvidence?.(target, messageSources);
              setOpen((o) => !o);
            }}
            aria-pressed={selected}
            className={cn(
              "ml-1.5 mr-0.5 inline-flex size-4 items-center justify-center rounded-full font-mono text-[10px] font-medium transition-colors hover:bg-primary hover:text-primary-foreground",
              selected
                ? "bg-primary text-primary-foreground ring-2 ring-primary/40"
                : "bg-muted text-muted-foreground"
            )}
          />
        }
      >
        {citation.id}
      </PopoverTrigger>
      <PopoverContent className="w-80 text-xs">
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
          <Button
            variant="link"
            size="xs"
            className="mt-1 h-auto p-0"
            onClick={() => {
              onOpenFullSource(target);
              setOpen(false);
            }}
          >
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
 * marker it couldn't resolve was already stripped server-side.
 *
 * Swaps each [N] for a raw <cite> tag and renders the WHOLE content through
 * one ReactMarkdown pass (rehype-raw lets the literal tag through, same
 * pattern InspectorPanel.tsx uses for embedded OCR <table> HTML). Markdown
 * block structure — lists, paragraphs — is parsed once and stays intact.
 * An earlier version split the raw string on each [N] and re-parsed every
 * fragment independently; a marker landing inside a list item ("- doc_001
 * ... [1]") made that fragment's own reparse treat "- doc_001 ..." as a
 * complete <ul><li>, a block element, forcing the citation chip that
 * followed onto its own line. Falls back to plain ReactMarkdown when there
 * are no sources to resolve against or no markers in the text. */
export function AnswerContent({
  content,
  sources,
  selectedCitation,
  onSelectEvidence,
  onOpenFullSource,
}: {
  content: string;
  sources: Source[];
  selectedCitation?: InspectTarget | null;
  onSelectEvidence?: (target: InspectTarget, messageSources: Source[]) => void;
  onOpenFullSource?: (target: InspectTarget) => void;
}) {
  INLINE_CITATION_RE.lastIndex = 0;
  if (sources.length === 0 || !INLINE_CITATION_RE.test(content)) {
    return <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>;
  }

  const withCiteTags = content.replace(
    INLINE_CITATION_RE,
    (_, n) => `<cite data-citation="${n}"></cite>`
  );
  const messageSources = citedOnlySources(content, sources);

  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      rehypePlugins={[rehypeRaw]}
      components={{
        cite: ({ node }: { node?: { properties?: Record<string, unknown> } }) => {
          const n = Number(node?.properties?.dataCitation);
          const source = sources[n - 1];
          if (!source) return <>{`[${n}]`}</>;
          return (
            <CitationChip
              citation={toCitation(source, n)}
              messageSources={messageSources}
              selectedCitation={selectedCitation}
              onSelectEvidence={onSelectEvidence}
              onOpenFullSource={onOpenFullSource}
            />
          );
        },
      }}
    >
      {withCiteTags}
    </ReactMarkdown>
  );
}

/** "Sources used · N" compact summary — one row per source, not the full
 * retrieved-chunk cards (those live under Technical details). Every row's
 * primary click selects Evidence; only the popover's "Open full source" link
 * leaves the chat. */
function SourceRow({
  citation,
  sources,
  selectedCitation,
  onSelectEvidence,
  onOpenFullSource,
}: {
  citation: Citation;
  sources: Source[];
  selectedCitation?: InspectTarget | null;
  onSelectEvidence?: (target: InspectTarget, messageSources: Source[]) => void;
  onOpenFullSource?: (target: InspectTarget) => void;
}) {
  const target: InspectTarget = { filename: citation.sourceId, page: citation.page, sheet: citation.sheet };
  const selected = isSameTarget(target, selectedCitation);
  return (
    <div
      className={cn(
        "flex items-center gap-1.5 rounded px-1 py-0.5 text-[11px] text-muted-foreground",
        selected && "bg-primary/10"
      )}
    >
      <CitationChip
        citation={citation}
        messageSources={sources}
        selectedCitation={selectedCitation}
        onSelectEvidence={onSelectEvidence}
        onOpenFullSource={onOpenFullSource}
      />
      <button
        type="button"
        className="min-w-0 flex-1 truncate text-left hover:text-foreground hover:underline"
        onClick={() => onSelectEvidence?.(target, sources)}
      >
        {citation.sourceName}
        {citation.page != null && ` · Page ${citation.page}`}
        {citation.sheet && ` · ${citation.sheet}`}
      </button>
    </div>
  );
}

/** "Sources used" only lists sources the answer actually cites via [N] — the
 * rest of the retrieved candidates (reranked out, or retrieved but unused)
 * move to Technical details so a one-fact answer doesn't look backed by 8
 * sources. Falls back to showing everything when the answer cites nothing
 * (e.g. a table/SQL answer with no inline markers). */
function SourcesUsed({
  content,
  sources,
  selectedCitation,
  onSelectEvidence,
  onOpenFullSource,
}: {
  content: string;
  sources: Source[];
  selectedCitation?: InspectTarget | null;
  onSelectEvidence?: (target: InspectTarget, messageSources: Source[]) => void;
  onOpenFullSource?: (target: InspectTarget) => void;
}) {
  if (sources.length === 0) return null;
  // "Unsupported" means nothing actually backed the answer -- and an answer
  // that cites nothing inline isn't "backed by every retrieved candidate"
  // either. Both used to fall through to the same "show every retrieved
  // candidate" block, which read as "N sources used" for an answer that
  // used zero of them -- reproduced live: a correct, single-fact answer
  // with no [N] marker showed 8 unrelated retrieved pages as if all 8 were
  // its evidence. Neither case fabricates a citation; both say plainly that
  // nothing specific was pinned down, leaving the raw candidates in
  // Technical details for inspection instead of asserting they were used.
  const isUnsupported = content.trim().toLowerCase() === "unsupported";
  const cited = citedIndices(content);
  const citations = sources.map((s, i) => toCitation(s, i + 1));
  const noCitations = !isUnsupported && cited.size === 0;
  const shown = isUnsupported || noCitations ? [] : citations.filter((c) => cited.has(c.id));
  const remaining = citations.length - shown.length;
  const shownSources = citedOnlySources(content, sources);

  if (isUnsupported) {
    return (
      <p className="mt-2 text-[10px] text-muted-foreground">
        {citations.length} retrieved candidate{citations.length > 1 ? "s" : ""} didn&apos;t support an
        answer — see Technical details.
      </p>
    );
  }

  if (noCitations) {
    return (
      <p className="mt-2 text-[10px] text-muted-foreground">
        Answered without pinning a specific citation — {citations.length} retrieved candidate
        {citations.length > 1 ? "s" : ""} in Technical details.
      </p>
    );
  }

  return (
    <div className="mt-2">
      <p className="text-[11px] font-medium text-muted-foreground">Sources used · {shown.length}</p>
      <div className="mt-1 space-y-1">
        {shown.map((c) => (
          <SourceRow
            key={c.id}
            citation={c}
            sources={shownSources}
            selectedCitation={selectedCitation}
            onSelectEvidence={onSelectEvidence}
            onOpenFullSource={onOpenFullSource}
          />
        ))}
      </div>
      {remaining > 0 && (
        <p className="mt-1 text-[10px] text-muted-foreground">
          {remaining} more retrieved candidate{remaining > 1 ? "s" : ""} in Technical details
        </p>
      )}
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
  onSelectEvidence,
  onOpenFullSource,
  selectedCitation = null,
  onRetry,
  hasSources,
  onExamplePick,
  scopedDocIds = [],
}: Props) {
  const bottomRef = useRef<HTMLDivElement>(null);
  const { is_admin: isAdmin } = useAdminSession();

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, streaming]);

  if (messages.length === 0 && !streaming) {
    let headline = "Ask across your organisation's knowledge base.";
    let subtext =
      "Get answers grounded in approved PDFs, spreadsheets, policies and business records, with citations to the exact page, sheet or row.";
    if (scopedDocIds.length === 1) {
      headline = `Ask about ${scopedDocIds[0].split("/").pop()}.`;
      subtext = "Answers will be restricted to this source and linked to the original evidence.";
    } else if (scopedDocIds.length > 1) {
      headline = `Ask across ${scopedDocIds.length} selected sources.`;
      subtext = "Compare and combine information from the selected documents, with evidence from each source.";
    }

    return (
      <div className="flex flex-1 select-none flex-col items-center justify-center px-8 text-center">
        <h2 className="text-lg font-semibold text-foreground">
          {hasSources
            ? headline
            : isAdmin
              ? "Build your organisation's knowledge base."
              : "Ask across your organisation's knowledge base."}
        </h2>
        <p className="mt-1.5 max-w-sm text-sm text-muted-foreground">
          {hasSources
            ? subtext
            : isAdmin
              ? "Add PDFs, spreadsheets and business documents so users can ask verified questions across them."
              : "No approved sources are available yet. Contact your administrator to add content to the knowledge base."}
        </p>
        {hasSources && (
          <div className="mt-5 flex max-w-md flex-col gap-1.5 select-text">
            <p className="mb-0.5 text-[11px] font-medium tracking-wide text-muted-foreground">
              Try asking
            </p>
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
        ) : (() => {
          // A streamed answer's message exists (empty, then filling in) from
          // the moment the request starts — the active turn's placeholder is
          // always the last message while streaming (see ChatPanel's
          // send()/retry()). Shows the waiting indicator until the first
          // token lands, then live-typed content; the footer (sources,
          // feedback, retry) only makes sense once the answer is final.
          const isActiveStream = streaming && i === messages.length - 1;
          return (
            <div key={msg.id} className="flex items-start justify-start">
              <div className="max-w-[85%]">
                <div className="prose-ui rounded-xl rounded-bl-sm border border-border bg-card px-4 py-2 text-sm text-card-foreground">
                  {isActiveStream && !msg.content ? (
                    <WaitingIndicator />
                  ) : (
                    <AnswerContent
                      content={msg.content}
                      sources={msg.sources ?? []}
                      selectedCitation={selectedCitation}
                      onSelectEvidence={onSelectEvidence}
                      onOpenFullSource={onOpenFullSource}
                    />
                  )}
                </div>
                {!isActiveStream && (
                  <>
                    {msg.sources && msg.sources.length > 0 ? (
                      <SourcesUsed
                        content={msg.content}
                        sources={msg.sources}
                        selectedCitation={selectedCitation}
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
                  </>
                )}
              </div>
            </div>
          );
        })()
      )}
      <div ref={bottomRef} />
    </div>
  );
}
