"use client";

import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import { ThumbsUp, ThumbsDown, Check } from "lucide-react";
import { submitFeedback, type Source, type FeedbackReason } from "@/lib/api";
import { cn } from "@/lib/utils";

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
}

interface Props {
  messages: Message[];
  streaming: boolean;
}

function LoadingPill() {
  return <div className="h-2 w-12 animate-pulse rounded-full bg-ink-200" />;
}

/** Numbered citation chips under an assistant message; click one to expand its
 * page/row evidence inline, so a non-technical user doesn't need the trace panel. */
function SourceDrawer({ sources }: { sources: Source[] }) {
  const [openIdx, setOpenIdx] = useState<number | null>(null);

  return (
    <div className="mt-2 flex flex-wrap items-start gap-1.5">
      {sources.map((s, i) => {
        const basename = s.filename.split("/").pop() ?? s.filename;
        const open = openIdx === i;
        return (
          <div key={i} className={open ? "w-full" : undefined}>
            <button
              onClick={() => setOpenIdx(open ? null : i)}
              className={cn(
                "inline-flex items-center gap-1 rounded-md border px-1.5 py-0.5 font-mono text-[10px] transition-colors",
                open
                  ? "border-brand bg-brand/10 text-brand-dark"
                  : "border-ink-200 text-ink-500 hover:bg-ink-100"
              )}
            >
              [{i + 1}] {basename}
              {s.page != null && ` · p.${s.page}`}
            </button>
            {open && (
              <div className="mt-1 rounded-md border border-ink-200 bg-surface px-3 py-2 text-[11px] leading-relaxed text-ink-600">
                {s.section && <p className="mb-1 font-medium text-ink-700">{s.section}</p>}
                <p className="whitespace-pre-wrap">{s.quote || s.excerpt || "—"}</p>
                {s.location && (
                  <p className="mt-1 font-mono text-[10px] text-ink-400">{s.location}</p>
                )}
              </div>
            )}
          </div>
        );
      })}
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
      <p className="mt-1.5 flex items-center gap-1 text-[10px] text-ink-400">
        <Check className="h-3 w-3" /> Thanks for the feedback
      </p>
    );
  }

  return (
    <div className="mt-1.5 flex items-center gap-1.5">
      <button
        onClick={() => submit("up", null)}
        aria-label="Good answer"
        className={cn(
          "rounded p-1 transition-colors",
          rating === "up" ? "text-emerald-600" : "text-ink-300 hover:text-ink-600"
        )}
      >
        <ThumbsUp className="h-3.5 w-3.5" />
      </button>
      <button
        onClick={() => setRating(rating === "down" ? null : "down")}
        aria-label="Bad answer"
        className={cn(
          "rounded p-1 transition-colors",
          rating === "down" ? "text-red-600" : "text-ink-300 hover:text-ink-600"
        )}
      >
        <ThumbsDown className="h-3.5 w-3.5" />
      </button>
      {rating === "down" && (
        <select
          value={reason}
          onChange={(e) => {
            const value = e.target.value as FeedbackReason;
            setReason(value);
            submit("down", value);
          }}
          className="rounded border border-ink-200 bg-surface px-1.5 py-0.5 text-[10px] text-ink-600"
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

export default function MessageList({ messages, streaming }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, streaming]);

  if (messages.length === 0 && !streaming) {
    return (
      <div className="flex flex-1 select-none flex-col items-center justify-center px-8 text-center">
        <svg
          className="mb-3 h-10 w-10 text-ink-300"
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
        <p className="text-sm text-ink-500">Upload a document and ask anything.</p>
      </div>
    );
  }

  return (
    <div className="flex-1 space-y-4 overflow-y-auto px-6 py-4">
      {messages.map((msg, i) =>
        msg.role === "user" ? (
          <div key={msg.id} className="flex items-end justify-end">
            <div className="max-w-[70%] whitespace-pre-wrap rounded-xl rounded-br-sm bg-brand px-4 py-2 text-sm text-white">
              {msg.content}
            </div>
          </div>
        ) : (
          <div key={msg.id} className="flex items-start justify-start">
            <div className="max-w-[85%]">
              <div className="prose-ui rounded-xl rounded-bl-sm border border-ink-200 bg-surface px-4 py-2 text-sm text-ink-800">
                <ReactMarkdown>{msg.content}</ReactMarkdown>
              </div>
              {msg.sources && msg.sources.length > 0 && (
                <SourceDrawer sources={msg.sources} />
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
          <div className="rounded-xl rounded-bl-sm border border-ink-200 bg-surface px-4 py-3">
            <LoadingPill />
          </div>
        </div>
      )}
      <div ref={bottomRef} />
    </div>
  );
}
