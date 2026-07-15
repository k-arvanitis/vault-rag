"use client";

import { useRef, useState, useCallback, useEffect } from "react";
import { Send, Square, SquarePen } from "lucide-react";
import { queryDocuments, saveConversation, getDocuments, type Document, type InspectTarget, type Source } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import MessageList, { type Message } from "./MessageList";
import SourceScope from "./SourceScope";

let msgCounter = 0;
function nextId() {
  return `msg-${++msgCounter}`;
}

import { type Trace } from "./TraceSidebar";

interface Props {
  onToast: (msg: string, variant?: "error") => void;
  resetSignal?: number;
  onTrace?: (trace: Trace | null) => void;
  /** Primary citation action — selects a specific citation's evidence,
   * scoped to the message it came from (see MessageList.tsx). */
  onSelectEvidence?: (target: InspectTarget, messageSources: Source[]) => void;
  /** Secondary action — opens the full-screen document inspector. */
  onOpenFullSource?: (target: InspectTarget) => void;
  initialMessages?: Message[];
  initialConversationId?: string | null;
  /** Preselects the source-scope control — e.g. arriving from a Sources
   * screen's "Ask about this source" action. */
  initialScopedDocId?: string | null;
  onConversationSaved?: (id: string) => void;
}

export default function ChatPanel({
  onToast,
  resetSignal = 0,
  onTrace,
  onSelectEvidence,
  onOpenFullSource,
  initialMessages,
  initialConversationId = null,
  initialScopedDocId = null,
  onConversationSaved,
}: Props) {
  const [messages, setMessages] = useState<Message[]>(initialMessages ?? []);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [conversationId, setConversationId] = useState<string | null>(initialConversationId);
  const [documents, setDocuments] = useState<Document[]>([]);
  const [scopedDocIds, setScopedDocIds] = useState<string[]>(
    initialScopedDocId ? [initialScopedDocId] : []
  );
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  // Elapsed-time counter for the honest waiting state (see MessageList) — no
  // fake sequential "searching... verifying..." progress, just how long the
  // real request has actually been running.
  useEffect(() => {
    if (!streaming) {
      setElapsedSeconds(0);
      return;
    }
    const start = Date.now();
    const id = setInterval(() => setElapsedSeconds(Math.floor((Date.now() - start) / 1000)), 1000);
    return () => clearInterval(id);
  }, [streaming]);

  useEffect(() => {
    getDocuments()
      .then(setDocuments)
      .catch(() => {
        // Backend unreachable — the source-scope control degrades to "All sources" only.
      });
  }, [resetSignal]);

  const hasSources = documents.length > 0;

  // Focus the question input as soon as there's something to ask about — but
  // only on the empty-conversation state, not on every documents refetch.
  useEffect(() => {
    if (hasSources && messages.length === 0) {
      textareaRef.current?.focus();
    }
  }, [hasSources, messages.length]);

  // initialScopedDocId arrives asynchronously (page.tsx reads it from the URL in
  // its own effect, after this component's first mount already captured the
  // prop's initial value) — react to it changing, not just its initial value.
  useEffect(() => {
    if (initialScopedDocId) setScopedDocIds([initialScopedDocId]);
  }, [initialScopedDocId]);

  useEffect(() => {
    if (resetSignal > 0) {
      setMessages([]);
      setConversationId(null);
      onTrace?.(null);
    }
  }, [resetSignal, onTrace]);

  const persist = useCallback(
    async (allMessages: Message[]) => {
      try {
        const saved = await saveConversation(conversationId, allMessages);
        if (!conversationId) {
          setConversationId(saved.id);
          onConversationSaved?.(saved.id);
        }
      } catch {
        // Best-effort — a failed save shouldn't interrupt the chat itself.
      }
    },
    [conversationId, onConversationSaved]
  );

  const send = useCallback(async () => {
    const question = input.trim();
    if (!question || streaming) return;

    setInput("");
    textareaRef.current?.focus();

    const userMsg: Message = { id: nextId(), role: "user", content: question };
    setMessages((prev) => [...prev, userMsg]);
    setStreaming(true);

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const data = await queryDocuments(question, scopedDocIds, controller.signal);
      const assistantMsg: Message = {
        id: nextId(),
        role: "assistant",
        content: data.answer,
        sources: data.sources ?? [],
      };
      setMessages((prev) => {
        const next = [...prev, assistantMsg];
        persist(next);
        return next;
      });
      onTrace?.({
        sources: data.sources ?? [],
        rejected_sources: data.rejected_sources ?? [],
        sql: data.sql ?? [],
        tools_used: data.tools_used ?? [],
      });
    } catch (err) {
      if ((err as Error).name === "AbortError") {
        onToast("Cancelled");
      } else {
        onToast((err as Error).message || "Query failed", "error");
      }
    } finally {
      abortRef.current = null;
      setStreaming(false);
    }
  }, [input, streaming, onToast, onTrace, persist, scopedDocIds]);

  const cancel = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  const newConversation = useCallback(() => {
    if (streaming) return;
    setMessages([]);
    setConversationId(null);
    setInput("");
    onTrace?.(null);
    textareaRef.current?.focus();
  }, [streaming, onTrace]);

  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  };

  return (
    <div className="flex h-full min-w-0 flex-1 flex-col bg-background">
      <div className="flex shrink-0 items-center justify-between border-b border-border bg-card px-6 py-2">
        <SourceScope documents={documents} scopedDocIds={scopedDocIds} onChange={setScopedDocIds} />
        <Button
          variant="outline"
          size="sm"
          onClick={newConversation}
          disabled={streaming || messages.length === 0}
        >
          <SquarePen data-icon="inline-start" />
          New conversation
        </Button>
      </div>

      <MessageList
        messages={messages}
        streaming={streaming}
        elapsedSeconds={elapsedSeconds}
        onSelectEvidence={onSelectEvidence}
        onOpenFullSource={onOpenFullSource}
        hasSources={hasSources}
        onExamplePick={(text) => {
          setInput(text);
          textareaRef.current?.focus();
        }}
      />

      {/* Input bar */}
      <div className="shrink-0 border-t border-border px-6 py-4">
        <div className="flex items-end gap-2 rounded-lg border border-input bg-card px-3 py-2 transition-colors focus-within:border-ring focus-within:ring-3 focus-within:ring-ring/50">
          <Textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder={hasSources ? "Ask about your documents…" : "Add a source to get started…"}
            rows={1}
            className="max-h-40 min-h-0 flex-1 resize-none border-0 bg-transparent px-0 py-0 text-sm leading-relaxed shadow-none focus-visible:ring-0"
            disabled={streaming || !hasSources}
          />
          {streaming ? (
            <Button onClick={cancel} variant="outline" aria-label="Cancel" size="icon-sm">
              <Square className="fill-current" />
            </Button>
          ) : (
            <Button onClick={send} disabled={!input.trim() || !hasSources} aria-label="Send" size="icon-sm">
              <Send />
            </Button>
          )}
        </div>
        <p className="mt-1.5 text-center text-[10px] text-muted-foreground">
          Enter to send · Shift+Enter for newline
        </p>
      </div>
    </div>
  );
}
