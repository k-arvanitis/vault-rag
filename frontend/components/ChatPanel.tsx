"use client";

import { useRef, useState, useCallback, useEffect } from "react";
import { Send, Square, SquarePen } from "lucide-react";
import {
  streamQueryDocuments,
  saveConversation,
  getDocuments,
  documentsExist,
  type Document,
  type InspectTarget,
  type Source,
  type QueryHistoryTurn,
} from "@/lib/api";
import { useAdminSession } from "@/lib/useAdminSession";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import MessageList, { type Message } from "./MessageList";
import SourceScope from "./SourceScope";

let msgCounter = 0;
function nextId() {
  return `msg-${++msgCounter}`;
}

// Pairs consecutive user/assistant messages into the turns sent as `history`
// so a follow-up like "who must that notice be given to?" can be resolved
// against what was actually asked/answered before it.
function toHistory(messages: Message[]): QueryHistoryTurn[] {
  const turns: QueryHistoryTurn[] = [];
  for (let i = 0; i < messages.length - 1; i++) {
    const user = messages[i];
    const assistant = messages[i + 1];
    if (user.role === "user" && assistant.role === "assistant" && assistant.content) {
      turns.push({ question: user.content, answer: assistant.content });
      i++;
    }
  }
  return turns;
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
  /** Currently selected Evidence citation, so the matching chip/row can be
   * highlighted (see MessageList's CitationChip). */
  selectedCitation?: InspectTarget | null;
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
  selectedCitation = null,
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
  const [hasSourcesNonAdmin, setHasSourcesNonAdmin] = useState(false);
  const { is_admin: isAdmin } = useAdminSession();
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

  // getDocuments()/getStats() are admin-only (a non-admin viewer must not be
  // able to enumerate the corpus, see api.py's require_admin on those
  // routes) -- a non-admin instead gets only a has-anything-to-ask-about
  // boolean, and the per-document scope picker doesn't render at all (see
  // isAdmin check below).
  const refetchDocuments = useCallback(() => {
    if (isAdmin) {
      getDocuments()
        .then(setDocuments)
        .catch(() => {
          // Backend unreachable — the source-scope control degrades to "All sources" only.
        });
    } else {
      documentsExist()
        .then(setHasSourcesNonAdmin)
        .catch(() => setHasSourcesNonAdmin(false));
    }
  }, [isAdmin]);

  useEffect(() => {
    refetchDocuments();
  }, [resetSignal, refetchDocuments]);

  // An upload/delete/reindex in the admin Sources page (a different route)
  // doesn't touch this component's state -- without this, "Ask across: All
  // N sources" stays stale until a full reload. Refetch when the tab
  // regains focus, the same pattern browsers use for "did anything change
  // while I was away".
  useEffect(() => {
    window.addEventListener("focus", refetchDocuments);
    return () => window.removeEventListener("focus", refetchDocuments);
  }, [refetchDocuments]);

  const hasSources = isAdmin ? documents.length > 0 : hasSourcesNonAdmin;

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

  // Shared by send() (new user turn) and retry() (re-ask an existing question
  // in place) — a retry passes the same history as the original ask (turns
  // before it), writing the result into the given assistant message slot
  // instead of appending. onToken fires as raw tokens stream in (perceived-latency
  // UX — see streamQueryDocuments); onDone fires once with the final,
  // cleaned answer, which callers should use to replace the raw
  // concatenation, not append to it.
  const ask = useCallback(
    async (
      question: string,
      onToken: (token: string) => void,
      onDone: (data: Awaited<ReturnType<typeof streamQueryDocuments>>) => void,
      history?: QueryHistoryTurn[]
    ) => {
      setStreaming(true);
      const controller = new AbortController();
      abortRef.current = controller;
      try {
        const data = await streamQueryDocuments(
          question,
          onToken,
          scopedDocIds,
          controller.signal,
          history
        );
        onDone(data);
        onTrace?.({
          sources: data.sources ?? [],
          rejected_sources: data.rejected_sources ?? [],
          sql: data.sql ?? [],
          tools_used: data.tools_used ?? [],
          answer: data.answer,
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
    },
    [onToast, onTrace, scopedDocIds]
  );

  const send = useCallback(async () => {
    const question = input.trim();
    if (!question || streaming) return;

    setInput("");
    textareaRef.current?.focus();

    const history = toHistory(messages);
    const userMsg: Message = { id: nextId(), role: "user", content: question };
    const assistantId = nextId();
    setMessages((prev) => [...prev, userMsg, { id: assistantId, role: "assistant", content: "" }]);

    await ask(
      question,
      (token) => {
        setMessages((prev) =>
          prev.map((m) => (m.id === assistantId ? { ...m, content: m.content + token } : m))
        );
      },
      (data) => {
        setMessages((prev) => {
          const next = prev.map((m) =>
            m.id === assistantId ? { ...m, content: data.answer, sources: data.sources ?? [] } : m
          );
          persist(next);
          return next;
        });
      },
      history
    );
  }, [input, streaming, ask, persist, messages]);

  const retry = useCallback(
    async (assistantMessageId: string) => {
      if (streaming) return;
      const idx = messages.findIndex((m) => m.id === assistantMessageId);
      const userMsg = idx > 0 ? messages[idx - 1] : null;
      if (!userMsg || userMsg.role !== "user") return;
      const history = toHistory(messages.slice(0, idx - 1));

      setMessages((prev) =>
        prev.map((m) => (m.id === assistantMessageId ? { ...m, content: "" } : m))
      );

      await ask(
        userMsg.content,
        (token) => {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantMessageId ? { ...m, content: m.content + token } : m
            )
          );
        },
        (data) => {
          setMessages((prev) => {
            const next = prev.map((m) =>
              m.id === assistantMessageId
                ? { ...m, content: data.answer, sources: data.sources ?? [] }
                : m
            );
            persist(next);
            return next;
          });
        },
        history
      );
    },
    [messages, streaming, ask, persist]
  );

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
        {isAdmin ? (
          <SourceScope documents={documents} scopedDocIds={scopedDocIds} onChange={setScopedDocIds} />
        ) : (
          <span className="text-xs text-muted-foreground">Ask across: All sources</span>
        )}
        {messages.length > 0 && (
          <Button variant="outline" size="sm" onClick={newConversation} disabled={streaming}>
            <SquarePen data-icon="inline-start" />
            New conversation
          </Button>
        )}
      </div>

      <MessageList
        messages={messages}
        streaming={streaming}
        elapsedSeconds={elapsedSeconds}
        selectedCitation={selectedCitation}
        onSelectEvidence={onSelectEvidence}
        onOpenFullSource={onOpenFullSource}
        onRetry={retry}
        hasSources={hasSources}
        scopedDocIds={scopedDocIds}
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
            placeholder={
              !hasSources
                ? "Add a source to get started…"
                : scopedDocIds.length === 1
                  ? `Ask about ${scopedDocIds[0].split("/").pop()}…`
                  : scopedDocIds.length > 1
                    ? `Ask across ${scopedDocIds.length} selected sources…`
                    : isAdmin
                      ? `Ask across all ${documents.length} approved sources…`
                      : "Ask across the approved knowledge base…"
            }
            rows={1}
            className="max-h-40 min-h-0 flex-1 resize-none rounded-none border-0 bg-transparent px-0 py-0 text-sm leading-relaxed shadow-none focus-visible:ring-0 dark:bg-transparent"
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
