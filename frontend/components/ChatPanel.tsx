"use client";

import { useRef, useState, useCallback, useEffect } from "react";
import { Send, Loader2, SquarePen } from "lucide-react";
import { queryDocuments, saveConversation, type InspectTarget } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import MessageList, { type Message } from "./MessageList";

let msgCounter = 0;
function nextId() {
  return `msg-${++msgCounter}`;
}

import { type Trace } from "./TraceSidebar";

interface Props {
  onToast: (msg: string, variant?: "error") => void;
  resetSignal?: number;
  onTrace?: (trace: Trace | null) => void;
  onInspect?: (target: InspectTarget) => void;
  initialMessages?: Message[];
  initialConversationId?: string | null;
  onConversationSaved?: (id: string) => void;
}

export default function ChatPanel({
  onToast,
  resetSignal = 0,
  onTrace,
  onInspect,
  initialMessages,
  initialConversationId = null,
  onConversationSaved,
}: Props) {
  const [messages, setMessages] = useState<Message[]>(initialMessages ?? []);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [conversationId, setConversationId] = useState<string | null>(initialConversationId);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

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

    try {
      const data = await queryDocuments(question);
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
      onToast((err as Error).message || "Query failed", "error");
    } finally {
      setStreaming(false);
    }
  }, [input, streaming, onToast, onTrace, persist]);

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
      <div className="flex shrink-0 items-center justify-end border-b border-border bg-card px-6 py-2">
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

      <MessageList messages={messages} streaming={streaming} onInspect={onInspect} />

      {/* Input bar */}
      <div className="shrink-0 border-t border-border px-6 py-4">
        <div className="flex items-end gap-2 rounded-lg border border-input bg-card px-3 py-2 transition-colors focus-within:border-ring focus-within:ring-3 focus-within:ring-ring/50">
          <Textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder="Ask about your documents…"
            rows={1}
            className="max-h-40 min-h-0 flex-1 resize-none border-0 bg-transparent px-0 py-0 text-sm leading-relaxed shadow-none focus-visible:ring-0"
            disabled={streaming}
          />
          <Button
            onClick={send}
            disabled={!input.trim() || streaming}
            aria-label="Send"
            size="icon-sm"
          >
            {streaming ? <Loader2 className="animate-spin" /> : <Send />}
          </Button>
        </div>
        <p className="mt-1.5 text-center text-[10px] text-muted-foreground">
          Enter to send · Shift+Enter for newline
        </p>
      </div>
    </div>
  );
}
