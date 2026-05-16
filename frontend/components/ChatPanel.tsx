"use client";

import { useRef, useState, useCallback, useEffect } from "react";
import { Send, Loader2 } from "lucide-react";
import { queryDocuments } from "@/lib/api";
import { cn } from "@/lib/utils";
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
}

export default function ChatPanel({ onToast, resetSignal = 0, onTrace }: Props) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (resetSignal > 0) {
      setMessages([]);
      onTrace?.(null);
    }
  }, [resetSignal, onTrace]);

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
      };
      setMessages((prev) => [...prev, assistantMsg]);
      onTrace?.({
        sources: data.sources ?? [],
        sql: data.sql ?? [],
        tools_used: data.tools_used ?? [],
      });
    } catch (err) {
      onToast((err as Error).message || "Query failed", "error");
    } finally {
      setStreaming(false);
    }
  }, [input, streaming, onToast, onTrace]);

  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  };

  return (
    <div className="flex h-full min-w-0 flex-1 flex-col bg-ink-50">
      <MessageList messages={messages} streaming={streaming} />

      {/* Input bar */}
      <div className="shrink-0 border-t border-ink-200 px-6 py-4">
        <div className="flex items-end gap-2 rounded-lg border border-ink-200 bg-surface px-3 py-2 transition-colors focus-within:border-brand">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder="Ask about your documents…"
            rows={1}
            className="max-h-40 flex-1 resize-none overflow-y-auto bg-transparent text-sm leading-relaxed text-ink-800 outline-none placeholder:text-ink-400"
            style={{ minHeight: "24px" }}
            onInput={(e) => {
              const el = e.currentTarget;
              el.style.height = "auto";
              el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
            }}
            disabled={streaming}
          />
          <button
            onClick={send}
            disabled={!input.trim() || streaming}
            aria-label="Send"
            className={cn(
              "shrink-0 rounded-md p-1.5 transition-colors",
              input.trim() && !streaming
                ? "bg-brand text-white hover:bg-brand-dark"
                : "cursor-not-allowed text-ink-300"
            )}
          >
            {streaming ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
          </button>
        </div>
        <p className="mt-1.5 text-center text-[10px] text-ink-400">
          Enter to send · Shift+Enter for newline
        </p>
      </div>
    </div>
  );
}
