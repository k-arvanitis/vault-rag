"use client";

import { useRef, useState, useCallback } from "react";
import { Send, Loader2, Trash2 } from "lucide-react";
import { queryDocuments, clearCollection } from "@/lib/api";
import { cn } from "@/lib/utils";
import MessageList, { type Message } from "./MessageList";

let msgCounter = 0;
function nextId() {
  return `msg-${++msgCounter}`;
}

interface Props {
  collectionName?: string;
  onToast: (msg: string, variant?: "error") => void;
  onCollectionCleared: () => void;
}

export default function ChatPanel({ collectionName = "My Collection", onToast, onCollectionCleared }: Props) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [confirmClear, setConfirmClear] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

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
        sources: data.sources,
      };
      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err) {
      onToast((err as Error).message || "Query failed", "error");
    } finally {
      setStreaming(false);
    }
  }, [input, streaming, onToast]);

  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  };

  const handleClear = async () => {
    try {
      await clearCollection();
      setMessages([]);
      setConfirmClear(false);
      onCollectionCleared();
    } catch (err) {
      onToast((err as Error).message || "Clear failed", "error");
      setConfirmClear(false);
    }
  };

  return (
    <div className="flex-1 flex flex-col h-full min-w-0">
      {/* Header */}
      <header className="flex items-center justify-between px-6 py-3 border-b border-zinc-800 shrink-0">
        <span className="text-sm font-medium text-zinc-300">{collectionName}</span>
        {!confirmClear ? (
          <button
            onClick={() => setConfirmClear(true)}
            className="flex items-center gap-1.5 text-xs text-zinc-500 hover:text-red-400 transition-colors px-2 py-1 rounded-md hover:bg-zinc-900"
          >
            <Trash2 className="h-3.5 w-3.5" />
            Clear Collection
          </button>
        ) : (
          <div className="flex items-center gap-2">
            <span className="text-xs text-zinc-400">Are you sure?</span>
            <button
              onClick={handleClear}
              className="text-xs text-red-400 hover:text-red-300 font-medium transition-colors"
            >
              Yes, clear
            </button>
            <button
              onClick={() => setConfirmClear(false)}
              className="text-xs text-zinc-500 hover:text-zinc-400 transition-colors"
            >
              Cancel
            </button>
          </div>
        )}
      </header>

      {/* Messages */}
      <MessageList messages={messages} streaming={streaming} />

      {/* Input bar */}
      <div className="shrink-0 px-6 py-4 border-t border-zinc-800">
        <div className="flex items-end gap-3 bg-zinc-900 rounded-xl border border-zinc-700 px-4 py-3 focus-within:border-zinc-500 transition-colors">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder="Ask about your documents…"
            rows={1}
            className={cn(
              "flex-1 resize-none bg-transparent text-sm text-zinc-100 placeholder-zinc-600",
              "outline-none leading-relaxed max-h-40 overflow-y-auto"
            )}
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
            className={cn(
              "shrink-0 rounded-lg p-1.5 transition-colors",
              input.trim() && !streaming
                ? "text-zinc-100 hover:bg-zinc-700"
                : "text-zinc-600 cursor-not-allowed"
            )}
          >
            {streaming ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Send className="h-4 w-4" />
            )}
          </button>
        </div>
        <p className="text-[10px] text-zinc-700 mt-1.5 text-center">
          Enter to send · Shift+Enter for newline
        </p>
      </div>
    </div>
  );
}
