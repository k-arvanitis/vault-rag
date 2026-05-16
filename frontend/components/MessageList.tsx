"use client";

import { useEffect, useRef } from "react";
import ReactMarkdown from "react-markdown";

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
}

interface Props {
  messages: Message[];
  streaming: boolean;
}

function LoadingPill() {
  return <div className="h-2 w-12 animate-pulse rounded-full bg-ink-200" />;
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
      {messages.map((msg) =>
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
