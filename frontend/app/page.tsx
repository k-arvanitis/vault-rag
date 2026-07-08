"use client";

import { useCallback, useEffect, useState } from "react";
import { X, FlaskConical, MessageSquareWarning, History } from "lucide-react";
import { checkHealth, type Conversation } from "@/lib/api";
import Sidebar from "@/components/Sidebar";
import ChatPanel from "@/components/ChatPanel";
import ToastContainer, { type ToastItem } from "@/components/Toast";
import InspectorPanel from "@/components/InspectorPanel";
import EvalPanel from "@/components/EvalPanel";
import FeedbackPanel from "@/components/FeedbackPanel";
import HistoryPanel from "@/components/HistoryPanel";
import ThemeToggle from "@/components/ThemeToggle";
import TraceSidebar, { type Trace } from "@/components/TraceSidebar";

let toastCounter = 0;

export default function Home() {
  const [toasts, setToasts] = useState<ToastItem[]>([]);
  const [offline, setOffline] = useState(false);
  const [refreshKey, setRefreshKey] = useState(0);
  const [inspecting, setInspecting] = useState<string | null>(null);
  const [showEval, setShowEval] = useState(false);
  const [showFeedback, setShowFeedback] = useState(false);
  const [showHistory, setShowHistory] = useState(false);
  const [loadedConversation, setLoadedConversation] = useState<Conversation | null>(null);
  const [conversationLoadKey, setConversationLoadKey] = useState(0);
  const [trace, setTrace] = useState<Trace | null>(null);

  const addToast = useCallback((message: string, variant?: "error") => {
    const id = `toast-${++toastCounter}`;
    setToasts((prev) => [...prev, { id, message, variant }]);
  }, []);

  const dismissToast = useCallback((id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  useEffect(() => {
    let cancelled = false;
    const probe = async () => {
      const ok = await checkHealth();
      if (!cancelled) setOffline(!ok);
    };
    probe();
    const id = setInterval(probe, 5000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  const handleCollectionCleared = useCallback(() => {
    setRefreshKey((k) => k + 1);
    setTrace(null);
    addToast("Collection cleared");
  }, [addToast]);

  const handleSelectConversation = useCallback((conv: Conversation) => {
    setLoadedConversation(conv);
    setTrace(null);
    setConversationLoadKey((k) => k + 1);
  }, []);

  return (
    <div className="flex h-screen flex-col">
      <header className="flex items-center justify-between border-b border-ink-200 bg-surface px-6 py-2.5">
        <div className="flex items-baseline gap-3">
          <span className="text-xl font-bold tracking-tight text-ink-800">Vault RAG</span>
          <span className="hidden font-mono text-[11px] uppercase tracking-widest text-ink-400 sm:inline">
            document intelligence
          </span>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowHistory(true)}
            className="flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-xs font-medium text-ink-500 transition-colors hover:bg-ink-100 hover:text-ink-800"
          >
            <History className="h-3.5 w-3.5" />
            History
          </button>
          <button
            onClick={() => setShowFeedback(true)}
            className="flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-xs font-medium text-ink-500 transition-colors hover:bg-ink-100 hover:text-ink-800"
          >
            <MessageSquareWarning className="h-3.5 w-3.5" />
            Feedback
          </button>
          <button
            onClick={() => setShowEval(true)}
            className="flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-xs font-medium text-ink-500 transition-colors hover:bg-ink-100 hover:text-ink-800"
          >
            <FlaskConical className="h-3.5 w-3.5" />
            Evaluation
          </button>
          <ThemeToggle />
        </div>
      </header>

      {offline && (
        <div className="flex items-center justify-between border-b border-amber-200 bg-amber-50 px-4 py-2 text-sm text-amber-800">
          <p>
            <strong>Backend offline</strong> — start the Python server (<code className="font-mono">make api</code>).
          </p>
          <button
            onClick={() => setOffline(false)}
            className="text-amber-700 hover:text-amber-900"
            aria-label="Dismiss"
          >
            <X className="h-3.5 w-3.5" />
          </button>
        </div>
      )}

      <div className="flex flex-1 overflow-hidden">
        <Sidebar
          key={refreshKey}
          onToast={addToast}
          onInspect={setInspecting}
          onCollectionCleared={handleCollectionCleared}
        />
        <ChatPanel
          key={`chat-${conversationLoadKey}`}
          onToast={addToast}
          resetSignal={refreshKey}
          onTrace={setTrace}
          initialMessages={loadedConversation?.messages}
          initialConversationId={loadedConversation?.id ?? null}
        />

        {inspecting ? (
          <InspectorPanel filename={inspecting} onClose={() => setInspecting(null)} />
        ) : (
          <TraceSidebar trace={trace} />
        )}
      </div>

      {showEval && <EvalPanel onClose={() => setShowEval(false)} />}
      {showFeedback && <FeedbackPanel onClose={() => setShowFeedback(false)} />}
      {showHistory && (
        <HistoryPanel onClose={() => setShowHistory(false)} onSelect={handleSelectConversation} />
      )}
      <ToastContainer toasts={toasts} onDismiss={dismissToast} />
    </div>
  );
}
