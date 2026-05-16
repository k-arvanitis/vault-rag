"use client";

import { useCallback, useEffect, useState } from "react";
import { X } from "lucide-react";
import { checkHealth } from "@/lib/api";
import Sidebar from "@/components/Sidebar";
import ChatPanel from "@/components/ChatPanel";
import ToastContainer, { type ToastItem } from "@/components/Toast";
import InspectorPanel from "@/components/InspectorPanel";
import ThemeToggle from "@/components/ThemeToggle";
import TraceSidebar, { type Trace } from "@/components/TraceSidebar";

let toastCounter = 0;

export default function Home() {
  const [toasts, setToasts] = useState<ToastItem[]>([]);
  const [offline, setOffline] = useState(false);
  const [refreshKey, setRefreshKey] = useState(0);
  const [inspecting, setInspecting] = useState<string | null>(null);
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

  return (
    <div className="flex h-screen flex-col">
      <header className="flex items-center justify-between border-b border-ink-200 bg-surface px-6 py-2.5">
        <div className="flex items-baseline gap-3">
          <span className="text-xl font-bold tracking-tight text-ink-800">Vault RAG</span>
          <span className="hidden font-mono text-[11px] uppercase tracking-widest text-ink-400 sm:inline">
            document intelligence
          </span>
        </div>
        <ThemeToggle />
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
        <ChatPanel onToast={addToast} resetSignal={refreshKey} onTrace={setTrace} />

        {inspecting ? (
          <InspectorPanel filename={inspecting} onClose={() => setInspecting(null)} />
        ) : (
          <TraceSidebar trace={trace} />
        )}
      </div>

      <ToastContainer toasts={toasts} onDismiss={dismissToast} />
    </div>
  );
}
