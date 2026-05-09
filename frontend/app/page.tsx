"use client";

import { useCallback, useEffect, useState } from "react";
import { X } from "lucide-react";
import { checkHealth } from "@/lib/api";
import Sidebar from "@/components/Sidebar";
import ChatPanel from "@/components/ChatPanel";
import ToastContainer, { type ToastItem } from "@/components/Toast";
import InspectorPanel from "@/components/InspectorPanel";

let toastCounter = 0;

export default function Home() {
  const [toasts, setToasts] = useState<ToastItem[]>([]);
  const [offline, setOffline] = useState(false);
  const [refreshKey, setRefreshKey] = useState(0);
  const [inspecting, setInspecting] = useState<string | null>(null);

  const addToast = useCallback((message: string, variant?: "error") => {
    const id = `toast-${++toastCounter}`;
    setToasts((prev) => [...prev, { id, message, variant }]);
  }, []);

  const dismissToast = useCallback((id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  useEffect(() => {
    checkHealth().then((ok) => setOffline(!ok));
  }, []);

  const handleCollectionCleared = useCallback(() => {
    setRefreshKey((k) => k + 1);
    addToast("Collection cleared");
  }, [addToast]);

  return (
    <div className="flex h-screen overflow-hidden">
      {offline && (
        <div className="fixed top-0 inset-x-0 z-40 flex items-center justify-between bg-yellow-950 border-b border-yellow-800 px-4 py-2">
          <p className="text-xs text-yellow-300">
            Backend offline — start the Python server
          </p>
          <button onClick={() => setOffline(false)} className="text-yellow-400 hover:text-yellow-300">
            <X className="h-3.5 w-3.5" />
          </button>
        </div>
      )}

      <Sidebar key={refreshKey} onToast={addToast} onInspect={setInspecting} />
      <ChatPanel onToast={addToast} onCollectionCleared={handleCollectionCleared} />

      {inspecting && (
        <InspectorPanel filename={inspecting} onClose={() => setInspecting(null)} />
      )}

      <ToastContainer toasts={toasts} onDismiss={dismissToast} />
    </div>
  );
}
