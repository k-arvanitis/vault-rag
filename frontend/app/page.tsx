"use client";

import { useCallback, useEffect, useState } from "react";
import { Compass } from "lucide-react";
import { toast } from "sonner";
import { checkHealth, type Conversation, type InspectTarget } from "@/lib/api";
import { SidebarProvider, SidebarInset } from "@/components/ui/sidebar";
import { Button } from "@/components/ui/button";
import { Sheet, SheetContent, SheetHeader, SheetTitle } from "@/components/ui/sheet";
import Sidebar from "@/components/Sidebar";
import AppHeader from "@/components/AppHeader";
import ChatPanel from "@/components/ChatPanel";
import InspectorPanel from "@/components/InspectorPanel";
import HistoryPanel from "@/components/HistoryPanel";
import { type Trace } from "@/components/TraceSidebar";
import RightPanelTabs from "@/components/RightPanelTabs";

export default function Home() {
  const [offline, setOffline] = useState(false);
  const [refreshKey, setRefreshKey] = useState(0);
  const [inspecting, setInspecting] = useState<InspectTarget | null>(null);
  const [lastCited, setLastCited] = useState<InspectTarget | null>(null);
  const [showHistory, setShowHistory] = useState(false);
  const [showTraceSheet, setShowTraceSheet] = useState(false);
  const [loadedConversation, setLoadedConversation] = useState<Conversation | null>(null);
  const [conversationLoadKey, setConversationLoadKey] = useState(0);
  const [trace, setTrace] = useState<Trace | null>(null);
  const [initialScopedDocId, setInitialScopedDocId] = useState<string | null>(null);

  // Picks up "Open" / "Ask about this source" links from /sources — read directly
  // from the URL rather than useSearchParams(), which requires wrapping this page
  // in a Suspense boundary for a one-time value read on mount.
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const inspectParam = params.get("inspect");
    if (inspectParam) setInspecting({ filename: inspectParam });
    const docParam = params.get("doc");
    if (docParam) setInitialScopedDocId(docParam);
    if (inspectParam || docParam) window.history.replaceState(null, "", "/");
  }, []);

  const addToast = useCallback((message: string, variant?: "error") => {
    if (variant === "error") toast.error(message);
    else toast(message);
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
    setLastCited(null);
    addToast("Collection cleared");
  }, [addToast]);

  const handleSelectConversation = useCallback((conv: Conversation) => {
    setLoadedConversation(conv);
    setTrace(null);
    setLastCited(null);
    setConversationLoadKey((k) => k + 1);
  }, []);

  const handleInspect = useCallback((target: InspectTarget) => {
    setInspecting(target);
    setLastCited(target);
  }, []);

  return (
    <SidebarProvider className="h-screen">
      <Sidebar
        key={refreshKey}
        onToast={addToast}
        onInspect={(filename) => handleInspect({ filename })}
        onCollectionCleared={handleCollectionCleared}
        offline={offline}
      />
      <SidebarInset className="overflow-hidden">
        <AppHeader
          offline={offline}
          onDismissOffline={() => setOffline(false)}
          onShowHistory={() => setShowHistory(true)}
        />

        <div className="flex flex-1 overflow-hidden">
          <ChatPanel
            key={`chat-${conversationLoadKey}`}
            onToast={addToast}
            resetSignal={refreshKey}
            onTrace={setTrace}
            onInspect={handleInspect}
            initialMessages={loadedConversation?.messages}
            initialConversationId={loadedConversation?.id ?? null}
            initialScopedDocId={initialScopedDocId}
          />

          {inspecting ? (
            <InspectorPanel
              filename={inspecting.filename}
              page={inspecting.page}
              sheet={inspecting.sheet}
              onClose={() => setInspecting(null)}
            />
          ) : (
            <>
              <div className="hidden h-full lg:flex">
                <RightPanelTabs trace={trace} onInspect={handleInspect} selectedTarget={lastCited} />
              </div>
              <Button
                variant="outline"
                size="icon"
                className="fixed bottom-4 right-4 z-20 rounded-full shadow-md lg:hidden"
                onClick={() => setShowTraceSheet(true)}
                aria-label="View trace"
              >
                <Compass />
              </Button>
              <Sheet open={showTraceSheet} onOpenChange={setShowTraceSheet}>
                <SheetContent side="right" className="w-full p-0 sm:max-w-sm">
                  <SheetHeader className="border-b border-border">
                    <SheetTitle>Sources</SheetTitle>
                  </SheetHeader>
                  <RightPanelTabs trace={trace} onInspect={handleInspect} selectedTarget={lastCited} />
                </SheetContent>
              </Sheet>
            </>
          )}
        </div>
      </SidebarInset>

      {showHistory && (
        <HistoryPanel onClose={() => setShowHistory(false)} onSelect={handleSelectConversation} />
      )}
    </SidebarProvider>
  );
}
