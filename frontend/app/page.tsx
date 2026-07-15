"use client";

import { useCallback, useEffect, useState } from "react";
import { Compass } from "lucide-react";
import { toast } from "sonner";
import { checkHealth, type Conversation, type InspectTarget, type Source } from "@/lib/api";
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
  const [showHistory, setShowHistory] = useState(false);
  const [showTraceSheet, setShowTraceSheet] = useState(false);
  const [loadedConversation, setLoadedConversation] = useState<Conversation | null>(null);
  const [conversationLoadKey, setConversationLoadKey] = useState(0);
  const [trace, setTrace] = useState<Trace | null>(null);
  const [initialScopedDocId, setInitialScopedDocId] = useState<string | null>(null);

  // Evidence is deliberately NOT derived from `trace` (the latest turn) —
  // it's whichever citation the user actually clicked, which can belong to
  // an older message. See MessageList's onSelectEvidence / review item A.
  const [evidenceSources, setEvidenceSources] = useState<Source[]>([]);
  const [selectedCitation, setSelectedCitation] = useState<InspectTarget | null>(null);
  const [rightPanelTab, setRightPanelTab] = useState<"evidence" | "technical">("evidence");

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

  const handleSelectConversation = useCallback((conv: Conversation) => {
    setLoadedConversation(conv);
    setTrace(null);
    setEvidenceSources([]);
    setSelectedCitation(null);
    setConversationLoadKey((k) => k + 1);
  }, []);

  // A new answer arriving: show its own evidence by default (no citation
  // highlighted yet — the user hasn't clicked one for THIS turn). Clicking an
  // older message's citation afterward overrides this via handleSelectEvidence.
  const handleTrace = useCallback((t: Trace | null) => {
    setTrace(t);
    setEvidenceSources(t?.sources ?? []);
    setSelectedCitation(null);
  }, []);

  // Primary citation action — select this citation's evidence and switch to
  // the Evidence tab. Deliberately does not open the full inspector or leave
  // the chat (see review item A: "the citation click itself should first
  // update the Evidence panel").
  const handleSelectEvidence = useCallback((target: InspectTarget, messageSources: Source[]) => {
    setEvidenceSources(messageSources);
    setSelectedCitation(target);
    setRightPanelTab("evidence");
  }, []);

  // Secondary action — opens the full-screen document inspector. Used by
  // "Open full source" links, the sidebar's inspect icon, and Sources-screen
  // row actions. Does not touch evidence/citation selection.
  const handleOpenFullSource = useCallback((target: InspectTarget) => {
    setInspecting(target);
  }, []);

  return (
    <SidebarProvider className="h-screen">
      <Sidebar
        key={refreshKey}
        onToast={addToast}
        onInspect={(filename) => handleOpenFullSource({ filename })}
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
            onTrace={handleTrace}
            onSelectEvidence={handleSelectEvidence}
            onOpenFullSource={handleOpenFullSource}
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
                <RightPanelTabs
                  trace={trace}
                  evidenceSources={evidenceSources}
                  onOpenFullSource={handleOpenFullSource}
                  selectedTarget={selectedCitation}
                  tab={rightPanelTab}
                  onTabChange={setRightPanelTab}
                />
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
                  <RightPanelTabs
                    trace={trace}
                    evidenceSources={evidenceSources}
                    onOpenFullSource={handleOpenFullSource}
                    selectedTarget={selectedCitation}
                    tab={rightPanelTab}
                    onTabChange={setRightPanelTab}
                  />
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
