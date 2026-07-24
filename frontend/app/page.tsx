"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Compass } from "lucide-react";
import { toast } from "sonner";
import {
  checkHealth,
  sourceInspectTarget,
  type Conversation,
  type InspectTarget,
  type Source,
} from "@/lib/api";
import { citedOnlySources } from "@/lib/product";
import { SidebarProvider, SidebarInset, useSidebar } from "@/components/ui/sidebar";
import { Button } from "@/components/ui/button";
import { Sheet, SheetContent, SheetHeader, SheetTitle } from "@/components/ui/sheet";
import Sidebar from "@/components/Sidebar";
import AppHeader from "@/components/AppHeader";
import ChatPanel from "@/components/ChatPanel";
import InspectorPanel from "@/components/InspectorPanel";
import HistoryPanel from "@/components/HistoryPanel";
import { type Trace } from "@/components/TraceSidebar";
import RightPanelTabs from "@/components/RightPanelTabs";

// Collapses the left source sidebar while the Evidence panel is expanded,
// restoring its prior state afterward -- reclaiming the sidebar's space is
// how expanded Evidence gets ~45% of the screen without covering the chat
// (side-by-side verification is the point). Must live inside SidebarProvider
// to reach useSidebar, so it can't just be inline in Home().
function SidebarAutoCollapse({ collapse }: { collapse: boolean }) {
  const { open, setOpen } = useSidebar();
  const priorOpen = useRef(open);
  useEffect(() => {
    if (collapse) {
      priorOpen.current = open;
      setOpen(false);
    } else {
      setOpen(priorOpen.current);
    }
    // Only react to `collapse` flipping -- `open`/`setOpen` intentionally
    // excluded so restoring doesn't retrigger this effect.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [collapse]);
  return null;
}

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
  const [evidenceExpanded, setEvidenceExpanded] = useState(false);

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

  // A new answer arriving: default Evidence to the first source the answer
  // actually cites — [N] in the answer text is already the source's 1-based
  // position in `sources` (see build_citation_map in answer_pipeline.py), so
  // this stays in sync with the inline chip and the Sources-used row for that
  // same [N]. Falls back to no selection if the answer cites nothing. Clicking
  // an older message's citation afterward overrides this via handleSelectEvidence.
  const handleTrace = useCallback((t: Trace | null) => {
    setTrace(t);
    const sources = citedOnlySources(t?.answer ?? "", t?.sources ?? []);
    setEvidenceSources(sources);
    const firstCitation = t?.answer?.match(/\[(\d+)\]/);
    const firstSource = firstCitation ? sources[Number(firstCitation[1]) - 1] : undefined;
    setSelectedCitation(firstSource ? sourceInspectTarget(firstSource) : null);
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
      <SidebarAutoCollapse collapse={evidenceExpanded} />
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
            selectedCitation={selectedCitation}
            initialMessages={loadedConversation?.messages}
            initialConversationId={loadedConversation?.id ?? null}
            initialScopedDocId={initialScopedDocId}
          />

          {inspecting ? (
            <InspectorPanel
              filename={inspecting.filename}
              page={inspecting.page}
              sheet={inspecting.sheet}
              quote={inspecting.quote}
              isAggregate={inspecting.isAggregate}
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
                  expanded={evidenceExpanded}
                  onExpandedChange={setEvidenceExpanded}
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
