"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Maximize2, Minimize2 } from "lucide-react";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import TraceSidebar, { type Trace } from "@/components/TraceSidebar";
import EvidencePanel from "@/components/EvidencePanel";
import type { InspectTarget, Source } from "@/lib/api";

const WIDTH_STORAGE_KEY = "vault-rag:evidence-panel-width";
const DEFAULT_WIDTH = 420;
const MIN_WIDTH = 320;
const MAX_WIDTH = 540;
const MAX_WIDTH_RATIO = 0.5;
const EXPANDED_WIDTH_RATIO = 0.45;

interface Props {
  /** Technical details tab always reflects the latest turn — this is
   * debug/advanced information, not the citation trust path (see Evidence). */
  trace: Trace | null;
  /** Evidence tab's source list — the specific message whose citation was
   * last clicked, NOT necessarily the latest turn (see app/page.tsx). */
  evidenceSources: Source[];
  onOpenFullSource?: (target: InspectTarget) => void;
  /** Citation last selected — Evidence tab scrolls/highlights it. */
  selectedTarget?: InspectTarget | null;
  tab: "evidence" | "technical";
  onTabChange: (tab: "evidence" | "technical") => void;
  /** Lifted so expand can also collapse the left source sidebar (see
   * app/page.tsx) -- side-by-side verification is the point, so expanded
   * mode reclaims the sidebar's space instead of covering the chat. */
  expanded?: boolean;
  onExpandedChange?: (expanded: boolean) => void;
}

/** Evidence / Technical details tab split, in a resizable panel.
 *
 * Evidence is driven by whichever citation the user actually clicked
 * (evidenceSources/selectedTarget); Technical details always shows the
 * latest turn's tool trace — the two are deliberately not the same data
 * source, see the per-prop comments above. Width is user-draggable (a plain
 * pointer-drag handle, not a full split-pane library -- one resizable
 * boundary doesn't need one) and persisted to localStorage so it survives a
 * reload. */
export default function RightPanelTabs({
  trace,
  evidenceSources,
  onOpenFullSource,
  selectedTarget,
  tab,
  onTabChange,
  expanded = false,
  onExpandedChange,
}: Props) {
  const [width, setWidth] = useState(DEFAULT_WIDTH);
  const dragging = useRef(false);

  useEffect(() => {
    const stored = Number(localStorage.getItem(WIDTH_STORAGE_KEY));
    if (stored && stored >= MIN_WIDTH) setWidth(stored);
  }, []);

  const onPointerMove = useCallback((e: PointerEvent) => {
    if (!dragging.current) return;
    const maxWidth = Math.min(MAX_WIDTH, window.innerWidth * MAX_WIDTH_RATIO);
    const next = Math.min(maxWidth, Math.max(MIN_WIDTH, window.innerWidth - e.clientX));
    setWidth(next);
  }, []);

  const onPointerUp = useCallback(() => {
    if (!dragging.current) return;
    dragging.current = false;
    document.body.style.cursor = "";
    setWidth((w) => {
      localStorage.setItem(WIDTH_STORAGE_KEY, String(w));
      return w;
    });
    window.removeEventListener("pointermove", onPointerMove);
    window.removeEventListener("pointerup", onPointerUp);
  }, [onPointerMove]);

  const startDrag = useCallback(() => {
    if (expanded) return;
    dragging.current = true;
    document.body.style.cursor = "col-resize";
    window.addEventListener("pointermove", onPointerMove);
    window.addEventListener("pointerup", onPointerUp);
  }, [expanded, onPointerMove, onPointerUp]);

  const effectiveWidth = expanded ? window.innerWidth * EXPANDED_WIDTH_RATIO : width;

  return (
    <div className="relative flex h-full" style={{ width: effectiveWidth }}>
      {!expanded && (
        <div
          role="separator"
          aria-orientation="vertical"
          aria-label="Resize evidence panel"
          onPointerDown={startDrag}
          className="w-1 shrink-0 cursor-col-resize bg-transparent transition-colors hover:bg-border active:bg-border"
        />
      )}
      <div className="flex h-full min-w-0 flex-1 flex-col border-l border-border">
        <Tabs
          value={tab}
          onValueChange={(v) => onTabChange(v as "evidence" | "technical")}
          className="flex h-full min-h-0 flex-col"
        >
          <div className="mt-2 flex shrink-0 items-center justify-between gap-2 px-3">
            <TabsList className="w-fit">
              <TabsTrigger value="evidence">Evidence</TabsTrigger>
              <TabsTrigger value="technical">Technical details</TabsTrigger>
            </TabsList>
            {onExpandedChange && (
              <Button
                variant="ghost"
                size="icon-xs"
                aria-label={expanded ? "Collapse evidence panel" : "Expand evidence panel"}
                onClick={() => onExpandedChange(!expanded)}
              >
                {expanded ? <Minimize2 /> : <Maximize2 />}
              </Button>
            )}
          </div>
          <TabsContent value="evidence" className="min-h-0 flex-1">
            <EvidencePanel
              sources={evidenceSources}
              onInspect={onOpenFullSource}
              selectedTarget={selectedTarget}
            />
          </TabsContent>
          <TabsContent value="technical" className="min-h-0 flex-1">
            <TraceSidebar trace={trace} onInspect={onOpenFullSource} />
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}
