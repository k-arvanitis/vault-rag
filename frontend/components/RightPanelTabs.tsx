"use client";

import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import TraceSidebar, { type Trace } from "@/components/TraceSidebar";
import EvidencePanel from "@/components/EvidencePanel";
import type { InspectTarget, Source } from "@/lib/api";

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
}

/** Evidence / Technical details tab split. Evidence is driven by whichever
 * citation the user actually clicked (evidenceSources/selectedTarget);
 * Technical details always shows the latest turn's tool trace — the two are
 * deliberately not the same data source, see the per-prop comments above. */
export default function RightPanelTabs({
  trace,
  evidenceSources,
  onOpenFullSource,
  selectedTarget,
  tab,
  onTabChange,
}: Props) {
  return (
    <div className="flex h-full w-full flex-col lg:w-[320px] lg:shrink-0">
      <Tabs value={tab} onValueChange={(v) => onTabChange(v as "evidence" | "technical")} className="flex h-full min-h-0 flex-col">
        <TabsList className="mx-3 mt-2 w-fit shrink-0">
          <TabsTrigger value="evidence">Evidence</TabsTrigger>
          <TabsTrigger value="technical">Technical details</TabsTrigger>
        </TabsList>
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
  );
}
