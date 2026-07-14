"use client";

import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import TraceSidebar, { type Trace } from "@/components/TraceSidebar";
import EvidencePanel from "@/components/EvidencePanel";
import type { InspectTarget } from "@/lib/api";

interface Props {
  trace: Trace | null;
  onInspect?: (target: InspectTarget) => void;
  /** Last citation the user clicked — Evidence tab selects/scrolls to it. */
  selectedTarget?: InspectTarget | null;
}

/** Evidence (default) / Technical trace tab split. Both tabs read the same
 * `trace.sources` — no separate fetch or parse for the Evidence view. */
export default function RightPanelTabs({ trace, onInspect, selectedTarget }: Props) {
  return (
    <div className="flex h-full w-full flex-col lg:w-[320px] lg:shrink-0">
      <Tabs defaultValue="evidence" className="flex h-full min-h-0 flex-col">
        <TabsList className="mx-3 mt-2 w-fit shrink-0">
          <TabsTrigger value="evidence">Evidence</TabsTrigger>
          <TabsTrigger value="technical">Technical details</TabsTrigger>
        </TabsList>
        <TabsContent value="evidence" className="min-h-0 flex-1">
          <EvidencePanel
            sources={trace?.sources ?? []}
            onInspect={onInspect}
            selectedTarget={selectedTarget}
          />
        </TabsContent>
        <TabsContent value="technical" className="min-h-0 flex-1">
          <TraceSidebar trace={trace} onInspect={onInspect} />
        </TabsContent>
      </Tabs>
    </div>
  );
}
