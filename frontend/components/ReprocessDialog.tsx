"use client";

import { useState } from "react";
import { reindexDocument, getIngestStatus } from "@/lib/api";
import { trackJob } from "@/lib/jobTracker";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog";
import { cn } from "@/lib/utils";

type Pipeline = "auto" | "ocr" | "text";

const OPTIONS: { value: Pipeline; label: string; description: string }[] = [
  { value: "auto", label: "Automatic", description: "Let the parser decide per page — the normal choice." },
  { value: "ocr", label: "Process with OCR", description: "Force every page through OCR, even ones with a text layer." },
  { value: "text", label: "Use text layer", description: "Force the text layer reader, even on pages that look scanned." },
];

interface Props {
  filename: string;
  onReprocessed: () => void;
  onToast: (msg: string, variant?: "error") => void;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

/** Advanced re-ingestion routing lives here, not on ordinary upload — upload
 * is always automatic. This is the correction path: pick a specific pipeline
 * only when the automatic result needs fixing (see TODO/review item C).
 * Always fully controlled — callers own their own trigger button and just
 * flip `open` (see Sidebar.tsx / app/sources/page.tsx), rather than this
 * component rendering its own DialogTrigger; composing a Tooltip-wrapped
 * button as a render-prop trigger broke the click in practice. */
export default function ReprocessDialog({ filename, onReprocessed, onToast, open, onOpenChange }: Props) {
  const [pipeline, setPipeline] = useState<Pipeline>("auto");
  const basename = filename.split("/").pop() ?? filename;

  const confirm = async () => {
    onOpenChange(false);
    try {
      const { job_id } = await reindexDocument(filename, pipeline);
      onToast(`Reprocessing ${basename}…`);
      trackJob(filename, "reindex", job_id, getIngestStatus, onReprocessed);
      onReprocessed();
    } catch (e) {
      onToast(e instanceof Error ? e.message : "Reprocess failed", "error");
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Reprocess &ldquo;{basename}&rdquo;</DialogTitle>
          <DialogDescription>
            Choose how this document should be re-read. Automatic is correct for almost every file —
            only override it if the automatic result missed text or garbled a scanned page.
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-2">
          {OPTIONS.map((opt) => (
            <button
              key={opt.value}
              type="button"
              onClick={() => setPipeline(opt.value)}
              className={cn(
                "w-full rounded-lg border px-3 py-2 text-left transition-colors",
                pipeline === opt.value
                  ? "border-ring bg-muted"
                  : "border-border hover:border-foreground/20"
              )}
            >
              <p className="text-sm font-medium text-foreground">{opt.label}</p>
              <p className="text-xs text-muted-foreground">{opt.description}</p>
            </button>
          ))}
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button onClick={confirm}>Reprocess</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
