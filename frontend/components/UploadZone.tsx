"use client";

import { useCallback, useRef, useState } from "react";
import { Upload, CheckCircle2 } from "lucide-react";
import { ingestFile, getIngestStatus } from "@/lib/api";
import { trackJob } from "@/lib/jobTracker";
import { cn } from "@/lib/utils";
import { Progress } from "@/components/ui/progress";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";

interface UploadingFile {
  id: string;
  name: string;
  stage: string;
  progress: number;
  error?: string;
}

// User-facing stages only: Uploading -> Processing -> Ready. The backend's
// finer-grained stage names (parsing/chunking/embedding) are an internal
// pipeline detail, not something a normal user needs to track.
const STAGE_PROGRESS: Record<string, number> = {
  parsing: 25,
  chunking: 55,
  embedding: 80,
  done: 100,
  failed: 100,
};

function displayStage(rawStage: string): string {
  const s = rawStage.toLowerCase();
  if (s === "done") return "Ready";
  if (s === "failed") return "Failed";
  return "Processing";
}

const ACCEPTED = ".pdf,.xlsx,.xls,.csv,.docx,.doc,.md,.png,.jpg,.jpeg";
const ACCEPTED_EXT = ACCEPTED.split(",").map((e) => e.slice(1));

interface Props {
  onUploaded: () => void;
  onToast: (msg: string, variant?: "error") => void;
  /** Disables uploads and shows an offline notice while the backend is unreachable. */
  offline?: boolean;
}

function isAccepted(filename: string) {
  const ext = filename.split(".").pop()?.toLowerCase();
  return !!ext && ACCEPTED_EXT.includes(ext);
}

/** Ordinary upload is always automatic — per-page OCR vs. text-layer routing
 * is a correction tool for when the automatic result is wrong, not a normal
 * upload-time decision. That control lives on the Reprocess action instead
 * (see ReprocessDialog.tsx). */
export default function UploadZone({ onUploaded, onToast, offline }: Props) {
  const [dragging, setDragging] = useState(false);
  const [files, setFiles] = useState<UploadingFile[]>([]);
  const inputRef = useRef<HTMLInputElement>(null);

  const processFile = useCallback(
    async (file: File) => {
      const id = `${Date.now()}-${file.name}`;

      if (!isAccepted(file.name)) {
        setFiles((prev) => [
          ...prev,
          { id, name: file.name, stage: "Unsupported", progress: 100, error: "Unsupported file type" },
        ]);
        onToast(`${file.name}: unsupported file type`, "error");
        return;
      }

      const entry: UploadingFile = { id, name: file.name, stage: "Uploading", progress: 5 };
      setFiles((prev) => [...prev, entry]);

      const update = (patch: Partial<UploadingFile>) =>
        setFiles((prev) => prev.map((f) => (f.id === id ? { ...f, ...patch } : f)));

      try {
        const { job_id } = await ingestFile(file, "auto");
        update({ stage: "Processing", progress: STAGE_PROGRESS.parsing });
        trackJob(file.name, "ingest", job_id, getIngestStatus, onUploaded);

        await new Promise<void>((resolve, reject) => {
          const poll = setInterval(async () => {
            try {
              const status = await getIngestStatus(job_id);
              const prog = STAGE_PROGRESS[status.stage.toLowerCase()] ?? 50;
              update({ stage: displayStage(status.stage), progress: prog });

              if (status.status === "done") {
                clearInterval(poll);
                update({ stage: "Ready", progress: 100 });
                resolve();
              } else if (status.status === "failed") {
                clearInterval(poll);
                reject(new Error("Ingestion failed"));
              }
            } catch (e) {
              clearInterval(poll);
              reject(e);
            }
          }, 2000);
        });

        onUploaded();
        setTimeout(() => setFiles((prev) => prev.filter((f) => f.id !== id)), 2000);
      } catch (err) {
        update({ stage: "Failed", progress: 100, error: (err as Error).message });
        onToast(`Failed to upload ${file.name}`, "error");
      }
    },
    [onUploaded, onToast]
  );

  const handleFiles = (list: FileList | null) => {
    if (!list || offline) return;
    Array.from(list).forEach((f) => processFile(f));
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    handleFiles(e.dataTransfer.files);
  };

  return (
    <div className="space-y-2 border-t border-border pt-3">
      {offline && (
        <Alert>
          <AlertDescription>The document service is temporarily unavailable — uploads paused.</AlertDescription>
        </Alert>
      )}

      <div
        className={cn(
          "rounded-lg border-2 border-dashed p-3 text-center transition-colors",
          offline
            ? "cursor-not-allowed border-border opacity-50"
            : dragging
            ? "cursor-pointer border-ring bg-muted"
            : "cursor-pointer border-border hover:border-foreground/20 hover:bg-muted"
        )}
        onDragOver={(e) => {
          if (offline) return;
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onClick={() => !offline && inputRef.current?.click()}
      >
        <Upload className="mx-auto mb-1 size-4 text-muted-foreground" />
        <p className="text-xs font-medium text-foreground">Upload documents</p>
        <p className="mt-0.5 text-[10px] text-muted-foreground">PDF · Excel · CSV · DOCX · MD · Image</p>
        <input
          ref={inputRef}
          type="file"
          multiple
          accept={ACCEPTED}
          className="hidden"
          disabled={offline}
          onChange={(e) => handleFiles(e.target.files)}
        />
      </div>

      {files.length > 0 && (
        <div className="space-y-2 pt-1">
          {files.map((f) => (
            <div key={f.id} className="text-xs">
              <div className="mb-1 flex items-center justify-between gap-1.5">
                <Tooltip>
                  <TooltipTrigger render={<span className="max-w-[180px] truncate text-foreground" />}>
                    {f.name}
                  </TooltipTrigger>
                  <TooltipContent>{f.name}</TooltipContent>
                </Tooltip>
                <span className="flex shrink-0 items-center gap-1 text-[10px] text-muted-foreground">
                  {f.progress === 100 && !f.error && <CheckCircle2 className="size-3 text-emerald-600" />}
                  {f.stage}
                </span>
              </div>
              <Progress
                value={f.progress}
                className="[&_[data-slot=progress-track]]:h-1.5"
              >
                <div
                  className={cn(
                    "sr-only",
                    f.error && "not-sr-only text-[10px] text-destructive"
                  )}
                >
                  {f.error}
                </div>
              </Progress>
              {f.error && (
                <Alert variant="destructive" className="mt-1 py-1.5">
                  <AlertDescription className="text-[10px]">{f.error}</AlertDescription>
                </Alert>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
