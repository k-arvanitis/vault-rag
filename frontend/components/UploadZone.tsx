"use client";

import { useCallback, useRef, useState } from "react";
import { Upload } from "lucide-react";
import { ingestFile, getIngestStatus } from "@/lib/api";
import { cn } from "@/lib/utils";

interface UploadingFile {
  id: string;
  name: string;
  stage: string;
  progress: number;
  error?: string;
}

const STAGES = ["Parsing", "Chunking", "Embedding", "Indexed"];
const STAGE_PROGRESS: Record<string, number> = {
  parsing: 10,
  chunking: 40,
  embedding: 70,
  done: 100,
  failed: 100,
};

const ACCEPTED = ".pdf,.xlsx,.xls,.csv,.docx,.doc,.md,.png,.jpg,.jpeg";

type Pipeline = "auto" | "ocr" | "text";

interface Props {
  onUploaded: () => void;
  onToast: (msg: string, variant?: "error") => void;
}

const PIPELINE_LABELS: Record<Pipeline, string> = {
  auto: "Auto",
  ocr: "Force OCR",
  text: "Force Text",
};

export default function UploadZone({ onUploaded, onToast }: Props) {
  const [dragging, setDragging] = useState(false);
  const [files, setFiles] = useState<UploadingFile[]>([]);
  const [pipeline, setPipeline] = useState<Pipeline>("auto");
  const inputRef = useRef<HTMLInputElement>(null);

  const processFile = useCallback(
    async (file: File, selectedPipeline: Pipeline) => {
      const id = `${Date.now()}-${file.name}`;
      const entry: UploadingFile = { id, name: file.name, stage: "Parsing", progress: 5 };
      setFiles((prev) => [...prev, entry]);

      const update = (patch: Partial<UploadingFile>) =>
        setFiles((prev) => prev.map((f) => (f.id === id ? { ...f, ...patch } : f)));

      try {
        const { job_id } = await ingestFile(file, selectedPipeline);
        update({ stage: "Parsing", progress: STAGE_PROGRESS.parsing });

        await new Promise<void>((resolve, reject) => {
          const poll = setInterval(async () => {
            try {
              const status = await getIngestStatus(job_id);
              const stageName =
                status.stage.charAt(0).toUpperCase() + status.stage.slice(1).toLowerCase();
              const prog = STAGE_PROGRESS[status.stage.toLowerCase()] ?? 50;
              update({ stage: stageName, progress: prog });

              if (status.status === "done") {
                clearInterval(poll);
                update({ stage: "Indexed", progress: 100 });
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

  const handleFiles = (list: FileList | null, selectedPipeline: Pipeline) => {
    if (!list) return;
    Array.from(list).forEach((f) => processFile(f, selectedPipeline));
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    handleFiles(e.dataTransfer.files, pipeline);
  };

  return (
    <div className="space-y-2 border-t border-ink-200 pt-3">
      <div
        className={cn(
          "cursor-pointer rounded-lg border-2 border-dashed p-3 text-center transition-colors",
          dragging
            ? "border-brand bg-ink-100"
            : "border-ink-200 hover:border-ink-300 hover:bg-ink-100"
        )}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onClick={() => inputRef.current?.click()}
      >
        <Upload className="mx-auto mb-1 h-4 w-4 text-ink-400" />
        <p className="text-xs font-medium text-ink-700">Upload documents</p>
        <p className="mt-0.5 text-[10px] text-ink-400">PDF · Excel · CSV · DOCX · MD · Image</p>
        <input
          ref={inputRef}
          type="file"
          multiple
          accept={ACCEPTED}
          className="hidden"
          onChange={(e) => handleFiles(e.target.files, pipeline)}
        />
      </div>

      {/* PDF pipeline selector — ignored for non-PDF file types */}
      <div className="flex items-center gap-1">
        <span className="shrink-0 text-[10px] text-ink-400">PDF pipeline:</span>
        <div className="ml-1 flex gap-0.5">
          {(["auto", "ocr", "text"] as Pipeline[]).map((p) => (
            <button
              key={p}
              onClick={() => setPipeline(p)}
              className={cn(
                "rounded px-2 py-0.5 text-[10px] transition-colors",
                pipeline === p
                  ? "bg-brand text-white"
                  : "text-ink-500 hover:bg-ink-100 hover:text-ink-700"
              )}
            >
              {PIPELINE_LABELS[p]}
            </button>
          ))}
        </div>
      </div>

      {files.length > 0 && (
        <div className="space-y-2 pt-1">
          {files.map((f) => (
            <div key={f.id} className="text-xs">
              <div className="mb-1 flex items-center justify-between">
                <span className="max-w-[180px] truncate text-ink-700">{f.name}</span>
                <span className={cn("ml-1 shrink-0 text-[10px]", f.error ? "text-red-600" : "text-ink-500")}>
                  {f.stage}
                </span>
              </div>
              <div className="h-1.5 overflow-hidden rounded-full bg-ink-100">
                <div
                  className={cn(
                    "h-full rounded-full transition-all duration-500",
                    f.error ? "bg-red-500" : f.progress === 100 ? "bg-emerald-500" : "bg-brand"
                  )}
                  style={{ width: `${f.progress}%` }}
                />
              </div>
              {f.stage !== "Indexed" && !f.error && (
                <div className="mt-1 flex gap-1.5">
                  {STAGES.map((s) => (
                    <span
                      key={s}
                      className={cn("text-[9px]", f.stage === s ? "text-ink-700" : "text-ink-300")}
                    >
                      {s}
                    </span>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
