"use client";

import { useCallback, useRef, useState } from "react";
import { Upload, X } from "lucide-react";
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
    <div className="mt-auto pt-3 border-t border-zinc-800">
      <div
        className={cn(
          "rounded-lg border-2 border-dashed p-3 text-center cursor-pointer transition-colors",
          dragging ? "border-zinc-500 bg-zinc-800/50" : "border-zinc-700 hover:border-zinc-600"
        )}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onClick={() => inputRef.current?.click()}
      >
        <Upload className="mx-auto mb-1 h-4 w-4 text-zinc-500" />
        <p className="text-xs text-zinc-400">Upload Documents</p>
        <p className="text-[10px] text-zinc-600 mt-0.5">PDF · Excel · CSV · DOCX · MD · Image</p>
        <input
          ref={inputRef}
          type="file"
          multiple
          accept={ACCEPTED}
          className="hidden"
          onChange={(e) => handleFiles(e.target.files, pipeline)}
        />
      </div>

      {/* Pipeline selector — PDF only, ignored for other file types */}
      <div className="mt-2 flex items-center gap-1">
        <span className="text-[10px] text-zinc-600 shrink-0">PDF pipeline:</span>
        <div className="flex gap-0.5 ml-1">
          {(["auto", "ocr", "text"] as Pipeline[]).map((p) => (
            <button
              key={p}
              onClick={() => setPipeline(p)}
              className={cn(
                "text-[10px] px-2 py-0.5 rounded transition-colors",
                pipeline === p
                  ? "bg-zinc-700 text-zinc-200"
                  : "text-zinc-600 hover:text-zinc-400"
              )}
            >
              {PIPELINE_LABELS[p]}
            </button>
          ))}
        </div>
      </div>

      {files.length > 0 && (
        <div className="mt-2 space-y-2">
          {files.map((f) => (
            <div key={f.id} className="text-xs">
              <div className="flex items-center justify-between mb-1">
                <span className="text-zinc-300 truncate max-w-[180px]">{f.name}</span>
                <span
                  className={cn(
                    "text-[10px] ml-1 shrink-0",
                    f.error ? "text-red-400" : "text-zinc-500"
                  )}
                >
                  {f.stage}
                </span>
              </div>
              <div className="h-1 bg-zinc-800 rounded-full overflow-hidden">
                <div
                  className={cn(
                    "h-full rounded-full transition-all duration-500",
                    f.error ? "bg-red-600" : f.progress === 100 ? "bg-emerald-600" : "bg-zinc-400"
                  )}
                  style={{ width: `${f.progress}%` }}
                />
              </div>
              {f.stage !== "Indexed" && !f.error && (
                <div className="flex gap-1 mt-1">
                  {STAGES.map((s) => (
                    <span
                      key={s}
                      className={cn(
                        "text-[9px]",
                        f.stage === s ? "text-zinc-300" : "text-zinc-700"
                      )}
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
