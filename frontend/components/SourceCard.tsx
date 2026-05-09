import { type Source } from "@/lib/api";
import { cn } from "@/lib/utils";

interface Props {
  source: Source;
  index: number;
}

export default function SourceCard({ source, index }: Props) {
  const isTable = source.location.startsWith("sheet");
  const scoreColor =
    source.score == null
      ? ""
      : source.score >= 0.85
      ? "text-emerald-500"
      : source.score >= 0.65
      ? "text-yellow-500"
      : "text-red-500";

  return (
    <div className="rounded-md bg-zinc-800/60 border border-zinc-700/40 overflow-hidden">
      {/* header row */}
      <div className="flex items-center gap-2 px-3 py-2 border-b border-zinc-700/40 bg-zinc-800/80">
        <span className="text-[10px] font-mono text-zinc-500 shrink-0">[{index}]</span>
        <span className="text-[11px] font-medium text-zinc-200 truncate flex-1 min-w-0">
          {source.filename.split("/").pop()}
        </span>
        <div className="flex items-center gap-1.5 shrink-0">
          {source.location && (
            <span className={cn(
              "text-[9px] px-1.5 py-0.5 rounded font-mono",
              isTable ? "bg-green-900/60 text-green-400" : "bg-zinc-700 text-zinc-400"
            )}>
              {source.location}
            </span>
          )}
          {source.score != null && (
            <span className={cn("text-[9px] font-mono", scoreColor)}>
              {source.score.toFixed(3)}
            </span>
          )}
        </div>
      </div>

      {/* section heading */}
      {source.section && (
        <div className="px-3 pt-2 pb-0">
          <p className="text-[11px] font-medium text-zinc-300 leading-tight">{source.section}</p>
        </div>
      )}

      {/* excerpt */}
      <p className="px-3 py-2 text-[11px] text-zinc-500 leading-relaxed line-clamp-3">
        {source.excerpt || "—"}
      </p>
    </div>
  );
}
