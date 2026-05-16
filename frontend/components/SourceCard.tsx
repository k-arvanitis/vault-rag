import { type Source } from "@/lib/api";
import { cn } from "@/lib/utils";

interface Props {
  source: Source;
  index: number;
}

export default function SourceCard({ source, index }: Props) {
  const isDocSummary = source.location === "document summary";
  const isSheetSummary = source.location.startsWith("sheet summary");
  const isSummary = isDocSummary || isSheetSummary;
  const isTable = source.location.startsWith("sheet") && !isSummary;

  const scoreChip =
    source.score == null
      ? ""
      : source.score >= 0.85
      ? "bg-emerald-100 text-emerald-700"
      : source.score >= 0.65
      ? "bg-amber-100 text-amber-700"
      : "bg-ink-100 text-ink-500";

  const basename = source.filename.split("/").pop() ?? source.filename;
  const sheetName = isTable ? source.location.replace(/^sheet:\s*/, "") : "";

  return (
    <div className="overflow-hidden rounded-md border border-ink-200 bg-surface">
      {/* header row */}
      <div className="flex items-center gap-2 border-b border-ink-100 px-3 py-2">
        <span className="shrink-0 font-mono text-[10px] text-ink-400">[{index}]</span>
        <span className="min-w-0 flex-1 truncate text-[11px] font-medium text-ink-800">{basename}</span>
        <div className="flex shrink-0 items-center gap-1.5">
          {isSummary && (
            <span className="rounded bg-brand/10 px-1.5 py-0.5 text-[10px] text-brand-dark">
              {isDocSummary ? "doc summary" : "sheet summary"}
            </span>
          )}
          {source.page != null && !isSummary && (
            <span className="rounded bg-ink-100 px-1.5 py-0.5 text-[10px] text-ink-600">page {source.page}</span>
          )}
          {isTable && sheetName && (
            <span className="rounded bg-emerald-50 px-1.5 py-0.5 text-[10px] text-emerald-700">{sheetName}</span>
          )}
          {source.score != null && (
            <span className={cn("rounded px-1.5 py-0.5 font-mono text-[10px]", scoreChip)}>
              {source.score.toFixed(3)}
            </span>
          )}
        </div>
      </div>

      {/* section heading */}
      {source.section && (
        <p className="px-3 pt-2 text-[11px] font-medium leading-tight text-ink-700">{source.section}</p>
      )}

      {/* excerpt */}
      <p className="line-clamp-3 px-3 py-2 text-[11px] leading-relaxed text-ink-500">{source.excerpt || "—"}</p>

      {/* faint chunk footnote */}
      {source.location && !isTable && !isSummary && (
        <p className="px-3 pb-1.5 font-mono text-[10px] text-ink-400">{source.location}</p>
      )}
    </div>
  );
}
