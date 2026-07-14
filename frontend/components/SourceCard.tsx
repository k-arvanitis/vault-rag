"use client";

import { useState } from "react";
import { ScanSearch } from "lucide-react";
import { type Source } from "@/lib/api";
import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { Collapsible, CollapsibleTrigger } from "@/components/ui/collapsible";

const EXCERPT_COLLAPSE_LENGTH = 220;

interface Props {
  source: Source;
  index: number;
  /** Set when this chunk was retrieved but excluded by the reranker. */
  rejected?: boolean;
  /** Opens this source's location in the document inspector, when the caller supports it. */
  onInspect?: () => void;
}

export default function SourceCard({ source, index, rejected, onInspect }: Props) {
  const [expanded, setExpanded] = useState(false);
  const isDocSummary = source.location === "document summary";
  const isSheetSummary = source.location.startsWith("sheet summary");
  const isSummary = isDocSummary || isSheetSummary;
  const isTable = source.location.startsWith("sheet") && !isSummary;

  // Restrained three-tier strength read on the reranker score — no rainbow palette.
  const scoreTier: "strong" | "moderate" | "weak" | null =
    source.score == null ? null : source.score >= 0.85 ? "strong" : source.score >= 0.65 ? "moderate" : "weak";
  const scoreVariant =
    rejected || scoreTier === "weak" ? "outline" : scoreTier === "moderate" ? "secondary" : "default";

  const basename = source.filename.split("/").pop() ?? source.filename;
  const sheetName = isTable ? source.location.replace(/^sheet:\s*/, "") : "";
  const excerpt = source.excerpt || "—";
  const isLong = excerpt.length > EXCERPT_COLLAPSE_LENGTH;

  return (
    <div
      className={cn(
        "overflow-hidden rounded-md border",
        rejected ? "border-border bg-muted/50" : "border-border bg-card"
      )}
    >
      {/* header row */}
      <div className="flex items-center gap-2 border-b border-border px-3 py-2">
        <span className="shrink-0 font-mono text-[10px] text-muted-foreground">[{index}]</span>
        <span
          className={cn(
            "min-w-0 flex-1 truncate text-[11px] font-medium",
            rejected ? "text-muted-foreground" : "text-foreground"
          )}
        >
          {basename}
        </span>
        <div className="flex shrink-0 items-center gap-1.5">
          {isSummary && (
            <Badge variant="secondary" className="text-[10px]">
              {isDocSummary ? "doc summary" : "sheet summary"}
            </Badge>
          )}
          {source.page != null && !isSummary && (
            <Badge variant="outline" className="text-[10px]">
              page {source.page}
            </Badge>
          )}
          {isTable && sheetName && (
            <Badge variant="outline" className="text-[10px]">
              {sheetName}
            </Badge>
          )}
          {source.score != null && (
            <Badge variant={scoreVariant} className="font-mono text-[10px]">
              {source.score.toFixed(3)}
            </Badge>
          )}
          {onInspect && (
            <Tooltip>
              <TooltipTrigger render={<Button variant="ghost" size="icon-xs" onClick={onInspect} aria-label="Open in inspector" />}>
                <ScanSearch />
              </TooltipTrigger>
              <TooltipContent>Open in inspector</TooltipContent>
            </Tooltip>
          )}
        </div>
      </div>

      {/* section heading */}
      {source.section && (
        <p
          className={cn(
            "px-3 pt-2 text-[11px] font-medium leading-tight",
            rejected ? "text-muted-foreground" : "text-foreground"
          )}
        >
          {source.section}
        </p>
      )}

      {/* excerpt — collapsed by default only when long */}
      {isLong ? (
        <Collapsible open={expanded} onOpenChange={setExpanded}>
          <p
            className={cn(
              "px-3 py-2 text-[11px] leading-relaxed text-muted-foreground",
              !expanded && "line-clamp-3"
            )}
          >
            {excerpt}
          </p>
          <CollapsibleTrigger
            render={<Button variant="link" size="xs" className="ml-2 mb-1.5 h-auto p-0 text-[10px]" />}
          >
            {expanded ? "Show less" : "Show more"}
          </CollapsibleTrigger>
        </Collapsible>
      ) : (
        <p className="px-3 py-2 text-[11px] leading-relaxed text-muted-foreground">{excerpt}</p>
      )}

      {/* faint chunk footnote */}
      {source.location && !isTable && !isSummary && (
        <p className="px-3 pb-1.5 font-mono text-[10px] text-muted-foreground/80">{source.location}</p>
      )}
    </div>
  );
}
