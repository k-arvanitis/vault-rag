"use client";

import { useEffect, useState } from "react";
import { ChevronLeft, ChevronRight, FileSearch, Loader2 } from "lucide-react";
import {
  getPdfCrop,
  getPdfHighlight,
  getPdfPage,
  getTableSheet,
  sourceInspectTarget,
  type InspectTarget,
  type Source,
} from "@/lib/api";
import { findMatchedRowIndex } from "@/lib/tableMatch";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Table, TableHeader, TableRow, TableHead, TableBody, TableCell } from "@/components/ui/table";

const IS_PDF = (f: string) => f.toLowerCase().endsWith(".pdf");

// Generous margin (PDF points) around the highlight bbox for the focused crop
// -- wide enough to always hit the page's real edges (fitz clips to the
// actual page rect, so an oversized request is harmless) and tall enough to
// read as "a couple lines of context", not just the bare cited phrase.
const CROP_MARGIN_X = 250;
const CROP_MARGIN_Y = 70;

function focusedCropBbox(
  bbox: [number, number, number, number]
): [number, number, number, number] {
  const [x0, y0, x1, y1] = bbox;
  return [
    Math.max(0, x0 - CROP_MARGIN_X),
    Math.max(0, y0 - CROP_MARGIN_Y),
    x1 + CROP_MARGIN_X,
    y1 + CROP_MARGIN_Y,
  ];
}

/** Renders the cited passage on its PDF page, if locatable.
 *
 * Defaults to a focused crop around the highlight (a couple lines of real
 * context, not a whole page shrunk into a narrow column) -- the highlight
 * bbox is computed live via exact text search against the PDF's real text
 * layer (api.py's /pdf/{page}/highlight, backed by fitz.search_for), then
 * reused as a crop region (api.py's /pdf/{page}/crop, the same endpoint that
 * already crops figures) padded with a margin. Falls back to the full page
 * with an overlay box when no bbox is locatable, or on user request via the
 * "View full page" toggle. Overlay position is a percentage of the image's
 * natural pixel size, not a fixed px*scale offset -- the <img> is CSS-scaled
 * to fit its container, so only percentages stay aligned regardless of
 * displayed size. */
function EvidencePage({ source }: { source: Source }) {
  const [bbox, setBbox] = useState<[number, number, number, number] | null>(null);
  const [cropSrc, setCropSrc] = useState<string | null>(null);
  const [pageSrc, setPageSrc] = useState<string | null>(null);
  const [naturalSize, setNaturalSize] = useState<{ w: number; h: number } | null>(null);
  const [loading, setLoading] = useState(false);
  const [showFullPage, setShowFullPage] = useState(false);

  useEffect(() => {
    setBbox(null);
    setCropSrc(null);
    setPageSrc(null);
    setNaturalSize(null);
    setShowFullPage(false);
    if (!IS_PDF(source.filename) || source.page == null) return;
    const page = source.page;
    setLoading(true);
    getPdfHighlight(source.filename, page, source.quote)
      .catch(() => ({ bbox: null }))
      .then((highlight) => {
        setBbox(highlight.bbox);
        if (highlight.bbox) {
          return getPdfCrop(source.filename, page, focusedCropBbox(highlight.bbox)).then((res) =>
            setCropSrc(`data:image/png;base64,${res.image_b64}`)
          );
        }
        return getPdfPage(source.filename, page).then((res) =>
          setPageSrc(`data:image/png;base64,${res.image_b64}`)
        );
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [source.filename, source.page, source.quote]);

  useEffect(() => {
    if (showFullPage && !pageSrc && source.page != null) {
      getPdfPage(source.filename, source.page).then((res) =>
        setPageSrc(`data:image/png;base64,${res.image_b64}`)
      );
    }
  }, [showFullPage, pageSrc, source.filename, source.page]);

  if (!IS_PDF(source.filename) || source.page == null) return null;

  if (loading) {
    return (
      <div className="mt-2 flex justify-center py-6">
        <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
      </div>
    );
  }

  if (!cropSrc && !pageSrc) return null;

  const displayFullPage = showFullPage || !cropSrc;
  const imgSrc = displayFullPage ? pageSrc : cropSrc;
  if (!imgSrc) {
    return (
      <div className="mt-2 flex justify-center py-6">
        <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
      </div>
    );
  }

  const [x0, y0, x1, y1] = bbox ?? [0, 0, 0, 0];

  return (
    <div className="mt-2 w-full">
      <div className="relative w-full">
        <img
          key={imgSrc}
          src={imgSrc}
          alt={`Page ${source.page}`}
          className="w-full rounded border border-border"
          onLoad={(e) =>
            setNaturalSize({
              w: e.currentTarget.naturalWidth,
              h: e.currentTarget.naturalHeight,
            })
          }
        />
        {displayFullPage && bbox && naturalSize && (
          <div
            className="pointer-events-none absolute rounded-sm border-2 border-amber-500 bg-amber-400/20"
            style={{
              left: `${(x0 / (naturalSize.w / 1.5)) * 100}%`,
              top: `${(y0 / (naturalSize.h / 1.5)) * 100}%`,
              width: `${((x1 - x0) / (naturalSize.w / 1.5)) * 100}%`,
              height: `${((y1 - y0) / (naturalSize.h / 1.5)) * 100}%`,
            }}
          />
        )}
      </div>
      <div className="mt-1 flex items-center justify-between">
        {!bbox ? (
          <p className="text-[10px] text-muted-foreground">
            Exact region unavailable — showing full page.
          </p>
        ) : (
          <span />
        )}
        {bbox && (
          <Button
            variant="link"
            size="xs"
            className="h-auto p-0 text-[10px]"
            onClick={() => setShowFullPage((v) => !v)}
          >
            {displayFullPage ? "View focused crop" : "View full page"}
          </Button>
        )}
      </div>
    </div>
  );
}

/** Shows the actual source figure/chart, cropped from the PDF at ingestion-time
 * bbox — a companion to the VLM's text description, not a replacement for it. */
function FigureCrop({ source }: { source: Source }) {
  const [imgSrc, setImgSrc] = useState<string | null>(null);

  useEffect(() => {
    setImgSrc(null);
    if (!source.figure_bbox || source.page == null) return;
    getPdfCrop(source.filename, source.page, source.figure_bbox)
      .then((res) => setImgSrc(`data:image/png;base64,${res.image_b64}`))
      .catch(() => setImgSrc(null));
  }, [source.filename, source.page, source.figure_bbox]);

  if (!imgSrc) return null;

  return (
    <div className="mt-2">
      <img
        src={imgSrc}
        alt="Source figure"
        className="w-full rounded border border-border"
      />
      <p className="mt-1 text-[10px] text-muted-foreground">
        Cropped directly from the source page — the AI-generated description used to make this figure
        searchable isn&apos;t shown separately here yet.
      </p>
    </div>
  );
}

const SHEET_CONTEXT_ROWS = 3;

/** Renders the source sheet's actual rows around the cited passage, with a
 * best-effort row highlight — matches a row when one of its cell values
 * appears verbatim in the citation's quote (see lib/tableMatch.ts). No exact
 * row/cell reference is tracked at ingestion time (spreadsheets only ever
 * get a sheet-level summary chunk — see TODO.md), so this is a heuristic
 * over the live sheet data, not a stored coordinate; when nothing matches,
 * it falls back to the sheet's first rows with a note, same "never
 * fabricate" pattern as the PDF bbox highlight below. Clicking the matched
 * row opens the full Document Inspector with the same row highlighted
 * there (see InspectorPanel.tsx's use of findMatchedRowIndex). */
function SpreadsheetEvidence({
  source,
  onInspect,
}: {
  source: Source;
  onInspect?: (target: InspectTarget) => void;
}) {
  const [rows, setRows] = useState<string[][] | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    setRows(null);
    if (!source.sheet) return;
    setLoading(true);
    getTableSheet(source.filename, source.sheet)
      .then((res) => setRows(res.raw_rows))
      .catch(() => setRows(null))
      .finally(() => setLoading(false));
  }, [source.filename, source.sheet]);

  if (!source.sheet) return null;
  if (loading) {
    return (
      <div className="mt-2 flex justify-center py-6">
        <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
      </div>
    );
  }
  if (!rows || rows.length === 0) return null;

  const [header, ...body] = rows;
  const matchedIdx = findMatchedRowIndex(body, source.quote);

  const start = matchedIdx >= 0 ? Math.max(0, matchedIdx - SHEET_CONTEXT_ROWS) : 0;
  const end =
    matchedIdx >= 0
      ? Math.min(body.length, matchedIdx + SHEET_CONTEXT_ROWS + 1)
      : Math.min(body.length, SHEET_CONTEXT_ROWS * 2 + 1);
  const visible = body.slice(start, end);

  return (
    <div className="mt-2">
      <div className="max-h-64 overflow-auto rounded border border-border">
        <Table>
          <TableHeader>
            <TableRow>
              {header.map((h, i) => (
                <TableHead key={i} className="whitespace-nowrap text-[10px]">
                  {h}
                </TableHead>
              ))}
            </TableRow>
          </TableHeader>
          <TableBody>
            {visible.map((row, i) => {
              const rowIdx = start + i;
              const isMatch = rowIdx === matchedIdx;
              return (
                <TableRow
                  key={rowIdx}
                  className={cn(
                    isMatch && "bg-amber-100 dark:bg-amber-950",
                    onInspect && "cursor-pointer hover:bg-muted"
                  )}
                  onClick={() => onInspect?.(sourceInspectTarget(source))}
                >
                  {row.map((cell, j) => (
                    <TableCell key={j} className="whitespace-nowrap text-[10px]">
                      {cell}
                    </TableCell>
                  ))}
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </div>
      <p className="mt-1 text-[10px] text-muted-foreground">
        {matchedIdx < 0
          ? "Exact row unavailable — showing the sheet's first rows and the quoted summary below."
          : "Highlighted row is a best-effort text match, not a stored cell reference — check it against the quote below."}
        {onInspect && " Click a row to open it in the document inspector."}
      </p>
    </div>
  );
}

interface Props {
  sources: Source[];
  onInspect?: (target: InspectTarget) => void;
  /** Last citation the user clicked — selects/scrolls to the matching source. */
  selectedTarget?: InspectTarget | null;
}

export default function EvidencePanel({ sources, onInspect, selectedTarget }: Props) {
  const [index, setIndex] = useState(0);

  useEffect(() => {
    if (!selectedTarget) return;
    const i = sources.findIndex(
      (s) =>
        s.filename === selectedTarget.filename &&
        (s.page ?? undefined) === selectedTarget.page &&
        (s.sheet ?? undefined) === selectedTarget.sheet
    );
    if (i >= 0) setIndex(i);
  }, [selectedTarget, sources]);

  useEffect(() => {
    setIndex((i) => (i < sources.length ? i : 0));
  }, [sources]);

  if (sources.length === 0) {
    return (
      <aside className="h-full w-full overflow-y-auto bg-background p-3">
        <div className="rounded-lg border border-dashed border-border bg-card px-3 py-6 text-center">
          <p className="text-xs font-medium text-foreground">Verify every answer</p>
          <p className="mt-1.5 text-xs text-muted-foreground">
            Click a citation to inspect the exact supporting page, sheet or row.
          </p>
        </div>
      </aside>
    );
  }

  const source = sources[index];

  return (
    <aside className="h-full w-full overflow-y-auto bg-background p-3">
      <div className="flex items-center justify-between gap-2">
        <Button
          variant="ghost"
          size="icon-xs"
          disabled={index === 0}
          onClick={() => setIndex((i) => Math.max(0, i - 1))}
          aria-label="Previous citation"
        >
          <ChevronLeft />
        </Button>
        <span className="text-[11px] text-muted-foreground">
          {index + 1} / {sources.length}
        </span>
        <Button
          variant="ghost"
          size="icon-xs"
          disabled={index === sources.length - 1}
          onClick={() => setIndex((i) => Math.min(sources.length - 1, i + 1))}
          aria-label="Next citation"
        >
          <ChevronRight />
        </Button>
      </div>

      <div className="mt-2 space-y-1">
        <p className="truncate text-sm font-medium text-foreground">{source.document_title}</p>
        <div className="flex flex-wrap items-center gap-1.5">
          {source.page != null && <Badge variant="secondary">p.{source.page}</Badge>}
          {source.sheet && <Badge variant="secondary">{source.sheet}</Badge>}
          {source.section && <span className="text-[11px] text-muted-foreground">{source.section}</span>}
        </div>
      </div>

      {source.quote && (
        <div className="mt-2">
          <p className="text-[10px] font-medium uppercase tracking-wide text-muted-foreground">
            Supporting evidence
          </p>
          <blockquote className="mt-1 border-l-2 border-amber-500 pl-2 text-sm leading-relaxed text-foreground">
            &ldquo;{source.quote}&rdquo;
          </blockquote>
        </div>
      )}

      <FigureCrop source={source} />
      <EvidencePage source={source} />
      <SpreadsheetEvidence source={source} onInspect={onInspect} />

      {onInspect && (
        <Button
          variant="outline"
          size="sm"
          className="mt-3 gap-1.5"
          onClick={() => onInspect(sourceInspectTarget(source))}
        >
          <FileSearch className="size-3.5" />
          Open full document
        </Button>
      )}
    </aside>
  );
}
