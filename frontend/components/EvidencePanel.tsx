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
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Table, TableHeader, TableRow, TableHead, TableBody, TableCell } from "@/components/ui/table";

const IS_PDF = (f: string) => f.toLowerCase().endsWith(".pdf");

/** Renders one PDF page with the cited passage highlighted, if locatable.
 *
 * The highlight bbox is computed live via exact text search against the PDF's
 * real text layer (api.py's /pdf/{page}/highlight, backed by fitz.search_for)
 * — nothing is precomputed or stored at ingestion time. Overlay position is a
 * percentage of the image's natural pixel size, not a fixed px*scale offset —
 * the <img> is CSS-scaled to fit its container, so only percentages stay
 * aligned regardless of displayed size. */
function EvidencePage({ source }: { source: Source }) {
  const [imgSrc, setImgSrc] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [naturalSize, setNaturalSize] = useState<{ w: number; h: number } | null>(null);
  const [bbox, setBbox] = useState<[number, number, number, number] | null>(null);

  useEffect(() => {
    setImgSrc(null);
    setNaturalSize(null);
    setBbox(null);
    if (!IS_PDF(source.filename) || source.page == null) return;
    setLoading(true);
    Promise.all([
      getPdfPage(source.filename, source.page),
      getPdfHighlight(source.filename, source.page, source.quote).catch(() => ({
        bbox: null,
      })),
    ])
      .then(([page, highlight]) => {
        setImgSrc(`data:image/png;base64,${page.image_b64}`);
        setBbox(highlight.bbox);
      })
      .catch(() => setImgSrc(null))
      .finally(() => setLoading(false));
  }, [source.filename, source.page, source.quote]);

  if (!IS_PDF(source.filename) || source.page == null) return null;

  if (loading) {
    return (
      <div className="mt-2 flex justify-center py-6">
        <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
      </div>
    );
  }

  if (!imgSrc) return null;

  const [x0, y0, x1, y1] = bbox ?? [0, 0, 0, 0];

  return (
    <div className="relative mt-2 w-full">
      <img
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
      {bbox && naturalSize && (
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
      {!bbox && (
        <p className="mt-1 text-[10px] text-muted-foreground">
          Exact region unavailable — showing full page and quote.
        </p>
      )}
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
    <img
      src={imgSrc}
      alt="Source figure"
      className="mt-2 w-full rounded border border-border"
    />
  );
}

const SHEET_CONTEXT_ROWS = 3;

function normalizeCell(v: string): string {
  return v.trim().toLowerCase();
}

/** Renders the source sheet's actual rows around the cited passage, with a
 * best-effort row highlight — matches a row when one of its cell values
 * appears verbatim in the citation's quote. No exact row/cell reference is
 * tracked at ingestion time (spreadsheets only ever get a sheet-level
 * summary chunk — see TODO.md), so this is a heuristic over the live sheet
 * data, not a stored coordinate; when nothing matches, it falls back to the
 * sheet's first rows with a note, same "never fabricate" pattern as the PDF
 * bbox highlight below. */
function SpreadsheetEvidence({ source }: { source: Source }) {
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
  const quoteNorm = normalizeCell(source.quote || "");
  const matchedIdx = quoteNorm
    ? body.findIndex((row) =>
        row.some((cell) => {
          const cellNorm = normalizeCell(cell || "");
          return cellNorm.length > 2 && quoteNorm.includes(cellNorm);
        })
      )
    : -1;

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
                <TableRow key={rowIdx} className={isMatch ? "bg-amber-100 dark:bg-amber-950" : undefined}>
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
      {matchedIdx < 0 && (
        <p className="mt-1 text-[10px] text-muted-foreground">
          Exact row unavailable — showing the sheet's first rows and the quoted summary below.
        </p>
      )}
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
      <aside className="h-full w-full overflow-y-auto bg-background p-3 lg:w-[320px] lg:shrink-0 lg:border-l lg:border-border">
        <div className="rounded-lg border border-dashed border-border bg-card px-3 py-6 text-center">
          <p className="text-[11px] uppercase tracking-wide text-muted-foreground">Evidence</p>
          <p className="mt-1.5 text-xs text-muted-foreground">
            Ask a question — cited passages for the latest answer appear here.
          </p>
        </div>
      </aside>
    );
  }

  const source = sources[index];

  return (
    <aside className="h-full w-full overflow-y-auto bg-background p-3 lg:w-[320px] lg:shrink-0 lg:border-l lg:border-border">
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

      <FigureCrop source={source} />
      <EvidencePage source={source} />
      <SpreadsheetEvidence source={source} />

      <blockquote className="mt-2 border-l-2 border-border pl-2 text-xs italic text-muted-foreground">
        &ldquo;{source.quote}&rdquo;
      </blockquote>

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
