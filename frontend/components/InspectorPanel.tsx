"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import { X, ChevronLeft, ChevronRight, Loader2 } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeRaw from "rehype-raw";
import {
  getDocumentChunks,
  getDocumentMarkdown,
  getPdfPage,
  getTableSheet,
  type Chunk,
  type MarkdownPage,
  type TableSheetResponse,
} from "@/lib/api";
import { findMatchedRowIndex } from "@/lib/tableMatch";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { ResizablePanelGroup, ResizablePanel, ResizableHandle } from "@/components/ui/resizable";

interface Props {
  filename: string;
  onClose: () => void;
  /** Jumps straight to this PDF page or spreadsheet sheet, when opened from a citation. */
  page?: number;
  sheet?: string;
  /** The citation's quote text -- when set, the same row-match heuristic
   * SpreadsheetEvidence uses highlights that row here too (see
   * lib/tableMatch.ts). */
  quote?: string;
}

// remark-gfm renders markdown pipe-tables; rehype-raw renders embedded HTML
// (e.g. OCR'd <table> blocks) as real elements instead of leaking raw tags.
const MD_PLUGINS = { remarkPlugins: [remarkGfm], rehypePlugins: [rehypeRaw] };

const IS_PDF = (f: string) => f.toLowerCase().endsWith(".pdf");
const IS_TABLE = (f: string) => /\.(xlsx|xls|csv)$/i.test(f);

/** cleaned_md's file starts with a "[File: ... | Sheet: ...]" header line plus
 * a schema/sample-values summary (the SQL generator's own context) before the
 * actual cleaned markdown table -- confusing to show verbatim as prose. Drop
 * everything before the first table row. */
export function extractTableMarkdown(cleanedMd: string): string {
  const lines = cleanedMd.split("\n");
  const start = lines.findIndex((l) => l.trimStart().startsWith("|"));
  return start === -1 ? cleanedMd : lines.slice(start).join("\n");
}

function PanelSkeleton() {
  return (
    <div className="flex-1 space-y-3 p-5">
      <Skeleton className="h-4 w-1/3" />
      <Skeleton className="h-32 w-full" />
      <Skeleton className="h-32 w-full" />
    </div>
  );
}

// ── PDF Inspector ──────────────────────────────────────────────────────────────

function PdfPage({
  currentPage,
  pdfMissing,
  imgLoading,
  imgSrc,
}: {
  currentPage: number;
  pdfMissing: boolean;
  imgLoading: boolean;
  imgSrc: string | null;
}) {
  if (pdfMissing) {
    return (
      <Alert className="m-3">
        <AlertDescription>Original PDF not available locally.</AlertDescription>
      </Alert>
    );
  }
  if (imgLoading) {
    return (
      <div className="mt-10 flex justify-center">
        <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
      </div>
    );
  }
  if (imgSrc) {
    return <img src={imgSrc} alt={`Page ${currentPage}`} className="w-full rounded border border-border" />;
  }
  return null;
}

function PdfInspector({ filename, initialPage }: { filename: string; initialPage?: number }) {
  const [loading, setLoading] = useState(true);
  const [pages, setPages] = useState<MarkdownPage[]>([]);
  const [hasMarkers, setHasMarkers] = useState(false);
  const [fullText, setFullText] = useState<string | null>(null);
  const [currentPage, setCurrentPage] = useState(initialPage ?? 1);
  const [imgSrc, setImgSrc] = useState<string | null>(null);
  const [imgLoading, setImgLoading] = useState(false);
  const [summary, setSummary] = useState<string | null>(null);
  const [showSummary, setShowSummary] = useState(false);
  const [pdfMissing, setPdfMissing] = useState(false);

  useEffect(() => {
    setLoading(true);
    Promise.all([getDocumentMarkdown(filename), getDocumentChunks(filename)])
      .then(([md, chunks]) => {
        setHasMarkers(md.has_page_markers);
        setPages(md.pages);
        setFullText(md.full_text);
        setSummary(chunks.summary);
        setCurrentPage(initialPage ?? 1);
      })
      .finally(() => setLoading(false));
  }, [filename, initialPage]);

  useEffect(() => {
    if (!hasMarkers || pages.length === 0) return;
    setImgLoading(true);
    setImgSrc(null);
    getPdfPage(filename, currentPage)
      .then((r) => setImgSrc(`data:image/png;base64,${r.image_b64}`))
      .catch(() => setPdfMissing(true))
      .finally(() => setImgLoading(false));
  }, [filename, currentPage, hasMarkers, pages.length]);

  const totalPages = pages.length || 0;

  if (loading) return <PanelSkeleton />;

  const mdPage = pages.find((p) => p.page === currentPage);

  const parsedContent = mdPage ? (
    <div className="prose-ui text-xs leading-relaxed text-foreground">
      <ReactMarkdown {...MD_PLUGINS}>{mdPage.content}</ReactMarkdown>
    </div>
  ) : (
    <p className="text-xs text-muted-foreground">No content for this page.</p>
  );

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      {/* Summary */}
      {summary && (
        <div className="px-5 pt-3">
          <Button
            variant="link"
            size="xs"
            className="mb-2 h-auto p-0 text-xs"
            onClick={() => setShowSummary((v) => !v)}
          >
            {showSummary ? "Hide" : "Show"} document summary
          </Button>
          {showSummary && (
            <div className="prose-ui mb-3 max-h-40 overflow-y-auto rounded-md border border-border bg-card p-3 text-xs text-foreground">
              <ReactMarkdown {...MD_PLUGINS}>{summary.replace("## Document Summary\n\n", "")}</ReactMarkdown>
            </div>
          )}
        </div>
      )}

      {hasMarkers ? (
        <>
          {/* Page nav */}
          <div className="flex shrink-0 items-center gap-3 border-b border-border bg-card px-5 py-2">
            <Button
              variant="ghost"
              size="icon-xs"
              onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
              disabled={currentPage <= 1}
              aria-label="Previous page"
            >
              <ChevronLeft />
            </Button>
            <span className="text-xs tabular-nums text-muted-foreground">
              Page {currentPage} / {totalPages}
            </span>
            <Button
              variant="ghost"
              size="icon-xs"
              onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
              disabled={currentPage >= totalPages}
              aria-label="Next page"
            >
              <ChevronRight />
            </Button>
            {mdPage?.pipeline && <Badge variant="secondary" className="ml-auto font-mono">{mdPage.pipeline}</Badge>}
          </div>

          {/* Desktop: side by side, resizable. Mobile/tablet: tabs. */}
          <div className="hidden min-h-0 flex-1 lg:flex">
            <ResizablePanelGroup orientation="horizontal">
              <ResizablePanel defaultSize={50} minSize={25}>
                <ScrollArea className="h-full p-3">
                  <PdfPage
                    currentPage={currentPage}
                    pdfMissing={pdfMissing}
                    imgLoading={imgLoading}
                    imgSrc={imgSrc}
                  />
                </ScrollArea>
              </ResizablePanel>
              <ResizableHandle withHandle />
              <ResizablePanel defaultSize={50} minSize={25}>
                <ScrollArea className="h-full p-4">{parsedContent}</ScrollArea>
              </ResizablePanel>
            </ResizablePanelGroup>
          </div>

          <div className="min-h-0 flex-1 lg:hidden">
            <Tabs defaultValue="original" className="flex h-full flex-col">
              <TabsList className="mx-3 mt-2">
                <TabsTrigger value="original">Original</TabsTrigger>
                <TabsTrigger value="parsed">Parsed</TabsTrigger>
              </TabsList>
              <TabsContent value="original" className="min-h-0 flex-1">
                <ScrollArea className="h-full p-3">
                  <PdfPage
                    currentPage={currentPage}
                    pdfMissing={pdfMissing}
                    imgLoading={imgLoading}
                    imgSrc={imgSrc}
                  />
                </ScrollArea>
              </TabsContent>
              <TabsContent value="parsed" className="min-h-0 flex-1">
                <ScrollArea className="h-full p-4">{parsedContent}</ScrollArea>
              </TabsContent>
            </Tabs>
          </div>
        </>
      ) : (
        <div className="flex-1 overflow-y-auto p-5">
          <p className="mb-3 text-[10px] text-muted-foreground">No page markers — showing full parsed markdown.</p>
          <div className="prose-ui text-xs leading-relaxed text-foreground">
            <ReactMarkdown {...MD_PLUGINS}>{fullText ?? ""}</ReactMarkdown>
          </div>
        </div>
      )}
    </div>
  );
}

// ── Raw vs. cleaned sheet view ─────────────────────────────────────────────────

function SheetCompare({
  filename,
  sheet,
  quote,
}: {
  filename: string;
  sheet: string;
  quote?: string;
}) {
  const [data, setData] = useState<TableSheetResponse | null>(null);
  const [loading, setLoading] = useState(false);
  // Open by default -- the cleaned table is the primary thing a user wants
  // to see here, not something to hunt for behind a toggle.
  const [open, setOpen] = useState(true);
  const matchedRowRef = useRef<HTMLTableRowElement>(null);

  const load = useCallback(() => {
    if (data || loading) return;
    setLoading(true);
    getTableSheet(filename, sheet)
      .then(setData)
      .catch(() => setData(null))
      .finally(() => setLoading(false));
  }, [filename, sheet, data, loading]);

  useEffect(() => {
    load();
    // Only ever auto-loads once, on mount.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    matchedRowRef.current?.scrollIntoView({ block: "center" });
  }, [data, quote]);

  const rawBody = (data?.raw_rows ?? []).slice(1);
  const matchedIdx = findMatchedRowIndex(rawBody, quote);

  return (
    <div className="mb-2">
      <Button
        variant="link"
        size="xs"
        className="h-auto p-0 text-[10px]"
        onClick={() => {
          setOpen((v) => !v);
          if (!open) load();
        }}
      >
        {open ? "Hide" : "Show"} raw vs. cleaned
      </Button>

      {open && (
        <div className="mt-2 flex min-h-0 gap-3" style={{ maxHeight: "340px" }}>
          {/* Raw */}
          <div className="flex-1 overflow-auto rounded-md border border-border">
            <p className="sticky top-0 border-b border-border bg-muted px-2 py-1 text-[10px] text-muted-foreground">
              Raw (from file)
            </p>
            {loading ? (
              <div className="flex justify-center p-4">
                <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
              </div>
            ) : data?.raw_rows ? (
              <table className="w-full font-mono text-[9px] text-foreground">
                <tbody>
                  {data.raw_rows.map((row, ri) => {
                    // Row 0 is the header (fetched with header=None) --
                    // rawBody/matchedIdx are indexed from row 1.
                    const isMatch = ri > 0 && ri - 1 === matchedIdx;
                    return (
                    <tr
                      key={ri}
                      ref={isMatch ? matchedRowRef : undefined}
                      className={cn("border-b border-border", isMatch && "bg-amber-100 dark:bg-amber-950")}
                    >
                      {row.map((cell, ci) => (
                        <td
                          key={ci}
                          className="max-w-[180px] overflow-hidden text-ellipsis whitespace-nowrap px-2 py-0.5"
                          title={cell}
                        >
                          {cell}
                        </td>
                      ))}
                    </tr>
                    );
                  })}
                </tbody>
              </table>
            ) : (
              <p className="p-3 text-[10px] text-muted-foreground">Not available</p>
            )}
          </div>

          {/* Cleaned */}
          <div className="flex-1 overflow-auto rounded-md border border-border">
            <p className="sticky top-0 border-b border-border bg-muted px-2 py-1 text-[10px] text-muted-foreground">
              Cleaned table
            </p>
            {loading ? (
              <div className="flex justify-center p-4">
                <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
              </div>
            ) : data?.cleaned_md ? (
              <div className="prose-ui p-2 text-[9px] leading-relaxed text-foreground [&_table]:text-[9px]">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {extractTableMarkdown(data.cleaned_md)}
                </ReactMarkdown>
              </div>
            ) : (
              <p className="p-3 text-[10px] text-muted-foreground">Not available</p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

// ── Excel / CSV Inspector ──────────────────────────────────────────────────────

function TableInspector({
  filename,
  initialSheet,
  quote,
}: {
  filename: string;
  initialSheet?: string;
  quote?: string;
}) {
  const [loading, setLoading] = useState(true);
  const [summary, setSummary] = useState<string | null>(null);
  const [chunks, setChunks] = useState<Chunk[]>([]);
  // Collapsed by default -- this is what the SQL generator reads, not
  // something a normal user opening the inspector wants to see up front.
  const [showSummary, setShowSummary] = useState(false);

  useEffect(() => {
    setLoading(true);
    getDocumentChunks(filename)
      .then((r) => {
        setSummary(r.summary);
        setChunks(r.chunks);
      })
      .finally(() => setLoading(false));
  }, [filename]);

  if (loading) return <PanelSkeleton />;

  const sheets: Record<string, Chunk[]> = {};
  for (const c of chunks) {
    const sheet = c.metadata.sheet_name ?? "—";
    if (!sheets[sheet]) sheets[sheet] = [];
    sheets[sheet].push(c);
  }

  return (
    <ScrollArea className="min-h-0 flex-1">
      <div className="space-y-4 p-5">
        <p className="text-xs text-muted-foreground">{chunks.length} data chunks indexed</p>

        {summary && (
          <div>
            <Button
              variant="link"
              size="xs"
              className="mb-2 h-auto p-0 text-xs"
              onClick={() => setShowSummary((v) => !v)}
            >
              {showSummary ? "Hide" : "Show"} schema &amp; sample values — read by the SQL generator
            </Button>
            {showSummary && (
              <div className="prose-ui rounded-md border border-border bg-card p-3 text-xs text-foreground">
                <ReactMarkdown {...MD_PLUGINS}>
                  {summary.replace("## Document Summary\n\n", "").replace(/ 00:00:00/g, "")}
                </ReactMarkdown>
              </div>
            )}
          </div>
        )}

        {Object.entries(sheets).map(([sheet, sheetChunks]) => (
          <div
            key={sheet}
            ref={(el) => {
              if (sheet === initialSheet) el?.scrollIntoView({ block: "start" });
            }}
          >
            <p className="mb-2 text-xs font-semibold text-foreground">Sheet: {sheet}</p>

            <SheetCompare filename={filename} sheet={sheet} quote={sheet === initialSheet ? quote : undefined} />

            <p className="mb-2 mt-3 text-[10px] text-muted-foreground">
              Indexed chunk — this sheet's data is queried directly, not read from this summary
            </p>
            <div className="space-y-2">
              {sheetChunks.map((c, i) => {
                const meta = c.metadata;
                const chunkType = meta.chunk_type ?? "chunk";
                const rowRef = meta.row_ref;
                const numRows = meta.num_rows ?? 1;
                const rowLabel =
                  rowRef != null
                    ? rowRef === rowRef + numRows - 1
                      ? `row ${rowRef}`
                      : `rows ${rowRef}–${rowRef + numRows - 1}`
                    : null;
                const label = [`Chunk ${i + 1}`, chunkType, rowLabel].filter(Boolean).join(" · ");
                // Drop the leading "[File: … | Sheet: …]" line — already shown in the section header.
                const body = c.content.replace(/^\[File:[^\]]*\]\n/, "");

                return (
                  <details key={i} className="group rounded-md border border-border bg-card">
                    <summary className="flex cursor-pointer select-none items-center gap-2 px-3 py-2 text-[11px] text-muted-foreground hover:bg-muted">
                      <ChevronRight className="h-3 w-3 shrink-0 transition-transform group-open:rotate-90" />
                      {label}
                    </summary>
                    <pre className="overflow-x-auto whitespace-pre-wrap px-3 pb-3 font-mono text-[10px] leading-relaxed text-muted-foreground">
                      {body}
                    </pre>
                  </details>
                );
              })}
            </div>
          </div>
        ))}
      </div>
    </ScrollArea>
  );
}

// ── Panel shell ────────────────────────────────────────────────────────────────

export default function InspectorPanel({ filename, onClose, page, sheet, quote }: Props) {
  const basename = filename.split("/").pop() ?? filename;
  const isPdf = IS_PDF(filename);
  const isTable = IS_TABLE(filename);

  return (
    <div className="fixed inset-0 z-30 flex">
      <div className="flex h-full w-full flex-col bg-background">
        {/* header */}
        <div className="flex shrink-0 items-center gap-3 border-b border-border bg-card px-5 py-3">
          <div className="min-w-0 flex-1">
            <p className="truncate text-xs font-semibold text-foreground">{basename}</p>
            <p className="text-[10px] text-muted-foreground">
              {isPdf
                ? "PDF · side-by-side view"
                : isTable
                ? "Spreadsheet · chunk view"
                : "Document inspector"}
            </p>
          </div>
          <Separator orientation="vertical" className="h-5" />
          <Button variant="ghost" size="icon-sm" onClick={onClose} aria-label="Close inspector">
            <X />
          </Button>
        </div>

        {isPdf ? (
          <PdfInspector filename={filename} initialPage={page} />
        ) : isTable ? (
          <TableInspector filename={filename} initialSheet={sheet} quote={quote} />
        ) : (
          <div className="flex flex-1 items-center justify-center px-8 text-center">
            <p className="text-xs text-muted-foreground">
              A side-by-side preview isn&apos;t available for this file type yet — cited quotes below the answer
              still reflect the original text.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
