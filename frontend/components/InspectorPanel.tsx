"use client";

import { useEffect, useState, useCallback } from "react";
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

interface Props {
  filename: string;
  onClose: () => void;
}

// remark-gfm renders markdown pipe-tables; rehype-raw renders embedded HTML
// (e.g. OCR'd <table> blocks) as real elements instead of leaking raw tags.
const MD_PLUGINS = { remarkPlugins: [remarkGfm], rehypePlugins: [rehypeRaw] };

const IS_PDF = (f: string) => f.toLowerCase().endsWith(".pdf");
const IS_TABLE = (f: string) => /\.(xlsx|xls|csv)$/i.test(f);

function Spinner() {
  return (
    <div className="flex flex-1 items-center justify-center">
      <Loader2 className="h-5 w-5 animate-spin text-ink-400" />
    </div>
  );
}

// ── PDF Inspector ──────────────────────────────────────────────────────────────

function PdfInspector({ filename }: { filename: string }) {
  const [loading, setLoading] = useState(true);
  const [pages, setPages] = useState<MarkdownPage[]>([]);
  const [hasMarkers, setHasMarkers] = useState(false);
  const [fullText, setFullText] = useState<string | null>(null);
  const [currentPage, setCurrentPage] = useState(1);
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
        setCurrentPage(1);
      })
      .finally(() => setLoading(false));
  }, [filename]);

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

  if (loading) return <Spinner />;

  const mdPage = pages.find((p) => p.page === currentPage);

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      {/* Summary */}
      {summary && (
        <div className="px-5 pt-3">
          <button
            onClick={() => setShowSummary((v) => !v)}
            className="mb-2 text-xs text-ink-500 underline underline-offset-2 hover:text-ink-700"
          >
            {showSummary ? "Hide" : "Show"} document summary
          </button>
          {showSummary && (
            <div className="prose-ui mb-3 max-h-40 overflow-y-auto rounded-md border border-ink-200 bg-surface p-3 text-xs text-ink-700">
              <ReactMarkdown {...MD_PLUGINS}>{summary.replace("## Document Summary\n\n", "")}</ReactMarkdown>
            </div>
          )}
        </div>
      )}

      {hasMarkers ? (
        <>
          {/* Page nav */}
          <div className="flex shrink-0 items-center gap-3 border-b border-ink-200 bg-surface px-5 py-2">
            <button
              onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
              disabled={currentPage <= 1}
              className="rounded p-1 text-ink-500 hover:bg-ink-100 hover:text-ink-800 disabled:opacity-30"
              aria-label="Previous page"
            >
              <ChevronLeft className="h-4 w-4" />
            </button>
            <span className="text-xs tabular-nums text-ink-500">
              Page {currentPage} / {totalPages}
            </span>
            <button
              onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
              disabled={currentPage >= totalPages}
              className="rounded p-1 text-ink-500 hover:bg-ink-100 hover:text-ink-800 disabled:opacity-30"
              aria-label="Next page"
            >
              <ChevronRight className="h-4 w-4" />
            </button>
            {mdPage?.pipeline && (
              <span className="ml-auto rounded bg-ink-100 px-2 py-0.5 font-mono text-[10px] text-ink-500">
                {mdPage.pipeline}
              </span>
            )}
          </div>

          {/* Side-by-side */}
          <div className="flex min-h-0 flex-1 overflow-hidden">
            <div className="w-1/2 overflow-y-auto border-r border-ink-200 p-3">
              {pdfMissing ? (
                <p className="mt-4 text-center text-xs text-ink-400">Original PDF not available locally.</p>
              ) : imgLoading ? (
                <div className="mt-10 flex justify-center">
                  <Loader2 className="h-5 w-5 animate-spin text-ink-400" />
                </div>
              ) : imgSrc ? (
                <img src={imgSrc} alt={`Page ${currentPage}`} className="w-full rounded border border-ink-200" />
              ) : null}
            </div>

            <div className="w-1/2 overflow-y-auto p-4">
              {mdPage ? (
                <div className="prose-ui text-xs leading-relaxed text-ink-700">
                  <ReactMarkdown {...MD_PLUGINS}>{mdPage.content}</ReactMarkdown>
                </div>
              ) : (
                <p className="text-xs text-ink-400">No content for this page.</p>
              )}
            </div>
          </div>
        </>
      ) : (
        <div className="flex-1 overflow-y-auto p-5">
          <p className="mb-3 text-[10px] text-ink-400">No page markers — showing full parsed markdown.</p>
          <div className="prose-ui text-xs leading-relaxed text-ink-700">
            <ReactMarkdown {...MD_PLUGINS}>{fullText ?? ""}</ReactMarkdown>
          </div>
        </div>
      )}
    </div>
  );
}

// ── Raw vs. cleaned sheet view ─────────────────────────────────────────────────

function SheetCompare({ filename, sheet }: { filename: string; sheet: string }) {
  const [data, setData] = useState<TableSheetResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [open, setOpen] = useState(false);

  const load = useCallback(() => {
    if (data || loading) return;
    setLoading(true);
    getTableSheet(filename, sheet)
      .then(setData)
      .catch(() => setData(null))
      .finally(() => setLoading(false));
  }, [filename, sheet, data, loading]);

  return (
    <div className="mb-2">
      <button
        onClick={() => {
          setOpen((v) => !v);
          if (!open) load();
        }}
        className="text-[10px] text-ink-500 underline underline-offset-2 hover:text-ink-700"
      >
        {open ? "Hide" : "Show"} raw vs. cleaned
      </button>

      {open && (
        <div className="mt-2 flex min-h-0 gap-3" style={{ maxHeight: "340px" }}>
          {/* Raw */}
          <div className="flex-1 overflow-auto rounded-md border border-ink-200">
            <p className="sticky top-0 border-b border-ink-200 bg-ink-100 px-2 py-1 text-[10px] text-ink-500">
              Raw (from file)
            </p>
            {loading ? (
              <div className="flex justify-center p-4">
                <Loader2 className="h-4 w-4 animate-spin text-ink-400" />
              </div>
            ) : data?.raw_rows ? (
              <table className="w-full font-mono text-[9px] text-ink-600">
                <tbody>
                  {data.raw_rows.map((row, ri) => (
                    <tr key={ri} className="border-b border-ink-100">
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
                  ))}
                </tbody>
              </table>
            ) : (
              <p className="p-3 text-[10px] text-ink-400">Not available</p>
            )}
          </div>

          {/* Cleaned */}
          <div className="flex-1 overflow-auto rounded-md border border-ink-200">
            <p className="sticky top-0 border-b border-ink-200 bg-ink-100 px-2 py-1 text-[10px] text-ink-500">
              Cleaned markdown
            </p>
            {loading ? (
              <div className="flex justify-center p-4">
                <Loader2 className="h-4 w-4 animate-spin text-ink-400" />
              </div>
            ) : data?.cleaned_md ? (
              <pre className="whitespace-pre-wrap p-2 font-mono text-[9px] leading-relaxed text-ink-600">
                {data.cleaned_md}
              </pre>
            ) : (
              <p className="p-3 text-[10px] text-ink-400">Not available</p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

// ── Excel / CSV Inspector ──────────────────────────────────────────────────────

function TableInspector({ filename }: { filename: string }) {
  const [loading, setLoading] = useState(true);
  const [summary, setSummary] = useState<string | null>(null);
  const [chunks, setChunks] = useState<Chunk[]>([]);
  const [showSummary, setShowSummary] = useState(true);

  useEffect(() => {
    setLoading(true);
    getDocumentChunks(filename)
      .then((r) => {
        setSummary(r.summary);
        setChunks(r.chunks);
      })
      .finally(() => setLoading(false));
  }, [filename]);

  if (loading) return <Spinner />;

  const sheets: Record<string, Chunk[]> = {};
  for (const c of chunks) {
    const sheet = c.metadata.sheet_name ?? "—";
    if (!sheets[sheet]) sheets[sheet] = [];
    sheets[sheet].push(c);
  }

  return (
    <div className="min-h-0 flex-1 space-y-4 overflow-y-auto p-5">
      <p className="text-xs text-ink-500">{chunks.length} data chunks indexed</p>

      {summary && (
        <div>
          <button
            onClick={() => setShowSummary((v) => !v)}
            className="mb-2 text-xs text-ink-500 underline underline-offset-2 hover:text-ink-700"
          >
            {showSummary ? "Hide" : "Show"} schema &amp; sample values — read by the SQL generator
          </button>
          {showSummary && (
            <div className="prose-ui rounded-md border border-ink-200 bg-surface p-3 text-xs text-ink-700">
              <ReactMarkdown {...MD_PLUGINS}>
                {summary.replace("## Document Summary\n\n", "").replace(/ 00:00:00/g, "")}
              </ReactMarkdown>
            </div>
          )}
        </div>
      )}

      {Object.entries(sheets).map(([sheet, sheetChunks]) => (
        <div key={sheet}>
          <p className="mb-2 text-xs font-semibold text-ink-800">Sheet: {sheet}</p>

          <SheetCompare filename={filename} sheet={sheet} />

          <p className="mb-2 mt-3 text-[10px] text-ink-400">
            Indexed retrieval chunk — routes queries to the DuckDB table
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
                <details key={i} className="group rounded-md border border-ink-200 bg-surface">
                  <summary className="flex cursor-pointer select-none items-center gap-2 px-3 py-2 text-[11px] text-ink-600 hover:bg-ink-100">
                    <ChevronRight className="h-3 w-3 shrink-0 transition-transform group-open:rotate-90" />
                    {label}
                  </summary>
                  <pre className="overflow-x-auto whitespace-pre-wrap px-3 pb-3 font-mono text-[10px] leading-relaxed text-ink-600">
                    {body}
                  </pre>
                </details>
              );
            })}
          </div>
        </div>
      ))}
    </div>
  );
}

// ── Panel shell ────────────────────────────────────────────────────────────────

export default function InspectorPanel({ filename, onClose }: Props) {
  const basename = filename.split("/").pop() ?? filename;
  const isPdf = IS_PDF(filename);
  const isTable = IS_TABLE(filename);

  return (
    <div className="fixed inset-0 z-30 flex">
      <div className="flex h-full w-full flex-col bg-ink-50">
        {/* header */}
        <div className="flex shrink-0 items-center gap-3 border-b border-ink-200 bg-surface px-5 py-3">
          <div className="min-w-0 flex-1">
            <p className="truncate text-xs font-semibold text-ink-800">{basename}</p>
            <p className="text-[10px] text-ink-400">
              {isPdf
                ? "PDF · side-by-side view"
                : isTable
                ? "Spreadsheet · chunk view"
                : "Document inspector"}
            </p>
          </div>
          <button
            onClick={onClose}
            className="rounded-md p-1.5 text-ink-500 transition-colors hover:bg-ink-100 hover:text-ink-800"
            aria-label="Close inspector"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        {isPdf ? (
          <PdfInspector filename={filename} />
        ) : isTable ? (
          <TableInspector filename={filename} />
        ) : (
          <div className="flex flex-1 items-center justify-center">
            <p className="text-xs text-ink-400">Inspector not available for this file type.</p>
          </div>
        )}
      </div>
    </div>
  );
}
