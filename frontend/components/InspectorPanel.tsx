"use client";

import { useEffect, useState, useCallback } from "react";
import { X, ChevronLeft, ChevronRight, Loader2 } from "lucide-react";
import ReactMarkdown from "react-markdown";
import {
  getDocumentChunks,
  getDocumentMarkdown,
  getPdfPage,
  getTableSheet,
  type Chunk,
  type MarkdownPage,
  type TableSheetResponse,
} from "@/lib/api";
import { cn } from "@/lib/utils";

interface Props {
  filename: string;
  onClose: () => void;
}

const IS_PDF = (f: string) => f.toLowerCase().endsWith(".pdf");
const IS_TABLE = (f: string) => /\.(xlsx|xls|csv)$/i.test(f);

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
    Promise.all([
      getDocumentMarkdown(filename),
      getDocumentChunks(filename),
    ])
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

  if (loading) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <Loader2 className="h-5 w-5 animate-spin text-zinc-500" />
      </div>
    );
  }

  const mdPage = pages.find((p) => p.page === currentPage);

  return (
    <div className="flex-1 flex flex-col min-h-0">
      {/* Summary */}
      {summary && (
        <div className="px-5 pt-3">
          <button
            onClick={() => setShowSummary((v) => !v)}
            className="text-xs text-zinc-400 hover:text-zinc-300 mb-2 underline underline-offset-2"
          >
            {showSummary ? "Hide" : "Show"} document summary
          </button>
          {showSummary && (
            <div className="prose-dark text-xs text-zinc-300 bg-zinc-800/60 rounded-lg p-3 mb-3 border border-zinc-700/40 max-h-40 overflow-y-auto">
              <ReactMarkdown>{summary.replace("## Document Summary\n\n", "")}</ReactMarkdown>
            </div>
          )}
        </div>
      )}

      {hasMarkers ? (
        <>
          {/* Page nav */}
          <div className="flex items-center gap-3 px-5 py-2 border-b border-zinc-800 shrink-0">
            <button
              onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
              disabled={currentPage <= 1}
              className="p-1 rounded text-zinc-400 hover:text-zinc-200 disabled:opacity-30"
            >
              <ChevronLeft className="h-4 w-4" />
            </button>
            <span className="text-xs text-zinc-400 tabular-nums">
              Page {currentPage} / {totalPages}
            </span>
            <button
              onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
              disabled={currentPage >= totalPages}
              className="p-1 rounded text-zinc-400 hover:text-zinc-200 disabled:opacity-30"
            >
              <ChevronRight className="h-4 w-4" />
            </button>
            {mdPage?.pipeline && (
              <span className="ml-auto text-[10px] font-mono text-zinc-600 bg-zinc-800 px-2 py-0.5 rounded">
                {mdPage.pipeline}
              </span>
            )}
          </div>

          {/* Side-by-side */}
          <div className="flex-1 overflow-hidden flex min-h-0">
            {/* PDF image */}
            <div className="w-1/2 overflow-y-auto border-r border-zinc-800 p-3">
              {pdfMissing ? (
                <p className="text-xs text-zinc-600 mt-4 text-center">Original PDF not available locally.</p>
              ) : imgLoading ? (
                <div className="flex justify-center mt-10">
                  <Loader2 className="h-5 w-5 animate-spin text-zinc-600" />
                </div>
              ) : imgSrc ? (
                <img src={imgSrc} alt={`Page ${currentPage}`} className="w-full rounded shadow-lg" />
              ) : null}
            </div>

            {/* Markdown */}
            <div className="w-1/2 overflow-y-auto p-4">
              {mdPage ? (
                <div className="prose-dark text-xs text-zinc-300 leading-relaxed">
                  <ReactMarkdown>{mdPage.content}</ReactMarkdown>
                </div>
              ) : (
                <p className="text-xs text-zinc-600">No content for this page.</p>
              )}
            </div>
          </div>
        </>
      ) : (
        /* No page markers — show full markdown */
        <div className="flex-1 overflow-y-auto p-5">
          <p className="text-[10px] text-zinc-600 mb-3">No page markers — showing full parsed markdown.</p>
          <div className="prose-dark text-xs text-zinc-300 leading-relaxed">
            <ReactMarkdown>{fullText ?? ""}</ReactMarkdown>
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
        onClick={() => { setOpen((v) => !v); if (!open) load(); }}
        className="text-[10px] text-zinc-500 hover:text-zinc-300 underline underline-offset-2"
      >
        {open ? "Hide" : "Show"} raw vs. cleaned
      </button>

      {open && (
        <div className="mt-2 flex gap-3 min-h-0" style={{ maxHeight: "340px" }}>
          {/* Raw */}
          <div className="flex-1 overflow-auto border border-zinc-700/40 rounded-md">
            <p className="sticky top-0 bg-zinc-900 text-[10px] text-zinc-500 px-2 py-1 border-b border-zinc-700/40">Raw (from file)</p>
            {loading ? (
              <div className="flex justify-center p-4"><Loader2 className="h-4 w-4 animate-spin text-zinc-600" /></div>
            ) : data?.raw_rows ? (
              <table className="text-[9px] text-zinc-400 font-mono w-full">
                <tbody>
                  {data.raw_rows.map((row, ri) => (
                    <tr key={ri} className="border-b border-zinc-800/60">
                      {row.map((cell, ci) => (
                        <td key={ci} className="px-2 py-0.5 whitespace-nowrap max-w-[180px] overflow-hidden text-ellipsis" title={cell}>
                          {cell}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <p className="text-[10px] text-zinc-600 p-3">Not available</p>
            )}
          </div>

          {/* Cleaned */}
          <div className="flex-1 overflow-auto border border-zinc-700/40 rounded-md">
            <p className="sticky top-0 bg-zinc-900 text-[10px] text-zinc-500 px-2 py-1 border-b border-zinc-700/40">Cleaned markdown</p>
            {loading ? (
              <div className="flex justify-center p-4"><Loader2 className="h-4 w-4 animate-spin text-zinc-600" /></div>
            ) : data?.cleaned_md ? (
              <pre className="text-[9px] text-zinc-400 font-mono p-2 whitespace-pre-wrap leading-relaxed">
                {data.cleaned_md}
              </pre>
            ) : (
              <p className="text-[10px] text-zinc-600 p-3">Not available</p>
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

  if (loading) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <Loader2 className="h-5 w-5 animate-spin text-zinc-500" />
      </div>
    );
  }

  // Group by sheet
  const sheets: Record<string, Chunk[]> = {};
  for (const c of chunks) {
    const sheet = c.metadata.sheet_name ?? "—";
    if (!sheets[sheet]) sheets[sheet] = [];
    sheets[sheet].push(c);
  }

  return (
    <div className="flex-1 overflow-y-auto p-5 space-y-4 min-h-0">
      <p className="text-xs text-zinc-500">{chunks.length} data chunks indexed</p>

      {summary && (
        <div>
          <button
            onClick={() => setShowSummary((v) => !v)}
            className="text-xs text-zinc-400 hover:text-zinc-300 underline underline-offset-2 mb-2"
          >
            {showSummary ? "Hide" : "Show"} document summary
          </button>
          {showSummary && (
            <div className="prose-dark text-xs text-zinc-300 bg-zinc-800/60 rounded-lg p-3 border border-zinc-700/40">
              <ReactMarkdown>{summary.replace("## Document Summary\n\n", "").replace(/ 00:00:00/g, "")}</ReactMarkdown>
            </div>
          )}
        </div>
      )}

      {Object.entries(sheets).map(([sheet, sheetChunks]) => (
        <div key={sheet}>
          <p className="text-xs font-semibold text-zinc-300 mb-2">Sheet: {sheet}</p>

          <SheetCompare filename={filename} sheet={sheet} />

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
              const label = [
                `Chunk ${i + 1}`,
                chunkType,
                rowLabel,
              ]
                .filter(Boolean)
                .join(" · ");

              return (
                <details key={i} className="group rounded-md bg-zinc-800/40 border border-zinc-700/40">
                  <summary className="flex items-center gap-2 px-3 py-2 cursor-pointer text-[11px] text-zinc-400 select-none">
                    <ChevronRight className="h-3 w-3 shrink-0 group-open:rotate-90 transition-transform" />
                    {label}
                  </summary>
                  <pre className="px-3 pb-3 text-[10px] text-zinc-400 whitespace-pre-wrap font-mono leading-relaxed overflow-x-auto">
                    {c.content}
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
      {/* panel */}
      <div className="w-full flex flex-col h-full bg-zinc-950 shadow-2xl">
        {/* header */}
        <div className="flex items-center gap-3 px-5 py-3 border-b border-zinc-800 shrink-0">
          <div className="flex-1 min-w-0">
            <p className="text-xs font-semibold text-zinc-200 truncate">{basename}</p>
            <p className="text-[10px] text-zinc-600">
              {isPdf ? "PDF · side-by-side view" : isTable ? "Spreadsheet · chunk view" : "Document inspector"}
            </p>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-md text-zinc-500 hover:text-zinc-300 hover:bg-zinc-800 transition-colors"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        {isPdf ? (
          <PdfInspector filename={filename} />
        ) : isTable ? (
          <TableInspector filename={filename} />
        ) : (
          <div className="flex-1 flex items-center justify-center">
            <p className="text-xs text-zinc-600">Inspector not available for this file type.</p>
          </div>
        )}
      </div>
    </div>
  );
}
