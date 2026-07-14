/**
 * Product-level data contracts, independent of the current backend.
 *
 * Presentation components consume these, not raw api.ts shapes — so a future
 * swap of the vector DB, reranker, agent framework, or spreadsheet engine
 * only touches the adapter functions below, not every component.
 */
import type { Document, Source as ApiSource, QueryResponse, RejectedSource } from "./api";

export type SourceType = "pdf" | "spreadsheet" | "image" | "document";

export function inferSourceType(filename: string): SourceType {
  const ext = filename.split("/").pop()?.split(".").pop()?.toLowerCase() ?? "";
  if (ext === "pdf") return "pdf";
  if (["xlsx", "xls", "csv"].includes(ext)) return "spreadsheet";
  if (["png", "jpg", "jpeg"].includes(ext)) return "image";
  return "document";
}

export interface Citation {
  id: number;
  sourceId: string;
  sourceName: string;
  sourceType: SourceType;
  page?: number;
  sheet?: string;
  section?: string;
  quote: string;
  bbox?: [number, number, number, number];
  figureCropUrl?: string;
  rowRange?: [number, number];
  cellRange?: string;
}

/** Adapts a raw api.ts Source (a per-query citation) into the product-level Citation. */
export function toCitation(s: ApiSource, id: number): Citation {
  const basename = s.filename.split("/").pop() ?? s.filename;
  return {
    id,
    sourceId: s.filename,
    sourceName: basename,
    sourceType: inferSourceType(s.filename),
    page: s.page ?? undefined,
    sheet: s.sheet ?? undefined,
    section: s.section || undefined,
    quote: s.quote || s.excerpt || "",
    bbox: undefined, // resolved live by EvidencePanel via getPdfHighlight — not stored per-citation
    figureCropUrl: undefined, // resolved live by EvidencePanel via getPdfCrop from figure_bbox
  };
}

export type AnswerStatus = "complete" | "unsupported" | "error";

export interface Answer {
  id: string;
  text: string;
  citations: Citation[];
  status: AnswerStatus;
}

export function toAnswer(id: string, data: QueryResponse): Answer {
  const status: AnswerStatus =
    data.answer.trim().toLowerCase() === "unsupported" ? "unsupported" : "complete";
  return {
    id,
    text: data.answer,
    citations: data.sources.map((s, i) => toCitation(s, i + 1)),
    status,
  };
}

export interface AnswerTrace {
  retrieved: ApiSource[];
  rejected: RejectedSource[];
  tools: string[];
  sql: string[];
}

export function toAnswerTrace(data: QueryResponse): AnswerTrace {
  return {
    retrieved: data.sources,
    rejected: data.rejected_sources ?? [],
    tools: data.tools_used ?? [],
    sql: data.sql ?? [],
  };
}

export type SourceStatus = "processing" | "ready" | "attention" | "failed";

export interface SourceLibraryItem {
  id: string;
  name: string;
  type: SourceType;
  status: SourceStatus;
  createdAt: string | null;
  updatedAt: string | null;
}

const STATUS_MAP: Record<Document["status"], SourceStatus> = {
  indexed: "ready",
  processing: "processing",
  failed: "failed",
};

/** User-facing label for a SourceStatus — never the backend's internal term. */
export const SOURCE_STATUS_LABEL: Record<SourceStatus, string> = {
  ready: "Ready",
  processing: "Processing",
  attention: "Needs attention",
  failed: "Failed",
};

export function toSourceLibraryItem(d: Document): SourceLibraryItem {
  return {
    id: d.filename,
    name: d.filename.split("/").pop() ?? d.filename,
    type: inferSourceType(d.filename),
    status: STATUS_MAP[d.status] ?? "processing",
    createdAt: d.last_indexed_at,
    updatedAt: d.last_indexed_at,
  };
}
