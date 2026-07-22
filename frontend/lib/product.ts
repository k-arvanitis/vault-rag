/**
 * Product-level data contracts, independent of the current backend.
 *
 * Presentation components consume these, not raw api.ts shapes — so a future
 * swap of the vector DB, reranker, agent framework, or spreadsheet engine
 * only touches the adapter functions below, not every component.
 */
import type { Document, Source as ApiSource, QueryResponse, RejectedSource } from "./api";
import type { TrackedJob } from "./jobTracker";

// Demo-only display names for sources whose backend display_title is null —
// either no extractable cover-page title (PDF) or no title concept at all
// (spreadsheets). Keyed by basename; purely cosmetic for the portfolio demo,
// not derived from document content the way display_title is.
const DEMO_TITLE_OVERRIDES: Record<string, string> = {
  "doc_002_services_contract_terms.pdf": "Services Contract Terms",
  "doc_003_fed_annual_report_2024.pdf": "Federal Reserve Annual Report 2024",
  "doc_004_foia_invoices_packet.pdf": "FOIA Invoices Packet",
  "doc_006_purchase_card_transactions_q1_2025_26.xlsx": "Purchase Card Transactions Q1 2025/26",
  "doc_007_published_spend_report_april_25.csv": "Published Spend Report — April 2025",
  "doc_011_spain_open_data_maturity_questionnaire_2025.xlsx": "Spain Open Data Maturity Questionnaire 2025",
  "doc_012_osse_afe_reporting_workbook_fy2025.xlsx": "OSSE AFE Reporting Workbook FY2025",
  "doc_013_osse_afe_grant_budget_finance_tracker_fy2026.xlsx": "OSSE AFE Grant Budget Finance Tracker FY2026",
  "doc_014_bristol_supplier_spend_april_2024.csv": "Bristol Supplier Spend — April 2024",
};

/** Single place every screen resolves a document's display name: real
 * extracted display_title (title-cased) > demo override > raw filename. */
export function resolveDisplayTitle(d: Pick<Document, "filename" | "display_title">): string {
  if (d.display_title) return toDisplayTitle(d.display_title);
  const basename = d.filename.split("/").pop() ?? d.filename;
  return DEMO_TITLE_OVERRIDES[basename] ?? basename;
}

export type SourceType = "pdf" | "spreadsheet" | "image" | "document";

// Words that stay lowercase in title case (unless first/last) — small,
// deliberately incomplete list covering common English function words in
// this corpus's document titles; good enough for a display heuristic, not a
// grammar engine.
const TITLE_CASE_MINOR_WORDS = new Set([
  "a", "an", "and", "as", "at", "but", "by", "for", "in", "nor", "of", "on",
  "or", "so", "the", "to", "up", "yet",
]);

/** Extracted document titles are often verbatim cover-page text, which is
 * frequently ALL CAPS (e.g. "POLICY FOR THE PROCUREMENT OF GOODS AND
 * SERVICES (PGS)") — display that in normal title case instead. Left
 * untouched when the title isn't all-uppercase (mixed case is assumed
 * already well-formed, e.g. "iPhone" or an acronym-heavy title). */
export function toDisplayTitle(title: string): string {
  if (title !== title.toUpperCase() || title === title.toLowerCase()) return title;
  const words = title.toLowerCase().split(" ");
  return words
    .map((w, i) => {
      if (i > 0 && i < words.length - 1 && TITLE_CASE_MINOR_WORDS.has(w)) return w;
      // Preserve parenthesised acronyms like "(PGS)" as-is rather than
      // title-casing to "(Pgs)".
      const paren = w.match(/^\((\w+)\)$/);
      if (paren && paren[1].length <= 5) return `(${paren[1].toUpperCase()})`;
      return w.charAt(0).toUpperCase() + w.slice(1);
    })
    .join(" ");
}

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

/** [N] marker numbers an answer's text actually cites. The backend renumbers
 * citations sequentially (1, 2, 3...) in order of first appearance and
 * reorders `sources` to match, so the cited sources are always exactly the
 * first N entries of the array -- see citedOnlySources. */
export function citedIndices(content: string): Set<number> {
  const indices = new Set<number>();
  const re = /\[(\d+)\]/g;
  let match: RegExpExecArray | null;
  while ((match = re.exec(content))) indices.add(Number(match[1]));
  return indices;
}

/** Sources actually cited in `content`, not the full retrieved-candidate
 * list -- e.g. a cross-document answer citing 2 of 8 retrieved sources
 * should show "1/2" in the Evidence panel, not "1/8". Falls back to the
 * full list when the answer cites nothing inline (e.g. a table/SQL answer). */
export function citedOnlySources(content: string, sources: ApiSource[]): ApiSource[] {
  const cited = citedIndices(content);
  return cited.size > 0 ? sources.slice(0, cited.size) : sources;
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

export type SourceStatus = "ready" | "processing" | "updating" | "attention" | "failed";

export interface SourceLibraryItem {
  id: string;
  name: string;
  /** Filename, always present — secondary/tooltip text when `name` is a
   * real display_title, primary text when there isn't one (see
   * toSourceLibraryItem). */
  filename: string;
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

/** User-facing label for a SourceStatus — never the backend's internal term.
 * The only 5 words allowed anywhere a source's status is shown; every screen
 * (sidebar, Sources table, reprocess dialog) reads this same map, not its own
 * copy — see toSourceLibraryItem below for the one place status is decided. */
export const SOURCE_STATUS_LABEL: Record<SourceStatus, string> = {
  ready: "Ready",
  processing: "Processing",
  updating: "Updating",
  attention: "Needs attention",
  failed: "Failed",
};

/** Single place that decides a document's displayed status. `job` is this
 * file's in-flight ingest/reindex job, if any (see lib/jobTracker.ts) — the
 * backend's /documents list alone can't distinguish a first-time ingest from
 * a reprocess, or a reprocess-failed-but-old-version-still-searchable state
 * from a total failure, so that signal has to be merged in here. */
export function toSourceLibraryItem(d: Document, job?: TrackedJob): SourceLibraryItem {
  let status: SourceStatus;
  if (job?.status === "processing") {
    status = job.kind === "reindex" ? "updating" : "processing";
  } else if (job?.status === "failed") {
    // A failed reprocess on a document that was already indexed leaves the
    // prior version searchable — that's degraded, not down, hence "attention"
    // rather than "failed".
    status = d.status === "indexed" ? "attention" : "failed";
  } else {
    status = STATUS_MAP[d.status] ?? "processing";
  }
  const basename = d.filename.split("/").pop() ?? d.filename;
  return {
    id: d.filename,
    name: resolveDisplayTitle(d),
    filename: basename,
    type: inferSourceType(d.filename),
    status,
    createdAt: d.last_indexed_at,
    updatedAt: d.last_indexed_at,
  };
}

/** Badge color tier — shared so every screen renders the same status the same way. */
export const SOURCE_STATUS_BADGE_VARIANT: Record<
  SourceStatus,
  "default" | "secondary" | "destructive" | "outline"
> = {
  ready: "default",
  processing: "secondary",
  updating: "secondary",
  attention: "destructive",
  failed: "destructive",
};
