const BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8001";

export interface Document {
  filename: string;
  /** Human-readable business title from the document's own cover page/title
   * line, when the ingest pipeline found one — null for sources with no
   * detectable title (falls back to the filename everywhere this is shown). */
  display_title: string | null;
  file_type: string;
  chunk_count: number;
  status: "indexed" | "processing" | "failed";
  last_indexed_at: string | null;
  page_count: number | null;
  sheet_count: number | null;
  row_count: number | null;
}

export interface IngestResponse {
  job_id: string;
  status: string;
}

export interface IngestStatus {
  status: "pending" | "processing" | "done" | "failed";
  stage: string;
  chunks_created: number;
  error?: string | null;
}

export interface Source {
  filename: string;
  document_id: string | null;
  document_title: string;
  section: string;
  location: string;
  page: number | null;
  sheet: string | null;
  excerpt: string;
  quote: string;
  chunk_id: number | null;
  score: number | null;
  figure_bbox: [number, number, number, number] | null;
  /** Bbox (PDF points) of the first OCR'd element in this chunk, when the
   * source page was scanned and parsed via the unstructured/CPU OCR path
   * (which computes layout coordinates) rather than LightOn OCR (which
   * doesn't). Used as a highlight/crop fallback for scanned pages, where
   * `/pdf/{page}/highlight`'s real-text-layer search always comes up empty. */
  ocr_bbox: [number, number, number, number] | null;
  /** True when this is a SQL aggregate answer (SUM/COUNT/AVG/...) -- the
   * quote is the WHERE clause's filter literal (e.g. "MATERIALS"), not the
   * computed result, so the highlighted row is one real matching example,
   * not the row containing the answer's number (no single row has it --
   * that's the point of an aggregate). See query_excel's citation-building
   * (src/tools/excel.py) for where this is set. */
  is_aggregate?: boolean;
}

export interface InspectTarget {
  filename: string;
  page?: number;
  sheet?: string;
  /** The citation's quote text -- when set, the inspector re-runs the same
   * best-effort "quote contains this row's cell value" match SpreadsheetEvidence
   * uses, to highlight the same row there. */
  quote?: string;
  /** See Source.is_aggregate -- carried through so the full inspector shows
   * the same "computed total, not a single row's value" note. */
  isAggregate?: boolean;
}

/** Derives an inspector jump target from a citation. */
export function sourceInspectTarget(s: Source): InspectTarget {
  return {
    filename: s.filename,
    page: s.page ?? undefined,
    sheet: s.sheet ?? undefined,
    quote: s.quote || undefined,
    isAggregate: s.is_aggregate,
  };
}

export interface RejectedSource {
  filename: string;
  score: number | null;
}

export interface QueryResponse {
  answer: string;
  sources: Source[];
  rejected_sources?: RejectedSource[];
  sql?: string[];
  tools_used?: string[];
}

export interface EvalSummary {
  question_count: number;
  [key: string]: unknown;
}

export type FeedbackReason =
  | "wrong_source"
  | "hallucinated"
  | "should_have_refused"
  | "missing_document"
  | "other";

export interface Feedback {
  id: string;
  question: string;
  answer: string;
  rating: "up" | "down";
  reason: FeedbackReason | null;
  sources: Source[];
  status: "open" | "resolved";
  action: string | null;
  note: string | null;
  created_at: string;
}

export interface Stats {
  total_docs: number;
  total_chunks: number;
}

export interface UsageEntry {
  timestamp: string;
  question: string;
  model: string | null;
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  cost_usd: number;
  latency_ms: number | null;
}

export interface UsageDaily {
  date: string;
  questions: number;
  total_tokens: number;
  cost_usd: number;
  avg_latency_ms: number | null;
}

export interface UsageStats {
  recent: UsageEntry[];
  daily: UsageDaily[];
  total_questions: number;
  total_tokens: number;
  total_cost_usd: number;
}

export interface ChunkMeta {
  chunk_type?: string;
  chunk_index?: number;
  section?: string;
  title?: string;
  subsection?: string;
  sheet_name?: string;
  row_ref?: number;
  num_rows?: number;
  part?: number;
  source_file?: string;
  file_name?: string;
  [key: string]: unknown;
}

export interface Chunk {
  content: string;
  metadata: ChunkMeta;
}

export interface DocumentChunksResponse {
  summary: string | null;
  chunks: Chunk[];
}

export interface MarkdownPage {
  page: number;
  content: string;
  pipeline: string;
}

export interface MarkdownResponse {
  has_page_markers: boolean;
  full_text: string | null;
  pages: MarkdownPage[];
}

export interface PdfPageResponse {
  image_b64: string;
  page: number;
  total_pages: number;
}

export interface TableSheetResponse {
  sheet: string;
  raw_rows: string[][] | null;
  cleaned_md: string | null;
}

async function request<T>(path: string, init?: RequestInit, timeoutMs?: number): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    ...init,
    // Required for the admin session cookie (ACCESS_MODE=admin_viewer) to
    // ride along on cross-origin requests -- harmless no-op in open mode.
    credentials: "include",
    signal: timeoutMs ? AbortSignal.timeout(timeoutMs) : init?.signal,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(text || `HTTP ${res.status}`);
  }
  return res.json() as Promise<T>;
}

// Inspector metadata calls should fail fast and visibly rather than leave a
// spinner stuck forever if the backend (or Qdrant) stalls -- unlike /ingest
// or /query, which can legitimately take minutes, these are quick lookups.
const INSPECTOR_TIMEOUT_MS = 20_000;

export async function getDocuments(): Promise<Document[]> {
  return request<Document[]>("/documents");
}

export async function getStats(): Promise<Stats> {
  return request<Stats>("/stats");
}

export async function getUsage(): Promise<UsageStats> {
  return request<UsageStats>("/usage");
}

/** Non-admin-safe corpus check -- whether there's anything to ask about, with
 * no document count or filenames (see api.py's /documents/exists). Used by
 * ChatPanel to enable the chat input for non-admin viewers, who can't call
 * getDocuments()/getStats() (admin-only, see require_admin on those routes). */
export async function documentsExist(): Promise<boolean> {
  const res = await request<{ has_documents: boolean }>("/documents/exists");
  return res.has_documents;
}

export async function ingestFile(file: File, pipeline: "auto" | "ocr" | "text" = "auto"): Promise<IngestResponse> {
  const form = new FormData();
  form.append("file", file);
  form.append("pipeline", pipeline);
  return request<IngestResponse>("/ingest", { method: "POST", body: form });
}

export async function getIngestStatus(jobId: string): Promise<IngestStatus> {
  return request<IngestStatus>(`/ingest/status/${jobId}`);
}

export type QueryHistoryTurn = { question: string; answer: string };

export async function queryDocuments(
  question: string,
  docId?: string | string[] | null,
  signal?: AbortSignal,
  history?: QueryHistoryTurn[]
): Promise<QueryResponse> {
  const doc_id = Array.isArray(docId) ? (docId.length ? docId : null) : docId ?? null;
  return request<QueryResponse>("/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, doc_id, history: history?.length ? history : null }),
    signal,
  });
}

/** SSE version of queryDocuments — calls onToken as each raw token arrives
 * (perceived-latency streaming), then resolves with the final, cleaned
 * QueryResponse once the "done" event lands. The streamed tokens are the
 * model's raw, uncleaned output (see stream_answer's docstring in
 * answer_pipeline.py) — callers should replace any partial text they built
 * from onToken with the resolved response's `answer`, not keep the raw
 * concatenation. Comparison/multi-part questions arrive as a single onToken
 * call with the whole answer, not live per-token — see the same docstring. */
export async function streamQueryDocuments(
  question: string,
  onToken: (token: string) => void,
  docId?: string | string[] | null,
  signal?: AbortSignal,
  history?: QueryHistoryTurn[]
): Promise<QueryResponse> {
  const doc_id = Array.isArray(docId) ? (docId.length ? docId : null) : docId ?? null;
  const res = await fetch(`${BASE_URL}/query/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, doc_id, history: history?.length ? history : null }),
    credentials: "include",
    signal,
  });
  if (!res.ok || !res.body) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(text || `HTTP ${res.status}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let final: QueryResponse | null = null;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n\n");
    buffer = lines.pop() ?? "";
    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      const event = JSON.parse(line.slice("data: ".length));
      if (event.done) {
        if (event.error) throw new Error(event.error);
        final = {
          answer: event.answer,
          sources: event.sources ?? [],
          rejected_sources: event.rejected_sources ?? [],
          sql: event.sql ?? [],
          tools_used: event.tools_used ?? [],
        };
      } else if (typeof event.token === "string") {
        onToken(event.token);
      }
    }
  }
  if (!final) throw new Error("Stream ended without a final response");
  return final;
}

export async function clearCollection(): Promise<void> {
  await fetch(`${BASE_URL}/collection`, { method: "DELETE", credentials: "include" });
}

export async function deleteDocument(filename: string): Promise<void> {
  const res = await fetch(`${BASE_URL}/documents/${encodeURIComponent(filename)}`, {
    method: "DELETE",
    credentials: "include",
  });
  if (!res.ok) throw new Error(`Delete failed (${res.status})`);
}

export async function reindexDocument(
  filename: string,
  pipeline: "auto" | "ocr" | "text" = "auto"
): Promise<IngestResponse> {
  const form = new FormData();
  form.append("pipeline", pipeline);
  return request<IngestResponse>(`/documents/${encodeURIComponent(filename)}/reindex`, {
    method: "POST",
    body: form,
  });
}

/** Set (title truthy) or clear (title null/blank) a source's admin display
 * title — see api.py's PATCH /documents/{filename}/title. Clearing reverts
 * to the extracted title or filename. */
export async function setDocumentTitle(filename: string, title: string | null): Promise<void> {
  await request(`/documents/${encodeURIComponent(filename)}/title`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title }),
  });
}

export async function getEvalSummary(): Promise<EvalSummary> {
  return request<EvalSummary>("/eval/summary");
}

export interface EvalJobStatus {
  status: "pending" | "running" | "done" | "failed";
  summary?: EvalSummary;
  error?: string;
}

export async function runEval(): Promise<IngestResponse> {
  return request<IngestResponse>("/eval/run", { method: "POST" });
}

export async function getEvalStatus(jobId: string): Promise<EvalJobStatus> {
  return request<EvalJobStatus>(`/eval/status/${jobId}`);
}

export async function submitFeedback(
  question: string,
  answer: string,
  rating: "up" | "down",
  reason: FeedbackReason | null,
  sources: Source[]
): Promise<Feedback> {
  return request<Feedback>("/feedback", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, answer, rating, reason, sources }),
  });
}

export async function getFeedback(): Promise<Feedback[]> {
  return request<Feedback[]>("/feedback");
}

export async function resolveFeedback(id: string, action: string, note?: string): Promise<Feedback> {
  return request<Feedback>(`/feedback/${id}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ action, note: note ?? null }),
  });
}

export interface ConversationMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
}

export interface ConversationSummary {
  id: string;
  title: string;
  message_count: number;
  created_at: string;
  updated_at: string;
}

export interface Conversation extends ConversationSummary {
  messages: ConversationMessage[];
}

export async function saveConversation(
  id: string | null,
  messages: ConversationMessage[]
): Promise<Conversation> {
  return request<Conversation>("/conversations", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id, messages }),
  });
}

export async function listConversations(): Promise<ConversationSummary[]> {
  return request<ConversationSummary[]>("/conversations");
}

export async function getConversation(id: string): Promise<Conversation> {
  return request<Conversation>(`/conversations/${id}`);
}

export async function deleteConversation(id: string): Promise<void> {
  await fetch(`${BASE_URL}/conversations/${id}`, { method: "DELETE" });
}

export async function checkHealth(): Promise<boolean> {
  try {
    // /health is a plain liveness probe, open to everyone -- /stats is now
    // admin-only (see api.py's require_admin), so using it here made every
    // non-admin session see a permanent false "offline" banner.
    const res = await fetch(`${BASE_URL}/health`, { signal: AbortSignal.timeout(3000) });
    return res.ok;
  } catch {
    return false;
  }
}

export async function getDocumentChunks(filename: string): Promise<DocumentChunksResponse> {
  return request<DocumentChunksResponse>(
    `/documents/${encodeURIComponent(filename)}/chunks`,
    undefined,
    INSPECTOR_TIMEOUT_MS
  );
}

export async function getDocumentMarkdown(filename: string): Promise<MarkdownResponse> {
  return request<MarkdownResponse>(
    `/documents/${encodeURIComponent(filename)}/markdown`,
    undefined,
    INSPECTOR_TIMEOUT_MS
  );
}

export async function getPdfPage(filename: string, page: number): Promise<PdfPageResponse> {
  return request<PdfPageResponse>(
    `/documents/${encodeURIComponent(filename)}/pdf/${page}`,
    undefined,
    INSPECTOR_TIMEOUT_MS
  );
}

export interface PdfHighlightResponse {
  bbox: [number, number, number, number] | null;
  coordinate_system: string | null;
}

/** Locates a cited quote on a born-digital PDF page, computed live via exact
 * text search against the PDF — no ingestion-time storage. */
export async function getPdfHighlight(
  filename: string,
  page: number,
  quote: string
): Promise<PdfHighlightResponse> {
  return request<PdfHighlightResponse>(
    `/documents/${encodeURIComponent(filename)}/pdf/${page}/highlight?quote=${encodeURIComponent(quote)}`,
    undefined,
    INSPECTOR_TIMEOUT_MS
  );
}

export interface PdfCropResponse {
  image_b64: string;
}

/** Crops a figure/chart region out of the source PDF page, so the evidence
 * panel can show the real image instead of only its VLM text description. */
export async function getPdfCrop(
  filename: string,
  page: number,
  bbox: [number, number, number, number]
): Promise<PdfCropResponse> {
  return request<PdfCropResponse>(
    `/documents/${encodeURIComponent(filename)}/pdf/${page}/crop?bbox=${bbox.join(",")}`,
    undefined,
    INSPECTOR_TIMEOUT_MS
  );
}

export async function getTableSheet(
  filename: string,
  sheet: string,
  quote?: string
): Promise<TableSheetResponse> {
  // quote lets the backend search the WHOLE sheet for the citation's real
  // row and return a window around it, instead of always "the first 60
  // rows" -- without it, a citation whose row falls past that cap can never
  // be found or highlighted no matter how good the client-side matching is.
  const query = quote ? `?quote=${encodeURIComponent(quote)}` : "";
  return request<TableSheetResponse>(
    `/documents/${encodeURIComponent(filename)}/table-sheet/${encodeURIComponent(sheet)}${query}`,
    undefined,
    INSPECTOR_TIMEOUT_MS
  );
}

export interface DriveStatus {
  configured: boolean;
  folder_id: string | null;
  last_synced_at: string | null;
  file_count: number;
}

export interface DriveFile {
  file_id: string;
  name: string;
  modified_time: string;
  local_path: string;
  indexed_at: string;
}

export interface DriveSyncResult {
  synced: { name: string; status: string; error?: string }[];
  removed: string[];
  last_synced_at: string;
}

export async function getDriveStatus(): Promise<DriveStatus> {
  return request<DriveStatus>("/connectors/google-drive/status");
}

export async function getDriveFiles(): Promise<DriveFile[]> {
  return request<DriveFile[]>("/connectors/google-drive/files");
}

export async function configureDrive(folderId: string): Promise<DriveStatus> {
  return request<DriveStatus>("/connectors/google-drive/configure", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ folder_id: folderId }),
  });
}

export async function syncDrive(removeDeleted = false): Promise<DriveSyncResult> {
  return request<DriveSyncResult>("/connectors/google-drive/sync", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ remove_deleted: removeDeleted }),
  });
}

export interface LLMCredentialsStatus {
  provider: string | null;
  model: string | null;
  key_set: boolean;
  key_last4: string | null;
  providers: string[];
}

export async function getLLMCredentials(): Promise<LLMCredentialsStatus> {
  return request<LLMCredentialsStatus>("/admin/llm-credentials");
}

/** apiKey omitted or blank keeps the existing stored key -- the GET endpoint
 * only ever returns a mask, so the form can't round-trip the real value. */
export async function setLLMCredentials(
  provider: string,
  apiKey: string | null,
  model: string | null
): Promise<{ status: string }> {
  return request<{ status: string }>("/admin/llm-credentials", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ provider, api_key: apiKey || null, model: model || null }),
  });
}

export async function deleteLLMCredentials(): Promise<{ status: string }> {
  return request<{ status: string }>("/admin/llm-credentials", { method: "DELETE" });
}

export interface AdminSession {
  access_mode: "open" | "admin_viewer";
  is_admin: boolean;
}

export async function getAdminSession(): Promise<AdminSession> {
  return request<AdminSession>("/admin/session");
}

export async function adminLogin(password: string): Promise<{ status: string }> {
  return request<{ status: string }>("/admin/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ password }),
  });
}

export async function adminLogout(): Promise<{ status: string }> {
  return request<{ status: string }>("/admin/logout", { method: "POST" });
}
