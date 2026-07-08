const BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8001";

export interface Document {
  filename: string;
  file_type: string;
  chunk_count: number;
  status: "indexed" | "processing" | "failed";
  last_indexed_at: string | null;
}

export interface IngestResponse {
  job_id: string;
  status: string;
}

export interface IngestStatus {
  status: "pending" | "processing" | "done" | "failed";
  stage: string;
  chunks_created: number;
}

export interface Source {
  filename: string;
  section: string;
  location: string;
  page: number | null;
  excerpt: string;
  quote: string;
  chunk_id: number | null;
  score: number | null;
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

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, init);
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(text || `HTTP ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export async function getDocuments(): Promise<Document[]> {
  return request<Document[]>("/documents");
}

export async function getStats(): Promise<Stats> {
  return request<Stats>("/stats");
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

export async function queryDocuments(question: string): Promise<QueryResponse> {
  return request<QueryResponse>("/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
}

export async function clearCollection(): Promise<void> {
  await fetch(`${BASE_URL}/collection`, { method: "DELETE" });
}

export async function deleteDocument(filename: string): Promise<void> {
  const res = await fetch(`${BASE_URL}/documents/${encodeURIComponent(filename)}`, {
    method: "DELETE",
  });
  if (!res.ok) throw new Error(`Delete failed (${res.status})`);
}

export async function reindexDocument(filename: string): Promise<IngestResponse> {
  return request<IngestResponse>(`/documents/${encodeURIComponent(filename)}/reindex`, {
    method: "POST",
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

export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${BASE_URL}/stats`, { signal: AbortSignal.timeout(3000) });
    return res.ok;
  } catch {
    return false;
  }
}

export async function getDocumentChunks(filename: string): Promise<DocumentChunksResponse> {
  return request<DocumentChunksResponse>(`/documents/${encodeURIComponent(filename)}/chunks`);
}

export async function getDocumentMarkdown(filename: string): Promise<MarkdownResponse> {
  return request<MarkdownResponse>(`/documents/${encodeURIComponent(filename)}/markdown`);
}

export async function getPdfPage(filename: string, page: number): Promise<PdfPageResponse> {
  return request<PdfPageResponse>(`/documents/${encodeURIComponent(filename)}/pdf/${page}`);
}

export async function getTableSheet(filename: string, sheet: string): Promise<TableSheetResponse> {
  return request<TableSheetResponse>(
    `/documents/${encodeURIComponent(filename)}/table-sheet/${encodeURIComponent(sheet)}`
  );
}
