/**
 * Minimal shared store for in-flight ingest/reindex jobs, keyed by filename.
 *
 * Single source of truth for "is this document mid-processing right now, and
 * was it a first-time ingest or a reprocess" — the signal the status badge
 * (Ready/Processing/Updating/Needs attention/Failed) needs and that the
 * backend's /documents list alone can't provide (it only reports whether a
 * document is currently indexed, not whether a job is running against it).
 */
import { useSyncExternalStore } from "react";

export type JobKind = "ingest" | "reindex";
export type JobStatus = "processing" | "failed";

export interface TrackedJob {
  kind: JobKind;
  status: JobStatus;
}

const jobs = new Map<string, TrackedJob>();
const listeners = new Set<() => void>();

function notify() {
  listeners.forEach((l) => l());
}

function subscribe(listener: () => void) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

function getSnapshot() {
  return jobs;
}

/** Poll a job to completion, updating the shared store as it goes. Clears the
 * entry a few seconds after success so the badge settles back to Ready once
 * the caller's own refresh() has had time to pick up the new document state. */
export function trackJob(
  filename: string,
  kind: JobKind,
  jobId: string,
  poll: (jobId: string) => Promise<{ status: string }>,
  onSettled?: () => void
) {
  jobs.set(filename, { kind, status: "processing" });
  notify();

  const interval = setInterval(async () => {
    try {
      const result = await poll(jobId);
      if (result.status === "done") {
        clearInterval(interval);
        onSettled?.();
        setTimeout(() => {
          jobs.delete(filename);
          notify();
        }, 2500);
      } else if (result.status === "failed") {
        clearInterval(interval);
        jobs.set(filename, { kind, status: "failed" });
        notify();
        onSettled?.();
      }
    } catch {
      clearInterval(interval);
      jobs.set(filename, { kind, status: "failed" });
      notify();
      onSettled?.();
    }
  }, 2000);
}

/** Re-renders the calling component whenever any job's state changes. */
export function useJobTracker(): Map<string, TrackedJob> {
  return useSyncExternalStore(subscribe, getSnapshot, getSnapshot);
}
