import { describe, it, expect, vi, afterEach } from "vitest";
import { streamQueryDocuments } from "./api";

function sseResponse(chunks: string[]): Response {
  const stream = new ReadableStream({
    start(controller) {
      const encoder = new TextEncoder();
      for (const chunk of chunks) {
        controller.enqueue(encoder.encode(chunk));
      }
      controller.close();
    },
  });
  return new Response(stream, { status: 200 });
}

describe("streamQueryDocuments", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("calls onToken for each token event and resolves with the final done event", async () => {
    const sse =
      'data: {"token": "Hello "}\n\n' +
      'data: {"token": "world."}\n\n' +
      'data: {"done": true, "answer": "Hello world.", "sources": [], "sql": [], ' +
      '"rejected_sources": [], "tools_used": ["search_documents"]}\n\n';
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(sseResponse([sse])));

    const tokens: string[] = [];
    const result = await streamQueryDocuments("What is X?", (t) => tokens.push(t));

    expect(tokens).toEqual(["Hello ", "world."]);
    expect(result.answer).toBe("Hello world.");
    expect(result.tools_used).toEqual(["search_documents"]);
  });

  it("handles an SSE event split across multiple stream chunks", () => {
    // Guards the buffering logic in streamQueryDocuments: a "data: ...\n\n"
    // frame can arrive split across separate stream reads (real network
    // behavior), and a naive per-chunk JSON.parse would throw on the
    // truncated first half.
    const half1 = 'data: {"token": "Hel';
    const half2 = 'lo"}\n\ndata: {"done": true, "answer": "Hello", "sources": [], ' +
      '"sql": [], "rejected_sources": [], "tools_used": []}\n\n';
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(sseResponse([half1, half2])));

    const tokens: string[] = [];
    return streamQueryDocuments("What is X?", (t) => tokens.push(t)).then((result) => {
      expect(tokens).toEqual(["Hello"]);
      expect(result.answer).toBe("Hello");
    });
  });

  it("throws when the done event carries an error", async () => {
    const sse = 'data: {"done": true, "error": "No connected db."}\n\n';
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(sseResponse([sse])));

    await expect(streamQueryDocuments("What is X?", () => {})).rejects.toThrow(
      "No connected db."
    );
  });

  it("throws on a non-ok HTTP response", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(new Response("Internal error", { status: 500 }))
    );

    await expect(streamQueryDocuments("What is X?", () => {})).rejects.toThrow();
  });
});
