import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { AnswerContent } from "./MessageList";
import type { Source } from "@/lib/api";

function makeSource(overrides: Partial<Source> = {}): Source {
  return {
    filename: "doc_001.pdf",
    document_id: "doc_001",
    document_title: "doc_001.pdf",
    section: "",
    location: "chunk 2",
    page: 1,
    sheet: null,
    excerpt: "The relevant excerpt.",
    quote: "The relevant excerpt.",
    chunk_id: 123,
    score: 0.9,
    figure_bbox: null,
    ocr_bbox: null,
    ...overrides,
  };
}

describe("AnswerContent", () => {
  it("renders a resolved [N] marker as a clickable citation chip", () => {
    const sources = [makeSource()];
    render(<AnswerContent content="The answer is X [1]." sources={sources} />);

    const chip = screen.getByRole("button", { name: /1/ });
    expect(chip).toBeInTheDocument();
    // The literal bracketed text is not left in the DOM as plain text.
    expect(screen.queryByText("[1]", { exact: false })).not.toBeInTheDocument();
  });

  it("selects the citation's own source when clicked", () => {
    const sources = [makeSource({ filename: "doc_001.pdf" }), makeSource({ filename: "doc_002.pdf" })];
    const onSelectEvidence = vi.fn();
    render(
      <AnswerContent
        content="First [1] and second [2]."
        sources={sources}
        onSelectEvidence={onSelectEvidence}
      />
    );

    screen.getByRole("button", { name: "2" }).click();
    expect(onSelectEvidence).toHaveBeenCalledWith(
      expect.objectContaining({ filename: "doc_002.pdf" }),
      sources
    );
  });

  it("falls back to plain markdown text when there are no sources to resolve against", () => {
    render(<AnswerContent content="The answer is X [1]." sources={[]} />);
    expect(screen.getByText("The answer is X [1].")).toBeInTheDocument();
    expect(screen.queryByRole("button")).not.toBeInTheDocument();
  });

  it("leaves an out-of-range marker as plain text instead of crashing", () => {
    // The backend only ever emits a resolvable [N] (see build_citation_map in
    // answer_pipeline.py) -- but the frontend must degrade gracefully rather
    // than throw if sources and the marker ever disagree.
    render(<AnswerContent content="Filed under [9]." sources={[makeSource()]} />);
    expect(screen.getByText("[9]", { exact: false })).toBeInTheDocument();
  });

  it("keeps a citation chip inline when its marker lands inside a list item", () => {
    // Reproduced live: a comparison answer formatted as "- **doc_001** ...
    // [1]" split the raw markdown string at the marker, so only the
    // fragment before it ("- **doc_001** ...") got reparsed -- markdown
    // read that alone as a complete <ul><li>, a block element, which forced
    // the chip that followed onto its own line. AnswerContent now parses
    // the whole message in one pass, so the list and the chip stay in the
    // same <li>.
    const sources = [makeSource()];
    render(
      <AnswerContent
        content={"- **doc_001** – requires legal review [1]. More text after."}
        sources={sources}
      />
    );

    const chip = screen.getByRole("button", { name: "1" });
    const item = chip.closest("li");
    expect(item).not.toBeNull();
    expect(item?.textContent).toContain("More text after.");
  });
});
