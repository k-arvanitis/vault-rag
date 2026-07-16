import { describe, it, expect } from "vitest";
import { extractTableMarkdown, parseMarkdownTable } from "./InspectorPanel";

describe("extractTableMarkdown", () => {
  it("drops the header/schema summary lines before the first table row", () => {
    const cleanedMd =
      "[File: doc_006.xlsx | Sheet: DataAnalysis]\n" +
      "Sheet summary: 6701 rows.\n" +
      "Columns: A, B\n" +
      "Sample values — A: 1, 2 | B: x, y\n" +
      "\n" +
      "| A | B |\n" +
      "| --- | --- |\n" +
      "| 1 | x |\n";

    const result = extractTableMarkdown(cleanedMd);
    expect(result.startsWith("| A | B |")).toBe(true);
    expect(result).not.toContain("Sheet summary");
    expect(result).not.toContain("[File:");
  });

  it("returns the input unchanged when there is no table row", () => {
    const noTable = "Just some prose with no pipe table.";
    expect(extractTableMarkdown(noTable)).toBe(noTable);
  });
});

describe("parseMarkdownTable", () => {
  it("splits header and data rows, skipping the separator line", () => {
    const md = "| A | B |\n| --- | --- |\n| 1 | x |\n| 2 | y |\n";
    const { header, rows } = parseMarkdownTable(md);
    expect(header).toEqual(["A", "B"]);
    expect(rows).toEqual([
      ["1", "x"],
      ["2", "y"],
    ]);
  });

  it("returns empty header/rows when there's no table", () => {
    expect(parseMarkdownTable("no table here")).toEqual({ header: [], rows: [] });
  });
});
