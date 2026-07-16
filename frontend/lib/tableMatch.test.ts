import { describe, it, expect } from "vitest";
import { findMatchedRowIndex } from "./tableMatch";

describe("findMatchedRowIndex", () => {
  const rows = [
    ["2025-04-01", "Google Ads", "402.89"],
    ["2025-04-01", "Amazon", "266.63"],
    ["2025-04-01", "Post Office", "253"],
  ];

  it("finds the row whose cell value appears in the quote", () => {
    expect(findMatchedRowIndex(rows, "NET Amount\n      266.63")).toBe(1);
  });

  it("returns -1 when the quote is empty or undefined", () => {
    expect(findMatchedRowIndex(rows, "")).toBe(-1);
    expect(findMatchedRowIndex(rows, undefined)).toBe(-1);
  });

  it("returns -1 when nothing in the quote matches any row", () => {
    expect(findMatchedRowIndex(rows, "no relation to any of these rows")).toBe(-1);
  });

  it("ignores cells too short to be a meaningful signal", () => {
    // "1" alone would coincidentally match many quotes -- cells of length <= 2
    // are excluded from matching.
    expect(findMatchedRowIndex([["1", "x"]], "some quote containing 1 somewhere")).toBe(-1);
  });
});
