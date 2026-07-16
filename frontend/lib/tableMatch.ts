/** Shared row-matching heuristic used by both SpreadsheetEvidence
 * (EvidencePanel.tsx) and the citation-row deep-link into the Document
 * Inspector (InspectorPanel.tsx) -- no exact row/cell reference is tracked at
 * ingestion time (spreadsheets only ever get a sheet-level summary chunk),
 * so a citation's row is found by checking whether one of its cell values
 * appears verbatim in the citation's quote. Kept in one place so the two
 * views agree on which row is "the" match. */
export function normalizeCell(v: string): string {
  return v.trim().toLowerCase();
}

/** Returns the index of the first row (in `body`, header already excluded)
 * whose one of its cells appears in `quote`, or -1 if none match / quote is
 * empty. */
export function findMatchedRowIndex(body: string[][], quote: string | undefined): number {
  const quoteNorm = normalizeCell(quote || "");
  if (!quoteNorm) return -1;
  return body.findIndex((row) =>
    row.some((cell) => {
      const cellNorm = normalizeCell(cell || "");
      return cellNorm.length > 2 && quoteNorm.includes(cellNorm);
    })
  );
}
