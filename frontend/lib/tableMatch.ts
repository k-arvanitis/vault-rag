/** Shared row-matching heuristic used by both SpreadsheetEvidence
 * (EvidencePanel.tsx) and the citation-row deep-link into the Document
 * Inspector (InspectorPanel.tsx) -- no exact row/cell reference is tracked at
 * ingestion time (spreadsheets only ever get a sheet-level summary chunk),
 * so a citation's row is found by checking whether its cell values appear
 * verbatim in the citation's quote. Mirrored server-side in api.py's
 * _find_matched_row_index (that copy searches the whole sheet, not just the
 * page this component was given) -- kept in sync deliberately so all three
 * call sites agree on which row is "the" match. */
export function normalizeCell(v: string): string {
  return v.trim().toLowerCase();
}

/** Returns the index of the row (in `body`, header already excluded) with
 * the MOST cells (len > 2) that appear verbatim in `quote`, or -1 if none
 * match / quote is empty. Ties broken by earliest row index.
 *
 * Scoring by match count rather than stopping at the first single-cell
 * match matters once a common value (e.g. a category name repeated across
 * many rows) could match several rows -- the quote also carries the row's
 * other distinguishing values (an exact amount, a date), which only the
 * true row matches on top of the common one; counting cells picks it. */
export function findMatchedRowIndex(body: string[][], quote: string | undefined): number {
  const quoteNorm = normalizeCell(quote || "");
  if (!quoteNorm) return -1;
  let bestIdx = -1;
  let bestCount = 0;
  body.forEach((row, i) => {
    const count = row.reduce((acc, cell) => {
      const cellNorm = normalizeCell(cell || "");
      return cellNorm.length > 2 && quoteNorm.includes(cellNorm) ? acc + 1 : acc;
    }, 0);
    if (count > bestCount) {
      bestCount = count;
      bestIdx = i;
    }
  });
  return bestIdx;
}
