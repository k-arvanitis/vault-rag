"""Multi-Modal RAG — CLI entry point.

Usage examples:

  # Ingest a PDF
  python -m src.ingest --pdf data/input/report.pdf --collection documents_chunks

  # Ingest an Excel / CSV file
  python -m src.ingest_table_rows data/input/tables.xlsx --collection documents_chunks

  # Ask a question
  python -m src.rag_agent --query "What are the wastewater reuse standards?"
"""

if __name__ == "__main__":
    print(__doc__)
