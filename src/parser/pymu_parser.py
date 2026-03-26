import pathlib

import pymupdf4llm

from src.parser.input_utils import resolve_to_pdf

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_FOLDER = PROJECT_ROOT / "data" / "output" / "pymupdf"


def convert_pdf_to_markdown(pdf_file: str, output_folder: str | pathlib.Path | None = None):
    """
    Convert a single text-based PDF to Markdown using PyMuPDF4LLM.

    Args:
        pdf_file: Path to a PDF or DOCX file
        output_folder: Deprecated. Output is always written to
            `data/output/pymupdf` under project root.
    """
    pdf_path = resolve_to_pdf(pdf_file)
    if output_folder is not None and pathlib.Path(output_folder) != DEFAULT_OUTPUT_FOLDER:
        print(
            f"Note: ignoring output_folder='{output_folder}'. "
            f"Using '{DEFAULT_OUTPUT_FOLDER}'."
        )

    output_path = DEFAULT_OUTPUT_FOLDER
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        md_text = pymupdf4llm.to_markdown(
            str(pdf_path),
            show_progress=True,
            pages=None
        )
        output_file = output_path / f"{pdf_path.stem}.md"
        output_file.write_text(md_text, encoding="utf-8")
        print(f"✓ Converted: {pdf_path.name}")
        print(f"Output: {output_file}")
    except Exception as e:
        print(f"✗ Error processing {pdf_path.name}: {e}")
