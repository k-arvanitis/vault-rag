from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DOCX_PDF_DIR = REPO_ROOT / "data/interim/docx_pdf"
ALLOWED_OFFICE_EXTENSIONS = {".docx", ".doc", ".pptx", ".ppt"}


def office_to_pdf(input_path: str | Path, output_dir: str | Path = DEFAULT_DOCX_PDF_DIR) -> Path:
    if shutil.which("libreoffice") is None:
        raise RuntimeError(
            "libreoffice is not installed. Install with: sudo apt-get install -y libreoffice"
        )

    input_file = Path(input_path).expanduser().resolve()
    suffix = input_file.suffix.lower()
    if suffix not in ALLOWED_OFFICE_EXTENSIONS:
        allowed = sorted(ALLOWED_OFFICE_EXTENSIONS)
        raise ValueError(f"Expected {allowed}, got: {suffix}")
    if not input_file.exists():
        raise FileNotFoundError(f"File not found: {input_file}")

    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = out_dir / f"{input_file.stem}.pdf"
    if pdf_path.exists() and pdf_path.stat().st_mtime >= input_file.stat().st_mtime:
        return pdf_path

    cmd = [
        "libreoffice",
        "--headless",
        "--convert-to",
        "pdf",
        "--outdir",
        str(out_dir),
        str(input_file),
    ]
    subprocess.run(cmd, check=True)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not created: {pdf_path}")
    return pdf_path


def resolve_to_pdf(input_path: str | Path, output_dir: str | Path = DEFAULT_DOCX_PDF_DIR) -> Path:
    path = Path(input_path).expanduser().resolve()
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        if not path.exists():
            raise FileNotFoundError(f"Input PDF not found: {path}")
        return path
    if suffix in ALLOWED_OFFICE_EXTENSIONS:
        return office_to_pdf(path, output_dir=output_dir)
    raise ValueError(f"Unsupported input type: {suffix}. Use .pdf, .doc, .docx, .ppt, or .pptx")
