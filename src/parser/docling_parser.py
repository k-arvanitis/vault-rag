from pathlib import Path
from typing import Tuple, Dict, Any
import os
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions, 
    TableFormerMode, 
    TesseractCliOcrOptions
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from src.parser.input_utils import resolve_to_pdf

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROCESSED_DIR = REPO_ROOT / "data/processed"
DEFAULT_DOCLING_OUTPUT_DIR = REPO_ROOT / "data/output/docling"

class DocumentProcessor:
    def __init__(
        self,
        output_dir: str = str(DEFAULT_PROCESSED_DIR),
        allow_external_plugins: bool | None = None,
        ocr_psm: int | None = 6,
    ):
        self.output_dir = Path(output_dir)
        if not self.output_dir.is_absolute():
            self.output_dir = REPO_ROOT / self.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._last_doc_stem: str | None = None

        # Configure pipeline for high-accuracy OCR and Table extraction
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True
        pipeline_options.generate_page_images = False    
        pipeline_options.generate_table_images = False   

        # Advanced table and language settings
        pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
        # psm=6 avoids Tesseract OSD on sparse pages where orientation detection often fails.
        pipeline_options.ocr_options = TesseractCliOcrOptions(lang=["eng", "ell"], psm=ocr_psm)
        if allow_external_plugins is None:
            allow_external_plugins = os.getenv("DOCLING_ALLOW_EXTERNAL_PLUGINS", "true").lower() in {
                "1", "true", "yes", "on"
            }
        pipeline_options.allow_external_plugins = allow_external_plugins

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

    def process(self, pdf_path: str) -> Tuple[Any, Dict[int, str]]:
        """Processes PDF or DOCX, saves page images, and returns (doc_object, page_map)"""
        pdf_file = resolve_to_pdf(pdf_path)
        self._last_doc_stem = pdf_file.stem
        result = self.converter.convert(str(pdf_file))
        doc = result.document
        
        # Setup specific output directory for this document
        doc_dir = self.output_dir / pdf_file.stem
        doc_dir.mkdir(parents=True, exist_ok=True)

        page_map = {}
        for page_no, page in doc.pages.items():
            if page.image:
                p_path = doc_dir / f"page_{page_no}.png"
                page.image.save(p_path)
                page_map[page_no] = str(p_path)

        return doc, page_map

    def get_markdown(self, doc_object: Any, file_name: str | None = None) -> str:
        """Extract markdown and always save it to data/output/docling/<file_name>.md."""
        if isinstance(doc_object, str):
            markdown = doc_object
        elif hasattr(doc_object, "export_to_markdown"):
            markdown = doc_object.export_to_markdown()
        else:
            raise TypeError(
                "get_markdown expected a Docling document object (with export_to_markdown) "
                f"or a markdown string, got {type(doc_object).__name__}."
            )

        resolved_name = file_name or self._last_doc_stem or "document"
        markdown_dir = DEFAULT_DOCLING_OUTPUT_DIR
        markdown_dir.mkdir(parents=True, exist_ok=True)
        markdown_path = markdown_dir / f"{resolved_name}.md"
        markdown_path.write_text(markdown, encoding="utf-8")

        return markdown
