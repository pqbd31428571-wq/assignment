import os

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch
torch._dynamo.config.disable = True

from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
)
from docling.document_converter import (
    DocumentConverter,
    ImageFormatOption,
    PdfFormatOption,
)

from app.models.document import Document
from .base_parser import BaseParser


class DoclingParser(BaseParser):
    """
    Parser for PDF, JPG, JPEG, and PNG documents using Docling.
    """

    SUPPORTED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png"}

    def __init__(self):
        # Use available CPU cores explicitly instead of Docling's
        # conservative default — speeds up every document, not just
        # images. Capped at 8 so it doesn't starve uvicorn/Streamlit
        # of threads on smaller machines.
        accelerator_options = AcceleratorOptions(
            num_threads=min(os.cpu_count() or 4, 8),
            device=AcceleratorDevice.AUTO,
        )

        pipeline_options = PdfPipelineOptions()
        pipeline_options.accelerator_options = accelerator_options

        # do_ocr=True: guarantees OCR always runs, including for
        # PDFs with embedded images that might otherwise be skipped.
        pipeline_options.do_ocr = True

        # do_table_structure=True: recovers row/column structure from
        # tables and table-like layouts (e.g. the OSI-model chart)
        # instead of flattening them into unstructured text — this is
        # what was actually causing the "Transport Layer protocols
        # not found" answer earlier.
        pipeline_options.do_table_structure = True

        # images_scale=2.0: upscales pages before OCR. Noticeably
        # improves small-text recognition on dense infographics/
        # screenshots, at the cost of somewhat slower parsing per
        # page — worth it for accuracy on image-heavy input.
        pipeline_options.images_scale = 2.0

        # Apply the SAME options to both PDFs and standalone images.
        # Without an explicit ImageFormatOption entry, .jpg/.png
        # files silently fall back to Docling's defaults instead of
        # this tuned configuration.
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
                InputFormat.IMAGE: ImageFormatOption(pipeline_options=pipeline_options),
            }
        )

    def parse(self, file_path: Path) -> Document:
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")

        try:
            result = self.converter.convert(str(file_path))
        except Exception as exc:
            raise RuntimeError(
                f"Docling failed to convert {file_path.name}: {exc}"
            ) from exc

        document = result.document
        content = document.export_to_markdown()

        return Document(
            document_id=file_path.stem,
            file_name=file_path.name,
            file_type=file_path.suffix.lower().lstrip("."),
            source_path=str(file_path),
            content=content,
        )