import logging
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
    RapidOcrOptions,
)
from docling.document_converter import (
    DocumentConverter,
    ImageFormatOption,
    PdfFormatOption,
)

from app.models.document import Document
from .base_parser import BaseParser

logger = logging.getLogger(__name__)


class DoclingParser(BaseParser):
    """
    Parser for PDF, JPG, JPEG, and PNG documents using Docling.
    """

    SUPPORTED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png"}

    def __init__(self):
        logger.info("Initializing DoclingParser (loading OCR/layout models)...")

        accelerator_options = AcceleratorOptions(
            num_threads=min(os.cpu_count() or 4, 8),
            device=AcceleratorDevice.AUTO,
        )

        ocr_options = RapidOcrOptions(
            force_full_page_ocr=True,
        )

        pdf_pipeline_options = PdfPipelineOptions()
        pdf_pipeline_options.accelerator_options = accelerator_options
        pdf_pipeline_options.do_ocr = True
        pdf_pipeline_options.do_table_structure = True
        pdf_pipeline_options.images_scale = 2.0
        pdf_pipeline_options.ocr_options = ocr_options

        image_pipeline_options = PdfPipelineOptions()
        image_pipeline_options.accelerator_options = accelerator_options
        image_pipeline_options.do_ocr = True
        image_pipeline_options.do_table_structure = True
        image_pipeline_options.ocr_options = ocr_options

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_pipeline_options),
                InputFormat.IMAGE: ImageFormatOption(pipeline_options=image_pipeline_options),
            }
        )

        logger.info("DoclingParser ready.")

    def parse(self, file_path: Path) -> Document:
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")

        try:
            result = self.converter.convert(str(file_path))
        except Exception as exc:
            logger.exception("Docling failed to convert %s", file_path.name)
            raise RuntimeError(
                f"Docling failed to convert {file_path.name}: {exc}"
            ) from exc

        document = result.document
        content = document.export_to_markdown()

        logger.info(
            "Parsed %s -> %d character(s) extracted",
            file_path.name, len(content),
        )

        return Document(
            document_id=file_path.stem,
            file_name=file_path.name,
            file_type=file_path.suffix.lower().lstrip("."),
            source_path=str(file_path),
            content=content,
        )