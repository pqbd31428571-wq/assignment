import os

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch
torch._dynamo.config.disable = True

from pathlib import Path
from docling.document_converter import DocumentConverter
from app.models.document import Document
from .base_parser import BaseParser


class DoclingParser(BaseParser):
    """
    Parser for PDF, JPG, JPEG, and PNG documents using Docling.
    """
    SUPPORTED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png"}

    def __init__(self):
        self.converter = DocumentConverter()

    def parse(self, file_path: Path) -> Document:
        """
        Parse a document using Docling.

        Args:
            file_path: Path of the document.

        Returns:
            A Document object containing extracted information.
        """

        file_path = Path(file_path)

        # 1. Check if the file exists
        if not file_path.exists():
            raise FileNotFoundError(
                f"File not found: {file_path}"
            )

        # 2. Check whether the file type is supported
        if file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {file_path.suffix}"
            )

        # 3. Process the document using Docling
        try:
            result = self.converter.convert(str(file_path))
        except Exception as exc:
            raise RuntimeError(
                f"Docling failed to convert {file_path.name}: {exc}"
            ) from exc

        # 4. Get the converted document
        document = result.document

        # 5. Export extracted content
        content = document.export_to_markdown()

        # 6. Create our application's Document object
        return Document(
            document_id=file_path.stem,
            file_name=file_path.name,
            file_type=file_path.suffix.lower().lstrip("."),
            source_path=str(file_path),
            content=content,
        )