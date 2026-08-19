from pathlib import Path

from .base_parser import BaseParser
from .docling_parser import DoclingParser


class ParserFactory:
    """
    Factory responsible for creating the appropriate parser for a
    given file.

    PERFORMANCE NOTE: parser instances are cached, because
    constructing a DoclingParser loads its OCR/layout models, which is
    expensive (several seconds). Without caching, every call to
    ParserFactory.create() — including every single /ingest upload
    from the Streamlit UI — would reload those models from scratch.
    """

    _parser_cache: dict[str, BaseParser] = {}

    @classmethod
    def create(cls, file_path: Path) -> BaseParser:
        """
        Create (or reuse) a parser based on the file extension.

        Args:
            file_path: Path of the document.

        Returns:
            A cached, reused parser object.
        """

        file_path = Path(file_path)
        extension = file_path.suffix.lower()

        if extension in DoclingParser.SUPPORTED_EXTENSIONS:
            if "docling" not in cls._parser_cache:
                cls._parser_cache["docling"] = DoclingParser()
            return cls._parser_cache["docling"]

        raise ValueError(
            f"No parser available for file type: {extension}"
        )