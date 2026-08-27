import logging
from pathlib import Path

from .base_parser import BaseParser
from .docling_parser import DoclingParser

logger = logging.getLogger(__name__)


class ParserFactory:
    """
    Factory responsible for creating the appropriate parser for a
    given file. Caches parser instances since constructing a
    DoclingParser loads its OCR/layout models, which is expensive.
    """

    _parser_cache: dict[str, BaseParser] = {}

    @classmethod
    def create(cls, file_path: Path) -> BaseParser:
        file_path = Path(file_path)
        extension = file_path.suffix.lower()

        if extension in DoclingParser.SUPPORTED_EXTENSIONS:
            if "docling" not in cls._parser_cache:
                logger.info("Creating and caching a new DoclingParser instance.")
                cls._parser_cache["docling"] = DoclingParser()
            return cls._parser_cache["docling"]

        raise ValueError(f"No parser available for file type: {extension}")