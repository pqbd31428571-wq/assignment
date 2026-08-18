from pathlib import Path

from .base_parser import BaseParser
from .docling_parser import DoclingParser


class ParserFactory:
    """
    Factory responsible for creating the appropriate
    parser for a given file.
    """

    @staticmethod
    def create(file_path: Path) -> BaseParser:
        """
        Create a parser based on the file extension.

        Args:
            file_path: Path of the document.

        Returns:
            An appropriate parser object.
        """

        file_path = Path(file_path)

        extension = file_path.suffix.lower()

        if extension in DoclingParser.SUPPORTED_EXTENSIONS:
            return DoclingParser()

        raise ValueError(
            f"No parser available for file type: {extension}"
        )