from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseParser(ABC):
    """
    Abstract base class for all document parsers.

    Every parser in the application must implement
    the parse() method.
    """

    @abstractmethod
    def parse(self, file_path: Path) -> dict[str, Any]:
        """
        Parse a document and return the extracted information.

        Args:
            file_path: Path to the document that needs to be parsed.

        Returns:
            A dictionary containing the extracted document information.
        """
        pass