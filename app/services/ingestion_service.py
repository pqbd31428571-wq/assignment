from pathlib import Path

from app.models.document import Document
from app.parsers.parser_factory import ParserFactory


class IngestionService:
    """
    Service responsible for discovering supported documents
    and sending them to the appropriate parser.
    """

    SUPPORTED_EXTENSIONS = {
        ".pdf",
        ".jpg",
        ".jpeg",
        ".png",
    }

    def ingest_directory(
        self,
        directory: Path
    ) -> list[Document]:
        """
        Find and parse all supported documents inside a directory.

        Args:
            directory: Directory containing input documents.

        Returns:
            List of parsed Document objects.
        """

        directory = Path(directory)

        if not directory.exists():
            raise FileNotFoundError(
                f"Directory not found: {directory}"
            )

        documents = []

        for file_path in sorted(directory.rglob("*")):

            # Ignore directories
            if not file_path.is_file():
                continue

            # Ignore unsupported files
            if file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                continue

            print(f"Processing: {file_path}")

            # Get appropriate parser
            parser = ParserFactory.create(file_path)

            # Parse the document
            document = parser.parse(file_path)

            documents.append(document)

        return documents