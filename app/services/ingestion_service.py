from pathlib import Path
from typing import Callable, List, Optional, Tuple

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
        directory: Path,
        skip_if: Optional[Callable[[Path], bool]] = None,
    ) -> List[Document]:
        """
        Find and parse all supported documents inside a directory
        (including subfolders, e.g. data/raw/pdf, data/raw/jpg).

        Args:
            directory: Directory containing input documents.
            skip_if: Optional predicate(file_path) -> bool. If it
                returns True, the file is skipped BEFORE parsing —
                so re-running bulk ingestion after adding a few new
                files doesn't re-OCR the ~100 already-indexed ones,
                only the new ones.

        Returns:
            List of successfully parsed Document objects. Files that
            fail to parse are logged and skipped rather than aborting
            the whole batch.
        """

        directory = Path(directory)

        if not directory.exists():
            raise FileNotFoundError(
                f"Directory not found: {directory}"
            )

        documents: List[Document] = []
        failed_files: List[Tuple[Path, str]] = []

        files = sorted(
            file_path
            for file_path in directory.rglob("*")
            if file_path.is_file()
            and file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS
        )

        for file_path in files:

            if skip_if and skip_if(file_path):
                print(f"Skipping already indexed: {file_path.name}")
                continue

            print(f"Processing: {file_path}")

            try:
                parser = ParserFactory.create(file_path)
                document = parser.parse(file_path)
                documents.append(document)

            except Exception as exc:
                print(f"Failed to parse {file_path.name}: {exc}")
                failed_files.append((file_path, str(exc)))

        if failed_files:
            print(f"\n{len(failed_files)} file(s) failed to parse:")
            for path, error in failed_files:
                print(f"  - {path.name}: {error}")

        return documents