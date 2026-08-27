import logging
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from app.models.document import Document
from app.parsers.parser_factory import ParserFactory

logger = logging.getLogger(__name__)


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
        directory = Path(directory)

        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        documents: List[Document] = []
        failed_files: List[Tuple[Path, str]] = []

        files = sorted(
            file_path
            for file_path in directory.rglob("*")
            if file_path.is_file()
            and file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS
        )

        logger.info("Found %d candidate file(s) in %s", len(files), directory)

        for file_path in files:

            if skip_if and skip_if(file_path):
                logger.info("Skipping already indexed: %s", file_path.name)
                continue

            logger.info("Processing: %s", file_path)

            try:
                parser = ParserFactory.create(file_path)
                document = parser.parse(file_path)
                documents.append(document)

            except Exception:
                # logger.exception automatically includes the full
                # traceback in the log — this is the single biggest
                # upgrade over print(): no more losing the stack
                # trace when a file fails to parse.
                logger.exception("Failed to parse %s", file_path.name)
                failed_files.append((file_path, "see traceback above"))

        if failed_files:
            logger.warning(
                "%d file(s) failed to parse: %s",
                len(failed_files),
                ", ".join(path.name for path, _ in failed_files),
            )

        logger.info(
            "Ingestion scan complete: %d parsed, %d failed",
            len(documents), len(failed_files),
        )

        return documents